"""Ollama worker adapter: a local model as an agentd worker.

Ollama only serves models, so this adapter is the agent loop: each turn it calls
`/api/chat` with tool definitions, runs the tool calls the model makes, feeds the results
back, and repeats until the model answers in text (at most --max-tool-rounds rounds).
The conversation is kept in memory across turns. The session loop (oneshot or
--interactive, room wakes, budgets, the closing handoff) is TurnAdapter's.

    worker_types:
      ollama:
        command: [/opt/agentd/.venv/bin/python, -m, agentd.workers.ollama,
                  --model, "qwen2.5:7b-instruct", --interactive]

Tools come only from the session's grant and profile; there is no shell and no web:

- room tools (rooms_read, rooms_send, rooms_note_get, rooms_note_put), when the session
  was invited into a room: the same functions and schemas as the room-tools MCP server,
  called in-process. A read-only invite (grant room.reply false) gets only the read tools.
- read_file / list_files, when a workspace is mounted (read or read_write), confined to it.
- write_artifact, when the workspace is mounted read_write: files go to the artifacts dir.

The model needs tool calling (e.g. qwen2.5, llama3.1+). Budgets are tokens: prompt plus
output tokens as Ollama reports them.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

import httpx

from agentd.workers import room_tools
from agentd.workers.claude_code import has_room, room_reply_allowed, system_prompt
from agentd.workers.common import emit
from agentd.workers.room_tools import RoomTools
from agentd.workers.turns import TurnAdapter, add_session_args

SUMMARY_LIMIT = 4000
PROGRESS_LIMIT = 2000
TOOL_RESULT_LIMIT = 20_000
READ_LIMIT = 100_000
LIST_LIMIT = 200


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + " …[truncated]"


class TurnError(Exception):
    """A turn can't continue (Ollama unreachable, model missing, ...)."""


class State:
    def __init__(self, model: str):
        self.model = model
        self.turns = 0
        self.usage = {"prompt_tokens": 0, "output_tokens": 0}
        self.last_result: str | None = None
        self.last_ok = False
        self.turn_done = False

    @property
    def total_tokens(self) -> int:
        return self.usage["prompt_tokens"] + self.usage["output_tokens"]

    def start_turn(self) -> None:
        self.turn_done = False

    def count(self, resp: dict) -> None:
        self.usage["prompt_tokens"] += int(resp.get("prompt_eval_count") or 0)
        self.usage["output_tokens"] += int(resp.get("eval_count") or 0)

    def result_fields(self) -> dict:
        return {
            "summary": _clip(self.last_result or "", SUMMARY_LIMIT),
            "model": self.model,
            "usage": dict(self.usage),
            "num_turns": self.turns,
        }


# ----- tools ---------------------------------------------------------------------------


def _schema(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


class Toolbox:
    """The tools this session may use, decided once from the grant and profile."""

    def __init__(self, env: dict[str, str], profile: dict):
        self.room: RoomTools | None = None
        self.read_only_room = False
        if has_room(env):
            self.room = RoomTools(
                env["ROOMSD_URL"],
                env["ROOMSD_ROOM_ID"],
                env["ROOMSD_TOKEN"],
                secrets=[v for k, v in env.items() if room_tools.is_secret_name(k)],
            )
            self.read_only_room = not room_reply_allowed(env)
        mode = env.get("AGENTD_WORKSPACE_MODE") or profile.get("workspace_mount")
        ws = env.get("AGENTD_WORKSPACE")
        self.workspace = Path(ws).resolve() if ws and mode in ("read", "read_write") else None
        art = env.get("AGENTD_ARTIFACTS_DIR")
        self.artifacts = Path(art).resolve() if art and mode == "read_write" else None
        self.schemas: list[dict] = []
        self.names: set[str] = set()
        self.sent: dict[str, int] = {}  # body -> message id, to refuse exact repeats

    async def load(self) -> None:
        schemas = []
        if self.room:
            server = room_tools.build_server(self.room, read_only=self.read_only_room)
            # Building the MCP server switches root logging to INFO on stderr, which would
            # turn every room poll into a log event. We only need its schemas.
            logging.getLogger().setLevel(logging.WARNING)
            for tool in await server.list_tools():
                schemas.append(
                    _schema(
                        tool.name,
                        tool.description or "",
                        tool.input_schema.get("properties", {}),
                        tool.input_schema.get("required", []),
                    )
                )
        if self.workspace:
            schemas.append(
                _schema(
                    "read_file",
                    "Read a text file from the workspace (path relative to the workspace).",
                    {"path": {"type": "string"}},
                    ["path"],
                )
            )
            schemas.append(
                _schema(
                    "list_files",
                    "List files and directories under a workspace path (default: the root).",
                    {"path": {"type": "string"}},
                    [],
                )
            )
        if self.artifacts:
            schemas.append(
                _schema(
                    "write_artifact",
                    "Save a deliverable file (report, patch) for the requester.",
                    {"name": {"type": "string"}, "content": {"type": "string"}},
                    ["name", "content"],
                )
            )
        self.schemas = schemas
        self.names = {s["function"]["name"] for s in schemas}
        self.params = {s["function"]["name"]: s["function"]["parameters"] for s in schemas}

    def _inside(self, root: Path, rel: str) -> Path:
        p = (root / rel).resolve()
        if not p.is_relative_to(root):
            where = "workspace" if root == self.workspace else "artifacts dir"
            raise ValueError(f"{rel!r} is outside the {where}")
        return p

    async def call(self, name: str, args: dict) -> str:
        """Run one tool call; failures come back as text the model can read."""
        if name not in self.names:
            return f"error: no tool named {name!r}"
        args = coerce(args, self.params.get(name) or {})
        try:
            if name == "rooms_send" and args.get("body") in self.sent:
                # Small models often don't take a tool result as "done" and post again.
                return (
                    f"error: you already posted exactly this as message "
                    f"#{self.sent[args['body']]}; don't repeat it. Answer in text if you're done."
                )
            if name.startswith("rooms_"):
                result = await asyncio.to_thread(getattr(self.room, name), **args)
                if name == "rooms_send":
                    self.sent[args["body"]] = result["id"]
                    return f"ok: posted to the room as message #{result['id']}."
                if name == "rooms_note_put" and result.get("written"):
                    return f"ok: note {result['key']!r} saved (revision {result['revision']})."
            elif name == "read_file":
                p = self._inside(self.workspace, args["path"])
                result = p.read_bytes()[:READ_LIMIT].decode(errors="replace")
            elif name == "list_files":
                p = self._inside(self.workspace, args.get("path") or ".")
                entries = sorted(f"{c.name}/" if c.is_dir() else c.name for c in p.iterdir())
                result = entries[:LIST_LIMIT]
            elif name == "write_artifact":
                p = self._inside(self.artifacts, args["name"])
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(args["content"])
                result = {"saved": str(p.relative_to(self.artifacts))}
            else:  # pragma: no cover - names is the allowlist
                return f"error: no tool named {name!r}"
        except Exception as e:  # noqa: BLE001 - the model reads the failure and adapts
            return f"error: {e}"
        text = result if isinstance(result, str) else json.dumps(result, default=str)
        return _clip(text, TOOL_RESULT_LIMIT)


def _types(prop: dict) -> set[str]:
    """JSON-schema types a property accepts (pydantic writes optionals as anyOf)."""
    out = {prop["type"]} if isinstance(prop.get("type"), str) else set(prop.get("type") or [])
    for alt in prop.get("anyOf") or []:
        out |= _types(alt)
    return out


def coerce(args: dict, schema: dict) -> dict:
    """Fix the type slips small models make, per the tool's schema: a bare string where a
    list is expected, numbers sent as strings. Anything else is passed through as is."""
    props = schema.get("properties") or {}
    out = dict(args)
    for key, value in args.items():
        types = _types(props.get(key) or {})
        if not isinstance(value, str) or "string" in types:
            continue
        if "array" in types:
            out[key] = [value]
        elif "integer" in types and value.strip().lstrip("-").isdigit():
            out[key] = int(value)
        elif "number" in types:
            try:
                out[key] = float(value)
            except ValueError:
                pass
        elif "boolean" in types and value.lower() in ("true", "false"):
            out[key] = value.lower() == "true"
    return out


def describe(name: str, args: dict) -> str:
    for key in ("path", "name", "body", "key", "after_id"):
        if isinstance(args.get(key), str | int):
            return f"{name}: {_clip(str(args[key]), 200)}"
    return name


# ----- the adapter process -------------------------------------------------------------


class Adapter(TurnAdapter):
    name = "ollama"

    def __init__(self, args: argparse.Namespace):
        super().__init__(args, State(args.model))
        self.tools = Toolbox(self.env, self.profile)
        self.messages: list[dict] = []
        self.announced = False

    def first_prompt(self, task: str) -> str:
        # Unlike Claude Code and Codex, a model served by Ollama isn't told what it is; in a
        # live room qwen2.5 introduced itself as "Ollama Model X (om-x)".
        preamble = system_prompt(self.profile, self.env) + (
            f" You are the model {self.args.model}, running locally through Ollama; if anyone "
            "asks which model you are, say exactly that. You act only through the tools you "
            "are given; you have no shell and no internet access. When you're done, answer "
            "in plain text."
        )
        self.messages = [{"role": "system", "content": preamble}]
        return task

    async def _chat(self, tools: list[dict] | None) -> dict:
        body = {
            "model": self.args.model,
            "messages": self.messages,
            "stream": False,
            "options": {"num_ctx": self.args.num_ctx},
        }
        if self.args.temperature is not None:
            body["options"]["temperature"] = self.args.temperature
        if tools:
            body["tools"] = tools
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(None, connect=10)) as client:
                r = await client.post(f"{self.args.url}/api/chat", json=body)
        except httpx.HTTPError as e:
            raise TurnError(f"cannot reach ollama at {self.args.url}: {e}") from e
        if r.status_code >= 400:
            try:
                detail = r.json().get("error", r.text)
            except ValueError:
                detail = r.text
            raise TurnError(f"ollama error {r.status_code}: {detail}")
        resp = r.json()
        self.state.count(resp)
        return resp

    async def run_turn(self, prompt: str) -> str | None:
        t = self.state
        if not self.announced:  # first turn: decide the tools once
            await self.tools.load()
            self.announced = True
            emit(
                "progress",
                message=f"ollama session (model {self.args.model}, tools: "
                f"{', '.join(sorted(self.tools.names)) or 'none'})",
            )
        self.messages.append({"role": "user", "content": prompt})
        try:
            for round_ in range(self.args.max_tool_rounds + 1):
                last = round_ == self.args.max_tool_rounds
                if last:
                    emit("progress", message="tool-call limit reached; asking for an answer")
                    self.messages.append(
                        {
                            "role": "user",
                            "content": "Tool-call limit for this turn reached. Answer now "
                            "with what you have, without calling tools.",
                        }
                    )
                resp = await self._chat(None if last else self.tools.schemas)
                msg = resp.get("message") or {}
                content = (msg.get("content") or "").strip()
                calls = [] if last else (msg.get("tool_calls") or [])
                self.messages.append(
                    {
                        "role": "assistant",
                        "content": msg.get("content") or "",
                        **({"tool_calls": calls} if calls else {}),
                    }
                )
                if content and calls:
                    emit("progress", message=_clip(content, PROGRESS_LIMIT))
                if not calls:
                    t.turns += 1
                    t.turn_done = True
                    t.last_ok = bool(content)
                    if content:
                        t.last_result = content
                        emit("progress", message=_clip(content, PROGRESS_LIMIT))
                        return None
                    emit("error", message="ollama gave an empty answer")
                    return None
                for call in calls:
                    fn = call.get("function") or {}
                    name = fn.get("name") or ""
                    args = fn.get("arguments") or {}
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except ValueError:
                            args = {}
                    emit("progress", message=describe(name, args), tool=name)
                    result = await self.tools.call(name, args if isinstance(args, dict) else {})
                    if result.startswith("error:"):
                        emit(
                            "progress",
                            message=f"{name} failed: {_clip(result[7:], 500)}",
                            tool=name,
                        )
                    self.messages.append({"role": "tool", "content": result, "tool_name": name})
        except TurnError as e:
            return str(e)
        return None  # pragma: no cover - the last round never has tool calls


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="agentd.workers.ollama")
    p.add_argument(
        "--url",
        default=(os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434").rstrip("/"),
        help="Ollama server (default $OLLAMA_HOST or http://127.0.0.1:11434)",
    )
    p.add_argument("--num-ctx", type=int, default=8192, help="context window per request")
    p.add_argument("--temperature", type=float)
    p.add_argument("--max-tool-rounds", type=int, default=8, help="tool-call rounds per turn")
    add_session_args(p, max_turn_seconds=300)
    args = p.parse_args(argv)
    if not args.model:
        p.error("--model is required (e.g. qwen2.5:7b-instruct)")
    if "://" not in args.url:
        args.url = f"http://{args.url}"
    return args


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(Adapter(parse_args(argv)).run())


if __name__ == "__main__":
    sys.exit(main())
