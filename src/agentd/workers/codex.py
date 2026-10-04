"""Codex CLI worker adapter.

Runs `codex exec --json` and translates between the two protocols. Codex takes one prompt
per process, so every turn is its own process: the first starts a thread, later turns
`codex exec resume <thread_id>`. Prompts go in on stdin, never in argv.

    agentd stdin  task / message  ->  one `codex exec` (or `resume`) process per turn
    codex stdout  thread / turn / item JSONL  ->  AGENT_EVENT progress / error / final

Configure as a worker type, e.g.

    worker_types:
      codex:
        command: [/opt/agentd/.venv/bin/python, -m, agentd.workers.codex,
                  --codex, /usr/local/bin/codex, --model, gpt-5-codex]

Modes are those of the Claude adapter: oneshot by default (the first successful turn ends
the session), or --interactive (needs_input after each turn; on stop, one bounded closing
turn writes the room handoff).

Permissions come only from the session's grant and profile (`sandbox_flags`), never from
the caller or the room:
- Codex's own sandbox confines its commands: `read-only` unless the workspace is mounted
  read_write (`workspace-write`, with network only when the profile allows it). Codex
  always has a shell inside that sandbox; there is no Claude-style "no shell" mode.
- approvals are off (`never`): nobody is there to answer, so anything outside the sandbox
  is refused.
- the operator's personal Codex config is ignored (`--ignore-user-config`; auth still
  comes from CODEX_HOME), and Codex's shell commands see only a minimal environment.
- budgets are tokens, not dollars (`--max-total-tokens`, profile `max_total_tokens`):
  uncached input plus output across the session, which ends once a turn takes it over.
"""

import argparse
import asyncio
import json
import os
import shlex
import sys
from pathlib import Path

import httpx

from agentd.workers import room_tools
from agentd.workers.claude_code import (
    CLOSING_PROMPT,
    RoomWatcher,
    frame_owner,
    frame_room,
    has_room,
    load_grant,
    room_reply_allowed,
    system_prompt,
)
from agentd.workers.common import announce_in_room, emit, join_room, post_room_status
from agentd.workers.wake import WakeGate

SUMMARY_LIMIT = 4000
PROGRESS_LIMIT = 2000
SANDBOXES = ("read-only", "workspace-write")


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + " …[truncated]"


def _toml(value) -> str:
    """A `-c key=value` value. JSON strings and arrays of strings are valid TOML."""
    return json.dumps(value)


# ----- policy --------------------------------------------------------------------------


def sandbox_flags(profile: dict, env: dict[str, str], external: bool = False) -> list[str]:
    """Codex sandbox flags for an agentd profile (see the module docstring)."""
    writable = profile.get("workspace_mount") == "read_write"
    mode = profile.get("codex_sandbox") or ("workspace-write" if writable else "read-only")
    if mode not in SANDBOXES:
        raise ValueError(f"codex_sandbox must be one of {SANDBOXES}, not {mode!r}")
    if external:
        # agentd's sandbox backend already confines the whole worker, and Codex's own
        # sandbox can't nest inside bubblewrap. Only for backend: sandbox.
        flags = ["--dangerously-bypass-approvals-and-sandbox"]
    else:
        flags = ["--sandbox", mode, "-c", 'approval_policy="never"']
        if mode == "workspace-write":
            network = bool(profile.get("network", True)) and load_grant(env).get("network", True)
            flags += ["-c", f"sandbox_workspace_write.network_access={str(network).lower()}"]
    artifacts = env.get("AGENTD_ARTIFACTS_DIR")
    if artifacts and writable:
        flags += ["--add-dir", artifacts]
    # Commands Codex runs see only core variables: never the room token or other secrets.
    flags += ["-c", 'shell_environment_policy.inherit="core"']
    return flags


def room_server_flags(env: dict[str, str]) -> list[str]:
    """The room-tools MCP server. Its env is passed by *name* (`env_vars`), so the invite
    token and secrets never appear in argv or in a config file."""
    names = ["ROOMSD_URL", "ROOMSD_ROOM_ID", "ROOMSD_TOKEN", "PATH", "HOME"]
    if not room_reply_allowed(env):
        names.append("ROOMSD_READ_ONLY")
    names += sorted(k for k in env if room_tools.is_secret_name(k))  # scanned, never posted
    server = f"mcp_servers.{room_tools.SERVER_NAME}"
    return [
        "-c",
        f"{server}.command={_toml(sys.executable)}",
        "-c",
        f"{server}.args={_toml(['-m', 'agentd.workers.room_tools'])}",
        "-c",
        f"{server}.env_vars={_toml(names)}",
        # Codex asks approval for every MCP call, and with approvals off it refuses them.
        # Room access was granted by the invite (read-only invites expose only read tools),
        # so this one server is pre-approved; nothing else is.
        "-c",
        f'{server}.default_tools_approval_mode="approve"',
    ]


def token_budget(args: argparse.Namespace, profile: dict) -> int | None:
    caps = [c for c in (args.max_total_tokens, profile.get("max_total_tokens")) if c]
    return min(int(c) for c in caps) if caps else None


def build_command(
    args: argparse.Namespace,
    profile: dict,
    env: dict[str, str],
    cwd: str,
    thread_id: str | None = None,
) -> list[str]:
    cmd = shlex.split(args.codex) + [
        "exec",
        "--json",
        "--skip-git-repo-check",
        "--ignore-user-config",
        "--color",
        "never",
        "-C",
        cwd,
    ]
    if args.model:
        cmd += ["--model", args.model]
    cmd += sandbox_flags(profile, env, external=args.external_sandbox)
    if has_room(env):
        cmd += room_server_flags(env)
    if thread_id:
        cmd += ["resume", thread_id]
    return [*cmd, "-"]  # the prompt comes on stdin


def worker_env(env: dict[str, str]) -> dict[str, str]:
    out = dict(env)
    if not room_reply_allowed(env):
        out["ROOMSD_READ_ONLY"] = "1"
    return out


# ----- translation ---------------------------------------------------------------------


class Translator:
    """Maps `codex exec --json` events to agentd events. Pure, so it's unit-testable."""

    def __init__(self, model: str | None = None):
        self.model = model
        self.thread_id: str | None = None
        self.turns = 0
        self.usage = {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0}
        self.turn_text: str | None = None  # last agent message of the current turn
        self.last_result: str | None = None  # last successful turn's answer
        self.last_ok = False
        self.turn_done = False

    @property
    def total_tokens(self) -> int:
        """What the budget counts: new input plus output. Codex resends the whole thread
        every turn, so raw input_tokens is mostly cache hits (about 90% in live runs) and
        would exhaust a budget on context that costs almost nothing."""
        u = self.usage
        return u["input_tokens"] - u["cached_input_tokens"] + u["output_tokens"]

    def start_turn(self) -> None:
        self.turn_text = None
        self.turn_done = False

    def handle(self, msg: dict) -> list[tuple[str, dict]]:
        kind = msg.get("type")
        if kind == "thread.started":
            first = self.thread_id is None
            self.thread_id = msg.get("thread_id")
            if not first:
                return []  # every resumed turn repeats it
            model = f", model {self.model}" if self.model else ""
            return [
                (
                    "progress",
                    {
                        "message": f"codex started (thread {self.thread_id}{model})",
                        "codex_thread_id": self.thread_id,
                    },
                )
            ]
        if kind == "item.started":
            return self._item_started(msg.get("item") or {})
        if kind == "item.completed":
            return self._item_completed(msg.get("item") or {})
        if kind == "turn.completed":
            self.turns += 1
            self.turn_done = True
            for k in self.usage:
                self.usage[k] += int((msg.get("usage") or {}).get(k) or 0)
            self.last_ok = self.turn_text is not None
            if self.last_ok:
                self.last_result = self.turn_text
                return []
            return [("error", {"message": "codex finished the turn without an answer"})]
        if kind == "turn.failed":
            self.turns += 1
            self.turn_done = True
            self.last_ok = False
            err = (msg.get("error") or {}).get("message") or "turn failed"
            return [("error", {"message": _clip(str(err), 2000), "subtype": "turn_failed"})]
        if kind == "error":
            return [("error", {"message": _clip(str(msg.get("message")), 2000)})]
        return []

    def _item_started(self, item: dict) -> list[tuple[str, dict]]:
        t = item.get("type")
        if t == "command_execution":
            return [
                (
                    "progress",
                    {"message": f"$ {_clip(str(item.get('command')), 300)}", "tool": "shell"},
                )
            ]
        if t == "mcp_tool_call":
            return [
                (
                    "progress",
                    {
                        "message": f"{item.get('tool')}",
                        "tool": f"mcp__{item.get('server')}__{item.get('tool')}",
                    },
                )
            ]
        return []

    def _item_completed(self, item: dict) -> list[tuple[str, dict]]:
        t = item.get("type")
        if t == "agent_message" and str(item.get("text", "")).strip():
            self.turn_text = item["text"]
            return [("progress", {"message": _clip(item["text"], PROGRESS_LIMIT)})]
        if t == "command_execution":
            code = item.get("exit_code")
            if code not in (0, None):
                return [
                    (
                        "progress",
                        {
                            "message": f"command exited {code}: "
                            f"{_clip(str(item.get('command')), 200)}",
                            "tool": "shell",
                        },
                    )
                ]
            return []
        if t == "mcp_tool_call" and (item.get("status") == "failed" or item.get("error")):
            err = item.get("error")
            err = err.get("message") if isinstance(err, dict) else err
            return [
                (
                    "progress",
                    {
                        "message": f"{item.get('tool')} failed: {_clip(str(err), 500)}",
                        "tool": f"mcp__{item.get('server')}__{item.get('tool')}",
                    },
                )
            ]
        if t == "file_change":
            paths = [c.get("path") for c in item.get("changes") or [] if isinstance(c, dict)]
            return [("progress", {"message": "changed: " + ", ".join(map(str, paths))[:500]})]
        if t == "web_search":
            return [
                (
                    "progress",
                    {
                        "message": f"web_search: {_clip(str(item.get('query')), 200)}",
                        "tool": "web_search",
                    },
                )
            ]
        if t == "error":
            return [
                ("progress", {"message": f"codex warning: {_clip(str(item.get('message')), 500)}"})
            ]
        return []

    def result_fields(self) -> dict:
        return {
            "summary": _clip(self.last_result or "", SUMMARY_LIMIT),
            "codex_thread_id": self.thread_id,
            "usage": dict(self.usage),
            "num_turns": self.turns,
        }


# ----- the adapter process -------------------------------------------------------------


class Adapter:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.translator = Translator(args.model)
        self.env = dict(os.environ)
        self.profile = json.loads(self.env.get("AGENTD_PROFILE") or "{}")
        self.cwd = os.getcwd()
        self.budget = token_budget(args, self.profile)
        self.inbox: asyncio.Queue[str] = asyncio.Queue()
        self.proc: asyncio.subprocess.Process | None = None
        self.in_turn = False
        self.stop_requested = False
        self.interrupted = False  # a stop arrived mid-turn: no closing summary
        self.finishing = False
        self.watcher: RoomWatcher | None = None
        self.gate: WakeGate | None = None
        self.preamble = system_prompt(self.profile, self.env)

    # ----- one turn = one codex process --------------------------------------------

    async def _turn(self, prompt: str, timeout: float | None = None, quiet: bool = False) -> bool:
        t = self.translator
        t.start_turn()
        cmd = build_command(self.args, self.profile, self.env, self.cwd, t.thread_id)
        try:
            self.proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                env=worker_env(self.env),
                limit=16 * 1024 * 1024,
            )
        except OSError as e:
            emit("error", message=f"cannot start codex ({shlex.split(self.args.codex)[0]}): {e}")
            return False
        self.in_turn = True
        try:
            self.proc.stdin.write(prompt.encode())
            await self.proc.stdin.drain()
            self.proc.stdin.close()
            await asyncio.wait_for(self._pump(self.proc), timeout)
        except TimeoutError:
            self.proc.terminate()
            quiet = True
            emit("progress", message="codex turn timed out")
        except (BrokenPipeError, ConnectionResetError):
            pass
        code = await self.proc.wait()
        self.in_turn = False
        if not t.turn_done and not self.stop_requested and not quiet:
            emit("error", message=f"codex exited with code {code} before finishing the turn")
            t.last_ok = False
        return t.turn_done and t.last_ok

    async def _pump(self, proc: asyncio.subprocess.Process) -> None:
        assert proc.stdout
        while line := await proc.stdout.readline():
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                print(line.decode(errors="replace").rstrip(), file=sys.stderr, flush=True)
                continue
            for kind, fields in self.translator.handle(msg):
                emit(kind, **fields)

    def _write_result_artifact(self) -> None:
        text = self.translator.last_result
        artifacts = self.env.get("AGENTD_ARTIFACTS_DIR")
        if text and artifacts:
            Path(artifacts, "result.md").write_text(text)
            emit("artifact", name="result.md", path="result.md", mime_type="text/markdown")

    def _over_budget(self) -> bool:
        if self.budget and self.translator.total_tokens >= self.budget:
            emit(
                "progress",
                message=f"token budget reached ({self.translator.total_tokens} of "
                f"{self.budget}); ending the session",
                usage=dict(self.translator.usage),
            )
            return True
        return False

    # ----- input: owner messages, room wakes, stop -----------------------------------

    async def _pump_gateway(self, gateway: asyncio.StreamReader) -> None:
        while (line := await gateway.readline()) and not self.stop_requested:
            msg = json.loads(line)
            if msg.get("type") == "message":
                await self._enqueue(frame_owner(msg.get("sender"), msg.get("message")))
            elif msg.get("type") == "stop":
                break
        self.stop_requested = True
        if self.in_turn and self.proc and self.proc.returncode is None:
            self.interrupted = True
            self.proc.terminate()  # don't wait for the turn to finish
        await self.inbox.put("")  # wake the main loop

    async def _enqueue(self, text: str) -> None:
        if self.finishing:
            emit("progress", message="message arrived after the task finished; not delivered")
            return
        await self.inbox.put(text)

    def _gate(self) -> WakeGate:
        if self.gate is None:
            self.gate = WakeGate(
                identity=self.watcher.identity,
                policy=self.args.room_wake,
                max_hop=self.args.room_max_hop,
                max_wakes=self.args.room_max_wakes,
                max_queue=self.args.room_max_queue,
            )
        return self.gate

    async def _room_turn(self, m: dict) -> None:
        if self.finishing or self.stop_requested:
            return
        gate = self._gate()
        action, reason = gate.offer(m, in_turn=self.in_turn or not self.inbox.empty())
        if action == "deliver":
            emit(
                "progress",
                message=f"room message #{m['id']} from {m['from']}",
                room_message_id=m["id"],
            )
            await self._enqueue(frame_room(m))
        elif action == "queue":
            emit("progress", message=f"queued room message #{m['id']}", room_message_id=m["id"])
        elif action == "suppress":
            emit(
                "progress",
                message=f"room wake suppressed ({reason})",
                room_wake_suppressed={"reason": reason, "count": gate.suppressed[reason]},
                room_message_id=m["id"],
            )
            if reason == "wake_budget_exhausted" and gate.suppressed[reason] == 1:
                post_room_status(
                    "wake budget exhausted: I'll stop responding to room messages in this "
                    "session; my owner can stop or re-summon me."
                )

    def _drain_room_queue(self) -> str | None:
        if self.gate is None:
            return None
        batch = self.gate.drain()
        if not batch:
            return None
        emit(
            "progress",
            message=f"coalesced {len(batch)} room messages into one turn",
            room_message_ids=[m["id"] for m in batch],
        )
        return "\n\n".join(frame_room(m) for m in batch)

    def _pending(self) -> list[str]:
        texts = []
        while not self.inbox.empty():
            if text := self.inbox.get_nowait():
                texts.append(text)
        if queued := self._drain_room_queue():
            texts.append(queued)
        return texts

    # ----- the session ---------------------------------------------------------------

    async def run(self) -> int:
        emit("progress", message="starting codex")
        gateway = await _stdin_reader()
        first = await _read_json(gateway)
        if not first or first.get("type") != "task":
            emit("error", message="expected a task message first")
            return 2
        # Join, fix where room wakes start, then announce (see the Claude adapter).
        if join_room() is not None and self.args.room_wake != "none":
            self.watcher = RoomWatcher.from_env(self.args)
            try:
                await self.watcher.start()
            except httpx.HTTPError as e:
                emit("error", message=f"room watcher stopped: {e}")
                self.watcher = None
        announce_in_room(join=False)

        inbox = asyncio.create_task(self._pump_gateway(gateway))
        watcher = asyncio.create_task(self.watcher.run(self._room_turn)) if self.watcher else None
        try:
            return await self._session(first["task"])
        finally:
            inbox.cancel()
            if watcher:
                watcher.cancel()
            self._report_new_artifacts()

    async def _session(self, task: str) -> int:
        t = self.translator
        prompt = f"<session-instructions>\n{self.preamble}\n</session-instructions>\n\n{task}"
        while True:
            ok = await self._turn(prompt)
            if ok:
                self._write_result_artifact()
            if self.stop_requested:
                break
            if not ok:
                return 1  # the error is reported; agentd marks the session failed
            if self._over_budget():
                break
            pending = self._pending()
            if pending:  # messages that arrived during the turn: answer them first
                prompt = "\n\n".join(pending)
                continue
            if not self.args.interactive:
                break
            emit("needs_input", question=t.result_fields()["summary"], turn=t.turns)
            text = await self.inbox.get()
            if self.stop_requested or not text:
                break
            prompt = "\n\n".join([text, *self._pending()])
        self.finishing = True
        if self.stop_requested and self._wants_closing_summary():
            before = (t.last_result, t.last_ok)
            emit("progress", message="writing a closing summary for the handoff")
            self.stop_requested = False  # let this one turn run
            ok = await self._turn(
                CLOSING_PROMPT, timeout=self.args.closing_summary_seconds, quiet=True
            )
            self.stop_requested = True
            if not ok:
                t.last_result, t.last_ok = before
        if t.last_result is None:
            return 0 if self.stop_requested else 1
        emit("final", **t.result_fields())
        return 0

    def _wants_closing_summary(self) -> bool:
        return (
            self.args.interactive
            and not self.interrupted
            and self.args.closing_summary_seconds > 0
            and self.translator.last_ok
            and self.translator.turns > 0
            and not self._over_budget_silent()
        )

    def _over_budget_silent(self) -> bool:
        return bool(self.budget and self.translator.total_tokens >= self.budget)

    def _report_new_artifacts(self) -> None:
        artifacts = self.env.get("AGENTD_ARTIFACTS_DIR")
        if not artifacts:
            return
        root = Path(artifacts)
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            rel = str(path.relative_to(root))
            if rel != "result.md":
                emit("artifact", name=rel, path=rel)


async def _stdin_reader() -> asyncio.StreamReader:
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader(limit=16 * 1024 * 1024)
    await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), sys.stdin)
    return reader


async def _read_json(reader: asyncio.StreamReader) -> dict | None:
    line = await reader.readline()
    return json.loads(line) if line else None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="agentd.workers.codex")
    p.add_argument("--codex", default="codex", help="codex command (shell-split)")
    p.add_argument("--model")
    p.add_argument(
        "--max-total-tokens",
        type=int,
        help="uncached input + output tokens per session; profile may lower it",
    )
    p.add_argument("--interactive", action="store_true", help="wait for messages between turns")
    p.add_argument(
        "--external-sandbox",
        action="store_true",
        help="only with agentd's sandbox backend: skip Codex's own sandbox, which can't nest",
    )
    p.add_argument(
        "--closing-summary-seconds",
        type=float,
        default=8,
        help="interactive: on stop, give codex this long to write the room handoff (0 = off);"
        " keep it under the gateway's stop_grace_seconds",
    )
    p.add_argument("--room-wake", choices=["mentions", "all", "none"], default="mentions")
    p.add_argument("--room-poll-seconds", type=float, default=3.0)
    p.add_argument("--room-max-hop", type=int, default=3)
    p.add_argument("--room-max-wakes", type=int, default=10)
    p.add_argument("--room-max-queue", type=int, default=20)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(Adapter(parse_args(argv)).run())


if __name__ == "__main__":
    sys.exit(main())
