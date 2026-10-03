"""Claude Code worker adapter (agentd milestone 2).

Runs `claude -p` in streaming-JSON mode and translates between the two protocols:

    agentd stdin  task / message  ->  claude stdin   {"type": "user", ...} turns
    claude stdout system/assistant/result  ->  AGENT_EVENT progress / error / final

Configure as a worker type, e.g.

    worker_types:
      claude:
        command: [/opt/agentd/.venv/bin/python, -m, agentd.workers.claude_code,
                  --model, sonnet, --max-budget-usd, "2"]

Modes:
  oneshot (default)  the first successful result ends the session with `final`. Messages
                     sent while that turn is running join the same conversation.
  --interactive      after each result emit `needs_input` and wait for the next message.
                     A stop request ends the session with `final` (the last result).

Tool permissions come only from the agentd profile (AGENTD_PROFILE), never from the
caller. See `tool_policy`. Anything that would prompt for permission is denied, since no
one is there to answer.
"""

import argparse
import asyncio
import json
import os
import shlex
import sys
from pathlib import Path

from agentd.workers.common import announce_in_room, emit

READ_TOOLS = ["Read", "Glob", "Grep"]
WRITE_TOOLS = ["Edit", "Write", "NotebookEdit"]
WEB_TOOLS = ["WebSearch", "WebFetch"]
SUMMARY_LIMIT = 4000
PROGRESS_LIMIT = 2000


# ----- policy --------------------------------------------------------------------------


def tool_policy(profile: dict) -> list[str]:
    """Claude CLI flags for an agentd profile.

    - No shell unless the profile explicitly sets `shell: true`. Without it the worker runs
      `--restricted` (no Bash or other code-running tools, and file tools confined to the
      working directories).
    - Write tools only when the workspace is mounted read_write.
    - Web tools only when `network` is true.
    - `claude_tools` / `claude_permission_mode` in the profile override the derived values.
    """
    writable = profile.get("workspace_mount") == "read_write"
    shell = profile.get("shell") is True
    tools = list(READ_TOOLS)
    if writable:
        tools += WRITE_TOOLS
    if profile.get("network", True):
        tools += WEB_TOOLS
    if shell:
        tools.append("Bash")
    tools = profile.get("claude_tools") or tools

    mode = profile.get("claude_permission_mode") or ("acceptEdits" if writable else "dontAsk")
    flags = ["--tools", *tools, "--allowedTools", *tools, "--permission-mode", mode]
    if not shell:
        flags.append("--restricted")
    return flags


def build_command(args: argparse.Namespace, profile: dict, env: dict[str, str]) -> list[str]:
    cmd = shlex.split(args.claude) + [
        "-p",
        "--input-format",
        "stream-json",
        "--output-format",
        "stream-json",
        "--verbose",
        "--no-session-persistence",
        "--strict-mcp-config",
        "--permission-prompts",
        "none",
    ]
    if args.model:
        cmd += ["--model", args.model]
    budgets = [b for b in (args.max_budget_usd, profile.get("max_budget_usd")) if b]
    if budgets:
        cmd += ["--max-budget-usd", str(min(float(b) for b in budgets))]
    cmd += tool_policy(profile)
    artifacts = env.get("AGENTD_ARTIFACTS_DIR")
    if artifacts and profile.get("workspace_mount") == "read_write":
        cmd += ["--add-dir", artifacts]
    cmd += ["--append-system-prompt", system_prompt(profile, env)]
    return cmd


def system_prompt(profile: dict, env: dict[str, str]) -> str:
    lines = [
        f"You are a helper worker in agentd session {env.get('AGENTD_SESSION_ID')} on "
        f"{env.get('AGENTD_INSTANCE_ID')}, running unattended: nobody will answer questions "
        "mid-task, so make reasonable assumptions and state them in your final answer.",
        "Your final message is returned to the requester as the session summary.",
    ]
    if profile.get("workspace_mount") == "read_write" and env.get("AGENTD_ARTIFACTS_DIR"):
        lines.append(f"Put deliverable files (reports, patches) in {env['AGENTD_ARTIFACTS_DIR']}.")
    else:
        lines.append("You have read-only access; describe changes rather than making them.")
    return " ".join(lines)


def user_turn(text: str) -> bytes:
    msg = {"type": "user", "message": {"role": "user", "content": text}}
    return (json.dumps(msg) + "\n").encode()


# ----- translation ---------------------------------------------------------------------


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + " …[truncated]"


def describe_tool(name: str, inp: dict) -> str:
    for key in ("command", "file_path", "path", "pattern", "url", "query"):
        if isinstance(inp.get(key), str):
            return f"{name}: {_clip(inp[key], 200)}"
    return name


class Translator:
    """Maps Claude stream-json messages to agentd events. Pure, so it's unit-testable."""

    def __init__(self):
        self.claude_session_id: str | None = None
        self.last_result: dict | None = None
        self.turns = 0

    def handle(self, msg: dict) -> list[tuple[str, dict]]:
        kind = msg.get("type")
        if kind == "system" and msg.get("subtype") == "init":
            self.claude_session_id = msg.get("session_id")
            return [
                (
                    "progress",
                    {
                        "message": f"claude started (model {msg.get('model')})",
                        "claude_session_id": self.claude_session_id,
                        "tools": msg.get("tools"),
                    },
                )
            ]
        if kind == "assistant":
            events = []
            for block in (msg.get("message") or {}).get("content") or []:
                if block.get("type") == "text" and block.get("text", "").strip():
                    events.append(("progress", {"message": _clip(block["text"], PROGRESS_LIMIT)}))
                elif block.get("type") == "tool_use":
                    events.append(
                        (
                            "progress",
                            {
                                "message": describe_tool(
                                    block.get("name"), block.get("input") or {}
                                ),
                                "tool": block.get("name"),
                            },
                        )
                    )
            return events
        if kind == "result":
            self.turns += 1
            self.last_result = msg
            if msg.get("is_error") or msg.get("subtype") != "success":
                return [
                    (
                        "error",
                        {
                            "message": _clip(str(msg.get("result") or msg.get("subtype")), 2000),
                            "subtype": msg.get("subtype"),
                        },
                    )
                ]
            return []
        return []

    def result_fields(self) -> dict:
        r = self.last_result or {}
        return {
            "summary": _clip(r.get("result") or "", SUMMARY_LIMIT),
            "claude_session_id": self.claude_session_id,
            "cost_usd": r.get("total_cost_usd"),
            "num_turns": r.get("num_turns"),
            "duration_ms": r.get("duration_ms"),
        }

    @property
    def last_ok(self) -> bool:
        r = self.last_result
        return bool(r) and not r.get("is_error") and r.get("subtype") == "success"


# ----- the adapter process -------------------------------------------------------------


class Adapter:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.translator = Translator()
        self.claude: asyncio.subprocess.Process | None = None
        self.in_turn = False
        self.finishing = False  # oneshot: the task's result is in; no more turns
        self.stop_requested = False

    async def run(self) -> int:
        emit("progress", message="starting claude")
        gateway = await _stdin_reader()
        first = await _read_json(gateway)
        if not first or first.get("type") != "task":
            emit("error", message="expected a task message first")
            return 2
        announce_in_room()

        profile = json.loads(os.environ.get("AGENTD_PROFILE") or "{}")
        cmd = build_command(self.args, profile, dict(os.environ))
        try:
            self.claude = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                limit=16 * 1024 * 1024,  # stream-json lines can carry whole tool results
            )
        except OSError as e:
            emit("error", message=f"cannot start claude ({shlex.split(self.args.claude)[0]}): {e}")
            return 2
        await self._send(first["task"])

        output = asyncio.create_task(self._pump_claude())
        inbox = asyncio.create_task(self._pump_gateway(gateway))
        await asyncio.wait({output, inbox}, return_when=asyncio.FIRST_COMPLETED)
        inbox.cancel()
        code = await self._finish(output)
        return code

    async def _send(self, text: str) -> None:
        if self.finishing or not (self.claude and self.claude.stdin):
            emit("progress", message="message arrived after the task finished; not delivered")
            return
        if not self.claude.stdin.is_closing():
            self.in_turn = True
            self.claude.stdin.write(user_turn(text))
            await self.claude.stdin.drain()

    async def _pump_gateway(self, gateway: asyncio.StreamReader) -> None:
        while (msg := await _read_json(gateway)) is not None:
            if msg.get("type") == "message":
                await self._send(f"[message from {msg.get('sender')}] {msg.get('message')}")
            elif msg.get("type") == "stop":
                break
        self.stop_requested = True
        if self.in_turn and self.claude and self.claude.returncode is None:
            self.claude.terminate()  # don't wait for the turn to finish
        else:
            self._close_claude_stdin()

    async def _pump_claude(self) -> None:
        assert self.claude and self.claude.stdout
        while line := await self.claude.stdout.readline():
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                print(line.decode(errors="replace").rstrip(), file=sys.stderr, flush=True)
                continue
            for kind, fields in self.translator.handle(msg):
                emit(kind, **fields)
            if msg.get("type") == "result":
                self.in_turn = False
                self._write_result_artifact()
                if not self.args.interactive or not self.translator.last_ok:
                    self.finishing = True
                    self._close_claude_stdin()  # done: let claude exit
                else:
                    emit(
                        "needs_input",
                        question=self.translator.result_fields()["summary"],
                        turn=self.translator.turns,
                    )

    def _close_claude_stdin(self) -> None:
        if self.claude and self.claude.stdin and not self.claude.stdin.is_closing():
            self.claude.stdin.close()

    def _write_result_artifact(self) -> None:
        text = (self.translator.last_result or {}).get("result")
        artifacts = os.environ.get("AGENTD_ARTIFACTS_DIR")
        if not text or not artifacts:
            return
        Path(artifacts, "result.md").write_text(text)
        emit("artifact", name="result.md", path="result.md", mime_type="text/markdown")

    async def _finish(self, output: asyncio.Task) -> int:
        assert self.claude
        try:
            code = await asyncio.wait_for(self.claude.wait(), self.args.exit_grace_seconds)
        except TimeoutError:
            self.claude.terminate()
            code = await self.claude.wait()
        await output
        self._report_new_artifacts()
        t = self.translator
        if t.last_ok and (not self.args.interactive or self.stop_requested):
            emit("final", **t.result_fields())
            return 0
        if self.stop_requested and t.last_result is None:
            return 0  # stopped mid-turn: agentd records the stop, there's nothing to report
        if t.last_result is None:
            emit("error", message=f"claude exited with code {code} before producing a result")
        return code or 1

    def _report_new_artifacts(self) -> None:
        """Announce files claude wrote into the artifacts dir (besides result.md)."""
        artifacts = os.environ.get("AGENTD_ARTIFACTS_DIR")
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
    p = argparse.ArgumentParser(prog="agentd.workers.claude_code")
    p.add_argument("--claude", default="claude", help="claude command (shell-split)")
    p.add_argument("--model")
    p.add_argument("--max-budget-usd", type=float, help="cap per session; profile may lower it")
    p.add_argument("--interactive", action="store_true", help="wait for messages between turns")
    p.add_argument("--exit-grace-seconds", type=float, default=30)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(Adapter(parse_args(argv)).run())


if __name__ == "__main__":
    sys.exit(main())
