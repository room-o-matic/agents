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
import shlex
import sys

from agentd.workers import room_tools
from agentd.workers.claude_code import has_room, load_grant, room_reply_allowed, system_prompt
from agentd.workers.common import emit
from agentd.workers.turns import TurnAdapter, add_session_args, token_budget

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


class Adapter(TurnAdapter):
    """One `codex exec` process per turn; the session loop is TurnAdapter's."""

    name = "codex"

    def __init__(self, args: argparse.Namespace):
        super().__init__(args, Translator(args.model))
        self.proc: asyncio.subprocess.Process | None = None

    def first_prompt(self, task: str) -> str:
        # Codex has no system-prompt flag: agentd's rules ride on the first turn.
        preamble = system_prompt(self.profile, self.env)
        return f"<session-instructions>\n{preamble}\n</session-instructions>\n\n{task}"

    def interrupt(self) -> None:
        if self.proc and self.proc.returncode is None:
            self.proc.terminate()

    async def run_turn(self, prompt: str) -> str | None:
        cmd = build_command(self.args, self.profile, self.env, self.cwd, self.state.thread_id)
        try:
            self.proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                env=worker_env(self.env),
                limit=16 * 1024 * 1024,
            )
        except OSError as e:
            return f"cannot start codex ({shlex.split(self.args.codex)[0]}): {e}"
        try:
            self.proc.stdin.write(prompt.encode())
            await self.proc.stdin.drain()
            self.proc.stdin.close()
            await self._pump(self.proc)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except asyncio.CancelledError:
            self.interrupt()
            raise
        code = await self.proc.wait()
        return f"codex exited with code {code} before finishing the turn"

    async def _pump(self, proc: asyncio.subprocess.Process) -> None:
        assert proc.stdout
        while line := await proc.stdout.readline():
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                print(line.decode(errors="replace").rstrip(), file=sys.stderr, flush=True)
                continue
            for kind, fields in self.state.handle(msg):
                emit(kind, **fields)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="agentd.workers.codex")
    p.add_argument("--codex", default="codex", help="codex command (shell-split)")
    p.add_argument(
        "--external-sandbox",
        action="store_true",
        help="only with agentd's sandbox backend: skip Codex's own sandbox, which can't nest",
    )
    add_session_args(p, max_turn_seconds=600)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(Adapter(parse_args(argv)).run())


__all__ = ["Adapter", "Translator", "build_command", "parse_args", "token_budget"]

if __name__ == "__main__":
    sys.exit(main())
