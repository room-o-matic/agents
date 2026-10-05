"""Ask a configured agent a question, with nothing else running.

A local agent is a worker type plus a profile, and optionally a workspace (e.g. a clean
clone of a repo used as a read-only knowledge base). `ask()` starts that worker as a child
process, exactly as the gateway would (same environment, same grant, same backends), sends
it one task, and returns its final answer. There's no lobbyd, roomsd or agentd service, no
port, and no database: the worker exists only while it answers.

    # ~/.config/agentd/agents.yaml (or $AGENTD_AGENTS)
    worker_types:
      claude: {command: [python, -m, agentd.workers.claude_code, --model, sonnet]}
    profiles:
      knowledge_read: {max_runtime_minutes: 10, workspace_mount: read, network: true,
                       claude_tools: [Read, Grep, Glob]}  # network: the worker's model
    agents:
      openvpn: {worker_type: claude, profile: knowledge_read, workspace: ~/kb/openvpn,
                description: "Answers questions about the OpenVPN network"}

`agentd ask <agent> <question>` and the `agentd mcp` server (for Claude Code) both use it.
The agents file is the whole policy: only the agents it names can run, with the profile it
gives them. Answers go back to the caller only; nothing is posted to a room.
"""

import asyncio
import getpass
import os
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

import yaml
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from agentd.backends.process import ProcessBackend, ProcessHandle, iter_lines
from agentd.backends.sandbox import LaunchSpec, SandboxBackend
from agentd.config import Profile, SandboxSettings, WorkerType
from agentd.ids import new_id
from agentd.protocol import parse_stdout_line
from agentd.supervisor import worker_environment

DEFAULT_PATH = Path("~/.config/agentd/agents.yaml")
MAX_LINE_BYTES = 1024 * 1024
STDERR_TAIL = 2000
CLIPPED = " …[truncated]"  # how adapters mark a summary cut to fit a room message
FULL_ANSWER_LIMIT = 256 * 1024


class LocalAgent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    worker_type: str
    profile: str
    workspace: Path | None = None
    description: str = ""


class AgentsFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    worker_types: dict[str, WorkerType]
    profiles: dict[str, Profile]
    agents: dict[str, LocalAgent] = Field(min_length=1)
    backend: Literal["process", "sandbox"] = "process"
    sandbox: SandboxSettings = SandboxSettings()
    env_allowlist: list[str] = ["PATH", "HOME", "LANG", "LC_ALL", "TZ"]
    stop_grace_seconds: float = Field(default=5.0, gt=0)

    @model_validator(mode="after")
    def _consistent(self):
        for name, a in self.agents.items():
            if a.worker_type not in self.worker_types:
                raise ValueError(f"agent {name!r}: unknown worker_type {a.worker_type!r}")
            if a.profile not in self.profiles:
                raise ValueError(f"agent {name!r}: unknown profile {a.profile!r}")
            mount = self.profiles[a.profile].workspace_mount
            if a.workspace and mount == "none":
                raise ValueError(f"agent {name!r}: profile {a.profile!r} doesn't mount a workspace")
            if self.profiles[a.profile].external_actions == "approval_required":
                raise ValueError(f"agent {name!r}: approval_required profiles can't run")
        return self


class AskError(Exception):
    """The agent couldn't answer: why, in words a person (or a model) can act on."""


def default_path() -> Path:
    return Path(os.environ.get("AGENTD_AGENTS") or DEFAULT_PATH).expanduser()


def load(path: Path | None = None) -> AgentsFile:
    path = path or default_path()
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except FileNotFoundError:
        raise AskError(f"no agents file at {path} (set AGENTD_AGENTS or create it)") from None
    return AgentsFile.model_validate(raw)


def describe(cfg: AgentsFile) -> list[dict]:
    return [
        {
            "agent": name,
            "description": a.description,
            "worker_type": a.worker_type,
            "profile": a.profile,
            "workspace": str(a.workspace.expanduser()) if a.workspace else None,
        }
        for name, a in cfg.agents.items()
    ]


def task_text(question: str, context: str | None) -> str:
    if not context:
        return question
    return f"{question}\n\nContext from earlier in this conversation:\n{context}"


async def ask(
    cfg: AgentsFile,
    name: str,
    question: str,
    *,
    context: str | None = None,
    timeout: float | None = None,
    on_event: Callable[[dict], None] | None = None,
) -> dict:
    """Run agent `name` on one question and return {agent, answer, cost_usd, session_id}.
    Raises AskError when it can't answer. The worker's whole process group is gone by the
    time this returns, whatever happened."""
    agent = cfg.agents.get(name)
    if agent is None:
        known = ", ".join(sorted(cfg.agents)) or "none"
        raise AskError(f"no agent named {name!r} (known: {known})")
    if not question.strip():
        raise AskError("the question is empty")
    profile = cfg.profiles[agent.profile]
    worker = cfg.worker_types[agent.worker_type]
    workspace = None
    if agent.workspace:
        workspace = agent.workspace.expanduser().resolve()
        if not workspace.is_dir():
            raise AskError(f"agent {name!r}: workspace {str(workspace)!r} is not a directory")
    limit = profile.max_runtime_minutes * 60
    timeout = min(timeout, limit) if timeout else limit

    session_id = new_id("ask")
    with tempfile.TemporaryDirectory(prefix="agentd-ask-") as tmp:
        scratch, artifacts = Path(tmp) / "scratch", Path(tmp) / "artifacts"
        scratch.mkdir()
        artifacts.mkdir()
        expires = datetime.now(UTC) + timedelta(seconds=timeout)
        env = worker_environment(
            cfg.env_allowlist,
            "local",
            session_id,
            agent.profile,
            profile,
            worker.env,
            artifacts,
            workspace,
            None,
            f"{getpass.getuser()}@local",
            expires.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            None,
        )
        backend = SandboxBackend(cfg.sandbox) if cfg.backend == "sandbox" else ProcessBackend()
        if isinstance(backend, SandboxBackend):
            try:
                backend.check()
            except RuntimeError as e:
                raise AskError(str(e)) from None
        spec = LaunchSpec(
            scratch_dir=scratch,
            artifacts_dir=artifacts,
            workspace=workspace,
            workspace_mode=profile.workspace_mount,
            network=profile.network,
        )
        try:
            handle = await backend.start(worker.command, env, workspace or scratch, spec)
        except OSError as e:
            raise AskError(f"agent {name!r} failed to start: {e}") from None
        text = task_text(question, context)
        result = await _drive(cfg, handle, session_id, text, timeout, on_event)
        result["answer"] = full_answer(result["answer"], artifacts)
    if result["answer"] is None:
        why = result["error"] or "it exited without an answer"
        if result["stderr"]:
            why += f" (stderr: {result['stderr'][-500:]})"
        raise AskError(f"agent {name!r} didn't answer: {why}")
    return {
        "agent": name,
        "answer": result["answer"],
        "cost_usd": result["cost_usd"],
        "session_id": session_id,
    }


def full_answer(answer: str | None, artifacts: Path) -> str | None:
    """Adapters clip `final.summary` to fit a room message (Claude: 4000 characters), but
    the Claude adapter also saves the whole result as result.md. Live, a detailed answer
    was cut mid-step; an ask has no room message to fit, so it returns the full text."""
    if not answer or not answer.endswith(CLIPPED):
        return answer
    try:
        full = (artifacts / "result.md").read_bytes()[:FULL_ANSWER_LIMIT].decode(errors="replace")
    except OSError:
        return answer
    return full if full.strip() else answer


async def _drive(
    cfg: AgentsFile,
    handle: ProcessHandle,
    session_id: str,
    task: str,
    timeout: float,
    on_event: Callable[[dict], None] | None,
) -> dict:
    """Send the task, then wait for the first `final` or for the worker to exit, whichever
    comes first (an interactive worker type would otherwise wait for more input)."""
    out = {"answer": None, "cost_usd": None, "error": None, "stderr": ""}
    answered = asyncio.Event()

    async def read_stdout() -> None:
        async for line in iter_lines(handle.stdout, MAX_LINE_BYTES):
            event = parse_stdout_line(line)
            if on_event:
                on_event(event)
            if event["type"] == "final" and out["answer"] is None:
                out["answer"] = event.get("summary") or ""
                out["cost_usd"] = event.get("cost_usd")
                answered.set()
            elif event["type"] == "needs_input" and out["answer"] is None:
                # An interactive worker type answers a turn with needs_input (its result is
                # the `question`) and waits for more. One answer is all an ask wants.
                out["answer"] = event.get("question") or ""
                answered.set()
            elif event["type"] in ("error", "protocol_error") and not out["error"]:
                out["error"] = event.get("message") or event.get("reason")

    async def read_stderr() -> None:
        async for line in iter_lines(handle.stderr, MAX_LINE_BYTES):
            out["stderr"] = (out["stderr"] + line + "\n")[-STDERR_TAIL:]

    readers = asyncio.gather(read_stdout(), read_stderr())
    exited = asyncio.ensure_future(handle.wait())
    final = asyncio.ensure_future(answered.wait())
    try:
        await handle.send({"type": "task", "session_id": session_id, "task": task})
        async with asyncio.timeout(timeout):
            await asyncio.wait({exited, final}, return_when=asyncio.FIRST_COMPLETED)
    except TimeoutError:
        out["error"] = out["error"] or f"no answer within {int(timeout)}s"
    except Exception as e:  # noqa: BLE001 - e.g. the worker exited before reading its task
        out["error"] = out["error"] or f"{type(e).__name__}: {e}"
    finally:
        final.cancel()
        # With an answer in hand, just end it: a polite stop would make an interactive
        # Claude worker spend a closing-summary turn nobody reads.
        await _end(cfg, handle, polite=out["answer"] is None)
        try:  # output closes with the group; don't wait on a pipe held open elsewhere
            await asyncio.wait_for(asyncio.shield(readers), cfg.stop_grace_seconds)
        except (TimeoutError, Exception):  # noqa: BLE001
            readers.cancel()
            await asyncio.gather(readers, return_exceptions=True)
        await exited
    return out


async def _end(cfg: AgentsFile, handle: ProcessHandle, *, polite: bool) -> None:
    """(Stop message,) then SIGTERM, then SIGKILL, until nothing in the group is left."""
    grace = cfg.stop_grace_seconds
    if polite and handle.group_alive():
        try:
            await asyncio.wait_for(handle.send({"type": "stop", "reason": "ask_done"}), 1)
        except Exception:  # noqa: BLE001 - it may not be reading any more
            pass
    for signal_group in (None, handle.terminate, handle.kill):
        if signal_group:
            signal_group()
        # Unasked, a worker that has answered gets a moment to exit on its own.
        wait = grace if polite or signal_group else min(0.5, grace)
        deadline = asyncio.get_running_loop().time() + wait
        while handle.group_alive() and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.05)
        if not handle.group_alive():
            break
    await handle.wait()


# ----- MCP server (`agentd mcp`): the agents as tools for a Claude Code session ----------


class AskTools:
    """Re-reads the agents file on every call, so an edit applies without restarting the
    session. Every failure comes back as a readable tool error."""

    def __init__(self, path: Path | None = None):
        self.path = path

    def _config(self) -> AgentsFile:
        try:
            return load(self.path)
        except AskError as e:
            raise ToolError(str(e)) from None
        except (ValidationError, yaml.YAMLError) as e:
            raise ToolError(f"the agents file is invalid: {e}") from None

    def agents_list(self) -> list[dict]:
        """The agents you can ask: each one's name, what it knows (description and, for a
        knowledge base, its workspace), and how it runs."""
        return describe(self._config())

    async def agent_ask(self, agent: str, question: str, context: str | None = None) -> dict:
        """Ask one agent a question and get its answer. Each ask starts the agent fresh and
        it exits afterwards, so for a follow-up pass what matters from earlier answers as
        `context`. The answer is the agent's own view, based on its files: check anything
        important before acting on it."""
        try:
            return await ask(self._config(), agent, question, context=context)
        except AskError as e:
            raise ToolError(str(e)) from None


def build_server(path: Path | None = None) -> MCPServer:
    server = MCPServer(
        "agents",
        instructions=(
            "Local agents you can ask questions, e.g. knowledge bases built from repos. Call "
            "agents_list to see who knows what, then agent_ask. Nothing runs between asks."
        ),
    )
    tools = AskTools(path)
    for name in ("agents_list", "agent_ask"):
        server.add_tool(getattr(tools, name), name=name)
    return server
