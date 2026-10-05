"""`agentd ask` / `agentd mcp`: a configured agent answers one question, with no lobbyd,
roomsd or agentd service, and nothing left running afterwards."""

import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import ValidationError

from agentd import cli, local
from agentd.local import AgentsFile, AskError, ask

FAKE_CLAUDE = Path(__file__).parent / "fake_claude.py"
CLAUDE = [
    sys.executable,
    "-m",
    "agentd.workers.claude_code",
    "--claude",
    f"{sys.executable} {FAKE_CLAUDE}",
    "--exit-grace-seconds",
    "2",
]


def config(tmp_path, **extra) -> dict:
    kb = tmp_path / "kb"
    kb.mkdir(exist_ok=True)
    (kb / "README.md").write_text("Botrick is the preferred hub.")
    argv = tmp_path / "argv.json"
    raw = {
        "worker_types": {
            "fake": {"command": [sys.executable, "-m", "agentd.workers.fake"]},
            "claude": {"command": CLAUDE, "env": {"FAKE_CLAUDE_ARGV_FILE": str(argv)}},
            "claude-chat": {
                "command": [*CLAUDE, "--interactive"],
                # A closing-summary turn would hang: an ask must never ask for one.
                "env": {"FAKE_CLAUDE_HANG_ON_CLOSE": "1"},
            },
        },
        "profiles": {
            "kb": {"max_runtime_minutes": 1, "workspace_mount": "read", "network": False},
            "plain": {"max_runtime_minutes": 1},
        },
        "agents": {
            "fake": {"worker_type": "fake", "profile": "plain"},
            "vpn": {
                "worker_type": "claude",
                "profile": "kb",
                "workspace": str(kb),
                "description": "OpenVPN network",
            },
            "chat": {"worker_type": "claude-chat", "profile": "kb", "workspace": str(kb)},
        },
        "stop_grace_seconds": 1,
    }
    raw.update(extra)
    return raw


@pytest.fixture
def cfg(tmp_path) -> AgentsFile:
    return AgentsFile.model_validate(config(tmp_path))


def run(cfg, name, question, **kw):
    return asyncio.run(ask(cfg, name, question, **kw))


def no_workers_left() -> bool:
    out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    return not any("-m agentd.workers." in line and "python" in line for line in out.splitlines())


def test_a_fake_worker_answers(cfg):
    r = run(cfg, "fake", "say hello")
    assert r["answer"] == "did: say hello" and r["agent"] == "fake"
    assert r["session_id"].startswith("ask_")


def test_claude_answers_from_its_workspace_with_read_tools_only(cfg, tmp_path):
    r = run(cfg, "vpn", "which hub is preferred?", context="we're debugging routing")
    assert r["answer"] and r["cost_usd"] is not None
    argv = json.loads((tmp_path / "argv.json").read_text())
    assert "--restricted" in argv  # no shell: the profile's grant reached the adapter
    tools = argv[argv.index("--tools") + 1 : argv.index("--allowedTools")]
    assert "Read" in tools and not {"Write", "Edit", "Bash", "WebFetch"} & set(tools)


def test_an_interactive_worker_is_ended_after_its_first_answer(cfg):
    """needs_input carries the answer; no closing-summary turn is asked for (it would hang
    here), so the ask returns quickly and leaves nothing running."""
    t0 = time.monotonic()
    r = run(cfg, "chat", "which hub is preferred?")
    assert r["answer"] and time.monotonic() - t0 < 5
    assert no_workers_left()


@pytest.mark.parametrize(
    "question, timeout, why",
    [
        ("crash now", None, "exited without an answer"),
        ("hang forever", 2, "no answer within 2s"),
        ("stubborn mule", 2, "no answer within 2s"),  # ignores stop and SIGTERM: killed
    ],
)
def test_failures_are_explicit_and_leave_nothing_running(cfg, question, timeout, why):
    with pytest.raises(AskError, match=why):
        run(cfg, "fake", question, timeout=timeout)
    assert no_workers_left()


def test_requests_that_cant_run(cfg, tmp_path):
    with pytest.raises(AskError, match="no agent named 'nope'.*known: chat, fake, vpn"):
        run(cfg, "nope", "hi")
    with pytest.raises(AskError, match="question is empty"):
        run(cfg, "fake", "  ")
    (tmp_path / "kb" / "README.md").unlink()
    (tmp_path / "kb").rmdir()
    with pytest.raises(AskError, match="is not a directory"):
        run(cfg, "vpn", "hi")


@pytest.mark.parametrize(
    "change, error",
    [
        ({"agents": {"x": {"worker_type": "nope", "profile": "kb"}}}, "unknown worker_type"),
        ({"agents": {"x": {"worker_type": "fake", "profile": "nope"}}}, "unknown profile"),
        (
            {"agents": {"x": {"worker_type": "fake", "profile": "plain", "workspace": "/tmp"}}},
            "doesn't mount a workspace",
        ),
        ({"agents": {"x": {"worker_type": "fake", "profile": "kb", "shell": True}}}, "Extra"),
    ],
)
def test_the_agents_file_is_validated(tmp_path, change, error):
    with pytest.raises(ValidationError, match=error):
        AgentsFile.model_validate(config(tmp_path, **change))


def test_cli(tmp_path, monkeypatch, capsys):
    path = tmp_path / "agents.yaml"
    path.write_text(yaml.safe_dump(config(tmp_path)))
    monkeypatch.setenv("AGENTD_AGENTS", str(path))
    assert cli.main(["ask", "fake", "say", "hello"]) == 0
    assert capsys.readouterr().out.strip() == "did: say hello"
    assert cli.main(["agents"]) == 0
    assert "vpn: claude/kb" in capsys.readouterr().out
    assert cli.main(["ask", "nope", "hi"]) == 1
    assert "no agent named" in capsys.readouterr().err
    monkeypatch.setenv("AGENTD_AGENTS", str(tmp_path / "missing.yaml"))
    assert cli.main(["agents"]) == 1
    assert "no agents file" in capsys.readouterr().err


def test_mcp_tools(tmp_path):
    path = tmp_path / "agents.yaml"
    path.write_text(yaml.safe_dump(config(tmp_path)))
    server = local.build_server(path)
    names = {t.name for t in asyncio.run(server.list_tools())}
    assert names == {"agents_list", "agent_ask"}
    listed = asyncio.run(server.call_tool("agents_list", {}))
    assert "OpenVPN network" in json.dumps(listed, default=str)
    answer = asyncio.run(server.call_tool("agent_ask", {"agent": "fake", "question": "say hi"}))
    assert "did: say hi" in json.dumps(answer, default=str)
    with pytest.raises(ToolError, match="no agent named"):
        asyncio.run(server.call_tool("agent_ask", {"agent": "nope", "question": "hi"}))
    path.write_text("agents: {}")  # edits apply on the next call, and stay readable
    with pytest.raises(ToolError, match="agents file is invalid"):
        asyncio.run(server.call_tool("agents_list", {}))
