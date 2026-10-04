"""Tests for the Codex CLI worker adapter (agentd.workers.codex). Uses tests/fake_codex.py,
which emits the JSONL that real Codex CLI 0.158 emits and never calls a model."""

import json
import sys
from pathlib import Path

import pytest
from fake_roomsd import FakeRoomsd
from fastapi.testclient import TestClient
from helpers import events, spawn, wait_event, wait_status

from agentd.app import create_app
from agentd.config import Profile, WorkerType
from agentd.workers import codex
from agentd.workers.claude_code import CLOSING_PROMPT
from agentd.workers.codex import Translator, build_command, parse_args, sandbox_flags

FAKE_CODEX = Path(__file__).parent / "fake_codex.py"
WORKER = "missy@test/agentd-test.codex"

# Captured from a live `codex exec --json` run (Codex CLI 0.158, 2026-10-04).
REAL_TURN = [
    {"type": "thread.started", "thread_id": "01a1049e-d079-7870-9d1d-a5c4a9ba6414"},
    {"type": "turn.started"},
    {
        "type": "item.completed",
        "item": {"id": "item_0", "type": "agent_message", "text": "I’ll read `note.txt` once."},
    },
    {
        "type": "item.started",
        "item": {
            "id": "item_1",
            "type": "command_execution",
            "command": "/bin/bash -lc 'cat note.txt'",
            "aggregated_output": "",
            "exit_code": None,
            "status": "in_progress",
        },
    },
    {
        "type": "item.completed",
        "item": {
            "id": "item_1",
            "type": "command_execution",
            "command": "/bin/bash -lc 'cat note.txt'",
            "aggregated_output": "hello\n",
            "exit_code": 0,
            "status": "completed",
        },
    },
    {
        "type": "item.completed",
        "item": {"id": "item_2", "type": "agent_message", "text": "DONE hello"},
    },
    {
        "type": "turn.completed",
        "usage": {
            "input_tokens": 27888,
            "cached_input_tokens": 25472,
            "cache_write_input_tokens": 0,
            "output_tokens": 115,
            "reasoning_output_tokens": 8,
        },
    },
]


def env_with(**kw):
    return {"AGENTD_ARTIFACTS_DIR": "/art", "PATH": "/bin", "HOME": "/h", **kw}


# ----- policy and command line ------------------------------------------------------------


def test_read_only_profile_runs_codex_read_only_without_approvals():
    cmd = build_command(parse_args([]), {"workspace_mount": "none"}, env_with(), "/w")
    assert cmd[:2] == ["codex", "exec"] and cmd[-1] == "-"  # the prompt comes on stdin
    assert cmd[cmd.index("--sandbox") + 1] == "read-only"
    assert 'approval_policy="never"' in cmd
    assert "--ignore-user-config" in cmd and "--json" in cmd
    assert 'shell_environment_policy.inherit="core"' in cmd  # commands never see secrets
    assert "--add-dir" not in cmd and "resume" not in cmd


def test_read_write_profile_and_network():
    profile = {"workspace_mount": "read_write", "network": False}
    flags = sandbox_flags(profile, env_with())
    assert flags[flags.index("--sandbox") + 1] == "workspace-write"
    assert "sandbox_workspace_write.network_access=false" in flags
    assert flags[flags.index("--add-dir") + 1] == "/art"
    on = sandbox_flags({"workspace_mount": "read_write"}, env_with())
    assert "sandbox_workspace_write.network_access=true" in on
    grant_off = env_with(AGENTD_GRANT=json.dumps({"network": False}))
    assert "sandbox_workspace_write.network_access=false" in sandbox_flags(
        {"workspace_mount": "read_write"}, grant_off
    )


def test_sandbox_override_is_bounded():
    assert "read-only" in sandbox_flags(
        {"workspace_mount": "read_write", "codex_sandbox": "read-only"}, env_with()
    )
    with pytest.raises(ValueError, match="codex_sandbox"):
        sandbox_flags({"codex_sandbox": "danger-full-access"}, env_with())
    ext = sandbox_flags({}, env_with(), external=True)
    assert "--dangerously-bypass-approvals-and-sandbox" in ext and "--sandbox" not in ext


def test_resume_and_model():
    cmd = build_command(parse_args(["--model", "gpt-x"]), {}, env_with(), "/w", "t-1")
    assert cmd[cmd.index("--model") + 1] == "gpt-x"
    assert cmd[cmd.index("resume") + 1] == "t-1" and cmd[-1] == "-"


def test_room_tools_get_the_token_by_name_never_by_value():
    env = env_with(
        ROOMSD_URL="http://r",
        ROOMSD_ROOM_ID="room_1",
        ROOMSD_TOKEN="rmsd_SECRET",
        ANTHROPIC_API_KEY="sk-ant-xyz-123456",
    )
    cmd = build_command(parse_args([]), {}, env, "/w")
    joined = " ".join(cmd)
    assert "rmsd_SECRET" not in joined and "sk-ant-xyz-123456" not in joined
    names = json.loads(
        next(c for c in cmd if c.startswith("mcp_servers.rooms.env_vars=")).split("=", 1)[1]
    )
    assert {"ROOMSD_URL", "ROOMSD_ROOM_ID", "ROOMSD_TOKEN", "ANTHROPIC_API_KEY"} <= set(names)
    assert "ROOMSD_READ_ONLY" not in names
    # Codex refuses unapproved MCP calls when approvals are off (seen live): only the room
    # server, granted by the invite, is pre-approved
    assert 'mcp_servers.rooms.default_tools_approval_mode="approve"' in cmd
    assert sum("approval_mode" in c for c in cmd) == 1
    ro = env_with(
        ROOMSD_URL="http://r",
        ROOMSD_ROOM_ID="room_1",
        ROOMSD_TOKEN="t",
        AGENTD_GRANT=json.dumps({"room": {"reply": False}}),
    )
    assert "ROOMSD_READ_ONLY" in " ".join(build_command(parse_args([]), {}, ro, "/w"))
    assert codex.worker_env(ro)["ROOMSD_READ_ONLY"] == "1"


def test_token_budget_takes_the_lower_cap():
    assert codex.token_budget(parse_args([]), {}) is None
    assert (
        codex.token_budget(parse_args(["--max-total-tokens", "900"]), {"max_total_tokens": 500})
        == 500
    )


# ----- translation of real events ---------------------------------------------------------


def test_translator_on_a_real_turn():
    t = Translator("gpt-x")
    out = [e for m in REAL_TURN for e in t.handle(m)]
    assert out[0] == (
        "progress",
        {
            "message": "codex started (thread 01a1049e-d079-7870-9d1d-a5c4a9ba6414, model gpt-x)",
            "codex_thread_id": "01a1049e-d079-7870-9d1d-a5c4a9ba6414",
        },
    )
    assert ("progress", {"message": "$ /bin/bash -lc 'cat note.txt'", "tool": "shell"}) in out
    assert all(k == "progress" for k, _ in out)
    assert t.last_ok and t.last_result == "DONE hello" and t.turns == 1
    assert t.usage == {"input_tokens": 27888, "cached_input_tokens": 25472, "output_tokens": 115}
    assert t.total_tokens == 27888 - 25472 + 115  # cache hits don't count against the budget
    # a resumed turn repeats thread.started: announced once
    assert t.handle(REAL_TURN[0]) == []


def test_translator_failures():
    t = Translator()
    t.start_turn()
    assert t.handle({"type": "turn.failed", "error": {"message": "boom"}}) == [
        ("error", {"message": "boom", "subtype": "turn_failed"})
    ]
    assert not t.last_ok
    t.start_turn()
    assert t.handle({"type": "turn.completed", "usage": {}})[0][0] == "error"  # no answer
    bad = {
        "type": "item.completed",
        "item": {"type": "command_execution", "command": "x", "exit_code": 2},
    }
    assert t.handle(bad)[0][1]["message"].startswith("command exited 2")


# ----- through agentd ---------------------------------------------------------------------


@pytest.fixture
def codex_client(settings, lobby, tmp_path):
    log = tmp_path / "codex.log"
    base = [
        sys.executable,
        "-m",
        "agentd.workers.codex",
        "--codex",
        f"{sys.executable} {FAKE_CODEX}",
        "--room-poll-seconds",
        "0.05",
    ]
    env = {"FAKE_CODEX_LOG": str(log)}
    s = settings.model_copy(
        update={
            "worker_types": {
                "codex": WorkerType(command=base, env=env),
                "codex-chat": WorkerType(command=[*base, "--interactive"], env=env),
                "codex-slow-close": WorkerType(
                    command=[*base, "--interactive", "--closing-summary-seconds", "0.5"],
                    env={**env, "FAKE_CODEX_HANG_ON_CLOSE": "1"},
                ),
                "codex-missing": WorkerType(
                    command=[
                        sys.executable,
                        "-m",
                        "agentd.workers.codex",
                        "--codex",
                        "/nonexistent/codex",
                    ]
                ),
            },
            "profiles": {
                **settings.profiles,
                "tokens": Profile(max_runtime_minutes=5, max_total_tokens=150),
            },
        }
    )
    with TestClient(create_app(s, verifier=lobby.verifier())) as c:
        c.calls = lambda: [json.loads(line) for line in log.read_text().splitlines()]
        yield c


def types(evs):
    return [e["type"] for e in evs]


def test_oneshot_completes_with_summary_usage_and_artifacts(codex_client, boostie):
    c = codex_client
    sid = spawn(c, boostie, "please run-command and write-artifact", worker_type="codex").json()[
        "session_id"
    ]
    s = wait_status(c, sid, boostie)
    assert s["status"] == "completed"
    assert s["summary"] == "done: please run-command and write-artifact"
    evs = events(c, sid, boostie)
    final = next(e for e in evs if e["type"] == "final")
    assert final["codex_thread_id"] == "thread-fake-1"
    assert final["usage"]["input_tokens"] == 100 and final["num_turns"] == 1
    assert {e["name"] for e in evs if e["type"] == "artifact"} == {"result.md", "report.md"}
    (call,) = c.calls()
    assert "session-instructions" in call["prompt"]  # agentd's rules ride on the first turn
    assert "please run-command" not in " ".join(call["argv"])  # prompts never in argv


def test_interactive_turns_resume_the_thread_then_closing_summary(codex_client, boostie):
    c = codex_client
    sid = spawn(c, boostie, "first", worker_type="codex-chat").json()["session_id"]
    wait_event(c, sid, boostie, lambda e: e["type"] == "needs_input")
    c.post(f"/v1/sessions/{sid}/messages", json={"message": "second"}, headers=boostie)
    wait_event(c, sid, boostie, lambda e: e["type"] == "needs_input" and e.get("turn") == 2)
    s = c.post(f"/v1/sessions/{sid}/stop", headers=boostie).json()
    assert s["status"] == "completed" and s["stop_reason"] == "caller_cancelled"
    assert s["summary"] == f"done: {CLOSING_PROMPT}"
    calls = c.calls()
    assert len(calls) == 3
    assert "resume" not in calls[0]["argv"]
    assert all(
        call["argv"][call["argv"].index("resume") + 1] == "thread-fake-1" for call in calls[1:]
    )
    assert '<owner-message from="boostie@test">' in calls[1]["prompt"]
    evs = events(c, sid, boostie)
    assert sum("codex started" in (e.get("message") or "") for e in evs) == 1


def test_closing_summary_timeout_keeps_last_result(codex_client, boostie):
    c = codex_client
    sid = spawn(c, boostie, "first", worker_type="codex-slow-close").json()["session_id"]
    wait_event(c, sid, boostie, lambda e: e["type"] == "needs_input")
    s = c.post(f"/v1/sessions/{sid}/stop", headers=boostie).json()
    assert s["status"] == "completed" and s["summary"].startswith("done: first")


def test_stop_mid_turn_ends_without_a_closing_turn(codex_client, boostie):
    c = codex_client
    sid = spawn(c, boostie, "hang", worker_type="codex-chat").json()["session_id"]
    wait_event(c, sid, boostie, lambda e: "codex started" in (e.get("message") or ""))
    s = c.post(f"/v1/sessions/{sid}/stop", headers=boostie).json()
    assert s["status"] == "stopped"
    assert len(c.calls()) == 1


@pytest.mark.parametrize(
    "task, message", [("fail-turn", "it broke"), ("die", "before finishing the turn")]
)
def test_failed_turns_fail_the_session(codex_client, boostie, task, message):
    c = codex_client
    sid = spawn(c, boostie, task, worker_type="codex").json()["session_id"]
    assert wait_status(c, sid, boostie)["status"] == "failed"
    errors = [e["message"] for e in events(c, sid, boostie) if e["type"] == "error"]
    assert any(message in m for m in errors)


def test_missing_codex_binary_fails_clearly(codex_client, boostie):
    c = codex_client
    sid = spawn(c, boostie, "hi", worker_type="codex-missing").json()["session_id"]
    assert wait_status(c, sid, boostie)["status"] == "failed"
    assert any(
        "cannot start codex" in e.get("message", "")
        for e in events(c, sid, boostie)
        if e["type"] == "error"
    )


def test_token_budget_ends_the_session(codex_client, boostie):
    c = codex_client
    sid = spawn(c, boostie, "big-usage", worker_type="codex-chat", profile="tokens").json()[
        "session_id"
    ]
    s = wait_status(c, sid, boostie)
    assert s["status"] == "completed"
    assert any("token budget reached" in (e.get("message") or "") for e in events(c, sid, boostie))
    assert len(c.calls()) == 1  # no further turns once over budget


def test_messages_sent_during_a_oneshot_turn_are_answered(codex_client, boostie):
    c = codex_client
    sid = spawn(c, boostie, "first", worker_type="codex").json()["session_id"]
    c.post(f"/v1/sessions/{sid}/messages", json={"message": "and also this"}, headers=boostie)
    s = wait_status(c, sid, boostie)
    assert s["status"] == "completed"
    prompts = [call["prompt"] for call in c.calls()]
    assert any("and also this" in p for p in prompts)


# ----- rooms --------------------------------------------------------------------------------


@pytest.fixture
def roomsd():
    fake = FakeRoomsd({"inv_worker": WORKER, "tok_boostie": "boostie@test"})
    yield fake
    fake.close()


def test_room_mention_wakes_a_resumed_turn(codex_client, roomsd, boostie):
    c = codex_client
    roomsd.require_join = True
    sid = spawn(
        c,
        boostie,
        "first",
        worker_type="codex-chat",
        room={"room_url": roomsd.room_url, "token": "inv_worker"},
    ).json()["session_id"]
    wait_event(c, sid, boostie, lambda e: e["type"] == "needs_input")
    assert WORKER in roomsd.participants  # joined and announced
    roomsd.post("boostie@test", "@agentd-test.codex what do you think?")
    wait_event(c, sid, boostie, lambda e: e["type"] == "needs_input" and e.get("turn") == 2)
    second = c.calls()[1]
    assert 'trust="untrusted"' in second["prompt"] and "what do you think?" in second["prompt"]
    assert "resume" in second["argv"]
    assert "inv_worker" not in " ".join(second["argv"])  # the invite never in argv
    c.post(f"/v1/sessions/{sid}/stop", headers=boostie)
    wait_event(c, sid, boostie, lambda e: e["type"] == "room_finalization")
    assert any(m["type"] == "handoff" for m in roomsd.messages)
