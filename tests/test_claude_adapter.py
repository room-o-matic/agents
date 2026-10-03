import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from helpers import events, spawn, wait_event, wait_status

from agentd.app import create_app
from agentd.config import DEFAULT_PROFILES, Profile, WorkerType
from agentd.workers.claude_code import Translator, build_command, parse_args, tool_policy

FAKE_CLAUDE = Path(__file__).parent / "fake_claude.py"


# ----- policy (pure) -------------------------------------------------------------------


def flags_after(flags: list[str], name: str) -> list[str]:
    """Values following `name` up to the next --flag."""
    i = flags.index(name) + 1
    vals = []
    while i < len(flags) and not flags[i].startswith("--"):
        vals.append(flags[i])
        i += 1
    return vals


def test_read_only_profile_is_restricted_without_write_or_shell():
    flags = tool_policy(DEFAULT_PROFILES["read_only_research"].model_dump())
    tools = flags_after(flags, "--tools")
    assert "--restricted" in flags
    assert set(tools) == {"Read", "Glob", "Grep", "WebSearch", "WebFetch"}
    assert flags_after(flags, "--permission-mode") == ["dontAsk"]


def test_workspace_coder_can_edit_but_has_no_shell():
    flags = tool_policy(DEFAULT_PROFILES["workspace_coder"].model_dump())
    tools = flags_after(flags, "--tools")
    assert {"Edit", "Write"} <= set(tools)
    assert "Bash" not in tools
    assert "--restricted" in flags
    assert flags_after(flags, "--permission-mode") == ["acceptEdits"]


def test_shell_only_when_profile_says_so():
    profile = {"workspace_mount": "read_write", "shell": True, "network": False}
    flags = tool_policy(profile)
    tools = flags_after(flags, "--tools")
    assert "Bash" in tools and "--restricted" not in flags
    assert not {"WebSearch", "WebFetch"} & set(tools)  # network: false
    assert tool_policy({"shell": "yes"}).count("--restricted") == 1  # only literal true


def test_profile_overrides():
    flags = tool_policy({"claude_tools": ["Read"], "claude_permission_mode": "plan"})
    assert flags_after(flags, "--tools") == ["Read"]
    assert flags_after(flags, "--permission-mode") == ["plan"]


def test_build_command_budget_and_artifacts_dir():
    args = parse_args(["--claude", "my claude", "--model", "sonnet", "--max-budget-usd", "5"])
    profile = {"workspace_mount": "read_write", "max_budget_usd": 2}
    env = {"AGENTD_ARTIFACTS_DIR": "/a", "AGENTD_SESSION_ID": "agt_1"}
    cmd = build_command(args, profile, env)
    assert cmd[:3] == ["my", "claude", "-p"]
    assert flags_after(cmd, "--max-budget-usd") == ["2.0"]  # the lower of the two
    assert flags_after(cmd, "--model") == ["sonnet"]
    assert flags_after(cmd, "--add-dir") == ["/a"]
    assert flags_after(cmd, "--permission-prompts") == ["none"]
    assert "agt_1" in flags_after(cmd, "--append-system-prompt")[0]
    read_only = build_command(args, {"workspace_mount": "none"}, env)
    assert "--add-dir" not in read_only


# ----- translation (pure) --------------------------------------------------------------


def test_translator_maps_stream_json():
    t = Translator()
    init = t.handle({"type": "system", "subtype": "init", "session_id": "s1", "model": "m"})
    assert init[0][0] == "progress" and t.claude_session_id == "s1"
    blocks = t.handle(
        {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "text", "text": "Looking."},
                    {"type": "thinking", "thinking": "hidden"},
                    {"type": "tool_use", "name": "Bash", "input": {"command": "ls -la"}},
                ]
            },
        }
    )
    assert [e[1]["message"] for e in blocks] == ["Looking.", "Bash: ls -la"]
    assert t.handle({"type": "user", "message": {"content": []}}) == []
    assert t.handle({"type": "result", "subtype": "success", "result": "ok"}) == []
    assert t.last_ok and t.result_fields()["summary"] == "ok"
    err = t.handle({"type": "result", "subtype": "error_max_turns", "is_error": True})
    assert err[0][0] == "error" and not t.last_ok


# ----- through agentd ------------------------------------------------------------------


@pytest.fixture
def claude_client(settings, lobby, tmp_path):
    argv_file = tmp_path / "argv.json"
    base = [
        sys.executable,
        "-m",
        "agentd.workers.claude_code",
        "--claude",
        f"{sys.executable} {FAKE_CLAUDE}",
        "--exit-grace-seconds",
        "2",
    ]
    env = {"FAKE_CLAUDE_ARGV_FILE": str(argv_file)}
    s = settings.model_copy(
        update={
            "worker_types": {
                "claude": WorkerType(command=base, env=env),
                "claude-chat": WorkerType(command=[*base, "--interactive"], env=env),
                "claude-missing": WorkerType(
                    command=[
                        sys.executable,
                        "-m",
                        "agentd.workers.claude_code",
                        "--claude",
                        "/nonexistent/claude",
                    ]
                ),
            },
            "profiles": {
                **settings.profiles,
                "budgeted": Profile(max_runtime_minutes=5, max_budget_usd=1),
            },
        }
    )
    with TestClient(create_app(s, verifier=lobby.verifier())) as c:
        c.argv_file = argv_file
        yield c


def test_oneshot_completes_with_summary_cost_and_artifacts(claude_client, boostie, settings):
    sid = spawn(claude_client, boostie, "please write-artifact", worker_type="claude").json()[
        "session_id"
    ]
    s = wait_status(claude_client, sid, boostie)
    assert (s["status"], s["exit_code"]) == ("completed", 0)
    assert s["summary"] == "done: please write-artifact"
    evs = events(claude_client, sid, boostie)
    final = next(e for e in evs if e["type"] == "final")
    assert final["cost_usd"] == 0.0123 and final["claude_session_id"].startswith("1111")
    names = {e["name"] for e in evs if e["type"] == "artifact"}
    assert names == {"result.md", "report.md"}
    assert any(e.get("message") == "Read: README.md" for e in evs)
    assert any(e["type"] == "log" and e["text"] == "not json at all" for e in evs)


def test_adapter_passes_profile_policy_to_claude(claude_client, boostie):
    sid = spawn(
        claude_client, boostie, "hi", worker_type="claude", profile="read_only_research"
    ).json()["session_id"]
    wait_status(claude_client, sid, boostie)
    argv = json.loads(claude_client.argv_file.read_text())
    assert "--restricted" in argv and "Write" not in argv and "Bash" not in argv


def test_budget_from_profile(claude_client, boostie):
    sid = spawn(claude_client, boostie, "hi", worker_type="claude", profile="budgeted").json()[
        "session_id"
    ]
    wait_status(claude_client, sid, boostie)
    argv = json.loads(claude_client.argv_file.read_text())
    assert argv[argv.index("--max-budget-usd") + 1] == "1.0"


def test_error_result_fails_session(claude_client, boostie):
    sid = spawn(claude_client, boostie, "fail-turn", worker_type="claude").json()["session_id"]
    s = wait_status(claude_client, sid, boostie)
    assert s["status"] == "failed"
    errs = [e for e in events(claude_client, sid, boostie) if e["type"] == "error"]
    assert errs[0]["message"] == "it broke"


def test_claude_dying_fails_session(claude_client, boostie):
    sid = spawn(claude_client, boostie, "die", worker_type="claude").json()["session_id"]
    s = wait_status(claude_client, sid, boostie)
    assert s["status"] == "failed"
    errs = [e["message"] for e in events(claude_client, sid, boostie) if e["type"] == "error"]
    assert "before producing a result" in errs[-1]


def test_missing_claude_binary_fails_clearly(claude_client, boostie):
    sid = spawn(claude_client, boostie, "hi", worker_type="claude-missing").json()["session_id"]
    assert wait_status(claude_client, sid, boostie)["status"] == "failed"
    errs = [e["message"] for e in events(claude_client, sid, boostie) if e["type"] == "error"]
    assert "cannot start claude" in errs[0]


def test_message_during_turn_joins_conversation(claude_client, boostie):
    sid = spawn(claude_client, boostie, "slow", worker_type="claude").json()["session_id"]
    wait_event(claude_client, sid, boostie, lambda e: e.get("message") == "working on: slow")
    r = claude_client.post(
        f"/v1/sessions/{sid}/messages", json={"message": "also check CI"}, headers=boostie
    )
    assert r.status_code == 200
    s = wait_status(claude_client, sid, boostie)
    assert s["status"] == "completed"
    assert s["summary"] == (
        'done: slow + <owner-message from="boostie@test">\nalso check CI\n</owner-message>'
    )


def test_stop_mid_turn_is_prompt(claude_client, boostie):
    sid = spawn(claude_client, boostie, "hang", worker_type="claude").json()["session_id"]
    wait_event(claude_client, sid, boostie, lambda e: e.get("message") == "working on: hang")
    s = claude_client.post(f"/v1/sessions/{sid}/stop", headers=boostie).json()
    assert s["status"] == "stopped"


def test_interactive_turns_then_stop_completes(claude_client, boostie):
    sid = spawn(claude_client, boostie, "first", worker_type="claude-chat").json()["session_id"]
    wait_event(claude_client, sid, boostie, lambda e: e["type"] == "needs_input")
    claude_client.post(f"/v1/sessions/{sid}/messages", json={"message": "second"}, headers=boostie)
    wait_event(
        claude_client,
        sid,
        boostie,
        lambda e: e["type"] == "needs_input" and e.get("turn") == 2,
    )
    s = claude_client.post(f"/v1/sessions/{sid}/stop", headers=boostie).json()
    assert s["status"] == "completed"
    assert s["summary"] == 'done: <owner-message from="boostie@test">\nsecond\n</owner-message>'
