"""Tests for the Ollama worker adapter (agentd.workers.ollama), against tests/fake_ollama.py
(the response shapes real Ollama 0.10 returns) and, for rooms, tests/fake_roomsd.py."""

import asyncio
import sys

import pytest
from fake_ollama import FakeOllama, tool_call
from fake_roomsd import FakeRoomsd
from fastapi.testclient import TestClient
from helpers import events, spawn, wait_event, wait_status

from agentd.app import create_app
from agentd.config import Profile, WorkerType
from agentd.workers.claude_code import CLOSING_PROMPT
from agentd.workers.ollama import Toolbox, coerce, parse_args

WORKER = "missy@test/agentd-test.ollama"


# ----- arguments and tools ------------------------------------------------------------------


def test_model_is_required_and_url_is_normalized():
    with pytest.raises(SystemExit):
        parse_args([])
    assert parse_args(["--model", "m", "--url", "box:11434"]).url == "http://box:11434"


def tools_for(env, profile=None):
    box = Toolbox(env, profile or {})
    asyncio.run(box.load())
    return box


def test_tools_follow_the_grant(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "a.txt").write_text("hello")
    art = tmp_path / "art"
    art.mkdir()
    assert tools_for({}).names == set()  # no room, no workspace: no tools at all
    read = tools_for(
        {
            "AGENTD_WORKSPACE": str(ws),
            "AGENTD_WORKSPACE_MODE": "read",
            "AGENTD_ARTIFACTS_DIR": str(art),
        }
    )
    assert read.names == {"read_file", "list_files"}
    rw = tools_for(
        {
            "AGENTD_WORKSPACE": str(ws),
            "AGENTD_WORKSPACE_MODE": "read_write",
            "AGENTD_ARTIFACTS_DIR": str(art),
        }
    )
    assert rw.names == {"read_file", "list_files", "write_artifact"}
    call = lambda n, **a: asyncio.run(rw.call(n, a))  # noqa: E731
    assert call("read_file", path="a.txt") == "hello"
    assert '"a.txt"' in call("list_files")
    assert call("read_file", path="../../etc/passwd").startswith("error:")  # confined
    assert call("write_artifact", name="r.md", content="# R") == '{"saved": "r.md"}'
    assert (art / "r.md").read_text() == "# R"
    assert call("write_artifact", name="../x", content="no").startswith("error:")
    assert call("run_shell", cmd="ls").startswith("error: no tool")


def test_read_only_invite_gets_only_read_tools():
    import json

    env = {"ROOMSD_URL": "http://r", "ROOMSD_ROOM_ID": "room_1", "ROOMSD_TOKEN": "t"}
    assert {"rooms_send", "rooms_note_put"} <= tools_for(env).names
    ro = {**env, "AGENTD_GRANT": json.dumps({"room": {"reply": False}})}
    assert tools_for(ro).names == {"rooms_read", "rooms_note_get"}


# ----- through agentd -----------------------------------------------------------------------


@pytest.fixture
def ollama():
    fake = FakeOllama()
    yield fake
    fake.close()


@pytest.fixture
def client(settings, lobby, ollama):
    base = [
        sys.executable,
        "-m",
        "agentd.workers.ollama",
        "--url",
        ollama.url,
        "--model",
        "fake-model",
        "--room-poll-seconds",
        "0.05",
    ]
    s = settings.model_copy(
        update={
            "worker_types": {
                "ollama": WorkerType(command=base),
                "ollama-chat": WorkerType(command=[*base, "--interactive"]),
                "ollama-2-rounds": WorkerType(command=[*base, "--max-tool-rounds", "2"]),
                "ollama-down": WorkerType(
                    command=[
                        sys.executable,
                        "-m",
                        "agentd.workers.ollama",
                        "--url",
                        "http://127.0.0.1:9",
                        "--model",
                        "m",
                    ]
                ),
            },
            "profiles": {
                **settings.profiles,
                "reader": Profile(max_runtime_minutes=5, workspace_mount="read"),
                "tokens": Profile(max_runtime_minutes=5, max_total_tokens=150),
            },
        }
    )
    with TestClient(create_app(s, verifier=lobby.verifier())) as c:
        yield c


def test_oneshot_answer(client, ollama, boostie):
    sid = spawn(
        client, boostie, "say hi", worker_type="ollama", profile="read_only_research"
    ).json()["session_id"]
    s = wait_status(client, sid, boostie)
    assert s["status"] == "completed" and s["summary"] == "done: say hi"
    final = next(e for e in events(client, sid, boostie) if e["type"] == "final")
    assert final["usage"] == {"prompt_tokens": 100, "output_tokens": 10}
    (req,) = ollama.requests
    assert req["model"] == "fake-model" and req["options"]["num_ctx"] == 8192
    assert req["messages"][0]["role"] == "system"
    assert "Never wait or poll" in req["messages"][0]["content"]
    assert "tools" not in req  # read_only_research, no room: nothing to call


def test_workspace_tool_loop(client, ollama, boostie, workspace_root):
    (workspace_root / "repo" / "README.md").write_text("Project Zed")
    ollama.script.extend([tool_call("read_file", path="README.md"), {"content": "It's Zed."}])
    sid = spawn(
        client,
        boostie,
        "what is the project called?",
        worker_type="ollama",
        profile="reader",
        workspace={"path": str(workspace_root / "repo")},
    ).json()["session_id"]
    s = wait_status(client, sid, boostie)
    assert s["status"] == "completed" and s["summary"] == "It's Zed."
    tool_msgs = [m for m in ollama.requests[1]["messages"] if m["role"] == "tool"]
    assert tool_msgs == [{"role": "tool", "content": "Project Zed", "tool_name": "read_file"}]
    assert {t["function"]["name"] for t in ollama.requests[0]["tools"]} == {
        "read_file",
        "list_files",
    }


def test_tool_call_limit_forces_an_answer(client, ollama, boostie, workspace_root):
    ollama.always = tool_call("list_files")
    sid = spawn(
        client,
        boostie,
        "loop",
        worker_type="ollama-2-rounds",
        profile="reader",
        workspace={"path": str(workspace_root / "repo")},
    ).json()["session_id"]
    s = wait_status(client, sid, boostie)
    assert s["status"] == "completed"
    assert len(ollama.requests) == 3 and "tools" not in ollama.requests[-1]
    assert "Tool-call limit" in ollama.requests[-1]["messages"][-1]["content"]


def test_interactive_keeps_history_and_writes_closing_summary(client, ollama, boostie):
    sid = spawn(
        client, boostie, "first", worker_type="ollama-chat", profile="read_only_research"
    ).json()["session_id"]
    wait_event(client, sid, boostie, lambda e: e["type"] == "needs_input")
    client.post(f"/v1/sessions/{sid}/messages", json={"message": "second"}, headers=boostie)
    wait_event(client, sid, boostie, lambda e: e["type"] == "needs_input" and e.get("turn") == 2)
    s = client.post(f"/v1/sessions/{sid}/stop", headers=boostie).json()
    assert s["status"] == "completed" and s["stop_reason"] == "caller_cancelled"
    assert s["summary"] == f"done: {CLOSING_PROMPT}"  # the closing turn wrote the handoff
    assert len(ollama.requests) == 3
    third = ollama.requests[2]["messages"]
    assert [m["role"] for m in third].count("user") == 3  # the whole conversation is kept
    assert third[-1]["content"] == CLOSING_PROMPT


@pytest.mark.parametrize(
    "setup, worker, message",
    [
        (lambda o: None, "ollama-down", "cannot reach ollama"),
        (
            lambda o: (setattr(o, "status", 404), setattr(o, "error", "model 'x' not found")),
            "ollama",
            "model 'x' not found",
        ),
    ],
)
def test_ollama_failures_are_explicit(client, ollama, boostie, setup, worker, message):
    setup(ollama)
    sid = spawn(client, boostie, "hi", worker_type=worker, profile="read_only_research").json()[
        "session_id"
    ]
    assert wait_status(client, sid, boostie)["status"] == "failed"
    assert any(
        message in e.get("message", "")
        for e in events(client, sid, boostie)
        if e["type"] == "error"
    )


def test_token_budget(client, ollama, boostie):
    ollama.tokens = (500, 10)
    sid = spawn(client, boostie, "hi", worker_type="ollama-chat", profile="tokens").json()[
        "session_id"
    ]
    s = wait_status(client, sid, boostie)
    assert s["status"] == "completed"
    assert any(
        "token budget reached" in (e.get("message") or "") for e in events(client, sid, boostie)
    )


def test_stop_mid_turn(client, ollama, boostie):
    ollama.delay = 30
    sid = spawn(
        client, boostie, "slow", worker_type="ollama-chat", profile="read_only_research"
    ).json()["session_id"]
    wait_event(client, sid, boostie, lambda e: "ollama session" in (e.get("message") or ""))
    s = client.post(f"/v1/sessions/{sid}/stop", headers=boostie).json()
    assert s["status"] == "stopped"


# ----- rooms --------------------------------------------------------------------------------


@pytest.fixture
def roomsd():
    fake = FakeRoomsd({"inv_worker": WORKER, "tok_boostie": "boostie@test"})
    yield fake
    fake.close()


def test_room_mention_answered_with_room_tools(client, ollama, roomsd, boostie):
    roomsd.require_join = True
    sid = spawn(
        client,
        boostie,
        "join and wait",
        worker_type="ollama-chat",
        profile="read_only_research",
        room={"room_url": roomsd.room_url, "token": "inv_worker"},
    ).json()["session_id"]
    wait_event(client, sid, boostie, lambda e: e["type"] == "needs_input")
    assert WORKER in roomsd.participants
    ollama.script.extend(
        [
            tool_call("rooms_send", body="Tallahassee.", type="answer", in_reply_to=1),
            {"content": "Answered in the room."},
        ]
    )
    roomsd.post("boostie@test", "@agentd-test.ollama capital of florida?")
    wait_event(client, sid, boostie, lambda e: e["type"] == "needs_input" and e.get("turn") == 2)
    posted = [m for m in roomsd.messages if m["from"] == WORKER and m["type"] == "answer"]
    assert posted and posted[0]["body"] == "Tallahassee."
    result = [m for m in ollama.requests[2]["messages"] if m["role"] == "tool"][-1]["content"]
    assert result.startswith("ok: posted to the room as message #")  # plain words for small models
    wake = ollama.requests[1]["messages"][-1]["content"]
    assert 'trust="untrusted"' in wake and "capital of florida?" in wake
    # a refused send comes back to the model as readable text, not a crash
    ollama.script.extend([tool_call("rooms_send", body="x" * 70_000), {"content": "ok"}])
    client.post(
        f"/v1/sessions/{sid}/messages", json={"message": "post a huge thing"}, headers=boostie
    )
    wait_event(client, sid, boostie, lambda e: e["type"] == "needs_input" and e.get("turn") == 3)
    tool_msg = [m for m in ollama.requests[-1]["messages"] if m["role"] == "tool"][-1]
    assert tool_msg["content"].startswith("error:") and "summarise" in tool_msg["content"]


def test_exact_repeat_posts_are_refused(client, ollama, roomsd, boostie):
    """Live with qwen2.5 7B: the model posted the same objection three times in one turn."""
    roomsd.require_join = True
    ollama.script.extend(
        [
            tool_call(
                "rooms_send", body="Objection: 2s polling is too frequent.", type="objection"
            ),
            tool_call(
                "rooms_send", body="Objection: 2s polling is too frequent.", type="objection"
            ),
            {"content": "Posted my objection."},
        ]
    )
    sid = spawn(
        client,
        boostie,
        "review",
        worker_type="ollama",
        profile="read_only_research",
        room={"room_url": roomsd.room_url, "token": "inv_worker"},
    ).json()["session_id"]
    assert wait_status(client, sid, boostie)["status"] == "completed"
    objections = [m for m in roomsd.messages if m["type"] == "objection"]
    assert len(objections) == 1
    refusal = [m for m in ollama.requests[2]["messages"] if m["role"] == "tool"][-1]["content"]
    assert refusal.startswith("error: you already posted exactly this")


def test_room_polling_is_not_logged_as_events(client, ollama, roomsd, boostie):
    sid = spawn(
        client,
        boostie,
        "wait",
        worker_type="ollama-chat",
        profile="read_only_research",
        room={"room_url": roomsd.room_url, "token": "inv_worker"},
    ).json()["session_id"]
    wait_event(client, sid, boostie, lambda e: e["type"] == "needs_input")
    import time

    time.sleep(0.5)  # ten room polls at 0.05s
    logs = [e for e in events(client, sid, boostie) if e["type"] == "log"]
    assert not any("HTTP Request" in e.get("text", "") for e in logs)
    client.post(f"/v1/sessions/{sid}/stop", headers=boostie)


def test_small_model_type_slips_are_coerced():
    """Live with qwen2.5 7B: `to` sent as a string made roomsd reject the answer (422)."""
    schema = {
        "properties": {
            "to": {"anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "null"}]},
            "in_reply_to": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
            "confidence": {"anyOf": [{"type": "number"}, {"type": "null"}]},
            "reply_requested": {"type": "boolean"},
            "body": {"type": "string"},
        }
    }
    got = coerce(
        {
            "to": "boostie@local",
            "in_reply_to": "10",
            "confidence": "0.8",
            "reply_requested": "true",
            "body": "42",
        },
        schema,
    )
    assert got == {
        "to": ["boostie@local"],
        "in_reply_to": 10,
        "confidence": 0.8,
        "reply_requested": True,
        "body": "42",
    }
    assert coerce({"confidence": "high"}, schema) == {"confidence": "high"}  # left for the API


def test_room_send_with_string_recipient_reaches_the_room(client, ollama, roomsd, boostie):
    roomsd.require_join = True
    ollama.script.extend(
        [
            tool_call(
                "rooms_send",
                body="5-10s with backoff.",
                type="answer",
                to="boostie@test",
                in_reply_to="1",
            ),
            {"content": "Answered."},
        ]
    )
    sid = spawn(
        client,
        boostie,
        "answer",
        worker_type="ollama",
        profile="read_only_research",
        room={"room_url": roomsd.room_url, "token": "inv_worker"},
    ).json()["session_id"]
    assert wait_status(client, sid, boostie)["status"] == "completed"
    assert [m["body"] for m in roomsd.messages if m["type"] == "answer"] == ["5-10s with backoff."]
