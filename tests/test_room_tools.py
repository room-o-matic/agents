import json
import subprocess
import sys
import time
from pathlib import Path

import pytest
from fake_roomsd import FakeRoomsd
from fastapi.testclient import TestClient
from helpers import events, spawn, wait_event, wait_status

from agentd.app import create_app
from agentd.config import WorkerType
from agentd.workers import room_tools
from agentd.workers.claude_code import build_command, parse_args
from agentd.workers.room_tools import RoomTools
from agentd.workers.wake import WakeGate

FAKE_CLAUDE = Path(__file__).parent / "fake_claude.py"
WORKER = "missy@test/agentd-test.claude"


@pytest.fixture
def roomsd():
    fake = FakeRoomsd({"inv_worker": WORKER, "tok_boostie": "boostie@test"})
    yield fake
    fake.close()


# ----- the tools themselves -----------------------------------------------------------


def test_read_send_and_notes(roomsd):
    roomsd.post("boostie@test", "hello")
    tools = RoomTools(roomsd.url, "room_1", "inv_worker")

    first = tools.rooms_read(include_notes=["summary"])
    assert first["you"] == WORKER
    assert [m["body"] for m in first["messages"]] == ["hello"]
    assert first["notes"] == {}

    sent = tools.rooms_send("on it", type="status")
    assert sent["posted_as"] == WORKER
    roomsd.post("boostie@test", "thanks")
    again = tools.rooms_read()  # only what's new since the last read
    assert [m["body"] for m in again["messages"]] == ["on it", "thanks"]
    assert tools.rooms_read()["messages"] == []
    assert len(tools.rooms_read(after_id=0)["messages"]) == 3

    tools.rooms_note_put("summary", {"state": "drafting"})
    note = tools.rooms_note_get("summary")
    assert (note["summary"], note["updated_by"]) == ({"state": "drafting"}, WORKER)
    assert note["provenance"]["trust"].startswith("untrusted")
    notes = tools.rooms_note_get()
    assert notes["summary"] == {"state": "drafting"} and "provenance" in notes


def test_note_put_never_clobbers_an_unread_change(roomsd):
    """Two workers share a note: a write based on a stale (or no) read is refused and comes
    back with the current value to merge, instead of overwriting (docs#20)."""
    a = RoomTools(roomsd.url, "room_1", "inv_worker")
    b = RoomTools(roomsd.url, "room_1", "tok_boostie")
    assert a.rooms_note_put("decisions", ["sqlite"])["revision"] == 1  # create
    lost = b.rooms_note_put("decisions", ["uv"])  # b never read it: create-only fails
    assert lost["conflict"] and not lost["written"]
    assert lost["current_value"] == ["sqlite"] and lost["current_revision"] == 1
    assert roomsd.notes["decisions"]["value"] == ["sqlite"]  # nothing overwritten
    # the conflict told b the current revision, so a merged retry goes through
    merged = b.rooms_note_put("decisions", [*lost["current_value"], "uv"])
    assert merged["written"] and merged["revision"] == 2
    # a's knowledge is now stale (revision 1): its write is refused too
    stale = a.rooms_note_put("decisions", ["sqlite", "polling"])
    assert stale["conflict"] and stale["current_value"] == ["sqlite", "uv"]
    assert a.rooms_note_put("decisions", ["sqlite", "uv", "polling"])["revision"] == 3


def test_note_get_and_read_record_revisions(roomsd):
    a = RoomTools(roomsd.url, "room_1", "inv_worker")
    b = RoomTools(roomsd.url, "room_1", "tok_boostie")
    a.rooms_note_put("summary", "v1")
    assert b.rooms_note_get("summary")["revision"] == 1
    assert b.rooms_note_put("summary", "v2")["written"]  # read first, so it may replace
    assert a.rooms_read(include_notes=["summary"])["note_revisions"] == {"summary": 2}
    assert a.rooms_note_put("summary", "v3")["revision"] == 3
    assert b.rooms_note_get()["revisions"] == {"summary": 3}


def test_note_put_explicit_revision(roomsd):
    a = RoomTools(roomsd.url, "room_1", "inv_worker")
    a.rooms_note_put("k", 1)
    b = RoomTools(roomsd.url, "room_1", "tok_boostie")
    assert b.rooms_note_put("k", 2, if_revision=1)["written"]
    assert b.rooms_note_put("k", 3, if_revision=1)["conflict"]


def test_note_put_schema_exposes_if_revision():
    server = room_tools.build_server(RoomTools("http://x", "room_1", "t"))
    tools = {t.name: t for t in server._tool_manager.list_tools()}
    assert "if_revision" in json.dumps(tools["rooms_note_put"].parameters)


def test_first_read_is_bounded(roomsd):
    for i in range(room_tools.HISTORY_ON_FIRST_READ + 10):
        roomsd.post("boostie@test", f"m{i}")
    msgs = RoomTools(roomsd.url, "room_1", "inv_worker").rooms_read()["messages"]
    assert len(msgs) == room_tools.HISTORY_ON_FIRST_READ
    assert msgs[-1]["body"] == f"m{room_tools.HISTORY_ON_FIRST_READ + 9}"


def test_errors_are_reported(roomsd):
    tools = RoomTools(roomsd.url, "room_1", "inv_worker")
    with pytest.raises(RuntimeError, match="roomsd 404"):
        tools.rooms_note_get("missing")
    with pytest.raises(ValueError, match="summarise"):
        tools.rooms_send("x" * (room_tools.BODY_LIMIT + 1))
    roomsd.revoked.add("inv_worker")
    with pytest.raises(RuntimeError, match="roomsd 401"):
        tools.rooms_send("after revoke")


# ----- over MCP stdio, the way claude talks to it ---------------------------------------


def test_mcp_server_lists_and_calls_tools(roomsd):
    env = {"ROOMSD_URL": roomsd.url, "ROOMSD_ROOM_ID": "room_1", "ROOMSD_TOKEN": "inv_worker"}
    proc = subprocess.Popen(
        [sys.executable, "-m", "agentd.workers.room_tools"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        env=env,
        text=True,
    )

    def rpc(id_, method, params=None):
        msg = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        if id_ is not None:
            msg["id"] = id_
        proc.stdin.write(json.dumps(msg) + "\n")
        proc.stdin.flush()
        if id_ is None:
            return None
        while True:
            reply = json.loads(proc.stdout.readline())
            if reply.get("id") == id_:
                return reply

    try:
        init = rpc(
            1,
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "0"},
            },
        )
        assert init["result"]["serverInfo"]["name"] == "rooms"
        rpc(None, "notifications/initialized")
        listed = rpc(2, "tools/list")["result"]["tools"]
        assert {t["name"] for t in listed} == set(room_tools.TOOL_NAMES)
        send = next(t for t in listed if t["name"] == "rooms_send")
        assert "body" in send["inputSchema"]["properties"]

        called = rpc(3, "tools/call", {"name": "rooms_send", "arguments": {"body": "via mcp"}})
        assert not called["result"].get("isError"), called
        assert roomsd.messages[-1]["body"] == "via mcp"
        assert roomsd.messages[-1]["from"] == WORKER
    finally:
        proc.stdin.close()
        proc.wait(timeout=10)


# ----- adapter wiring -----------------------------------------------------------------


def test_room_tools_attached_only_with_a_room():
    args = parse_args([])
    profile = {"workspace_mount": "none"}
    room_env = {
        "ROOMSD_URL": "http://r",
        "ROOMSD_ROOM_ID": "room_1",
        "ROOMSD_TOKEN": "t",
        "ROOMSD_ROOM_URL": "http://r/v1/rooms/room_1",
    }
    cmd = build_command(args, profile, room_env)
    config = json.loads(cmd[cmd.index("--mcp-config") + 1])
    server = config["mcpServers"]["rooms"]
    assert server["args"] == ["-m", "agentd.workers.room_tools"]
    assert server["env"]["ROOMSD_TOKEN"] == "t"
    allowed = cmd[cmd.index("--allowedTools") + 1 : cmd.index("--permission-mode")]
    assert set(room_tools.claude_tool_names()) <= set(allowed)
    # Room tools are allowed, not added to the built-in --tools set.
    tools = cmd[cmd.index("--tools") + 1 : cmd.index("--allowedTools")]
    assert not any(t.startswith("mcp__") for t in tools)
    assert "rooms_send" in cmd[cmd.index("--append-system-prompt") + 1]

    plain = build_command(args, profile, {})
    assert "--mcp-config" not in plain
    assert not any(t.startswith("mcp__") for t in plain)


def test_wake_policy():
    g = WakeGate(identity=WORKER)
    msg = {"from": "boostie@test", "type": "message", "body": ""}
    ids = iter(range(1, 100))

    def wake(**kw):
        return g.offer({**msg, "id": next(ids), **kw}, in_turn=False)[0]

    assert wake(body="general chatter") == "ignore"
    assert wake(body="@agentd-test.claude can you check CI?") == "deliver"
    assert wake(body=f"{WORKER}: ping") == "deliver"
    assert wake(**{"from": WORKER}, body="@agentd-test.claude me") == "ignore"  # never itself
    g.policy = "all"
    assert wake(body="general chatter") == "deliver"


@pytest.fixture
def room_client(settings, lobby, tmp_path):
    cmd = [
        sys.executable,
        "-m",
        "agentd.workers.claude_code",
        "--claude",
        f"{sys.executable} {FAKE_CLAUDE}",
        "--interactive",
        "--room-poll-seconds",
        "0.1",
        "--exit-grace-seconds",
        "2",
    ]
    s = settings.model_copy(update={"worker_types": {"claude-chat": WorkerType(command=cmd)}})
    with TestClient(create_app(s, verifier=lobby.verifier())) as c:
        yield c


def test_mention_in_room_wakes_worker(room_client, roomsd, boostie):
    roomsd.post("boostie@test", "history: should not wake anyone, @agentd-test.claude")
    sid = spawn(
        room_client,
        boostie,
        "first",
        worker_type="claude-chat",
        room={"room_url": roomsd.room_url, "token": "inv_worker"},
    ).json()["session_id"]
    wait_event(room_client, sid, boostie, lambda e: e["type"] == "needs_input")
    assert WORKER in roomsd.participants  # announced itself

    roomsd.post("boostie@test", "chatter that is not for the worker")
    roomsd.post("boostie@test", "@agentd-test.claude please review the schema")
    woke = wait_event(
        room_client, sid, boostie, lambda e: e["type"] == "needs_input" and e.get("turn") == 2
    )
    assert (
        '<room-message id="4" from="boostie@test" type="message" hop="0" trust="untrusted">\n'
        "@agentd-test.claude please review" in (woke["question"])
    )
    room_progress = [
        e for e in events(room_client, sid, boostie) if e.get("room_message_id") is not None
    ]
    assert [e["room_message_id"] for e in room_progress] == [4]  # not history, not chatter

    s = room_client.post(f"/v1/sessions/{sid}/stop", headers=boostie).json()
    assert s["status"] == "completed"
    wait_status(room_client, sid, boostie)
    # agentd's close-out (which runs just after the session turns terminal) posts the
    # handoff and revokes the invite.
    deadline = time.monotonic() + 10
    while "inv_worker" not in roomsd.revoked and time.monotonic() < deadline:
        time.sleep(0.05)
    assert roomsd.messages[-1]["type"] == "handoff"
    assert "inv_worker" in roomsd.revoked


# ----- standalone use: an owner's own Claude Code session (live test 2) ------------------


def test_tools_join_the_room_on_first_use(roomsd):
    """Inside agentd the adapter joins before starting the tools; a standalone session
    holding only an invite must join by itself instead of failing every call with 403."""
    roomsd.require_join = True
    roomsd.post("boostie@test", "hello")
    tools = RoomTools(roomsd.url, "room_1", "inv_worker")
    assert [m["body"] for m in tools.rooms_read()["messages"]] == ["hello"]
    assert WORKER in roomsd.participants
    tools.rooms_send("hi back")  # already joined: no second join needed
    assert roomsd.messages[-1]["from"] == WORKER


def test_errors_reach_the_model_through_mcp(roomsd):
    """The MCP SDK masks any non-ToolError as 'Error executing tool'; the model must see
    why a call failed (revoked invite, secret refused, too long) to act on it."""
    import asyncio

    from mcp.server.mcpserver.exceptions import ToolError, UnexpectedToolError

    tools = RoomTools(roomsd.url, "room_1", "inv_worker", secrets=["sk-ant-xyz-123456"])
    server = room_tools.build_server(tools)

    def failure(name, args):
        with pytest.raises(ToolError) as e:
            asyncio.run(server.call_tool(name, args))
        assert not isinstance(e.value, UnexpectedToolError)
        return str(e.value)

    assert "refusing to post" in failure("rooms_send", {"body": "key sk-ant-xyz-123456"})
    assert "summarise" in failure("rooms_send", {"body": "x" * (room_tools.BODY_LIMIT + 1)})
    roomsd.revoked.add("inv_worker")
    assert "roomsd 401" in failure("rooms_read", {})


def test_rooms_mcp_entry_point():
    from importlib.metadata import entry_points

    (ep,) = [e for e in entry_points(group="console_scripts") if e.name == "rooms-mcp"]
    assert ep.value == "agentd.workers.room_tools:main"
