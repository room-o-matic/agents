"""Regression tests for room-o-matic/docs#16 (worker side): who wakes a worker, and how
floods, reply loops and repeated deliveries stay bounded and visible."""

import sys
import time
from pathlib import Path

import pytest
from fake_roomsd import FakeRoomsd
from fastapi.testclient import TestClient
from helpers import events, spawn, wait_event, wait_status

from agentd.app import create_app
from agentd.config import WorkerType
from agentd.workers.wake import WakeGate

FAKE_CLAUDE = Path(__file__).parent / "fake_claude.py"
ME = "missy@test/agentd-test.claude"
_ids = iter(range(1, 10**6))


def msg(**kw):
    return {"id": next(_ids), "from": "boostie@test", "type": "message", "body": "", **kw}


def test_exact_addressing_and_similar_names():
    g = WakeGate(identity=ME)
    assert g.offer(msg(body="@agentd-test.claude-other please"), in_turn=False)[0] == "ignore"
    assert g.offer(msg(body="ping @agentd-test.claude."), in_turn=False)[0] == "deliver"
    assert g.offer(msg(body="x", to=[ME]), in_turn=False)[0] == "deliver"
    assert g.offer(msg(body="x", to=["missy@test/someone-else"]), in_turn=False)[0] == "ignore"
    assert g.offer(msg(body="email@agentd-test.claude.example"), in_turn=False)[0] == "ignore"


def test_informational_types_never_wake():
    g = WakeGate(identity=ME)
    for t in ("status", "handoff", "decision", "artifact", "task_update"):
        assert g.offer(msg(type=t, to=[ME]), in_turn=False) == ("ignore", "informational")
    assert g.wakes_used == 0


def test_reply_loops_are_cut_by_hop_depth():
    g = WakeGate(identity=ME, max_hop=3)
    decisions = [g.offer(msg(to=[ME], hop=h), in_turn=False)[0] for h in range(6)]
    assert decisions == ["deliver", "deliver", "deliver", "suppress", "suppress", "suppress"]
    assert g.suppressed == {"reply_chain_too_deep": 3}


def test_repeated_delivery_wakes_once():
    g = WakeGate(identity=ME)
    m = msg(to=[ME])
    assert g.offer(m, in_turn=False)[0] == "deliver"
    assert g.offer(dict(m), in_turn=False) == ("ignore", "duplicate")


def test_flood_is_bounded_and_coalesced():
    g = WakeGate(identity=ME, max_queue=5, max_wakes=10)
    assert g.offer(msg(to=[ME]), in_turn=False)[0] == "deliver"  # starts a turn
    outcomes = [g.offer(msg(to=[ME]), in_turn=True)[0] for _ in range(100)]
    assert outcomes.count("queue") == 5 and outcomes.count("suppress") == 95
    assert g.suppressed == {"queue_full": 95}
    assert len(g.drain()) == 5  # one coalesced turn for the queue
    assert g.wakes_used == 2 and g.drain() == []


def test_wake_budget():
    g = WakeGate(identity=ME, max_wakes=2)
    results = [g.offer(msg(to=[ME]), in_turn=False)[0] for _ in range(4)]
    assert results == ["deliver", "deliver", "suppress", "suppress"]
    assert g.suppressed == {"wake_budget_exhausted": 2}


# ----- through a running worker -------------------------------------------------------


@pytest.fixture
def roomsd():
    fake = FakeRoomsd({"inv_worker": ME, "tok_boostie": "boostie@test"})
    yield fake
    fake.close()


@pytest.fixture
def worker(settings, lobby):
    cmd = [
        sys.executable,
        "-m",
        "agentd.workers.claude_code",
        "--claude",
        f"{sys.executable} {FAKE_CLAUDE}",
        "--interactive",
        "--room-poll-seconds",
        "0.05",
        "--room-max-wakes",
        "2",
        "--room-max-queue",
        "3",
        "--exit-grace-seconds",
        "2",
    ]
    s = settings.model_copy(update={"worker_types": {"claude-chat": WorkerType(command=cmd)}})
    with TestClient(create_app(s, verifier=lobby.verifier())) as c:
        yield c


def turns(c, sid, h):
    return [e for e in events(c, sid, h) if e["type"] == "needs_input"]


def test_mention_flood_costs_bounded_turns_and_is_reported(worker, roomsd, boostie):
    c = worker
    sid = spawn(
        c,
        boostie,
        "first",
        worker_type="claude-chat",
        room={"room_url": roomsd.room_url, "token": "inv_worker"},
    ).json()["session_id"]
    wait_event(c, sid, boostie, lambda e: e["type"] == "needs_input")
    for i in range(40):
        roomsd.post("boostie@test", f"@agentd-test.claude flood {i}")
    wait_event(c, sid, boostie, lambda e: e.get("room_wake_suppressed"))
    time.sleep(1)
    roomsd.post("boostie@test", "@agentd-test.claude one more")  # past the wake budget
    wait_event(
        c,
        sid,
        boostie,
        lambda e: (e.get("room_wake_suppressed") or {}).get("reason") == "wake_budget_exhausted",
    )
    time.sleep(0.5)
    # 1 task turn + at most 2 room-triggered turns, however many mentions arrived
    assert len(turns(c, sid, boostie)) <= 3
    reasons = {
        e["room_wake_suppressed"]["reason"]
        for e in events(c, sid, boostie)
        if e.get("room_wake_suppressed")
    }
    assert {"queue_full", "wake_budget_exhausted"} <= reasons  # visible, not silent
    assert any(
        "wake budget exhausted" in m["body"] for m in roomsd.messages if m["from"] == ME
    )  # and announced in the room
    s = c.post(f"/v1/sessions/{sid}/stop", headers=boostie).json()  # operator stop works
    assert s["status"] in ("completed", "stopped")


def test_no_wakes_after_cancellation(worker, roomsd, boostie):
    c = worker
    sid = spawn(
        c,
        boostie,
        "first",
        worker_type="claude-chat",
        room={"room_url": roomsd.room_url, "token": "inv_worker"},
    ).json()["session_id"]
    wait_event(c, sid, boostie, lambda e: e["type"] == "needs_input")
    c.post(f"/v1/sessions/{sid}/stop", headers=boostie)
    wait_status(c, sid, boostie)
    wait_event(c, sid, boostie, lambda e: e["type"] == "room_finalization")  # docs#17
    before = len(events(c, sid, boostie))
    roomsd.post("boostie@test", "@agentd-test.claude are you there?")
    time.sleep(0.3)
    assert len(events(c, sid, boostie)) == before  # a stopped session never wakes


def test_mention_right_after_joining_wakes_the_worker(worker, roomsd, boostie):
    """The wake cursor is fixed before the worker announces itself, so a mention sent the
    moment it appears in the room (e.g. right after summon) isn't skipped as history."""
    c = worker
    sid = spawn(
        c,
        boostie,
        "first",
        worker_type="claude-chat",
        room={"room_url": roomsd.room_url, "token": "inv_worker"},
    ).json()["session_id"]
    deadline = time.monotonic() + 10
    while not any(m["from"] == ME for m in roomsd.messages):  # the join announcement
        assert time.monotonic() < deadline, "worker never announced itself"
        time.sleep(0.005)
    roomsd.post("boostie@test", "@agentd-test.claude are you there?")
    wait_event(c, sid, boostie, lambda e: e["type"] == "needs_input" and e.get("turn") == 2)
    c.post(f"/v1/sessions/{sid}/stop", headers=boostie)
