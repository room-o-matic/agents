"""Regression tests for room-o-matic/docs#3: cancellation is bounded even when the worker
won't read stdin, and lifecycle ownership covers the worker's whole process group."""

import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from helpers import events, spawn, wait_event, wait_status

from agentd.app import create_app

BIG = "x" * 64 * 1024


def alive(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except (FileNotFoundError, ProcessLookupError):
        return False
    return state != "Z"  # a zombie has exited; it's only waiting to be reaped


def worker_pid(client, sid) -> int:
    return client.app.state.conn.execute(
        "select pid from sessions where id = ?", (sid,)
    ).fetchone()["pid"]


@pytest.fixture
def small_buffer_client(settings, lobby):
    s = settings.model_copy(
        update={"max_pending_stdin_bytes": 256 * 1024, "send_timeout_seconds": 0.5}
    )
    with TestClient(create_app(s, verifier=lobby.verifier())) as c:
        yield c


def test_nonreading_worker_bounds_sends_and_stop(small_buffer_client, boostie):
    c = small_buffer_client
    sid = spawn(c, boostie, "deaf").json()["session_id"]
    wait_status(c, sid, boostie, {"running"})
    pid = worker_pid(c, sid)

    codes = []
    for _ in range(12):  # 768 KiB against a 256 KiB pending cap plus the pipe buffer
        t = time.monotonic()
        r = c.post(f"/v1/sessions/{sid}/messages", json={"message": BIG}, headers=boostie)
        assert time.monotonic() - t < 2, "a send blocked"
        codes.append(r.status_code)
    assert codes[0] == 200
    assert 409 in codes  # eventually refused instead of queueing without bound
    refused = c.post(f"/v1/sessions/{sid}/messages", json={"message": BIG}, headers=boostie)
    assert "unread input" in refused.json()["detail"]

    t = time.monotonic()
    s = c.post(f"/v1/sessions/{sid}/stop", headers=boostie).json()
    assert time.monotonic() - t < 5  # grace 1s: ask, wait, SIGTERM, SIGKILL
    assert s["status"] == "stopped"
    assert not alive(pid)


def test_concurrent_sends_and_stop_are_bounded(small_buffer_client, boostie):
    c = small_buffer_client
    sid = spawn(c, boostie, "deaf").json()["session_id"]
    wait_status(c, sid, boostie, {"running"})

    def flood():
        for _ in range(8):
            c.post(f"/v1/sessions/{sid}/messages", json={"message": BIG}, headers=boostie)

    senders = [threading.Thread(target=flood) for _ in range(3)]
    for th in senders:
        th.start()
    t = time.monotonic()
    s = c.post(f"/v1/sessions/{sid}/stop", headers=boostie).json()
    assert time.monotonic() - t < 6
    assert s["status"] == "stopped"
    for th in senders:
        th.join(timeout=10)
        assert not th.is_alive()


def test_stuck_session_does_not_stall_cleanup_of_others(small_buffer_client, boostie):
    c = small_buffer_client
    deaf = spawn(c, boostie, "deaf", idle_timeout_seconds=0.5).json()["session_id"]
    other = spawn(c, boostie, "interactive", idle_timeout_seconds=0.8).json()["session_id"]
    for sid in (deaf, other):
        s = wait_status(c, sid, boostie, timeout=10)
        assert (s["status"], s["stop_reason"]) == ("expired", "idle_timeout")


@pytest.mark.parametrize("variant", ["", "stubborn"])
def test_descendants_are_reaped_before_session_ends(client, boostie, variant):
    sid = spawn(client, boostie, f"orphan {variant}".strip()).json()["session_id"]
    child = wait_event(client, sid, boostie, lambda e: "child_pid" in e)["child_pid"]
    s = wait_status(client, sid, boostie, timeout=10)
    # By the time the session reports an end state, nothing in its group is left.
    assert not alive(child)
    assert s["status"] == "completed"
    assert client.get("/v1/instance", headers=boostie).json()["active_sessions"] == 0
    assert all(e["type"] != "protocol_error" for e in events(client, sid, boostie))
