"""Regression tests for room-o-matic/docs#17: room finalization is checked, retried,
visible, covers failed launches, shutdown and restarts, and bounds session lifetime."""

import time
from datetime import UTC, datetime, timedelta

import pytest
from fake_roomsd import FakeRoomsd
from fastapi.testclient import TestClient
from helpers import spawn, wait_status

from agentd import db
from agentd.app import create_app
from agentd.config import WorkerType
from agentd.ids import now_iso

WORKER = "missy@test/agentd-test.fake"


@pytest.fixture
def roomsd():
    fake = FakeRoomsd({"inv_worker": WORKER})
    yield fake
    fake.close()


@pytest.fixture
def fast(settings):
    return settings.model_copy(update={"finalize_retry_seconds": 0.05, "finalize_max_attempts": 3})


def room(roomsd, **kw):
    return {"room_url": roomsd.room_url, "token": "inv_worker", "invite_id": "inv_1", **kw}


def finalization(c, sid, h, want, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        s = c.get(f"/v1/sessions/{sid}", headers=h).json()
        if s["room_finalization"] in want:
            return s
        time.sleep(0.05)
    raise AssertionError(f"finalization stuck at {s['room_finalization']}: {s}")


def test_successful_finalization_is_recorded(client, roomsd, boostie):
    sid = spawn(client, boostie, room=room(roomsd)).json()["session_id"]
    s = finalization(client, sid, boostie, {"done"})
    assert s["room_invite_id"] == "inv_1" and s["room_finalization_error"] is None
    assert "inv_worker" in roomsd.revoked and roomsd.messages[-1]["type"] == "handoff"


def test_revoke_failure_retries_then_hands_to_owner(fast, lobby, roomsd, boostie):
    roomsd.revoke_status = 500
    with TestClient(create_app(fast, verifier=lobby.verifier())) as c:
        sid = spawn(c, boostie, room=room(roomsd)).json()["session_id"]
        wait_status(c, sid, boostie)
        s = finalization(c, sid, boostie, {"owner_required"})
    assert (
        "HTTP 500" in s["room_finalization_error"] and "3 attempts" in s["room_finalization_error"]
    )
    assert s["status"] == "completed"  # the session's own status is separate from cleanup


def test_retry_recovers_when_roomsd_comes_back(fast, lobby, roomsd, boostie):
    roomsd.revoke_status = 500
    with TestClient(
        create_app(fast.model_copy(update={"finalize_max_attempts": 50}), verifier=lobby.verifier())
    ) as c:
        sid = spawn(c, boostie, room=room(roomsd)).json()["session_id"]
        wait_status(c, sid, boostie)
        time.sleep(0.3)
        roomsd.revoke_status = 204
        finalization(c, sid, boostie, {"done", "revoked"})
    assert "inv_worker" in roomsd.revoked


def test_revoke_401_means_already_dead(client, roomsd, boostie):
    roomsd.revoke_status = 401
    sid = spawn(client, boostie, room=room(roomsd)).json()["session_id"]
    assert finalization(client, sid, boostie, {"done", "revoked"})["room_finalization"]


def test_post_failure_still_revokes(client, roomsd, boostie):
    roomsd.post_status = 423  # e.g. the room was paused
    sid = spawn(client, boostie, room=room(roomsd)).json()["session_id"]
    s = finalization(client, sid, boostie, {"revoked"})
    assert "closing message" in s["room_finalization_error"]
    assert "inv_worker" in roomsd.revoked


def test_failed_launch_still_revokes_the_invite(settings, lobby, roomsd, boostie):
    broken = settings.model_copy(
        update={"worker_types": {"fake": WorkerType(command=["/nonexistent/worker"])}}
    )
    with TestClient(create_app(broken, verifier=lobby.verifier())) as c:
        r = spawn(c, boostie, room=room(roomsd))
        assert r.json()["status"] == "failed"
        finalization(c, r.json()["session_id"], boostie, {"done", "revoked"})
    assert "inv_worker" in roomsd.revoked


def test_shutdown_hands_pending_finalization_to_owner(fast, lobby, roomsd, boostie):
    roomsd.revoke_status = 500
    slow = fast.model_copy(update={"finalize_max_attempts": 1000, "finalize_retry_seconds": 60})
    with TestClient(create_app(slow, verifier=lobby.verifier())) as c:
        sid = spawn(c, boostie, room=room(roomsd)).json()["session_id"]
        wait_status(c, sid, boostie)
        finalization(c, sid, boostie, {"pending"})
    conn = db.connect(slow.db_path)
    row = conn.execute(
        "select room_finalization, room_finalization_error from sessions where id = ?", (sid,)
    ).fetchone()
    conn.close()
    assert row[0] == "owner_required" and "shut down" in row[1]


def test_restart_hands_owed_finalization_to_owner(settings, lobby, boostie):
    db.init_db(settings.db_path)
    conn = db.connect(settings.db_path)
    now = now_iso()
    with conn:
        conn.execute(
            "insert into sessions (id, instance_id, requester_agent, profile, worker_type,"
            " status, task, idle_timeout_seconds, created_at, last_activity_at, expires_at,"
            " room_url, room_invite_id, room_finalization) values ('agt_old', 'agentd-test',"
            " 'boostie@test', 'workspace_coder', 'fake', 'completed', 't', 60, ?, ?, ?,"
            " 'http://r/v1/rooms/room_1', 'inv_9', 'pending')",
            (now, now, now),
        )
    conn.close()
    (settings.sessions_dir / "agt_old").mkdir(parents=True)
    with TestClient(create_app(settings, verifier=lobby.verifier())) as c:
        s = c.get("/v1/sessions/agt_old", headers=boostie).json()
    assert s["room_finalization"] == "owner_required"
    assert s["room_invite_id"] == "inv_9"  # what the owner needs to revoke it
    assert "restarted" in s["room_finalization_error"]


def test_session_cannot_outlive_its_room_invite(client, roomsd, boostie):
    expires = (datetime.now(UTC) + timedelta(seconds=1.5)).isoformat().replace("+00:00", "Z")
    sid = spawn(
        client, boostie, "interactive", room=room(roomsd, expires_at=expires), timeout_seconds=3600
    ).json()["session_id"]
    s = wait_status(client, sid, boostie, timeout=10)
    assert (s["status"], s["stop_reason"]) == ("expired", "hard_timeout")


def test_already_expired_invite_is_refused(client, roomsd, boostie):
    past = (datetime.now(UTC) - timedelta(seconds=5)).isoformat()
    r = spawn(client, boostie, room=room(roomsd, expires_at=past))
    assert r.status_code == 422 and "expired" in r.json()["detail"]
