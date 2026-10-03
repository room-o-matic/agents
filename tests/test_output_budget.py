"""Regression tests for room-o-matic/docs#22: a session's structured worker output has its
own byte/count/rate budget (separate from the raw-log cap), the terminal result survives
it, a flood doesn't starve other sessions, and old event logs are pruned."""

import time

import pytest
from fastapi.testclient import TestClient
from helpers import events, spawn, wait_status

from agentd.app import create_app


def types(evs):
    return [e["type"] for e in evs]


def make(settings, lobby, **limits):
    s = settings.model_copy(update=limits)
    return TestClient(create_app(s, verifier=lobby.verifier()))


@pytest.fixture
def budgeted(settings, lobby):
    # The original report: max_log_bytes=128 and 200 progress events of 1000 chars all
    # persisted. Here the structured budget is separate and much smaller than the flood.
    with make(
        settings,
        lobby,
        max_log_bytes=128,
        max_events=50,
        max_event_bytes=20_000,
        event_rate_per_second=10_000,
        event_burst=10_000,
    ) as c:
        yield c


def test_progress_flood_is_capped_with_one_record_and_final_kept(budgeted, boostie):
    sid = spawn(budgeted, boostie, "flood 200 1000").json()["session_id"]
    s = wait_status(budgeted, sid, boostie)
    assert s["status"] == "completed" and s["summary"] == "flooded"
    evs = events(budgeted, sid, boostie)
    assert types(evs).count("output_truncated") == 1
    progress = [e for e in evs if e["type"] == "progress"]
    assert len(progress) < 50
    assert sum(len(e["message"]) for e in progress) <= 20_000
    assert types(evs).count("final") == 1


def test_count_budget(budgeted, boostie):
    sid = spawn(budgeted, boostie, "flood 500 1").json()["session_id"]
    wait_status(budgeted, sid, boostie)
    evs = events(budgeted, sid, boostie)
    assert types(evs).count("progress") == 50  # gateway status events are not charged
    assert types(evs).count("output_truncated") == 1


def test_protocol_error_flood_is_capped(budgeted, boostie):
    sid = spawn(budgeted, boostie, "badflood 500").json()["session_id"]
    s = wait_status(budgeted, sid, boostie)
    assert s["status"] == "completed"
    evs = events(budgeted, sid, boostie)
    assert types(evs).count("protocol_error") <= 50
    assert types(evs).count("output_truncated") == 1


def test_repeated_final_is_recorded_once(budgeted, boostie):
    sid = spawn(budgeted, boostie, "finals 300").json()["session_id"]
    s = wait_status(budgeted, sid, boostie)
    assert s["status"] == "completed" and s["summary"] == "final 0"
    evs = events(budgeted, sid, boostie)
    assert types(evs).count("final") == 1
    assert types(evs).count("protocol_error") <= 50  # duplicates are budgeted errors


def test_rate_budget_throttles_and_says_so(settings, lobby, boostie):
    with make(settings, lobby, event_rate_per_second=5, event_burst=10) as c:
        sid = spawn(c, boostie, "flood 300 10").json()["session_id"]
        s = wait_status(c, sid, boostie)
        assert s["status"] == "completed"
        evs = events(c, sid, boostie)
        assert types(evs).count("progress") < 30
        assert types(evs).count("output_throttled") >= 1
        assert types(evs).count("final") == 1


def test_flood_does_not_starve_other_sessions(settings, lobby, boostie):
    with make(
        settings,
        lobby,
        event_rate_per_second=1e9,
        event_burst=10**9,
        max_events=10**9,
        max_event_bytes=10**12,
    ) as c:
        quiet = spawn(c, boostie, "interactive").json()["session_id"]
        loud = spawn(c, boostie, "flood 30000 200").json()["session_id"]
        worst = 0.0
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            t0 = time.monotonic()
            st = c.get(f"/v1/sessions/{loud}", headers=boostie).json()["status"]
            c.get(f"/v1/sessions/{quiet}", headers=boostie)
            worst = max(worst, time.monotonic() - t0)
            if st not in ("starting", "running"):
                break
            time.sleep(0.02)
        assert worst < 0.5, f"a status read took {worst:.2f}s during another session's flood"
        c.post(f"/v1/sessions/{quiet}/stop", headers=boostie)


def test_old_event_logs_are_pruned(settings, lobby, boostie):
    with make(settings, lobby, event_retention_days=0) as c:
        sid = spawn(c, boostie, "say hello").json()["session_id"]
        assert wait_status(c, sid, boostie)["status"] == "completed"
        time.sleep(0.01)
        sup = c.app.state.supervisor
        assert sup.prune_events() == 1
        assert events(c, sid, boostie) == []
        assert not (settings.sessions_dir / sid / "events.jsonl").exists()
        s = c.get(f"/v1/sessions/{sid}", headers=boostie).json()
        assert s["status"] == "completed" and s["summary"]  # the record stays
        assert sup.prune_events() == 0


def test_retention_disabled_keeps_everything(settings, lobby, boostie):
    with make(settings, lobby, event_retention_days=None) as c:
        sid = spawn(c, boostie, "say hello").json()["session_id"]
        wait_status(c, sid, boostie)
        assert c.app.state.supervisor.prune_events() == 0
        assert events(c, sid, boostie)
