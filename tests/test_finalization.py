"""Regression tests for room-o-matic/docs#2: malformed worker output and event-store
failures must never leave a dead worker active or make stop/shutdown hang."""

import time

import pytest
from helpers import events, spawn, wait_status

from agentd.protocol import parse_stdout_line


@pytest.mark.parametrize(
    ("line", "reason"),
    [
        ('{"type":"final","summary":{"not":"a string"}}', "final.summary must be str, got dict"),
        ('{"type":"final","summary":["a"]}', "final.summary must be str, got list"),
        ('{"type":"final","summary":42}', "final.summary must be str, got int"),
        ('{"type":"final","artifacts":["ok", 3]}', "final.artifacts must be list[str]"),
        ('{"type":"artifact","path":"x"}', "artifact.name is required"),
        ('{"type":"progress","message":{"x":1}}', "progress.message must be str"),
        ('{"type":"needs_input","choices":"yes"}', "needs_input.choices must be list[str]"),
    ],
)
def test_typed_payload_validation(line, reason):
    event = parse_stdout_line("AGENT_EVENT " + line)
    assert event["type"] == "protocol_error"
    assert event["reason"].startswith(reason)


def test_valid_payloads_pass_and_extra_fields_are_kept():
    event = parse_stdout_line(
        'AGENT_EVENT {"type":"final","summary":null,"cost_usd":0.1,"anything":{"x":[1]}}'
    )
    assert event == {"type": "final", "summary": None, "cost_usd": 0.1, "anything": {"x": [1]}}


@pytest.mark.parametrize("kind", ["dict", "list", "number"])
def test_malformed_final_fails_session_and_releases_capacity(client, boostie, kind):
    sid = spawn(client, boostie, f"badfinal {kind}").json()["session_id"]
    s = wait_status(client, sid, boostie)
    assert s["status"] == "failed"
    assert "without a final event" in s["stop_reason"]
    assert s["summary"] is None
    errs = [e for e in events(client, sid, boostie) if e["type"] == "protocol_error"]
    assert errs and errs[0]["reason"].startswith("final.summary must be str")
    assert client.get("/v1/instance", headers=boostie).json()["active_sessions"] == 0
    # stop on an ended session returns immediately instead of waiting on a lost signal
    t = time.monotonic()
    assert client.post(f"/v1/sessions/{sid}/stop", headers=boostie).status_code == 200
    assert time.monotonic() - t < 1


def test_event_store_failure_at_finalization_still_releases(client, boostie, monkeypatch):
    sup = client.app.state.supervisor
    real_append = sup.events.append

    def flaky_append(session_id, type, **payload):
        if type == "status" and payload.get("status") in ("completed", "failed"):
            raise RuntimeError("event store is down")
        return real_append(session_id, type, **payload)

    monkeypatch.setattr(sup.events, "append", flaky_append)
    sid = spawn(client, boostie, "say hello").json()["session_id"]
    deadline = time.monotonic() + 10
    while sup.active_count() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert sup.active_count() == 0
    row = sup.get_row(sid)
    # The failure is observable rather than the session silently looking healthy.
    assert row["status"] == "failed"
    assert row["stop_reason"].startswith("gateway error during finalization")
    t = time.monotonic()
    client.post(f"/v1/sessions/{sid}/stop", headers=boostie)
    assert time.monotonic() - t < 1


def test_event_store_failure_mid_run_ends_session(client, boostie, monkeypatch):
    sup = client.app.state.supervisor
    real_append = sup.events.append

    def flaky_append(session_id, type, **payload):
        if type == "progress":
            raise RuntimeError("event store is down")
        return real_append(session_id, type, **payload)

    monkeypatch.setattr(sup.events, "append", flaky_append)
    sid = spawn(client, boostie, "interactive").json()["session_id"]
    s = wait_status(client, sid, boostie)
    assert s["status"] == "failed"
    assert "gateway error" in s["stop_reason"]
    assert sup.active_count() == 0
