import json

from helpers import events, spawn, wait_event, wait_status


def types(evs):
    return [e["type"] for e in evs]


def test_spawn_runs_to_completion(client, boostie, settings):
    r = spawn(client, boostie, "say hello")
    assert r.status_code == 201
    body = r.json()
    sid = body["session_id"]
    assert sid.startswith("agt_")
    assert body["instance_id"] == "agentd-test"
    assert body["events_url"] == f"/v1/sessions/{sid}/events"

    s = wait_status(client, sid, boostie)
    assert s["status"] == "completed"
    assert s["summary"] == "did: say hello"
    assert s["exit_code"] == 0
    assert s["requester"]["agent"] == "boostie"

    evs = events(client, sid, boostie)
    statuses = [e["status"] for e in evs if e["type"] == "status"]
    assert statuses == ["starting", "running", "completed"]
    assert {"progress", "log", "artifact", "final"} <= set(types(evs))
    logs = {(e["stream"], e["text"]) for e in evs if e["type"] == "log"}
    assert ("stdout", "hello from the fake worker") in logs
    assert ("stderr", "a stderr line") in logs

    artifact = next(e for e in evs if e["type"] == "artifact")
    assert artifact["exists"] is True
    assert artifact["path"] == str(settings.sessions_dir / sid / "artifacts" / "notes.md")


def test_events_jsonl_mirrors_db(client, boostie, settings):
    sid = spawn(client, boostie).json()["session_id"]
    wait_status(client, sid, boostie)
    session_dir = settings.sessions_dir / sid
    lines = (session_dir / "events.jsonl").read_text().splitlines()
    assert [json.loads(line) for line in lines] == events(client, sid, boostie)
    assert json.loads((session_dir / "input.json").read_text())["task"] == "say hello"


def test_follow_up_messages(client, boostie):
    sid = spawn(client, boostie, "interactive").json()["session_id"]
    wait_status(client, sid, boostie, {"running"})

    r = client.post(f"/v1/sessions/{sid}/messages", json={"message": "hi"}, headers=boostie)
    assert r.json() == {"accepted": True, "status": "running"}
    echo = wait_event(client, sid, boostie, lambda e: e.get("message") == "echo from boostie: hi")
    assert echo["type"] == "progress"

    client.post(f"/v1/sessions/{sid}/messages", json={"message": "done"}, headers=boostie)
    assert wait_status(client, sid, boostie)["status"] == "completed"
    sent = [e for e in events(client, sid, boostie) if e["type"] == "message"]
    assert [(e["sender"], e["message"]) for e in sent] == [("boostie", "hi"), ("boostie", "done")]

    r = client.post(f"/v1/sessions/{sid}/messages", json={"message": "late"}, headers=boostie)
    assert r.status_code == 409


def test_stop(client, boostie):
    sid = spawn(client, boostie, "interactive").json()["session_id"]
    wait_status(client, sid, boostie, {"running"})
    r = client.post(
        f"/v1/sessions/{sid}/stop", json={"reason": "no longer needed"}, headers=boostie
    )
    assert r.status_code == 200
    s = r.json()
    assert s["status"] == "stopped"
    assert s["stop_reason"] == "no longer needed"
    assert s["exit_code"] == 0  # the worker honoured the stop message


def test_stop_escalates_to_kill(client, boostie):
    sid = spawn(client, boostie, "stubborn").json()["session_id"]
    wait_status(client, sid, boostie, {"running"})
    s = client.post(f"/v1/sessions/{sid}/stop", headers=boostie).json()
    assert s["status"] == "stopped"
    assert s["stop_reason"] == "caller_cancelled"
    assert s["exit_code"] == -9


def test_crash_without_final_fails(client, boostie):
    sid = spawn(client, boostie, "crash").json()["session_id"]
    s = wait_status(client, sid, boostie)
    assert s["status"] == "failed"
    assert s["exit_code"] == 3
    assert "without a final event" in s["stop_reason"]


def test_ready_timeout_fails(client, boostie):
    sid = spawn(client, boostie, "hang").json()["session_id"]
    s = wait_status(client, sid, boostie, timeout=15)
    assert s["status"] == "failed"
    assert "no event within" in s["stop_reason"]


def test_lingering_worker_reaped_after_final(client, boostie):
    sid = spawn(client, boostie, "linger").json()["session_id"]
    s = wait_status(client, sid, boostie)
    assert s["status"] == "completed"
    assert s["summary"] == "done, but not exiting"


def test_bad_worker_output_is_visible_not_fatal(client, boostie):
    sid = spawn(client, boostie, "badjson").json()["session_id"]
    assert wait_status(client, sid, boostie)["status"] == "completed"
    errors = [e for e in events(client, sid, boostie) if e["type"] == "protocol_error"]
    assert [e["reason"] for e in errors][0] == "invalid JSON"
    assert "event type must be one of" in errors[1]["reason"]  # worker tried to send status


def test_artifact_path_escape_rejected(client, boostie):
    sid = spawn(client, boostie, "escape").json()["session_id"]
    wait_status(client, sid, boostie)
    evs = events(client, sid, boostie)
    assert "artifact" not in types(evs)
    assert any(e.get("reason") == "artifact path escapes artifacts dir" for e in evs)


def test_log_volume_capped(client, boostie):
    sid = spawn(client, boostie, "chatty 100").json()["session_id"]
    assert wait_status(client, sid, boostie)["status"] == "completed"
    evs = events(client, sid, boostie)
    assert types(evs).count("log_truncated") == 1
    assert types(evs).count("log") < 100
    assert "final" in types(evs)  # structured events still flow after logs are capped


def test_idle_timeout_expires(client, boostie):
    sid = spawn(client, boostie, "interactive", idle_timeout_seconds=0.5).json()["session_id"]
    s = wait_status(client, sid, boostie)
    assert s["status"] == "expired"
    assert s["stop_reason"] == "idle_timeout"


def test_hard_timeout_expires(client, boostie):
    sid = spawn(client, boostie, "interactive", timeout_seconds=1, idle_timeout_seconds=100).json()[
        "session_id"
    ]
    s = wait_status(client, sid, boostie)
    assert s["status"] == "expired"
    assert s["stop_reason"] == "hard_timeout"


def test_sse_stream_replays_and_closes(client, boostie):
    sid = spawn(client, boostie).json()["session_id"]
    with client.stream("GET", f"/v1/sessions/{sid}/events", headers=boostie) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        frames = [f for f in r.read().decode().split("\n\n") if f.startswith("id:")]
    parsed = [json.loads(f.split("data: ", 1)[1]) for f in frames]
    assert parsed[-1] == {**parsed[-1], "type": "status", "status": "completed"}
    ids = [p["id"] for p in parsed]
    assert ids == sorted(ids)

    # Resume from Last-Event-ID: only later events.
    with client.stream(
        "GET", f"/v1/sessions/{sid}/events", headers={**boostie, "Last-Event-ID": str(ids[-2])}
    ) as r:
        resumed = [f for f in r.read().decode().split("\n\n") if f.startswith("id:")]
    assert len(resumed) == 1


def test_list_sessions(client, boostie, missy):
    sid = spawn(client, boostie, "interactive").json()["session_id"]
    wait_status(client, sid, boostie, {"running"})
    assert [s["session_id"] for s in client.get("/v1/sessions", headers=boostie).json()] == [sid]
    active = client.get("/v1/sessions", params={"active": True}, headers=boostie).json()
    assert [s["session_id"] for s in active] == [sid]
    assert client.get("/v1/sessions", headers=missy).json() == []
    client.post(f"/v1/sessions/{sid}/stop", headers=boostie)
    assert client.get("/v1/sessions", params={"active": True}, headers=boostie).json() == []


def test_instance_info(client, boostie):
    info = client.get("/v1/instance", headers=boostie).json()
    assert info["instance_id"] == "agentd-test"
    assert info["worker_types"] == ["fake"]
    assert "workspace_coder" in info["profiles"]
    assert info["max_sessions"] == 2
    assert info["registry_enabled"] is False
