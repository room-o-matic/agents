import json
import os
from datetime import datetime

from fastapi.testclient import TestClient
from helpers import spawn, wait_status

from agentd import auth, db
from agentd.app import create_app
from agentd.config import DEFAULT_WORKER_TYPES, Profile, Settings
from agentd.ids import now_iso


def test_requires_token(client):
    assert client.get("/v1/sessions").status_code == 401
    r = spawn(client, {"Authorization": "Bearer agtd_nope"})
    assert r.status_code == 401


def test_requester_identity_bound_to_token(client, boostie):
    r = spawn(client, boostie, requester={"agent": "missy", "surface": "discord"})
    assert r.status_code == 403
    r = spawn(client, boostie, requester={"agent": "boostie", "surface": "discord"})
    assert r.status_code == 201


def test_message_sender_bound_to_token(client, boostie):
    sid = spawn(client, boostie, "interactive").json()["session_id"]
    r = client.post(
        f"/v1/sessions/{sid}/messages", json={"message": "x", "sender": "missy"}, headers=boostie
    )
    assert r.status_code == 403


def test_sessions_private_to_requester(client, boostie, missy):
    sid = spawn(client, boostie, "interactive").json()["session_id"]
    for method, path in [
        ("GET", f"/v1/sessions/{sid}"),
        ("GET", f"/v1/sessions/{sid}/events?stream=false"),
        ("POST", f"/v1/sessions/{sid}/stop"),
    ]:
        assert client.request(method, path, headers=missy).status_code == 404
    r = client.post(f"/v1/sessions/{sid}/messages", json={"message": "x"}, headers=missy)
    assert r.status_code == 404


def test_unknown_profile_and_worker_type(client, boostie):
    assert spawn(client, boostie, profile="root").status_code == 422
    assert spawn(client, boostie, worker_type="codex").status_code == 422


def test_profile_worker_type_allowlist(tmp_path):
    settings = Settings(
        data_dir=tmp_path / "d2",
        profiles={"locked": Profile(max_runtime_minutes=1, worker_types=["other"])},
        worker_types=DEFAULT_WORKER_TYPES,
    )
    with TestClient(create_app(settings)) as c:
        headers = {"Authorization": "Bearer " + auth.create_token(c.app.state.conn, "boostie")}
        assert spawn(c, headers, profile="locked").status_code == 403


def test_workspace_must_be_under_allowed_root(client, boostie, workspace_root, tmp_path):
    ok = spawn(client, boostie, workspace={"path": str(workspace_root / "repo")})
    assert ok.status_code == 201

    outside = tmp_path / "elsewhere"
    outside.mkdir()
    assert spawn(client, boostie, workspace={"path": str(outside)}).status_code == 403
    sneaky = str(workspace_root / "repo" / ".." / ".." / "elsewhere")
    assert spawn(client, boostie, workspace={"path": sneaky}).status_code == 403
    missing = str(workspace_root / "nope")
    assert spawn(client, boostie, workspace={"path": missing}).status_code == 422


def test_profile_without_workspace_rejects_one(client, boostie, workspace_root):
    r = spawn(
        client,
        boostie,
        profile="read_only_research",
        workspace={"path": str(workspace_root / "repo")},
    )
    assert r.status_code == 403


def test_caller_timeout_cannot_exceed_profile(client, boostie):
    sid = spawn(client, boostie, "interactive", timeout_seconds=10**9).json()["session_id"]
    s = client.get(f"/v1/sessions/{sid}", headers=boostie).json()
    span = datetime.fromisoformat(s["expires_at"]) - datetime.fromisoformat(s["created_at"])
    assert span.total_seconds() <= 120 * 60 + 1


def test_capacity_limit(client, boostie):
    sids = [spawn(client, boostie, "interactive").json()["session_id"] for _ in range(2)]
    r = spawn(client, boostie, "interactive")
    assert r.status_code == 429
    client.post(f"/v1/sessions/{sids[0]}/stop", headers=boostie)
    assert spawn(client, boostie, "interactive").status_code == 201


def test_worker_env_is_allowlisted(client, boostie, monkeypatch, settings):
    monkeypatch.setenv("SUPER_SECRET", "hunter2")
    sid = spawn(client, boostie, "interactive").json()["session_id"]
    wait_status(client, sid, boostie, {"running"})
    pid = client.app.state.conn.execute("select pid from sessions where id = ?", (sid,)).fetchone()[
        "pid"
    ]
    env = dict(
        kv.split("=", 1)
        for kv in open(f"/proc/{pid}/environ", "rb").read().decode().split("\0")
        if "=" in kv
    )
    assert "SUPER_SECRET" not in env
    assert env["AGENTD_SESSION_ID"] == sid
    assert json.loads(env["AGENTD_PROFILE"])["workspace_mount"] == "read_write"
    assert env["PATH"] == os.environ["PATH"]


def test_room_token_redacted_on_disk(client, boostie, settings):
    r = spawn(client, boostie, room={"url": "http://127.0.0.1:1", "room_id": "r", "token": "s3"})
    sid = r.json()["session_id"]
    wait_status(client, sid, boostie)
    assert "s3" not in (settings.sessions_dir / sid / "input.json").read_text()


def test_restart_fails_orphaned_sessions(settings):
    db.init_db(settings.db_path)
    conn = db.connect(settings.db_path)
    now = now_iso()
    with conn:
        conn.execute(
            "insert into sessions (id, instance_id, requester_agent, profile, worker_type,"
            " status, task, idle_timeout_seconds, created_at, last_activity_at, expires_at,"
            " pid) values ('agt_old', 'agentd-test', 'boostie', 'workspace_coder', 'fake',"
            " 'running', 't', 60, ?, ?, ?, 999999)",
            (now, now, now),
        )
    conn.close()
    (settings.sessions_dir / "agt_old").mkdir(parents=True)

    with TestClient(create_app(settings)) as c:
        row = c.app.state.conn.execute("select * from sessions where id = 'agt_old'").fetchone()
        assert (row["status"], row["stop_reason"]) == ("failed", "gateway_restarted")
