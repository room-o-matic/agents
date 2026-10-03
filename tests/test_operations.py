"""Tests for room-o-matic/docs#24 (agentd): versioned schema, backups of the database and
sessions dir, restore that can't revive work, and readiness/metrics. Every drill runs on a
disposable copy under tmp_path."""

import hashlib
import sqlite3
import stat

import pytest
from fastapi.testclient import TestClient
from helpers import events, spawn, wait_status

from agentd import cli, db, ops, recovery
from agentd.app import create_app


def version_of(path):
    conn = sqlite3.connect(path)
    try:
        return conn.execute("pragma user_version").fetchone()[0]
    finally:
        conn.close()


def sql(settings, query, params=()):
    conn = sqlite3.connect(settings.db_path)
    try:
        with conn:
            return conn.execute(query, params).fetchall()
    finally:
        conn.close()


def test_fresh_database_is_versioned(client, settings):
    assert version_of(settings.db_path) == db.SCHEMA_VERSION


def test_newer_schema_refused_at_startup(client, settings, lobby):
    sql(settings, "pragma user_version = 7")
    with pytest.raises(ops.SchemaError, match="newer than this release"):
        with TestClient(create_app(settings, verifier=lobby.verifier())):
            pass


def test_pre_baseline_schema_refused_untouched(settings, lobby):
    settings.data_dir.mkdir(parents=True)
    conn = sqlite3.connect(settings.db_path)
    conn.execute("create table sessions (id text primary key, status text)")
    conn.commit()
    conn.close()
    before = hashlib.sha256(settings.db_path.read_bytes()).hexdigest()
    with pytest.raises(ops.SchemaError, match="predates the supported baseline"):
        with TestClient(create_app(settings, verifier=lobby.verifier())):
            pass
    assert hashlib.sha256(settings.db_path.read_bytes()).hexdigest() == before


def test_backup_includes_session_files_and_is_private(client, settings, boostie, tmp_path):
    sid = spawn(client, boostie).json()["session_id"]
    wait_status(client, sid, boostie)
    manifest = recovery.backup(settings, tmp_path / "bk")
    assert f"sessions/{sid}/events.jsonl" in manifest["files"]
    assert stat.S_IMODE((tmp_path / "bk").stat().st_mode) == 0o700
    assert ops.verify_backup(tmp_path / "bk")["service"] == "agentd"
    (tmp_path / "bk" / "sessions" / sid / "events.jsonl").write_text("tampered\n")
    with pytest.raises(ops.BackupError, match="checksum"):
        ops.verify_backup(tmp_path / "bk")


def test_restore_fails_live_work_and_never_reuses_ids(settings, lobby, boostie, tmp_path):
    with TestClient(create_app(settings, verifier=lobby.verifier())) as c:
        done = spawn(c, boostie).json()["session_id"]
        wait_status(c, done, boostie)
        live = spawn(c, boostie, "interactive").json()["session_id"]
        wait_status(c, live, boostie, statuses=("running",))
        sql(settings, "update sessions set room_finalization = 'pending' where id = ?", (done,))
        recovery.backup(settings, tmp_path / "bk")  # taken while `live` runs
        last_event = events(c, live, boostie)[-1]["id"]
        c.post(f"/v1/sessions/{live}/stop", headers=boostie)
        wait_status(c, live, boostie)

    report = recovery.restore(settings, tmp_path / "bk", force=True)
    assert report["integrity"] == "ok" and report["moved_aside"]
    assert report["invalidated"]["sessions_failed"] == 1
    assert report["invalidated"]["room_finalizations_to_owner"] == 1

    with TestClient(create_app(settings, verifier=lobby.verifier())) as c:
        s = c.get(f"/v1/sessions/{live}", headers=boostie).json()
        assert s["status"] == "failed" and s["stop_reason"] == "restored_from_backup"
        assert sql(settings, "select pid from sessions where id = ?", (live,))[0][0] is None
        assert c.get(f"/v1/sessions/{done}", headers=boostie).json()["status"] == "completed"
        assert events(c, done, boostie)  # history restored
        assert (settings.sessions_dir / done / "events.jsonl").exists()
        fresh = spawn(c, boostie).json()["session_id"]
        wait_status(c, fresh, boostie)
        assert events(c, fresh, boostie)[0]["id"] > last_event  # cursors never reused
        assert c.get("/readyz").json()["checks"]["sessions"]["orphaned"] == 0


def test_restore_refuses_existing_data_without_force(client, settings, tmp_path):
    recovery.backup(settings, tmp_path / "bk")
    with pytest.raises(ops.BackupError, match="--force"):
        recovery.restore(settings, tmp_path / "bk")


def test_cli_drill(client, settings, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli.Settings, "load", classmethod(lambda cls, path=None: settings))
    assert cli.main(["backup", "--out", str(tmp_path / "bk")]) == 0
    assert cli.main(["verify-backup", str(tmp_path / "bk")]) == 0
    assert cli.main(["restore", str(tmp_path / "bk"), "--force"]) == 0
    assert '"integrity": "ok"' in capsys.readouterr().out


def test_readyz_and_metrics(client, boostie):
    sid = spawn(client, boostie).json()["session_id"]  # also confirms JWKS
    wait_status(client, sid, boostie)
    r = client.get("/readyz")
    assert r.status_code == 200 and r.json()["ready"] is True
    text = client.get("/metrics").text
    assert "agentd_ready 1" in text and "agentd_orphaned_sessions 0" in text


def test_readyz_reports_orphaned_sessions(client, settings, boostie):
    sid = spawn(client, boostie).json()["session_id"]
    wait_status(client, sid, boostie)
    sql(settings, "update sessions set status = 'running' where id = ?", (sid,))  # no worker
    r = client.get("/readyz")
    assert r.status_code == 503 and r.json()["checks"]["sessions"]["orphaned"] == 1
    assert "agentd_orphaned_sessions 1" in client.get("/metrics").text


def test_readyz_fails_on_storage_pressure(settings, lobby):
    s = settings.model_copy(update={"min_free_bytes": 1 << 62})
    with TestClient(create_app(s, verifier=lobby.verifier())) as c:
        r = c.get("/readyz")
        assert r.status_code == 503 and r.json()["checks"]["storage"]["ok"] is False
