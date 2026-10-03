"""Regression tests for room-o-matic/docs#9: hosted work needs the gateway operator's
grant, untrusted callers need isolation, and approval is never inferred."""

import json
import os
import socket
import subprocess
import time

import pytest
from fastapi.testclient import TestClient
from helpers import events, spawn, wait_event, wait_status

from agentd.app import create_app
from agentd.config import CallerPolicy, Profile

TRUSTED_ALL = CallerPolicy(trust="trusted", profiles=["*"], worker_types=["*"], max_sessions=10)


def app_with(settings, lobby, **update):
    return TestClient(create_app(settings.model_copy(update=update), verifier=lobby.verifier()))


def test_default_deny(settings, lobby):
    with app_with(settings, lobby, callers={}) as c:
        r = spawn(c, lobby.headers("boostie"))
        assert r.status_code == 403 and "no caller policy" in r.json()["detail"]


def test_named_policy_beats_wildcard(settings, lobby):
    callers = {"*": TRUSTED_ALL, "mallory@test": CallerPolicy(trust="trusted", profiles=[])}
    with app_with(settings, lobby, callers=callers) as c:
        assert spawn(c, lobby.headers("boostie")).status_code == 201
        r = spawn(c, lobby.headers("mallory"))
        assert r.status_code == 403 and "may not use profile" in r.json()["detail"]


def test_profile_and_worker_type_must_be_granted(settings, lobby):
    callers = {
        "boostie@test": CallerPolicy(
            trust="trusted", profiles=["read_only_research"], worker_types=["fake"]
        )
    }
    with app_with(settings, lobby, callers=callers) as c:
        h = lobby.headers("boostie")
        assert spawn(c, h, profile="workspace_coder").status_code == 403
        assert spawn(c, h, profile="read_only_research").status_code == 201


def test_untrusted_callers_refused_on_process_backend(settings, lobby):
    callers = {"*": CallerPolicy(trust="untrusted", profiles=["*"], worker_types=["*"])}
    with app_with(settings, lobby, callers=callers) as c:
        r = spawn(c, lobby.headers("boostie"))
        assert r.status_code == 403 and "without isolation" in r.json()["detail"]


def test_caller_workspace_roots(settings, lobby, workspace_root):
    (workspace_root / "mine").mkdir()
    callers = {
        "*": CallerPolicy(
            trust="trusted",
            profiles=["*"],
            worker_types=["*"],
            workspace_roots=[workspace_root / "mine"],
        )
    }
    with app_with(settings, lobby, callers=callers) as c:
        h = lobby.headers("boostie")
        other = spawn(c, h, workspace={"path": str(workspace_root / "repo")})
        assert other.status_code == 403 and "may not use workspace" in other.json()["detail"]
        assert spawn(c, h, workspace={"path": str(workspace_root / "mine")}).status_code == 201


def test_per_caller_quota(settings, lobby):
    callers = {
        "*": CallerPolicy(trust="trusted", profiles=["*"], worker_types=["*"], max_sessions=1)
    }
    with app_with(settings, lobby, callers=callers) as c:
        assert spawn(c, lobby.headers("boostie"), "interactive").status_code == 201
        r = spawn(c, lobby.headers("boostie"), "interactive")
        assert r.status_code == 429 and "quota" in r.json()["detail"]
        assert spawn(c, lobby.headers("missy"), "interactive").status_code == 201  # separate


def test_approval_required_profiles_are_refused(client, boostie):
    r = spawn(client, boostie, profile="dangerous_needs_approval")
    assert r.status_code == 403 and "requires approval" in r.json()["detail"]


def test_spend_cap_is_the_lower_of_profile_and_caller(settings, lobby):
    callers = {
        "*": CallerPolicy(trust="trusted", profiles=["*"], worker_types=["*"], max_budget_usd=0.5)
    }
    profiles = {**settings.profiles, "capped": Profile(max_runtime_minutes=5, max_budget_usd=2)}
    with app_with(settings, lobby, callers=callers, profiles=profiles) as c:
        h = lobby.headers("boostie")
        sid = spawn(c, h, "interactive", profile="capped").json()["session_id"]
        wait_event(c, sid, h, lambda e: e["type"] == "progress")
        pid = c.app.state.conn.execute("select pid from sessions where id = ?", (sid,)).fetchone()[
            0
        ]
        raw = open(f"/proc/{pid}/environ", "rb").read().decode().split("\0")
        grant = json.loads(dict(kv.split("=", 1) for kv in raw if "=" in kv)["AGENTD_GRANT"])
        assert grant["max_budget_usd"] == 0.5


def test_callers_file_revocation_without_restart(settings, lobby, tmp_path):
    path = tmp_path / "callers.yaml"
    path.write_text("boostie@test: {trust: trusted, profiles: ['*'], worker_types: ['*']}\n")
    with app_with(settings, lobby, callers={}, callers_file=path) as c:
        h = lobby.headers("boostie")
        assert spawn(c, h).status_code == 201
        time.sleep(0.01)
        path.write_text("{}\n")  # operator revokes
        os.utime(path, (time.time() + 5, time.time() + 5))  # make the change observable
        assert spawn(c, h).status_code == 403


# ----- the sandbox backend ---------------------------------------------------------------


def bwrap_works() -> bool:
    try:
        return (
            subprocess.run(
                ["bwrap", "--ro-bind", "/", "/", "--unshare-all", "true"],
                capture_output=True,
                timeout=10,
            ).returncode
            == 0
        )
    except (OSError, subprocess.TimeoutExpired):
        return False


needs_bwrap = pytest.mark.skipif(not bwrap_works(), reason="bubblewrap/user namespaces unavailable")


@pytest.fixture
def listener():
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen()
    yield srv.getsockname()[1]
    srv.close()


@needs_bwrap
def test_sandbox_contains_a_hostile_worker(settings, lobby, tmp_path, workspace_root, listener):
    secret = tmp_path / "host-secret.txt"
    secret.write_text("do not read")
    untrusted = {"*": CallerPolicy(trust="untrusted", profiles=["*"], worker_types=["*"])}
    profiles = {
        **settings.profiles,
        "offline_reader": Profile(max_runtime_minutes=5, workspace_mount="read", network=False),
    }
    sandbox = settings.sandbox.model_copy(update={"memory_bytes": 1024**3})
    with app_with(
        settings, lobby, callers=untrusted, backend="sandbox", profiles=profiles, sandbox=sandbox
    ) as c:
        h = lobby.headers("mallory")
        sid = spawn(
            c,
            h,
            f"probe {secret} {listener}",
            profile="offline_reader",
            workspace={"path": str(workspace_root / "repo")},
        ).json()["session_id"]
        s = wait_status(c, sid, h, timeout=20)
        assert s["status"] == "completed", s
        report = next(e for e in events(c, sid, h) if "probe" in e)["probe"]
        assert report["read_outside"] == "FileNotFoundError"  # host files aren't there
        assert report["connect_host"] != "ok"  # no network namespace interfaces
        assert report["home"] == "/tmp/home" and report["home_entries"] == []
        assert report["big_alloc"] == "MemoryError"  # address-space limit
        assert report["write_workspace"] != "ok"  # read-only mount
        assert not (workspace_root / "repo" / "probe.txt").exists()

        beat = settings.sessions_dir / sid / "artifacts" / "heartbeat"
        size = beat.stat().st_size
        time.sleep(0.5)
        assert beat.stat().st_size == size  # the setsid'd descendant died with the sandbox


@needs_bwrap
def test_sandbox_runs_normal_work_for_untrusted_callers(settings, lobby):
    untrusted = {"*": CallerPolicy(trust="untrusted", profiles=["*"], worker_types=["*"])}
    with app_with(settings, lobby, callers=untrusted, backend="sandbox") as c:
        h = lobby.headers("mallory")
        sid = spawn(c, h, "say hello").json()["session_id"]
        s = wait_status(c, sid, h, timeout=20)
        assert (s["status"], s["summary"]) == ("completed", "did: say hello")
        assert (settings.sessions_dir / sid / "artifacts" / "notes.md").exists()


def test_sandbox_fails_closed_without_bwrap(settings, lobby):
    sandbox = settings.sandbox.model_copy(update={"bwrap": "/nonexistent/bwrap"})
    with pytest.raises(RuntimeError, match="not found"):
        with app_with(settings, lobby, backend="sandbox", sandbox=sandbox):
            pass
