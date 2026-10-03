"""Tests for room-o-matic/docs#15 (gateway side): a versioned capability contract."""

from fastapi.testclient import TestClient

from agentd.app import create_app
from agentd.config import CallerPolicy, Profile
from agentd.models import PROTOCOL


def caps(client, headers):
    return client.get("/v1/instance", headers=headers).json()["capabilities"]


def test_contract_shape(client, boostie):
    c = caps(client, boostie)
    assert c["protocol"] == PROTOCOL == "room-o-matic.agentd/1"
    assert c["kind"] == "gateway" and c["isolation"] == "none"
    assert {"operation_id", "room_invites", "task_grant"} <= set(c["features"])
    assert c["limits"]["task_bytes"] == 64 * 1024
    assert "SIGTERM" in c["cancellation"]


def test_pairs_follow_profile_allowlists_and_skip_unlaunchable(settings, lobby):
    profiles = {
        "only_fake": Profile(max_runtime_minutes=5, worker_types=["fake"]),
        "needs_approval": Profile(max_runtime_minutes=5, external_actions="approval_required"),
    }
    with TestClient(
        create_app(settings.model_copy(update={"profiles": profiles}), verifier=lobby.verifier())
    ) as c:
        pairs = caps(c, lobby.headers("boostie"))["pairs"]
    assert pairs == [{"profile": "only_fake", "worker_type": "fake"}]


def test_allowed_for_you_reflects_the_callers_grant(settings, lobby):
    callers = {
        "boostie@test": CallerPolicy(
            trust="trusted", profiles=["read_only_research"], worker_types=["fake"]
        ),
        "mallory@test": CallerPolicy(trust="untrusted", profiles=["*"], worker_types=["*"]),
    }
    with TestClient(
        create_app(settings.model_copy(update={"callers": callers}), verifier=lobby.verifier())
    ) as c:
        mine = caps(c, lobby.headers("boostie"))["allowed_for_you"]
        assert mine == [{"profile": "read_only_research", "worker_type": "fake"}]
        # untrusted on the process backend: nothing is allowed, and the client can tell
        assert caps(c, lobby.headers("mallory"))["allowed_for_you"] == []
        assert caps(c, lobby.headers("nobody"))["allowed_for_you"] == []
