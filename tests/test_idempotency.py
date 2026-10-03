"""Regression tests for room-o-matic/docs#13 (gateway side): spawn is idempotent per
caller-scoped operation_id, and outcomes can be reconciled by operation_id."""

from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient
from helpers import spawn, wait_status

from agentd.app import create_app

OP = "op-0123456789"


def sessions(client, headers):
    return client.get("/v1/sessions", headers=headers).json()


def test_retry_replays_the_original_session(client, boostie):
    first = spawn(client, boostie, "interactive", operation_id=OP)
    again = spawn(client, boostie, "interactive", operation_id=OP)
    assert (first.status_code, again.status_code) == (201, 200)
    assert again.json()["session_id"] == first.json()["session_id"]
    assert again.json()["replayed"] is True and first.json()["replayed"] is False
    assert len(sessions(client, boostie)) == 1  # no second worker


def test_reused_key_with_different_payload_conflicts(client, boostie):
    spawn(client, boostie, "interactive", operation_id=OP)
    r = spawn(client, boostie, "something else", operation_id=OP)
    assert r.status_code == 409 and "different spawn request" in r.json()["detail"]


def test_room_token_is_not_part_of_the_payload_identity(client, boostie):
    room = "http://127.0.0.1:1/v1/rooms/room_x"
    a = spawn(
        client,
        boostie,
        "interactive",
        operation_id=OP,
        room={"room_url": room, "token": "rmsd_first"},
    )
    b = spawn(
        client,
        boostie,
        "interactive",
        operation_id=OP,
        room={"room_url": room, "token": "rmsd_reminted"},
    )
    assert b.json()["session_id"] == a.json()["session_id"]


def test_concurrent_duplicates_start_one_worker(client, boostie):
    with ThreadPoolExecutor(6) as pool:
        results = list(
            pool.map(lambda _: spawn(client, boostie, "interactive", operation_id=OP), range(6))
        )
    ids = {r.json()["session_id"] for r in results}
    assert len(ids) == 1
    assert sorted(r.status_code for r in results) == [200] * 5 + [201]
    assert len(sessions(client, boostie)) == 1


def test_reconcile_by_operation_id(client, boostie, missy):
    assert client.get(f"/v1/sessions/by-operation/{OP}", headers=boostie).status_code == 404
    sid = spawn(client, boostie, "say hello", operation_id=OP).json()["session_id"]
    found = client.get(f"/v1/sessions/by-operation/{OP}", headers=boostie)
    assert found.status_code == 200 and found.json()["session_id"] == sid
    # scoped to the caller: another agent's operation ids are invisible
    assert client.get(f"/v1/sessions/by-operation/{OP}", headers=missy).status_code == 404
    other = spawn(client, missy, "say hello", operation_id=OP)
    assert other.status_code == 201 and other.json()["session_id"] != sid


def test_idempotency_survives_gateway_restart(settings, lobby, boostie):
    with TestClient(create_app(settings, verifier=lobby.verifier())) as c:
        sid = spawn(c, boostie, "say hello", operation_id=OP).json()["session_id"]
        wait_status(c, sid, boostie)
    with TestClient(create_app(settings, verifier=lobby.verifier())) as c:
        assert c.get(f"/v1/sessions/by-operation/{OP}", headers=boostie).json()["session_id"] == sid
        again = spawn(c, boostie, "say hello", operation_id=OP)
        assert again.status_code == 200 and again.json()["session_id"] == sid


def test_without_operation_id_each_spawn_is_new(client, boostie):
    a = spawn(client, boostie, "say hello").json()["session_id"]
    b = spawn(client, boostie, "say hello").json()["session_id"]
    assert a != b
