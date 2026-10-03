"""Regression tests for room-o-matic/docs#4: max_sessions holds under concurrent spawns,
and reservations are visible to the registry and released on failure."""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

from helpers import spawn, wait_status


class BarrierBackend:
    """Delegates to the real backend but holds every start() until `release` is set, so
    concurrent spawns all reach the launch await together (the race in the issue)."""

    def __init__(self, inner, fail: bool = False):
        self.inner = inner
        self.fail = fail
        self.started = 0
        self.release = threading.Event()

    async def start(self, command, env, cwd):
        self.started += 1
        while not self.release.is_set():
            await asyncio.sleep(0.01)
        if self.fail:
            raise OSError("no such worker binary")
        return await self.inner.start(command, env, cwd)


def gate(client, fail=False) -> BarrierBackend:
    sup = client.app.state.supervisor
    backend = BarrierBackend(sup.backend, fail=fail)
    sup.backend = backend
    return backend


def wait_until(pred, timeout=5):
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.02)
    return False


def test_concurrent_spawns_never_exceed_max(client, boostie):
    sup = client.app.state.supervisor
    backend = gate(client)  # max_sessions=2 in the test settings
    with ThreadPoolExecutor(5) as pool:
        futures = [pool.submit(spawn, client, boostie, "interactive") for _ in range(5)]
        try:
            # Two launches are parked at the barrier holding reservations; the rest were
            # refused at admission without reaching the backend.
            assert wait_until(lambda: sum(f.done() for f in futures) == 3)
            assert backend.started == 2
            assert sup.active_count() == 2
        finally:
            backend.release.set()  # a failing run must fail, not hang
        codes = sorted(f.result().status_code for f in futures)
    assert codes == [201, 201, 429, 429, 429]
    assert client.get("/v1/instance", headers=boostie).json()["active_sessions"] == 2


def test_reservations_count_for_registry_and_release_on_failed_launch(client, boostie):
    sup = client.app.state.supervisor
    backend = gate(client, fail=True)
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(spawn, client, boostie, "interactive")
        try:
            assert wait_until(lambda: backend.started == 1)
            # In flight: the slot is already occupied as far as the registry can tell.
            assert sup.active_count() == 1
            assert client.get("/v1/instance", headers=boostie).json()["active_sessions"] == 1
            sup.capacity_changed.clear()
        finally:
            backend.release.set()
        r = future.result()
    assert r.status_code == 201  # spawn returns the failed session, as before
    s = wait_status(client, r.json()["session_id"], boostie)
    assert s["status"] == "failed" and "failed to start" in s["stop_reason"]
    assert sup.active_count() == 0
    assert sup.capacity_changed.is_set()  # the registry heartbeat is woken promptly
