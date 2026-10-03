import time

from agentd.app import TERMINAL_STATUSES


def spawn(client, headers, task="say hello", **body):
    body = {"task": task, "profile": "workspace_coder", "worker_type": "fake", **body}
    return client.post("/v1/sessions", json=body, headers=headers)


def wait_status(client, sid, headers, statuses=TERMINAL_STATUSES, timeout=10) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        s = client.get(f"/v1/sessions/{sid}", headers=headers).json()
        if s["status"] in statuses:
            return s
        time.sleep(0.05)
    raise AssertionError(f"session {sid} still {s['status']} after {timeout}s")


def events(client, sid, headers, after_id=0) -> list[dict]:
    r = client.get(
        f"/v1/sessions/{sid}/events",
        params={"stream": False, "after_id": after_id},
        headers=headers,
    )
    assert r.status_code == 200
    return r.json()


def wait_event(client, sid, headers, predicate, timeout=10) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for ev in events(client, sid, headers):
            if predicate(ev):
                return ev
        time.sleep(0.05)
    raise AssertionError(f"no matching event for {sid} after {timeout}s")
