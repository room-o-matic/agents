"""agentd's outbound calls (see design/multi-server.md in the docs repo).

- Registry: heartbeat this instance into the lobbyd directory so orchestrators find it.
- Rooms: when a session that was invited into a room ends, post a closing message to
  that room's roomsd with the worker's invite token, then revoke the token so it dies
  with the session.
"""

import asyncio
import logging
from collections.abc import Callable

import httpx

from agentd.config import Settings
from agentd.models import PROTOCOL

log = logging.getLogger("agentd.lobby")


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


async def register(client: httpx.AsyncClient, settings: Settings, active_sessions: int) -> None:
    r = await client.put(
        f"{settings.lobbyd_url}/v1/registry/agentd/{settings.instance_id}",
        headers=_auth(settings.lobbyd_api_key),
        json={
            "base_url": settings.base_url,
            "worker_types": sorted(settings.worker_types),
            "profiles": sorted(settings.profiles),
            "max_sessions": settings.max_sessions,
            "active_sessions": active_sessions,
            "ttl_seconds": settings.registry_ttl_seconds,
            # docs#15: enough for discovery to rule out incompatible gateways early; the
            # full contract is at GET /v1/instance.
            "metadata": {
                "protocol": PROTOCOL,
                "kind": "gateway",
                "isolation": "sandbox" if settings.backend == "sandbox" else "none",
            },
        },
    )
    r.raise_for_status()


async def heartbeat_loop(
    settings: Settings, active_sessions: Callable[[], int], changed: asyncio.Event
) -> None:
    """Heartbeat every ttl/3, and immediately whenever the active session count changes."""
    interval = settings.registry_ttl_seconds / 3
    async with httpx.AsyncClient(timeout=10) as client:
        while True:
            changed.clear()
            try:
                await register(client, settings, active_sessions())
            except httpx.HTTPError as e:
                log.warning("registry heartbeat to %s failed: %s", settings.lobbyd_url, e)
            try:
                await asyncio.wait_for(changed.wait(), interval)
            except TimeoutError:
                pass


async def deregister(settings: Settings) -> None:
    async with httpx.AsyncClient(timeout=5) as client:
        try:
            await client.delete(
                f"{settings.lobbyd_url}/v1/registry/agentd/{settings.instance_id}",
                headers=_auth(settings.lobbyd_api_key),
            )
        except httpx.HTTPError as e:
            log.warning("registry deregister failed: %s", e)


async def close_out_room(url: str, room_id: str, token: str, *, msg_type: str, body: str) -> dict:
    """Join (idempotent, in case the worker never did), post the closing message, then
    revoke the invite, checking every response (docs#17). Never raises for HTTP errors:
    returns {"posted": bool, "revoked": bool, "error": str | None}. A 401 on revoke means the
    token is already dead, which counts as revoked."""
    posted, revoked, errors = False, False, []
    async with httpx.AsyncClient(base_url=url, headers=_auth(token), timeout=10) as client:
        try:
            r = await client.post(f"/v1/rooms/{room_id}/participants", json={})
            r.raise_for_status()
            r = await client.post(
                f"/v1/rooms/{room_id}/messages", json={"type": msg_type, "body": body}
            )
            r.raise_for_status()
            posted = True
        except httpx.HTTPError as e:
            errors.append(f"closing message: {e}")
        try:
            r = await client.post("/v1/auth/revoke")
            if r.status_code in (200, 204, 401):
                revoked = True
            else:
                errors.append(f"revoke: HTTP {r.status_code}")
        except httpx.HTTPError as e:
            errors.append(f"revoke: {e}")
    return {"posted": posted, "revoked": revoked, "error": "; ".join(errors) or None}
