"""agentd's side of the roomsd integration (see CLAUDE.md "Integration").

- Registry: heartbeat this instance into roomsd so orchestrators can find it.
- Rooms: when a session that was invited into a room ends, post a closing message with
  the worker's invite token, then revoke that token so it dies with the session.
"""

import asyncio
import logging
from collections.abc import Callable

import httpx

from agentd.config import Settings

log = logging.getLogger("agentd.rooms")


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


async def register(client: httpx.AsyncClient, settings: Settings, active_sessions: int) -> None:
    r = await client.put(
        f"{settings.roomsd_url}/v1/registry/agentd/{settings.instance_id}",
        headers=_auth(settings.roomsd_token),
        json={
            "base_url": settings.base_url,
            "worker_types": sorted(settings.worker_types),
            "profiles": sorted(settings.profiles),
            "max_sessions": settings.max_sessions,
            "active_sessions": active_sessions,
            "ttl_seconds": settings.registry_ttl_seconds,
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
                log.warning("registry heartbeat to %s failed: %s", settings.roomsd_url, e)
            try:
                await asyncio.wait_for(changed.wait(), interval)
            except TimeoutError:
                pass


async def deregister(settings: Settings) -> None:
    async with httpx.AsyncClient(timeout=5) as client:
        try:
            await client.delete(
                f"{settings.roomsd_url}/v1/registry/agentd/{settings.instance_id}",
                headers=_auth(settings.roomsd_token),
            )
        except httpx.HTTPError as e:
            log.warning("registry deregister failed: %s", e)


async def close_out_room(url: str, room_id: str, token: str, *, msg_type: str, body: str) -> None:
    """Join (idempotent, in case the worker never did), post, then revoke the token.

    Raises httpx.HTTPError; the caller records it as a room_error event.
    """
    async with httpx.AsyncClient(base_url=url, headers=_auth(token), timeout=10) as client:
        try:
            r = await client.post(f"/v1/rooms/{room_id}/participants", json={})
            r.raise_for_status()
            r = await client.post(
                f"/v1/rooms/{room_id}/messages", json={"type": msg_type, "body": body}
            )
            r.raise_for_status()
        finally:
            await client.post("/v1/auth/revoke")
