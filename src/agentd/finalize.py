"""Room finalization for sessions invited into a room (room-o-matic/docs#17).

When a session ends (or fails to launch), its room invite must stop working, and the
outcome must be visible rather than assumed. Finalization runs alongside the session's
terminal status and records its own state on the session row:

    pending        a close-out is owed (set at launch, so a crash leaves a trace)
    done           closing message posted and invite revoked
    revoked        invite revoked, but the closing message couldn't be posted
    owner_required agentd can't revoke any more: the retries were exhausted, or the gateway
                   restarted (the plaintext token is never persisted). The non-secret
                   room_invite_id is recorded so the inviter, who can revoke by id, takes over.

A 401 on revoke means the token is already dead, which counts as revoked. Failed attempts
retry with exponential backoff from the cleanup loop, up to max_attempts.
"""

import asyncio
import logging
import sqlite3
import time

from agentd import lobby_client
from agentd.events import EventStore
from agentd.models import RoomRef

log = logging.getLogger("agentd.finalize")


class RoomFinalizer:
    def __init__(
        self,
        conn: sqlite3.Connection,
        events: EventStore,
        *,
        max_attempts: int = 4,
        base_delay: float = 2.0,
        clock=time.monotonic,
    ):
        self.conn = conn
        self.events = events
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self._clock = clock
        # session_id -> (room, msg_type, body); the token lives only here, in memory
        self._pending: dict[str, tuple[RoomRef, str, str]] = {}
        self._next_at: dict[str, float] = {}
        self._tasks: set[asyncio.Task] = set()

    def _set(self, session_id: str, state: str, error: str | None = None, attempts=None) -> None:
        with self.conn:
            self.conn.execute(
                "update sessions set room_finalization = ?, room_finalization_error = ?,"
                " room_finalization_attempts = coalesce(?, room_finalization_attempts)"
                " where id = ?",
                (state, error, attempts, session_id),
            )
        self.events.append(session_id, "room_finalization", state=state, error=error)

    def start(self, session_id: str, room: RoomRef, msg_type: str, body: str) -> None:
        self._pending[session_id] = (room, msg_type, body)
        task = asyncio.create_task(self._attempt(session_id))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _attempt(self, session_id: str) -> None:
        item = self._pending.get(session_id)
        if item is None:
            return
        room, msg_type, body = item
        self._next_at.pop(session_id, None)
        outcome = await lobby_client.close_out_room(
            room.base_url, room.room_id, room.token, msg_type=msg_type, body=body
        )
        row = self.conn.execute(
            "select room_finalization_attempts from sessions where id = ?", (session_id,)
        ).fetchone()
        attempts = (row[0] if row else 0) + 1
        if outcome["revoked"]:
            self._pending.pop(session_id, None)
            state = "done" if outcome["posted"] else "revoked"
            self._set(session_id, state, outcome["error"], attempts)
            return
        if attempts >= self.max_attempts:
            self._pending.pop(session_id, None)  # drop the token: hand over to the owner
            self._set(
                session_id,
                "owner_required",
                f"could not revoke the room invite after {attempts} attempts: {outcome['error']}",
                attempts,
            )
            return
        self._next_at[session_id] = self._clock() + self.base_delay * 2 ** (attempts - 1)
        self._set(session_id, "pending", outcome["error"], attempts)

    async def retry_due(self) -> None:
        now = self._clock()
        due = [sid for sid, at in self._next_at.items() if at <= now]
        await asyncio.gather(*(self._attempt(sid) for sid in due))

    async def drain(self, timeout: float) -> None:
        """Shutdown: give in-flight close-outs a bounded chance, then hand any remaining
        ones to their owners."""
        if self._tasks:
            await asyncio.wait(set(self._tasks), timeout=timeout)
        for sid in list(self._pending):
            self._pending.pop(sid)
            self._next_at.pop(sid, None)
            self._set(sid, "owner_required", "gateway shut down before the invite was revoked")

    def recover(self) -> None:
        """Startup: finalizations a previous process owed can't be completed here (the token
        was in its memory), so they become the inviter's to revoke by room_invite_id."""
        rows = self.conn.execute(
            "select id from sessions where room_finalization = 'pending'"
        ).fetchall()
        for row in rows:
            self._set(row[0], "owner_required", "gateway restarted before the invite was revoked")
