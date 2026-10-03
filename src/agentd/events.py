import asyncio
import json
import sqlite3
from pathlib import Path

from agentd.ids import now_iso
from agentd.protocol import RESERVED_KEYS


class EventStore:
    """Appends session events to SQLite and the session's events.jsonl, and wakes
    streaming readers. Only touched from the event loop thread."""

    def __init__(self, conn: sqlite3.Connection, sessions_dir: Path):
        self._conn = conn
        self._sessions_dir = sessions_dir
        self._signals: dict[str, asyncio.Event] = {}

    def append(self, session_id: str, type: str, **payload) -> dict:
        payload = {k: v for k, v in payload.items() if k not in RESERVED_KEYS}
        time = now_iso()
        with self._conn:
            cur = self._conn.execute(
                "insert into events (session_id, type, time, payload_json) values (?, ?, ?, ?)",
                (session_id, type, time, json.dumps(payload)),
            )
        event = {"id": cur.lastrowid, "session_id": session_id, "type": type, "time": time}
        event.update(payload)
        with open(self._sessions_dir / session_id / "events.jsonl", "a") as f:
            f.write(json.dumps(event) + "\n")
        if signal := self._signals.pop(session_id, None):
            signal.set()
        return event

    def since(self, session_id: str, after_id: int = 0, limit: int = 500) -> list[dict]:
        rows = self._conn.execute(
            "select * from events where session_id = ? and id > ? order by id limit ?",
            (session_id, after_id, limit),
        ).fetchall()
        return [
            {
                "id": r["id"],
                "session_id": r["session_id"],
                "type": r["type"],
                "time": r["time"],
                **json.loads(r["payload_json"]),
            }
            for r in rows
        ]

    async def wait(self, session_id: str, timeout: float) -> bool:
        """Wait until the next append for this session (True) or the timeout (False).

        Callers read with since() and then call wait() with no await in between, so
        an append can't slip through unnoticed.
        """
        signal = self._signals.setdefault(session_id, asyncio.Event())
        try:
            await asyncio.wait_for(signal.wait(), timeout)
        except TimeoutError:
            return False
        return True
