"""MCP server that gives a worker tools for the roomsd room it was invited into.

The Claude adapter starts it through `--mcp-config` when the session has a room. It runs
with the worker's env and authenticates with the room invite (ROOMSD_TOKEN), so it can do
exactly what the invite allows: read, post and use notes in that one room as the
invited identity.

Tools (the gist's core set): rooms_read, rooms_send, rooms_note_get, rooms_note_put.
In Claude Code they appear as mcp__rooms__<tool>.
"""

import os
from typing import Literal

import httpx
from mcp.server.mcpserver import MCPServer

SERVER_NAME = "rooms"
TOOL_NAMES = ("rooms_read", "rooms_send", "rooms_note_get", "rooms_note_put")
HISTORY_ON_FIRST_READ = 30
BODY_LIMIT = 16 * 1024

MessageType = Literal[
    "message",
    "proposal",
    "objection",
    "question",
    "answer",
    "finding",
    "status",
    "decision_request",
    "artifact",
    "task_update",
    "handoff",
]


def claude_tool_names() -> list[str]:
    return [f"mcp__{SERVER_NAME}__{name}" for name in TOOL_NAMES]


class RoomTools:
    def __init__(self, base_url: str, room_id: str, token: str, *, transport=None):
        self.room_id = room_id
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {token}"},
            timeout=15,
            transport=transport,
        )
        self._cursor: int | None = None
        self._identity: str | None = None

    @classmethod
    def from_env(cls) -> "RoomTools":
        return cls(
            os.environ["ROOMSD_URL"], os.environ["ROOMSD_ROOM_ID"], os.environ["ROOMSD_TOKEN"]
        )

    def _call(self, method: str, path: str, **kw):
        r = self._http.request(method, path, **kw)
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except ValueError:
                detail = r.text
            raise RuntimeError(f"roomsd {r.status_code}: {detail}")
        return r.json() if r.content else None

    def identity(self) -> str:
        if self._identity is None:
            self._identity = self._call("GET", "/v1/auth/whoami")["agent"]
        return self._identity

    @staticmethod
    def _slim(m: dict) -> dict:
        keep = ("id", "from", "type", "topic", "body", "confidence", "reply_requested")
        return {k: m[k] for k in keep if m.get(k) is not None}

    # ----- tools ---------------------------------------------------------------------

    def rooms_read(
        self, after_id: int | None = None, include_notes: list[str] | None = None
    ) -> dict:
        """Read room messages. With no after_id, returns what's new since your last read
        (the first call returns recent history). Optionally include named notes, e.g.
        ["summary", "open_questions", "decisions"]."""
        start = after_id if after_id is not None else (self._cursor or 0)
        messages: list[dict] = []
        while True:
            page = self._call(
                "GET",
                f"/v1/rooms/{self.room_id}/messages",
                params={"after_id": start, "limit": 500},
            )
            messages += page["messages"]
            start = page["latest_message_id"]
            if len(page["messages"]) < 500:
                break
        if after_id is None and self._cursor is None:
            messages = messages[-HISTORY_ON_FIRST_READ:]
        if messages:
            self._cursor = max(self._cursor or 0, messages[-1]["id"])
        elif self._cursor is None:
            self._cursor = start
        result = {
            "you": self.identity(),
            "messages": [self._slim(m) for m in messages],
            "latest_message_id": self._cursor,
        }
        if include_notes:
            notes = self._call(
                "GET",
                f"/v1/rooms/{self.room_id}/notes",
                params={"keys": ",".join(include_notes)},
            )["notes"]
            result["notes"] = {k: v["value"] for k, v in notes.items()}
        return result

    def rooms_send(
        self,
        body: str,
        type: MessageType = "message",
        topic: str | None = None,
        confidence: float | None = None,
        reply_requested: bool | None = None,
    ) -> dict:
        """Post a message to the room. Use typed messages for important claims (proposal,
        objection, finding, question, answer, status) and include confidence for uncertain
        ones. Publish long content as a file artifact rather than a huge message."""
        if len(body.encode()) > BODY_LIMIT:
            raise ValueError(f"message is over {BODY_LIMIT} bytes; summarise it")
        payload = {
            "body": body,
            "type": type,
            "topic": topic,
            "confidence": confidence,
            "reply_requested": reply_requested,
        }
        m = self._call(
            "POST",
            f"/v1/rooms/{self.room_id}/messages",
            json={k: v for k, v in payload.items() if v is not None},
        )
        return {"id": m["id"], "posted_as": m["from"]}

    def rooms_note_get(self, key: str | None = None) -> dict:
        """Get one shared note by key, or all notes when key is omitted. Notes are the
        room's working memory (summary, open_questions, decisions, …)."""
        if key:
            note = self._call("GET", f"/v1/rooms/{self.room_id}/notes/{key}")
            return {key: note["value"], "updated_by": note["updated_by"]}
        notes = self._call("GET", f"/v1/rooms/{self.room_id}/notes")["notes"]
        return {k: v["value"] for k, v in notes.items()}

    def rooms_note_put(self, key: str, value: str | dict | list) -> dict:
        """Create or replace a shared note. Keys: letters, digits, _ . -"""
        note = self._call("PUT", f"/v1/rooms/{self.room_id}/notes/{key}", json={"value": value})
        return {"key": note["key"], "updated_by": note["updated_by"]}


def build_server(tools: RoomTools) -> MCPServer:
    server = MCPServer(
        SERVER_NAME,
        instructions="Tools for the roomsd room this worker was invited into.",
    )
    for name in TOOL_NAMES:
        server.add_tool(getattr(tools, name), name=name)
    return server


def main() -> None:
    build_server(RoomTools.from_env()).run("stdio")


if __name__ == "__main__":
    main()
