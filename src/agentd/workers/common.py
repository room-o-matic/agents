"""Helpers for workers speaking the agentd protocol (see agentd/protocol.py).

Workers run as separate processes with only allowlisted env, so this module sticks to
the standard library.
"""

import json
import os
import sys
import urllib.request

MARKER = "AGENT_EVENT "


def emit(type: str, **fields) -> None:
    print(MARKER + json.dumps({"type": type, **fields}), flush=True)


def read_msg() -> dict | None:
    """Next gateway message from stdin, or None at EOF."""
    line = sys.stdin.readline()
    return json.loads(line) if line else None


def _roomsd(method: str, path: str, body: dict | None = None) -> None:
    req = urllib.request.Request(
        os.environ["ROOMSD_URL"] + path,
        method=method,
        data=json.dumps(body or {}).encode(),
        headers={
            "Authorization": f"Bearer {os.environ['ROOMSD_TOKEN']}",
            "Content-Type": "application/json",
        },
    )
    urllib.request.urlopen(req, timeout=10).read()


def join_room() -> str | None:
    """If invited into a room, join it. Returns the room id, or None when there is no room
    or joining failed (reported as an error event)."""
    room = os.environ.get("ROOMSD_ROOM_ID")
    if not room or not os.environ.get("ROOMSD_URL"):
        return None
    try:
        _roomsd("POST", f"/v1/rooms/{room}/participants")
    except OSError as e:
        emit("error", message=f"could not join room {room}: {e}")
        return None
    return room


def announce_in_room(what: str = "started", join: bool = True) -> str | None:
    """If invited into a room, join it (unless already joined) and post a status message
    naming this session. Returns the room id, or None when there is no room."""
    room = os.environ.get("ROOMSD_ROOM_ID")
    if not room or not os.environ.get("ROOMSD_URL"):
        return None
    if join and join_room() is None:
        return room
    try:
        _roomsd(
            "POST",
            f"/v1/rooms/{room}/messages",
            {
                "type": "status",
                "body": f"session {os.environ['AGENTD_SESSION_ID']} on "
                f"{os.environ['AGENTD_INSTANCE_ID']} {what}",
            },
        )
        emit("progress", message=f"joined room {room}")
    except OSError as e:
        emit("error", message=f"could not join room {room}: {e}")
    return room


def post_room_status(body: str) -> None:
    """Best-effort status message to the worker's room (e.g. a limit being hit)."""
    room = os.environ.get("ROOMSD_ROOM_ID")
    if not room or not os.environ.get("ROOMSD_URL"):
        return
    try:
        _roomsd("POST", f"/v1/rooms/{room}/messages", {"type": "status", "body": body})
    except OSError as e:
        emit("error", message=f"could not post to room {room}: {e}")
