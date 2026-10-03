"""The worker protocol.

Gateway -> worker (stdin): one JSON object per line.
    {"type": "task", "session_id": "...", "task": "..."}
    {"type": "message", "sender": "boostie", "message": "..."}
    {"type": "stop", "reason": "..."}

Worker -> gateway (stdout): structured events are lines prefixed with AGENT_EVENT.
    AGENT_EVENT {"type": "progress", "message": "Reading repository layout"}
    AGENT_EVENT {"type": "artifact", "name": "notes.md", "path": "notes.md"}
    AGENT_EVENT {"type": "final", "summary": "Done"}
Any other stdout line, and every stderr line, becomes a `log` event. Raw output is never
the source of truth for session state.
"""

import json

MARKER = "AGENT_EVENT "

# Event types a worker may emit. Everything else is reserved for the gateway
# (status, log, message, protocol_error, room_error, log_truncated).
WORKER_EVENT_TYPES = frozenset({"progress", "artifact", "needs_input", "final", "error"})

# Keys the gateway owns on every stored event; worker payloads can't override them.
RESERVED_KEYS = frozenset({"id", "session_id", "time", "type"})


def parse_stdout_line(line: str) -> dict:
    if not line.startswith(MARKER):
        return {"type": "log", "stream": "stdout", "text": line}
    raw = line[len(MARKER) :]
    try:
        event = json.loads(raw)
    except json.JSONDecodeError:
        return {"type": "protocol_error", "reason": "invalid JSON", "text": raw}
    if not isinstance(event, dict) or event.get("type") not in WORKER_EVENT_TYPES:
        return {
            "type": "protocol_error",
            "reason": f"event type must be one of {sorted(WORKER_EVENT_TYPES)}",
            "text": raw,
        }
    return event


def encode(obj: dict) -> bytes:
    return (json.dumps(obj) + "\n").encode()
