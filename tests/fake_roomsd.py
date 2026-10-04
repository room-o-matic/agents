"""A tiny in-memory roomsd over real HTTP, for tests that run worker subprocesses.

Implements just the endpoints workers use: whoami, join, messages (read/post), notes, and
self-revoke. Bearer tokens map to identities via `tokens`.
"""

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse


class FakeRoomsd:
    def __init__(self, tokens: dict[str, str], room_id: str = "room_1"):
        self.tokens = dict(tokens)
        self.room_id = room_id
        self.messages: list[dict] = []
        self.notes: dict[str, dict] = {}
        self.participants: set[str] = set()
        self.require_join = False  # like roomsd: room routes 403 until the caller joins
        self.revoked: set[str] = set()
        self.revoke_status = 204  # docs#17 fault injection
        self.post_status: int | None = None
        self.lock = threading.Lock()
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        self.room_url = f"{self.url}/v1/rooms/{room_id}"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self._server.shutdown()

    def post(self, sender: str, body: str, type: str = "message") -> dict:
        with self.lock:
            m = {
                "id": len(self.messages) + 1,
                "room_id": self.room_id,
                "from": sender,
                "type": type,
                "body": body,
            }
            self.messages.append(m)
            return m

    def _handler(self):
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _reply(self, code: int, obj=None):
                data = json.dumps(obj).encode() if obj is not None else b""
                self.send_response(code)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _who(self) -> str | None:
                token = self.headers.get("authorization", "").removeprefix("Bearer ")
                return None if token in fake.revoked else fake.tokens.get(token)

            def _body(self):
                n = int(self.headers.get("content-length") or 0)
                return json.loads(self.rfile.read(n)) if n else {}

            def _route(self, method: str):
                who = self._who()
                if who is None:
                    return self._reply(401, {"detail": "bad token"})
                u = urlparse(self.path)
                q = {k: v[0] for k, v in parse_qs(u.query).items()}
                p = u.path
                room = f"/v1/rooms/{fake.room_id}"
                if p == "/v1/auth/whoami":
                    return self._reply(200, {"agent": who})
                if p == "/v1/auth/revoke" and method == "POST":
                    if fake.revoke_status != 204:
                        return self._reply(fake.revoke_status, {"detail": "injected"})
                    token = self.headers["authorization"].removeprefix("Bearer ")
                    fake.revoked.add(token)
                    return self._reply(204)
                if p == f"{room}/participants" and method == "POST":
                    fake.participants.add(who)
                    return self._reply(200, {"agent": who})
                if fake.require_join and p.startswith(room) and who not in fake.participants:
                    return self._reply(403, {"detail": "join the room first"})
                if p == f"{room}/messages" and method == "POST":
                    if fake.post_status is not None:
                        return self._reply(fake.post_status, {"detail": "injected"})
                    b = self._body()
                    m = fake.post(who, b["body"], b.get("type", "message"))
                    m.update({k: b[k] for k in ("topic", "confidence") if k in b})
                    return self._reply(201, m)
                if p == f"{room}/messages":
                    after, limit = int(q.get("after_id", 0)), int(q.get("limit", 100))
                    msgs = [m for m in fake.messages if m["id"] > after][:limit]
                    latest = msgs[-1]["id"] if msgs else after
                    return self._reply(200, {"messages": msgs, "latest_message_id": latest})
                if p == f"{room}/notes":
                    keys = q.get("keys", "").split(",") if q.get("keys") else list(fake.notes)
                    notes = {k: fake.notes[k] for k in keys if k in fake.notes}
                    return self._reply(200, {"room_id": fake.room_id, "notes": notes})
                if m := re.fullmatch(rf"{room}/notes/([\w.-]+)", p):
                    key = m.group(1)
                    if method == "PUT":
                        body = self._body()
                        current = fake.notes.get(key, {}).get("revision", 0)
                        want = body.get("if_revision")
                        if want is not None and want != current:
                            return self._reply(
                                412, {"detail": f"note {key!r} is at revision {current}"}
                            )
                        fake.notes[key] = {
                            "key": key,
                            "value": body["value"],
                            "updated_by": who,
                            "revision": current + 1,
                        }
                    if key not in fake.notes:
                        return self._reply(404, {"detail": "note not found"})
                    return self._reply(200, fake.notes[key])
                return self._reply(404, {"detail": f"no route {method} {p}"})

            def do_GET(self):
                self._route("GET")

            def do_POST(self):
                self._route("POST")

            def do_PUT(self):
                self._route("PUT")

        return Handler
