"""A scriptable stand-in for Ollama's /api/chat over real HTTP (the worker is a subprocess).

Responses have the shape real Ollama 0.10 returns (captured live): message.content,
message.tool_calls[].function.{name, arguments}, prompt_eval_count, eval_count.
Each request takes the next scripted message; with the script empty it answers
"done: <last user message>". Every request body is recorded in `requests`.
"""

import json
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def tool_call(name: str, **arguments) -> dict:
    return {"content": "", "tool_calls": [{"function": {"name": name, "arguments": arguments}}]}


class FakeOllama:
    def __init__(self):
        self.script: deque[dict] = deque()
        self.requests: list[dict] = []
        self.always: dict | None = None  # answer every request with this (e.g. endless tools)
        self.status = 200
        self.error = ""
        self.delay = 0.0
        self.tokens = (100, 10)  # prompt_eval_count, eval_count per response
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self._server.shutdown()

    def _reply(self, body: dict) -> dict:
        if self.always is not None and body.get("tools"):
            msg = self.always
        elif self.script:
            msg = self.script.popleft()
        else:
            users = [m for m in body["messages"] if m["role"] == "user"]
            last = users[-1]["content"].strip().splitlines()[-1] if users else ""
            msg = {"content": f"done: {last}"}
        return {
            "model": body["model"],
            "message": {"role": "assistant", **msg},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": self.tokens[0],
            "eval_count": self.tokens[1],
        }

    def _handler(self):
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                n = int(self.headers.get("content-length") or 0)
                body = json.loads(self.rfile.read(n))
                fake.requests.append(body)
                if fake.delay:
                    time.sleep(fake.delay)
                if fake.status != 200:
                    out, code = {"error": fake.error}, fake.status
                else:
                    out, code = fake._reply(body), 200
                data = json.dumps(out).encode()
                try:
                    self.send_response(code)
                    self.send_header("content-type", "application/json")
                    self.send_header("content-length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        return Handler
