"""Fake worker for milestone 1 and tests. Behaviour is picked by the task's first word:

    interactive  stay up, echo each message; finish on message "done"
    hang         never emit an event (exercises the ready timeout)
    crash        emit progress, then exit 3 without final
    linger       emit final but don't exit (exercises reaping after final)
    stubborn     ignore stop requests and SIGTERM (exercises SIGKILL)
    chatty N     print N plain log lines, then final
    badjson      emit malformed and reserved events, then final
    escape       emit an artifact whose path escapes the artifacts dir, then final
    badfinal K   emit a final whose summary is a K (dict|list|number), then exit
    deaf         emit progress, then never read stdin again
    orphan [stubborn]  start a child in the same process group with stdio redirected
                 (ignoring SIGTERM if stubborn), report its pid, emit final, exit
    (anything)   progress, a log line, an artifact, final

If ROOMSD_* env vars are set, it joins the room and posts a status message first, the
way a real worker announces itself.
"""

import os
import signal
import sys
import time
from pathlib import Path

from agentd.workers.common import announce_in_room, emit, read_msg


def write_artifact(name: str, text: str) -> None:
    Path(os.environ["AGENTD_ARTIFACTS_DIR"], name).write_text(text)
    emit("artifact", name=name, path=name, mime_type="text/markdown")


def main() -> int:
    first = read_msg()
    if not first or first.get("type") != "task":
        print("expected a task message first", file=sys.stderr)
        return 2
    task: str = first["task"]
    mode, _, arg = task.partition(" ")

    if mode == "hang":
        time.sleep(3600)
        return 0
    if mode == "stubborn":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        emit("progress", message="not stopping")
        while True:
            read_msg() or time.sleep(3600)

    announce_in_room()

    if mode == "crash":
        emit("progress", message="about to crash")
        return 3
    if mode == "linger":
        emit("final", summary="done, but not exiting")
        time.sleep(3600)
        return 0
    if mode == "chatty":
        emit("progress", message="chatting")
        for i in range(int(arg or 10)):
            print(f"line {i} " + "x" * 100, flush=True)
        emit("final", summary="chatted")
        return 0
    if mode == "badjson":
        print("AGENT_EVENT {not json", flush=True)
        emit("status", status="completed")  # reserved for the gateway
        emit("final", summary="survived bad output")
        return 0
    if mode == "deaf":
        emit("progress", message="not listening")
        time.sleep(3600)
        return 0
    if mode == "orphan":
        import subprocess

        ignore = "signal.signal(signal.SIGTERM, signal.SIG_IGN); " if arg == "stubborn" else ""
        child = subprocess.Popen(
            [sys.executable, "-c", f"import signal, time; {ignore}time.sleep(60)"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(0.2)  # let the child install its signal handler
        emit("progress", message="spawned child", child_pid=child.pid)
        emit("final", summary="done, child left running")
        return 0
    if mode == "badfinal":
        emit("progress", message="about to send a malformed final")
        bad = {"dict": {"not": "a string"}, "list": ["a", "b"], "number": 42}[arg or "dict"]
        emit("final", summary=bad)
        return 0
    if mode == "escape":
        emit("artifact", name="passwd", path="../../../etc/passwd")
        emit("final", summary="tried to escape")
        return 0
    if mode == "interactive":
        emit("progress", message="waiting for messages")
        while (msg := read_msg()) is not None:
            if msg["type"] == "stop":
                emit("progress", message=f"stopping: {msg.get('reason')}")
                return 0
            if msg["type"] == "message":
                if msg["message"] == "done":
                    write_artifact("transcript.md", "# Transcript\n")
                    emit("final", summary="finished on request")
                    return 0
                emit("progress", message=f"echo from {msg['sender']}: {msg['message']}")
        return 0

    emit("progress", message=f"working on: {task}")
    print("hello from the fake worker", flush=True)
    print("a stderr line", file=sys.stderr, flush=True)
    write_artifact("notes.md", f"# Notes\n\nTask: {task}\n")
    emit("final", summary=f"did: {task}", artifacts=["notes.md"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
