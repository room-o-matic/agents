"""The session loop shared by adapters that work in discrete turns (Codex CLI, Ollama).

A subclass supplies `run_turn(prompt)` (one turn: a CLI process, an HTTP tool loop, ...)
and a `state` object; everything about the session lives here:

- oneshot (the first successful turn ends the session) or --interactive (`needs_input`
  after each turn; owner messages and room mentions start the next one);
- messages that arrive during a turn are answered next, never dropped;
- owner and room framing, joining the room before announcing, mention wakes through the
  WakeGate (with its hop, queue and wake budgets);
- a token budget, a per-turn time limit (a turn must end: waiting inside one would keep
  every later message queued behind it), and stop: mid-turn it interrupts, between turns
  of an interactive session it asks for one bounded closing summary for the handoff;
- result.md and artifact announcements.

`state` must provide: `start_turn()`, `turn_done`, `last_ok`, `last_result`, `turns`,
`usage` (dict), `total_tokens` (what the budget counts) and `result_fields()`.
"""

import argparse
import asyncio
import contextlib
import json
import os
import sys
from pathlib import Path

import httpx

from agentd.workers.claude_code import CLOSING_PROMPT, RoomWatcher, frame_owner, frame_room
from agentd.workers.common import announce_in_room, emit, join_room, post_room_status
from agentd.workers.wake import WakeGate


def token_budget(args: argparse.Namespace, profile: dict) -> int | None:
    caps = [c for c in (args.max_total_tokens, profile.get("max_total_tokens")) if c]
    return min(int(c) for c in caps) if caps else None


def add_session_args(p: argparse.ArgumentParser, *, max_turn_seconds: float) -> None:
    p.add_argument("--model")
    p.add_argument(
        "--max-total-tokens",
        type=int,
        help="tokens per session (see the adapter); profile may lower it",
    )
    p.add_argument("--interactive", action="store_true", help="wait for messages between turns")
    p.add_argument(
        "--max-turn-seconds",
        type=float,
        default=max_turn_seconds,
        help="stop a turn that runs longer (0 = no limit); interactive sessions keep going",
    )
    p.add_argument(
        "--closing-summary-seconds",
        type=float,
        default=8,
        help="interactive: on stop, allow this long for the room handoff (0 = off); keep it"
        " under the gateway's stop_grace_seconds",
    )
    p.add_argument("--room-wake", choices=["mentions", "all", "none"], default="mentions")
    p.add_argument("--room-poll-seconds", type=float, default=3.0)
    p.add_argument("--room-max-hop", type=int, default=3)
    p.add_argument("--room-max-wakes", type=int, default=10)
    p.add_argument("--room-max-queue", type=int, default=20)


class TurnAdapter:
    name = "worker"  # used in progress and error messages

    def __init__(self, args: argparse.Namespace, state):
        self.args = args
        self.state = state
        self.env = dict(os.environ)
        self.profile = json.loads(self.env.get("AGENTD_PROFILE") or "{}")
        self.cwd = os.getcwd()
        self.budget = token_budget(args, self.profile)
        self.inbox: asyncio.Queue[str] = asyncio.Queue()
        self.in_turn = False
        self.stop_requested = False
        self.interrupted = False  # a stop arrived mid-turn: no closing summary
        self.turn_timed_out = False  # the last turn ran past --max-turn-seconds
        self.finishing = False
        self.watcher: RoomWatcher | None = None
        self.gate: WakeGate | None = None
        self._current: asyncio.Future | None = None

    # ----- subclass hooks --------------------------------------------------------------

    async def run_turn(self, prompt: str) -> str | None:
        """Run one turn, updating `self.state`. Return a reason if it didn't finish."""
        raise NotImplementedError

    def interrupt(self) -> None:
        """Abandon the running turn (stop, or the turn time limit)."""
        if self._current and not self._current.done():
            self._current.cancel()

    def first_prompt(self, task: str) -> str:
        return task

    # ----- one turn ----------------------------------------------------------------------

    async def _turn(self, prompt: str, timeout: float | None = None, quiet: bool = False) -> bool:
        t = self.state
        t.start_turn()
        self.in_turn = True
        self._current = asyncio.ensure_future(self.run_turn(prompt))
        failure = None
        try:
            failure = await asyncio.wait_for(asyncio.shield(self._current), timeout)
        except TimeoutError:
            self.interrupt()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._current
            if quiet:
                emit("progress", message=f"{self.name} turn timed out")
            else:
                self.turn_timed_out = True
                emit(
                    "error",
                    message=f"{self.name} turn ran over {timeout:g}s and was stopped; a turn "
                    "must end rather than wait for messages",
                )
            quiet = True
        except asyncio.CancelledError:
            if not self._current.cancelled():
                raise  # we are being cancelled ourselves
        finally:
            self.in_turn = False
        if not t.turn_done and not self.stop_requested and not quiet:
            emit("error", message=failure or f"{self.name} stopped before finishing the turn")
            t.last_ok = False
        return t.turn_done and t.last_ok

    def _write_result_artifact(self) -> None:
        text = self.state.last_result
        artifacts = self.env.get("AGENTD_ARTIFACTS_DIR")
        if text and artifacts:
            Path(artifacts, "result.md").write_text(text)
            emit("artifact", name="result.md", path="result.md", mime_type="text/markdown")

    def _over_budget(self, report: bool = True) -> bool:
        over = bool(self.budget and self.state.total_tokens >= self.budget)
        if over and report:
            emit(
                "progress",
                message=f"token budget reached ({self.state.total_tokens} of {self.budget});"
                " ending the session",
                usage=dict(self.state.usage),
            )
        return over

    # ----- input: owner messages, room wakes, stop -----------------------------------------

    async def _pump_gateway(self, gateway: asyncio.StreamReader) -> None:
        while (line := await gateway.readline()) and not self.stop_requested:
            msg = json.loads(line)
            if msg.get("type") == "message":
                await self._enqueue(frame_owner(msg.get("sender"), msg.get("message")))
            elif msg.get("type") == "stop":
                break
        self.stop_requested = True
        if self.in_turn:
            self.interrupted = True
            self.interrupt()  # don't wait for the turn to finish
        await self.inbox.put("")  # wake the main loop

    async def _enqueue(self, text: str) -> None:
        if self.finishing:
            emit("progress", message="message arrived after the task finished; not delivered")
            return
        await self.inbox.put(text)

    def _gate(self) -> WakeGate:
        if self.gate is None:
            self.gate = WakeGate(
                identity=self.watcher.identity,
                policy=self.args.room_wake,
                max_hop=self.args.room_max_hop,
                max_wakes=self.args.room_max_wakes,
                max_queue=self.args.room_max_queue,
            )
        return self.gate

    async def _room_turn(self, m: dict) -> None:
        if self.finishing or self.stop_requested:
            return
        gate = self._gate()
        action, reason = gate.offer(m, in_turn=self.in_turn or not self.inbox.empty())
        if action == "deliver":
            emit(
                "progress",
                message=f"room message #{m['id']} from {m['from']}",
                room_message_id=m["id"],
            )
            await self._enqueue(frame_room(m))
        elif action == "queue":
            emit("progress", message=f"queued room message #{m['id']}", room_message_id=m["id"])
        elif action == "suppress":
            emit(
                "progress",
                message=f"room wake suppressed ({reason})",
                room_wake_suppressed={"reason": reason, "count": gate.suppressed[reason]},
                room_message_id=m["id"],
            )
            if reason == "wake_budget_exhausted" and gate.suppressed[reason] == 1:
                post_room_status(
                    "wake budget exhausted: I'll stop responding to room messages in this "
                    "session; my owner can stop or re-summon me."
                )

    def _drain_room_queue(self) -> str | None:
        if self.gate is None:
            return None
        batch = self.gate.drain()
        if not batch:
            return None
        emit(
            "progress",
            message=f"coalesced {len(batch)} room messages into one turn",
            room_message_ids=[m["id"] for m in batch],
        )
        return "\n\n".join(frame_room(m) for m in batch)

    def _pending(self) -> list[str]:
        texts = []
        while not self.inbox.empty():
            if text := self.inbox.get_nowait():
                texts.append(text)
        if queued := self._drain_room_queue():
            texts.append(queued)
        return texts

    # ----- the session ---------------------------------------------------------------------

    async def run(self) -> int:
        emit("progress", message=f"starting {self.name}")
        gateway = await _stdin_reader()
        first = await _read_json(gateway)
        if not first or first.get("type") != "task":
            emit("error", message="expected a task message first")
            return 2
        # Join, fix where room wakes start, then announce (see the Claude adapter).
        if join_room() is not None and self.args.room_wake != "none":
            self.watcher = RoomWatcher.from_env(self.args)
            try:
                await self.watcher.start()
            except httpx.HTTPError as e:
                emit("error", message=f"room watcher stopped: {e}")
                self.watcher = None
        announce_in_room(join=False)

        inbox = asyncio.create_task(self._pump_gateway(gateway))
        watcher = asyncio.create_task(self.watcher.run(self._room_turn)) if self.watcher else None
        try:
            return await self._session(first["task"])
        finally:
            inbox.cancel()
            if watcher:
                watcher.cancel()
            self._report_new_artifacts()

    async def _session(self, task: str) -> int:
        t = self.state
        prompt = self.first_prompt(task)
        while True:
            self.turn_timed_out = False
            ok = await self._turn(prompt, timeout=self.args.max_turn_seconds or None)
            if ok:
                self._write_result_artifact()
            if self.stop_requested:
                break
            if not ok and not (self.turn_timed_out and self.args.interactive):
                return 1  # the error is reported; agentd marks the session failed
            if self._over_budget():
                break
            pending = self._pending()
            if pending:  # messages that arrived during the turn: answer them first
                prompt = "\n\n".join(pending)
                continue
            if not self.args.interactive:
                break
            emit("needs_input", question=t.result_fields()["summary"], turn=t.turns)
            text = await self.inbox.get()
            if self.stop_requested or not text:
                break
            prompt = "\n\n".join([text, *self._pending()])
        self.finishing = True
        if self.stop_requested and self._wants_closing_summary():
            before = (t.last_result, t.last_ok)
            emit("progress", message="writing a closing summary for the handoff")
            self.stop_requested = False  # let this one turn run
            ok = await self._turn(
                CLOSING_PROMPT, timeout=self.args.closing_summary_seconds, quiet=True
            )
            self.stop_requested = True
            if not ok:
                t.last_result, t.last_ok = before
        if t.last_result is None:
            return 0 if self.stop_requested else 1
        emit("final", **t.result_fields())
        return 0

    def _wants_closing_summary(self) -> bool:
        return (
            self.args.interactive
            and not self.interrupted
            and self.args.closing_summary_seconds > 0
            and self.state.last_ok
            and self.state.turns > 0
            and not self._over_budget(report=False)
        )

    def _report_new_artifacts(self) -> None:
        artifacts = self.env.get("AGENTD_ARTIFACTS_DIR")
        if not artifacts:
            return
        root = Path(artifacts)
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            rel = str(path.relative_to(root))
            if rel != "result.md":
                emit("artifact", name=rel, path=rel)


async def _stdin_reader() -> asyncio.StreamReader:
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader(limit=16 * 1024 * 1024)
    await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), sys.stdin)
    return reader


async def _read_json(reader: asyncio.StreamReader) -> dict | None:
    line = await reader.readline()
    return json.loads(line) if line else None
