"""Session lifecycle: spawn, route messages, consume worker output, stop, and clean up.

Everything here runs on the event loop thread; SQLite access is synchronous on the one
shared connection. Each live session has one `_run` task that owns the worker process
and is the only place a session reaches a terminal status (via `_finish`).
"""

import asyncio
import json
import logging
import os
import signal
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from agentd.backends.process import (
    ProcessBackend,
    ProcessHandle,
    WorkerNotReading,
    find_orphan,
    iter_lines,
)
from agentd.backends.sandbox import LaunchSpec, SandboxBackend
from agentd.config import CallerPolicy, Profile, Settings, WorkerType
from agentd.events import EventStore
from agentd.finalize import RoomFinalizer
from agentd.ids import iso_in, new_id, now_iso
from agentd.models import ACTIVE_STATUSES, RoomRef, SpawnRequest
from agentd.policy import CallerPolicies, PolicyError, authorize, caller_workspace_ok
from agentd.protocol import parse_stdout_line

log = logging.getLogger("agentd.supervisor")


def task_grant(
    session_id: str,
    profile_name: str,
    profile: Profile,
    workspace: Path | None,
    room: RoomRef | None,
    requester: str,
    expires_at: str,
    caller: CallerPolicy | None = None,
) -> dict:
    """The immutable authority a session runs under (docs#8). Fixed at spawn from the
    server-side profile and the authenticated requester; nothing a worker reads later
    (room messages, notes, artifacts, issue text) can widen it. Workers must derive every
    permission from this, and there is no approval channel: anything not granted is denied."""
    extra = profile.model_extra or {}
    return {
        "session_id": session_id,
        "requester": requester,
        "profile": profile_name,
        "workspace": (
            {"path": str(workspace), "mode": profile.workspace_mount} if workspace else None
        ),
        "network": profile.network,
        "external_actions": profile.external_actions,
        "approval": "none",
        "expires_at": expires_at,
        # The lower of the profile's and the caller's spend caps (docs#9).
        "max_budget_usd": min(
            (b for b in (extra.get("max_budget_usd"), caller and caller.max_budget_usd) if b),
            default=None,
        ),
        "room": (
            {"room_url": room.room_url, "reply": extra.get("room_reply", True) is not False}
            if room
            else None
        ),
    }


class SpawnReplay(Exception):  # noqa: N818 - control flow, not an error
    """The spawn is a retry of an earlier operation; carries the original session row."""

    def __init__(self, row: sqlite3.Row):
        super().__init__(row["id"])
        self.row = row


class SpawnError(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


@dataclass
class LiveSession:
    session_id: str
    handle: ProcessHandle
    artifacts_dir: Path
    room: RoomRef | None
    stdin_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    finished: asyncio.Event = field(default_factory=asyncio.Event)
    ready: bool = False
    final_summary: str | None = None
    final_seen: bool = False
    log_bytes: int = 0
    log_truncated: bool = False
    # docs#22: structured-event budget
    event_bytes: int = 0
    event_count: int = 0
    output_truncated: bool = False
    rate_tokens: float = -1  # -1: start with a full burst
    rate_at: float = 0
    throttling: bool = False
    dropped_events: int = 0
    # Set by stop(); applied when the process exits unless the worker already sent final.
    stop_status: str | None = None
    stop_reason: str | None = None
    tasks: list[asyncio.Task] = field(default_factory=list)


class Supervisor:
    def __init__(
        self,
        settings: Settings,
        conn: sqlite3.Connection,
        events: EventStore,
        backend: ProcessBackend | None = None,
    ):
        self.settings = settings
        self.conn = conn
        self.events = events
        limits = {
            "max_pending_bytes": settings.max_pending_stdin_bytes,
            "send_timeout": settings.send_timeout_seconds,
        }
        if backend is None and settings.backend == "sandbox":
            backend = SandboxBackend(settings.sandbox, **limits)
            backend.check()  # fail closed: no isolation available, no gateway
        self.backend = backend or ProcessBackend(**limits)
        self._background: set[asyncio.Task] = set()
        # Slots reserved by spawns that passed admission but aren't in _live yet. Taken
        # synchronously at admission (no await in between) so concurrent spawns can't
        # oversubscribe max_sessions (docs#4).
        self._launching = 0
        self._launching_by: dict[str, int] = {}
        self.policies = CallerPolicies(settings)
        self.finalizer = RoomFinalizer(
            conn,
            events,
            max_attempts=settings.finalize_max_attempts,
            base_delay=settings.finalize_retry_seconds,
        )
        self._live: dict[str, LiveSession] = {}
        # Set whenever active_count() changes, so the registry heartbeat can report promptly.
        self.capacity_changed = asyncio.Event()

    # ----- queries -------------------------------------------------------------------

    def _caller_active(self, principal: str) -> int:
        live = sum(1 for sid in self._live if self._requester(sid) == principal)
        return live + self._launching_by.get(principal, 0)

    def _requester(self, session_id: str) -> str | None:
        row = self.get_row(session_id)
        return row["requester_agent"] if row else None

    def active_count(self) -> int:
        """Live sessions plus launches in progress: what admission and the registry see."""
        return len(self._live) + self._launching

    def get_row(self, session_id: str) -> sqlite3.Row | None:
        return self.conn.execute("select * from sessions where id = ?", (session_id,)).fetchone()

    # ----- spawn ---------------------------------------------------------------------

    def _resolve_workspace(self, req: SpawnRequest, profile: Profile) -> Path | None:
        if req.workspace is None:
            return None
        if profile.workspace_mount == "none":
            raise SpawnError(403, f"profile {req.profile!r} does not allow a workspace")
        path = Path(req.workspace.path).resolve()
        roots = [r.resolve() for r in self.settings.workspace_roots]
        if not any(path.is_relative_to(root) for root in roots):
            raise SpawnError(403, f"workspace {str(path)!r} is not under an allowed root")
        if not path.is_dir():
            raise SpawnError(422, f"workspace {str(path)!r} is not a directory")
        return path

    def _worker_env(
        self,
        session_id: str,
        profile_name: str,
        profile: Profile,
        worker_env: dict[str, str],
        artifacts_dir: Path,
        workspace: Path | None,
        room: RoomRef | None,
        requester: str,
        expires_at: str,
        caller: CallerPolicy,
    ) -> dict[str, str]:
        env = {k: os.environ[k] for k in self.settings.env_allowlist if k in os.environ}
        env.update(worker_env)
        env.update(
            AGENTD_SESSION_ID=session_id,
            AGENTD_INSTANCE_ID=self.settings.instance_id,
            AGENTD_PROFILE_NAME=profile_name,
            AGENTD_PROFILE=profile.model_dump_json(),
            AGENTD_ARTIFACTS_DIR=str(artifacts_dir),
            AGENTD_GRANT=json.dumps(
                task_grant(
                    session_id,
                    profile_name,
                    profile,
                    workspace,
                    room,
                    requester,
                    expires_at,
                    caller,
                )
            ),
        )
        if workspace:
            env.update(
                AGENTD_WORKSPACE=str(workspace), AGENTD_WORKSPACE_MODE=profile.workspace_mount
            )
        if room:
            env.update(
                ROOMSD_URL=room.base_url,
                ROOMSD_ROOM_ID=room.room_id,
                ROOMSD_ROOM_URL=room.room_url,
                ROOMSD_TOKEN=room.token,
            )
        return env

    def by_operation(self, requester_agent: str, operation_id: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "select * from sessions where requester_agent = ? and operation_id = ?",
            (requester_agent, operation_id),
        ).fetchone()

    async def spawn(self, req: SpawnRequest, requester_agent: str) -> sqlite3.Row:
        """Start a session, or with an operation_id seen before, return that session
        (docs#13). Check and insert happen with no await in between, so concurrent
        duplicates can't both start a worker."""
        if req.operation_id:
            existing = self.by_operation(requester_agent, req.operation_id)
            if existing is not None:
                if existing["payload_hash"] != req.payload_hash():
                    raise SpawnError(
                        409, "operation_id was already used for a different spawn request"
                    )
                raise SpawnReplay(existing)
        s = self.settings
        profile = s.profiles.get(req.profile)
        if profile is None:
            raise SpawnError(422, f"unknown profile {req.profile!r}; have {sorted(s.profiles)}")
        worker = s.worker_types.get(req.worker_type)
        if worker is None:
            raise SpawnError(
                422, f"unknown worker_type {req.worker_type!r}; have {sorted(s.worker_types)}"
            )
        if profile.worker_types is not None and req.worker_type not in profile.worker_types:
            raise SpawnError(403, f"profile {req.profile!r} does not allow {req.worker_type!r}")
        if profile.external_actions == "approval_required":
            # There is no approval transaction in this API, and approval is never inferred
            # from a peer, task metadata or model output (docs#9): refuse rather than run.
            raise SpawnError(
                403, f"profile {req.profile!r} requires approval, which this gateway can't grant"
            )
        try:
            caller = authorize(
                self.policies.lookup(requester_agent),
                principal=requester_agent,
                profile_name=req.profile,
                worker_type=req.worker_type,
                backend=s.backend,
            )
        except PolicyError as e:
            raise SpawnError(e.status_code, e.detail) from e
        workspace = self._resolve_workspace(req, profile)
        if workspace is not None and not caller_workspace_ok(caller, workspace):
            raise SpawnError(403, f"{requester_agent} may not use workspace {str(workspace)!r}")
        room = req.room
        if self.active_count() >= s.max_sessions:
            raise SpawnError(429, f"instance at capacity ({s.max_sessions} active sessions)")
        if self._caller_active(requester_agent) >= caller.max_sessions:
            raise SpawnError(
                429, f"{requester_agent} is at its quota ({caller.max_sessions} sessions)"
            )
        self._launching += 1
        self.capacity_changed.set()
        try:
            self._launching_by[requester_agent] = self._launching_by.get(requester_agent, 0) + 1
            return await self._launch(
                req, requester_agent, profile, worker, workspace, room, caller
            )
        finally:
            self._launching -= 1
            self._launching_by[requester_agent] -= 1
            self.capacity_changed.set()

    async def _launch(
        self,
        req: SpawnRequest,
        requester_agent: str,
        profile: Profile,
        worker: WorkerType,
        workspace: Path | None,
        room: RoomRef | None,
        caller: CallerPolicy,
    ) -> sqlite3.Row:
        """Everything after admission. Runs while holding a reserved slot; by the time it
        returns the session is either in _live or terminal."""
        s = self.settings
        # A caller timeout can only lower the profile's maximum.
        hard = profile.max_runtime_minutes * 60
        if req.timeout_seconds is not None:
            hard = min(hard, req.timeout_seconds)
        if room is not None and room.expires_at:
            # The session can't outlive its room access (docs#17): cap the hard deadline.
            try:
                remaining = (
                    datetime.fromisoformat(room.expires_at.replace("Z", "+00:00"))
                    - datetime.now(UTC)
                ).total_seconds()
            except ValueError as e:
                raise SpawnError(422, f"room.expires_at isn't ISO 8601: {e}") from e
            if remaining <= 0:
                raise SpawnError(422, "the room invite has already expired")
            hard = min(hard, remaining)
        idle = min(req.idle_timeout_seconds or s.default_idle_timeout_seconds, hard)

        session_id = new_id("agt")
        session_dir = s.sessions_dir / session_id
        artifacts_dir = session_dir / "artifacts"
        scratch_dir = session_dir / "scratch"
        artifacts_dir.mkdir(parents=True)
        scratch_dir.mkdir()
        # The room token is a credential: kept in memory and the worker env only.
        redacted = req.model_dump(mode="json")
        if redacted.get("room"):
            redacted["room"]["token"] = "[redacted]"
        (session_dir / "input.json").write_text(json.dumps(redacted, indent=2))

        now = now_iso()
        expires_at = iso_in(hard)
        with self.conn:
            self.conn.execute(
                "insert into sessions (id, instance_id, requester_agent, requester_surface,"
                " requester_conversation_id, parent_session_id, profile, worker_type, status,"
                " task, workspace_path, room_url, idle_timeout_seconds, created_at,"
                " last_activity_at, expires_at, metadata_json, operation_id, payload_hash,"
                " room_invite_id, room_finalization)"
                " values (?, ?, ?, ?, ?, ?, ?, ?, 'starting', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    s.instance_id,
                    requester_agent,
                    req.requester.surface,
                    req.requester.conversation_id,
                    req.parent_session_id,
                    req.profile,
                    req.worker_type,
                    req.task,
                    str(workspace) if workspace else None,
                    room.room_url if room else None,
                    idle,
                    now,
                    now,
                    expires_at,
                    json.dumps(req.metadata) if req.metadata is not None else None,
                    req.operation_id,
                    req.payload_hash() if req.operation_id else None,
                    room.invite_id if room else None,
                    "pending" if room else None,  # set now so a crash leaves a trace
                ),
            )
        self.events.append(session_id, "status", status="starting")

        env = self._worker_env(
            session_id,
            req.profile,
            profile,
            worker.env,
            artifacts_dir,
            workspace,
            room,
            requester_agent,
            expires_at,
            caller,
        )
        try:
            spec = LaunchSpec(
                scratch_dir=scratch_dir,
                artifacts_dir=artifacts_dir,
                workspace=workspace,
                workspace_mode=profile.workspace_mount,
                network=profile.network,
            )
            handle = await self.backend.start(worker.command, env, workspace or scratch_dir, spec)
        except OSError as e:
            self._set_terminal(session_id, "failed", f"worker failed to start: {e}")
            if room is not None:  # never leave a live invite behind a failed launch
                self.finalizer.start(
                    session_id,
                    room,
                    "status",
                    f"Session {session_id} on {s.instance_id} failed to start: {e}",
                )
            return self.get_row(session_id)
        except BaseException:
            # Cancelled mid-launch: never leave the row looking like it's starting.
            self._set_terminal(session_id, "failed", "launch cancelled")
            raise

        with self.conn:
            self.conn.execute("update sessions set pid = ? where id = ?", (handle.pid, session_id))
        live = LiveSession(session_id, handle, artifacts_dir, room)
        self._live[session_id] = live  # the reservation is released by spawn's finally
        live.tasks.append(asyncio.create_task(self._run(live), name=f"run-{session_id}"))
        try:
            await handle.send({"type": "task", "session_id": session_id, "task": req.task})
        except (BrokenPipeError, ConnectionResetError, WorkerNotReading):
            pass  # The worker died immediately; _run records the failure.
        return self.get_row(session_id)

    # ----- running -------------------------------------------------------------------

    def _touch(self, session_id: str) -> None:
        with self.conn:
            self.conn.execute(
                "update sessions set last_activity_at = ? where id = ?", (now_iso(), session_id)
            )

    def _mark_running(self, live: LiveSession) -> None:
        live.ready = True
        with self.conn:
            self.conn.execute(
                "update sessions set status = 'running' where id = ? and status = 'starting'",
                (live.session_id,),
            )
        self.events.append(live.session_id, "status", status="running")

    def _record_log(self, live: LiveSession, stream: str, text: str) -> None:
        if live.log_truncated:
            return
        live.log_bytes += len(text)
        if live.log_bytes > self.settings.max_log_bytes:
            live.log_truncated = True
            self.events.append(
                live.session_id,
                "log_truncated",
                reason=f"worker output exceeded {self.settings.max_log_bytes} bytes",
            )
            return
        self.events.append(live.session_id, "log", stream=stream, text=text)

    def _admit_event(self, live: LiveSession, event: dict) -> bool:
        """Charge a structured worker event to the session's budgets (docs#22)."""
        if live.output_truncated:
            live.dropped_events += 1
            return False
        st = self.settings
        now = time.monotonic()
        if live.rate_tokens < 0:
            live.rate_tokens = st.event_burst
        else:
            live.rate_tokens = min(
                st.event_burst, live.rate_tokens + (now - live.rate_at) * st.event_rate_per_second
            )
        live.rate_at = now
        if live.rate_tokens < 1:
            live.dropped_events += 1
            if not live.throttling:
                live.throttling = True
                self._charge(live, 0)
                self.events.append(
                    live.session_id,
                    "output_throttled",
                    reason=f"worker events over {st.event_rate_per_second}/s;"
                    " dropping until the rate falls",
                )
            return False
        live.rate_tokens -= 1
        live.throttling = False
        size = len(json.dumps(event))
        if live.event_count + 1 > st.max_events or live.event_bytes + size > st.max_event_bytes:
            live.output_truncated = True
            live.dropped_events += 1
            self.events.append(
                live.session_id,
                "output_truncated",
                reason=f"worker events exceeded {st.max_events} events or"
                f" {st.max_event_bytes} bytes; further events are dropped",
            )
            return False
        self._charge(live, size)
        return True

    @staticmethod
    def _charge(live: LiveSession, size: int) -> None:
        live.event_count += 1
        live.event_bytes += size

    def _record_worker_event(self, live: LiveSession, event: dict) -> None:
        sid = live.session_id
        kind = event["type"]
        if kind == "final" and live.final_seen:
            event = {"type": "protocol_error", "reason": "duplicate final ignored"}
            kind = "protocol_error"
        if kind != "protocol_error" and not live.ready:
            self._mark_running(live)
        if kind == "artifact":
            name, path = event.get("name"), event.get("path") or event.get("name")
            if not isinstance(name, str) or not isinstance(path, str):
                event = {"type": "protocol_error", "reason": "artifact needs name and path"}
            else:
                resolved = (live.artifacts_dir / path).resolve()
                if not resolved.is_relative_to(live.artifacts_dir.resolve()):
                    event = {
                        "type": "protocol_error",
                        "reason": "artifact path escapes artifacts dir",
                        "text": path,
                    }
                else:
                    event = {**event, "path": str(resolved), "exists": resolved.is_file()}
        elif kind == "final":
            # The terminal result is never dropped by the output budget.
            live.final_seen = True
            live.final_summary = event.get("summary")
            live.tasks.append(asyncio.create_task(self._reap_after_final(live)))
            self.events.append(sid, **event)
            return
        if self._admit_event(live, event):
            self.events.append(sid, **event)

    async def _read_stdout(self, live: LiveSession) -> None:
        async for line in iter_lines(live.handle.stdout, self.settings.max_line_bytes):
            self._touch(live.session_id)
            event = parse_stdout_line(line)
            if event["type"] == "log":
                self._record_log(live, "stdout", line)
            else:
                self._record_worker_event(live, event)
            await asyncio.sleep(0)  # docs#22: a chatty worker must not starve the loop

    async def _read_stderr(self, live: LiveSession) -> None:
        async for line in iter_lines(live.handle.stderr, self.settings.max_line_bytes):
            self._touch(live.session_id)
            self._record_log(live, "stderr", line)
            await asyncio.sleep(0)

    async def _ready_watchdog(self, live: LiveSession) -> None:
        await asyncio.sleep(self.settings.ready_timeout_seconds)
        if not live.ready and not live.finished.is_set():
            await self.stop(
                live.session_id,
                f"worker emitted no event within {self.settings.ready_timeout_seconds}s",
                status="failed",
            )

    async def _reap_after_final(self, live: LiveSession) -> None:
        """A worker that sent final should exit; make sure it does."""
        try:
            await asyncio.wait_for(live.finished.wait(), self.settings.stop_grace_seconds)
        except TimeoutError:
            await self._terminate(live)

    async def _read_output(self, live: LiveSession) -> None:
        await asyncio.gather(self._read_stdout(live), self._read_stderr(live))

    async def _wait_group_gone(self, live: LiveSession, timeout: float) -> bool:
        deadline = asyncio.get_running_loop().time() + timeout
        while live.handle.group_alive():
            if asyncio.get_running_loop().time() >= deadline:
                return False
            await asyncio.sleep(0.05)
        return True

    async def _reap_group(self, live: LiveSession) -> None:
        """After the leader exits, end every process left in its group (SIGTERM, then
        SIGKILL). Lifecycle ownership isn't released while descendants remain."""
        if not live.handle.group_alive():
            return
        grace = self.settings.stop_grace_seconds
        live.handle.terminate()
        if await self._wait_group_gone(live, grace):
            return
        live.handle.kill()
        if not await self._wait_group_gone(live, grace):
            self.events.append(
                live.session_id,
                "protocol_error",
                reason="worker process group still alive after SIGKILL",
            )

    async def _run(self, live: LiveSession) -> None:
        sid = live.session_id
        grace = self.settings.stop_grace_seconds
        watchdog = asyncio.create_task(self._ready_watchdog(live))
        readers = asyncio.create_task(self._read_output(live))
        leader = asyncio.create_task(live.handle.wait())
        try:
            done, _ = await asyncio.wait({readers, leader}, return_when=asyncio.FIRST_COMPLETED)
            if readers in done and readers.exception():
                raise readers.exception()
            exit_code = await leader
            await self._reap_group(live)
            # Output normally closes with the group. A process that escaped the group can
            # hold the pipes open; don't let it keep the session alive.
            try:
                await asyncio.wait_for(asyncio.shield(readers), grace)
            except TimeoutError:
                readers.cancel()
                self.events.append(
                    sid,
                    "protocol_error",
                    reason="worker output still open after its process group exited",
                )
            else:
                if readers.exception():
                    raise readers.exception()
        except Exception as e:  # noqa: BLE001 - any supervisor bug must still end the session
            log.exception("session %s supervisor error", sid)
            readers.cancel()
            await self._terminate(live)
            exit_code = await leader
            await self._reap_group(live)
            live.stop_status, live.stop_reason = "failed", f"gateway error: {e}"
        finally:
            watchdog.cancel()

        if live.final_seen:
            status, reason = "completed", None
        elif live.stop_status:
            status, reason = live.stop_status, live.stop_reason
        else:
            status, reason = "failed", f"worker exited with code {exit_code} without a final event"
        try:
            self._set_terminal(sid, status, reason, exit_code=exit_code, summary=live.final_summary)
        except Exception as e:  # noqa: BLE001 - the session must still be released
            log.exception("session %s: recording terminal status failed", sid)
            status, reason = "failed", f"gateway error during finalization: {e}"
            try:
                self._set_terminal(sid, status, reason, exit_code=exit_code)
            except Exception:  # noqa: BLE001
                log.exception("session %s: recording failure status also failed", sid)
        finally:
            # Always release bookkeeping, or capacity leaks and stop()/shutdown() hang.
            self._live.pop(sid, None)
            self.capacity_changed.set()
            live.finished.set()
        if live.room:
            await self._close_out_room(live, status, reason)

    def _set_terminal(
        self,
        session_id: str,
        status: str,
        reason: str | None,
        *,
        exit_code: int | None = None,
        summary: str | None = None,
    ) -> None:
        if not isinstance(summary, str):
            summary = None  # protocol validation should have caught it; never bind non-text
        with self.conn:
            self.conn.execute(
                "update sessions set status = ?, stop_reason = ?, stopped_at = ?,"
                " exit_code = ?, summary = ? where id = ?",
                (status, reason, now_iso(), exit_code, summary, session_id),
            )
        self.events.append(session_id, "status", status=status, reason=reason, exit_code=exit_code)

    async def _close_out_room(self, live: LiveSession, status: str, reason: str | None) -> None:
        row = self.get_row(live.session_id)
        body = f"Session {live.session_id} on {self.settings.instance_id} ended: {status}"
        if reason:
            body += f" ({reason})"
        if row["summary"]:
            body += f". Summary: {row['summary']}"
        msg_type = "handoff" if status == "completed" else "status"
        self.finalizer.start(live.session_id, live.room, msg_type, body)

    # ----- caller actions ------------------------------------------------------------

    async def send_message(self, session_id: str, message: str, sender: str) -> str | None:
        """Deliver a caller message. Returns None on success, else why it was refused."""
        live = self._live.get(session_id)
        if live is None or live.final_seen or live.stop_status:
            return "session is not accepting messages"
        try:
            async with asyncio.timeout(self.settings.send_timeout_seconds):
                async with live.stdin_lock:
                    await live.handle.send(
                        {"type": "message", "sender": sender, "message": message}
                    )
        except WorkerNotReading as e:
            return str(e)
        except (BrokenPipeError, ConnectionResetError):
            return "worker input is closed"
        except TimeoutError:
            return "worker input is busy"
        self.events.append(session_id, "message", sender=sender, message=message)
        self._touch(session_id)
        return None

    async def _terminate(self, live: LiveSession) -> None:
        """SIGTERM the process group, then SIGKILL after the grace period."""
        live.handle.terminate()
        try:
            await asyncio.wait_for(live.handle.wait(), self.settings.stop_grace_seconds)
        except TimeoutError:
            live.handle.kill()

    async def stop(self, session_id: str, reason: str, status: str = "stopped") -> None:
        """Ask the worker to stop, then SIGTERM, then SIGKILL. Returns once the session has
        ended (process group gone). The deadline starts first: a worker that isn't reading
        stdin, or a send holding the input lock, can only delay escalation by the grace."""
        live = self._live.get(session_id)
        if live is None:
            return
        if live.stop_status is None:
            live.stop_status, live.stop_reason = status, reason
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.settings.stop_grace_seconds

        async def ask() -> None:
            async with live.stdin_lock:
                try:
                    await live.handle.send({"type": "stop", "reason": reason})
                except (BrokenPipeError, ConnectionResetError, WorkerNotReading):
                    pass
                live.handle.close_stdin()

        try:
            await asyncio.wait_for(ask(), self.settings.stop_grace_seconds)
        except TimeoutError:
            pass
        try:
            await asyncio.wait_for(live.finished.wait(), max(0.0, deadline - loop.time()))
        except TimeoutError:
            await self._terminate(live)
        await live.finished.wait()

    # ----- housekeeping --------------------------------------------------------------

    def _stop_in_background(self, session_id: str, reason: str, status: str) -> None:
        task = asyncio.create_task(self.stop(session_id, reason, status=status))
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    async def cleanup_once(self) -> None:
        """Start stops for timed-out sessions without awaiting them, so one slow stop
        can't stall timeouts for every other session; retry owed room finalizations."""
        await self.finalizer.retry_due()
        self.prune_events()
        now = datetime.now(UTC)
        for sid, live in list(self._live.items()):
            if live.stop_status is not None:
                continue  # already stopping
            row = self.get_row(sid)
            if datetime.fromisoformat(row["expires_at"]) <= now:
                self._stop_in_background(sid, "hard_timeout", "expired")
                continue
            idle = (now - datetime.fromisoformat(row["last_activity_at"])).total_seconds()
            if idle >= row["idle_timeout_seconds"]:
                self._stop_in_background(sid, "idle_timeout", "expired")

    def prune_events(self) -> int:
        """Retention (docs#22): drop the event log of sessions that ended more than
        event_retention_days ago. The session row, status and summary stay."""
        days = self.settings.event_retention_days
        if days is None:
            return 0
        cutoff = iso_in(-days * 86400)
        placeholders = ",".join("?" * len(ACTIVE_STATUSES))
        ids = [
            r[0]
            for r in self.conn.execute(
                f"select id from sessions where stopped_at is not null and stopped_at < ?"
                f" and status not in ({placeholders})"
                " and exists (select 1 from events e where e.session_id = sessions.id)",
                (cutoff, *ACTIVE_STATUSES),
            )
        ]
        for sid in ids:
            with self.conn:
                self.conn.execute("delete from events where session_id = ?", (sid,))
            (self.settings.sessions_dir / sid / "events.jsonl").unlink(missing_ok=True)
        if ids:
            log.info("pruned event logs of %d ended sessions", len(ids))
        return len(ids)

    async def cleanup_loop(self) -> None:
        while True:
            await asyncio.sleep(self.settings.cleanup_interval_seconds)
            try:
                await self.cleanup_once()
            except Exception:  # noqa: BLE001 - keep the loop alive
                log.exception("cleanup pass failed")

    def recover(self) -> None:
        """On startup, fail sessions a previous gateway process left active."""
        placeholders = ",".join("?" * len(ACTIVE_STATUSES))
        rows = self.conn.execute(
            f"select id, pid from sessions where status in ({placeholders})", ACTIVE_STATUSES
        ).fetchall()
        for row in rows:
            if row["pid"] and find_orphan(row["pid"], row["id"]):
                try:
                    os.killpg(row["pid"], signal.SIGKILL)
                except ProcessLookupError:
                    pass
            self._set_terminal(row["id"], "failed", "gateway_restarted")
        self.finalizer.recover()

    async def shutdown(self) -> None:
        await asyncio.gather(*(self.stop(sid, "gateway_shutdown") for sid in list(self._live)))
        await self.finalizer.drain(timeout=2 * self.settings.stop_grace_seconds)
