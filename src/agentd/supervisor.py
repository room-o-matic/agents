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
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import httpx

from agentd import rooms_client
from agentd.backends.process import ProcessBackend, ProcessHandle, find_orphan, iter_lines
from agentd.config import Profile, Settings
from agentd.events import EventStore
from agentd.ids import iso_in, new_id, now_iso
from agentd.models import ACTIVE_STATUSES, RoomRef, SpawnRequest
from agentd.protocol import parse_stdout_line

log = logging.getLogger("agentd.supervisor")


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
        self.backend = backend or ProcessBackend()
        self._live: dict[str, LiveSession] = {}
        # Set whenever active_count() changes, so the registry heartbeat can report promptly.
        self.capacity_changed = asyncio.Event()

    # ----- queries -------------------------------------------------------------------

    def active_count(self) -> int:
        return len(self._live)

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
    ) -> dict[str, str]:
        env = {k: os.environ[k] for k in self.settings.env_allowlist if k in os.environ}
        env.update(worker_env)
        env.update(
            AGENTD_SESSION_ID=session_id,
            AGENTD_INSTANCE_ID=self.settings.instance_id,
            AGENTD_PROFILE_NAME=profile_name,
            AGENTD_PROFILE=profile.model_dump_json(),
            AGENTD_ARTIFACTS_DIR=str(artifacts_dir),
        )
        if workspace:
            env.update(
                AGENTD_WORKSPACE=str(workspace), AGENTD_WORKSPACE_MODE=profile.workspace_mount
            )
        if room:
            env.update(
                ROOMSD_URL=room.url or "", ROOMSD_ROOM_ID=room.room_id, ROOMSD_TOKEN=room.token
            )
        return env

    async def spawn(self, req: SpawnRequest, requester_agent: str) -> sqlite3.Row:
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
        workspace = self._resolve_workspace(req, profile)
        room = req.room
        if room is not None:
            room = room.model_copy(update={"url": room.url or s.roomsd_url})
            if not room.url:
                raise SpawnError(422, "room.url is required (no roomsd_url configured)")
        if len(self._live) >= s.max_sessions:
            raise SpawnError(429, f"instance at capacity ({s.max_sessions} active sessions)")

        # A caller timeout can only lower the profile's maximum.
        hard = profile.max_runtime_minutes * 60
        if req.timeout_seconds is not None:
            hard = min(hard, req.timeout_seconds)
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
        with self.conn:
            self.conn.execute(
                "insert into sessions (id, instance_id, requester_agent, requester_surface,"
                " requester_conversation_id, parent_session_id, profile, worker_type, status,"
                " task, workspace_path, room_id, idle_timeout_seconds, created_at,"
                " last_activity_at, expires_at, metadata_json)"
                " values (?, ?, ?, ?, ?, ?, ?, ?, 'starting', ?, ?, ?, ?, ?, ?, ?, ?)",
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
                    room.room_id if room else None,
                    idle,
                    now,
                    now,
                    iso_in(hard),
                    json.dumps(req.metadata) if req.metadata is not None else None,
                ),
            )
        self.events.append(session_id, "status", status="starting")

        env = self._worker_env(
            session_id, req.profile, profile, worker.env, artifacts_dir, workspace, room
        )
        try:
            handle = await self.backend.start(worker.command, env, workspace or scratch_dir)
        except OSError as e:
            self._set_terminal(session_id, "failed", f"worker failed to start: {e}")
            return self.get_row(session_id)

        with self.conn:
            self.conn.execute("update sessions set pid = ? where id = ?", (handle.pid, session_id))
        live = LiveSession(session_id, handle, artifacts_dir, room)
        self._live[session_id] = live
        self.capacity_changed.set()
        live.tasks.append(asyncio.create_task(self._run(live), name=f"run-{session_id}"))
        try:
            await handle.send({"type": "task", "session_id": session_id, "task": req.task})
        except (BrokenPipeError, ConnectionResetError):
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

    def _record_worker_event(self, live: LiveSession, event: dict) -> None:
        sid = live.session_id
        kind = event["type"]
        if kind == "protocol_error":
            self.events.append(sid, **event)
            return
        if not live.ready:
            self._mark_running(live)
        if kind == "artifact":
            name, path = event.get("name"), event.get("path") or event.get("name")
            if not isinstance(name, str) or not isinstance(path, str):
                self.events.append(sid, "protocol_error", reason="artifact needs name and path")
                return
            resolved = (live.artifacts_dir / path).resolve()
            if not resolved.is_relative_to(live.artifacts_dir.resolve()):
                self.events.append(
                    sid, "protocol_error", reason="artifact path escapes artifacts dir", text=path
                )
                return
            event = {**event, "path": str(resolved), "exists": resolved.is_file()}
        elif kind == "final":
            live.final_seen = True
            live.final_summary = event.get("summary")
            live.tasks.append(asyncio.create_task(self._reap_after_final(live)))
        self.events.append(sid, **event)

    async def _read_stdout(self, live: LiveSession) -> None:
        async for line in iter_lines(live.handle.stdout, self.settings.max_line_bytes):
            self._touch(live.session_id)
            event = parse_stdout_line(line)
            if event["type"] == "log":
                self._record_log(live, "stdout", line)
            else:
                self._record_worker_event(live, event)

    async def _read_stderr(self, live: LiveSession) -> None:
        async for line in iter_lines(live.handle.stderr, self.settings.max_line_bytes):
            self._touch(live.session_id)
            self._record_log(live, "stderr", line)

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

    async def _run(self, live: LiveSession) -> None:
        sid = live.session_id
        watchdog = asyncio.create_task(self._ready_watchdog(live))
        try:
            await asyncio.gather(self._read_stdout(live), self._read_stderr(live))
            exit_code = await live.handle.wait()
        except Exception as e:  # noqa: BLE001 - any supervisor bug must still end the session
            log.exception("session %s supervisor error", sid)
            await self._terminate(live)
            exit_code = await live.handle.wait()
            live.stop_status, live.stop_reason = "failed", f"gateway error: {e}"
        finally:
            watchdog.cancel()

        if live.final_seen:
            status, reason = "completed", None
        elif live.stop_status:
            status, reason = live.stop_status, live.stop_reason
        else:
            status, reason = "failed", f"worker exited with code {exit_code} without a final event"
        self._set_terminal(sid, status, reason, exit_code=exit_code, summary=live.final_summary)
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
        try:
            await rooms_client.close_out_room(
                live.room.url,
                live.room.room_id,
                live.room.token,
                msg_type="handoff" if status == "completed" else "status",
                body=body,
            )
        except httpx.HTTPError as e:
            self.events.append(live.session_id, "room_error", reason=str(e))

    # ----- caller actions ------------------------------------------------------------

    async def send_message(self, session_id: str, message: str, sender: str) -> bool:
        live = self._live.get(session_id)
        if live is None or live.final_seen or live.stop_status:
            return False
        async with live.stdin_lock:
            try:
                await live.handle.send({"type": "message", "sender": sender, "message": message})
            except (BrokenPipeError, ConnectionResetError):
                return False
        self.events.append(session_id, "message", sender=sender, message=message)
        self._touch(session_id)
        return True

    async def _terminate(self, live: LiveSession) -> None:
        """SIGTERM the process group, then SIGKILL after the grace period."""
        live.handle.terminate()
        try:
            await asyncio.wait_for(live.handle.wait(), self.settings.stop_grace_seconds)
        except TimeoutError:
            live.handle.kill()

    async def stop(self, session_id: str, reason: str, status: str = "stopped") -> None:
        """Ask the worker to stop, then SIGTERM, then SIGKILL. Returns once it has exited."""
        live = self._live.get(session_id)
        if live is None:
            return
        if live.stop_status is None:
            live.stop_status, live.stop_reason = status, reason
        async with live.stdin_lock:
            try:
                await live.handle.send({"type": "stop", "reason": reason})
            except (BrokenPipeError, ConnectionResetError):
                pass
            live.handle.close_stdin()
        try:
            await asyncio.wait_for(live.finished.wait(), self.settings.stop_grace_seconds)
        except TimeoutError:
            await self._terminate(live)
        await live.finished.wait()

    # ----- housekeeping --------------------------------------------------------------

    async def cleanup_once(self) -> None:
        now = datetime.now(UTC)
        stops = []
        for sid in list(self._live):
            row = self.get_row(sid)
            if datetime.fromisoformat(row["expires_at"]) <= now:
                stops.append(self.stop(sid, "hard_timeout", status="expired"))
                continue
            idle = (now - datetime.fromisoformat(row["last_activity_at"])).total_seconds()
            if idle >= row["idle_timeout_seconds"]:
                stops.append(self.stop(sid, "idle_timeout", status="expired"))
        await asyncio.gather(*stops)

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

    async def shutdown(self) -> None:
        await asyncio.gather(*(self.stop(sid, "gateway_shutdown") for sid in list(self._live)))
