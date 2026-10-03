import asyncio
import contextlib
import json
import logging
import shutil
import sqlite3
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response, status
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from agentd import db, lobby_client, ops
from agentd.config import Settings
from agentd.events import EventStore
from agentd.models import (
    ACTIVE_STATUSES,
    Capabilities,
    InstanceInfo,
    MessageAccepted,
    MessageRequest,
    Pair,
    Requester,
    Session,
    SpawnRequest,
    SpawnResponse,
    StopRequest,
)
from agentd.supervisor import SpawnError, SpawnReplay, Supervisor
from agentd.verify import InvalidToken, TokenVerifier

log = logging.getLogger("agentd")

SSE_KEEPALIVE_SECONDS = 15
TERMINAL_STATUSES = frozenset({"completed", "failed", "stopped", "expired"})

_bearer = HTTPBearer(auto_error=False)


def events_url(session_id: str) -> str:
    return f"/v1/sessions/{session_id}/events"


def session_from_row(row: sqlite3.Row) -> Session:
    return Session(
        session_id=row["id"],
        instance_id=row["instance_id"],
        status=row["status"],
        requester=Requester(
            agent=row["requester_agent"],
            surface=row["requester_surface"],
            conversation_id=row["requester_conversation_id"],
        ),
        parent_session_id=row["parent_session_id"],
        profile=row["profile"],
        worker_type=row["worker_type"],
        task=row["task"],
        workspace_path=row["workspace_path"],
        room_url=row["room_url"],
        created_at=row["created_at"],
        last_activity_at=row["last_activity_at"],
        expires_at=row["expires_at"],
        stopped_at=row["stopped_at"],
        stop_reason=row["stop_reason"],
        exit_code=row["exit_code"],
        summary=row["summary"],
        events_url=events_url(row["id"]),
        room_invite_id=row["room_invite_id"],
        room_finalization=row["room_finalization"],
        room_finalization_error=row["room_finalization_error"],
    )


def capabilities(settings: Settings, caller) -> Capabilities:
    pairs = [
        Pair(profile=p, worker_type=w)
        for p, prof in sorted(settings.profiles.items())
        if prof.external_actions != "approval_required"  # refused at launch (docs#9)
        for w in sorted(settings.worker_types)
        if prof.worker_types is None or w in prof.worker_types
    ]
    allowed = [
        pair
        for pair in pairs
        if caller is not None
        and caller.allows(caller.profiles, pair.profile)
        and caller.allows(caller.worker_types, pair.worker_type)
        and (settings.backend == "sandbox" or caller.trust == "trusted")
    ]
    return Capabilities(
        features=["operation_id", "sse_events", "room_invites", "task_grant", "caller_policy"],
        pairs=pairs,
        allowed_for_you=allowed,
        profile_runtime_seconds={
            p: int(prof.max_runtime_minutes * 60) for p, prof in settings.profiles.items()
        },
        limits={
            "task_bytes": 64 * 1024,
            "message_bytes": 64 * 1024,
            "max_pending_stdin_bytes": settings.max_pending_stdin_bytes,
            "max_sessions": settings.max_sessions,
        },
        cancellation=(
            f"POST /stop: stop message, then SIGTERM after {settings.stop_grace_seconds}s, then"
            " SIGKILL; ends 'stopped' (or 'completed' if the worker already sent final)"
        ),
        isolation="sandbox" if settings.backend == "sandbox" else "none",
    )


def create_app(settings: Settings | None = None, verifier: TokenVerifier | None = None) -> FastAPI:
    settings = settings or Settings.load()
    verifier = verifier or TokenVerifier(
        issuer=settings.lobbyd_url,
        domain=settings.lobbyd_domain,
        audience=settings.base_url,
        jwks_url=settings.lobbyd_jwks_url,
    )

    registry_health = ops.LoopHealth()

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # docs#24: upgrades the schema (after a pre-upgrade backup) or refuses to start.
        db.init_db(settings.db_path, backup_dir=settings.backup_dir)
        settings.sessions_dir.mkdir(parents=True, exist_ok=True)
        conn = db.connect(settings.db_path)
        supervisor = Supervisor(settings, conn, EventStore(conn, settings.sessions_dir))
        supervisor.recover()
        app.state.conn = conn
        app.state.supervisor = supervisor
        tasks = [asyncio.create_task(supervisor.cleanup_loop())]
        if settings.registry_enabled:
            tasks.append(
                asyncio.create_task(
                    lobby_client.heartbeat_loop(
                        settings,
                        supervisor.active_count,
                        supervisor.capacity_changed,
                        registry_health,
                    )
                )
            )
        try:
            yield
        finally:
            for t in tasks:
                t.cancel()
            await supervisor.shutdown()
            if settings.registry_enabled:
                await lobby_client.deregister(settings)
            conn.close()

    app = FastAPI(title="agentd", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings

    async def current_agent(
        creds: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
    ) -> str:
        """Callers present lobbyd access tokens for this instance; identity is name@domain."""
        agent = None
        if creds:
            try:
                # A thread, since a cache miss fetches the JWKS synchronously.
                claims = await asyncio.to_thread(verifier.verify, creds.credentials)
            except InvalidToken:
                pass
            else:
                agent = claims.identity if claims.scope == "agent" else None
        if agent is None:
            raise HTTPException(
                status.HTTP_401_UNAUTHORIZED,
                "missing or invalid bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return agent

    Agent = Annotated[str, Depends(current_agent)]

    def sup(request: Request) -> Supervisor:
        return request.app.state.supervisor

    def own_session(request: Request, session_id: str, agent: str) -> sqlite3.Row:
        """Sessions are visible only to their requester; others get 404, not 403."""
        row = sup(request).get_row(session_id)
        if row is None or row["requester_agent"] != agent:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "session not found")
        return row

    def assert_identity(agent: str, claimed: str | None) -> None:
        if claimed is not None and claimed != agent:
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                f"token belongs to {agent!r}; cannot act as {claimed!r}",
            )

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    def checks(request: Request) -> dict:
        """Readiness (docs#24): can this gateway actually run sessions?"""
        supervisor = sup(request)
        conn = request.app.state.conn
        version = ops.read_schema_version(conn)
        marks = ",".join("?" * len(ACTIVE_STATUSES))
        active_ids = {
            r[0]
            for r in conn.execute(
                f"select id from sessions where status in ({marks})", ACTIVE_STATUSES
            )
        }
        # Launches in progress have a row before they join _live.
        orphaned = max(0, len(active_ids - set(supervisor._live)) - supervisor._launching)
        finalization = dict(
            conn.execute(
                "select room_finalization, count(*) from sessions"
                " where room_finalization in ('pending', 'owner_required')"
                " group by room_finalization"
            ).fetchall()
        )
        db_error = ops.db_writable(settings.db_path)
        free = shutil.disk_usage(settings.data_dir).free
        jwks = verifier.health()
        age = registry_health.age()
        return {
            "database": {"ok": db_error is None, "error": db_error},
            "schema": {"ok": version == db.SCHEMA_VERSION, "version": version},
            "storage": {"ok": free >= settings.min_free_bytes, "free_bytes": free},
            "jwks": {"ok": not jwks["failing_closed"], **jwks},
            # Sessions the database says are active but no live worker backs: a gateway
            # bug or a crash in progress; not ready until recover() has dealt with them.
            "sessions": {
                "ok": orphaned == 0,
                "active": supervisor.active_count(),
                "orphaned": orphaned,
                "room_finalization_pending": finalization.get("pending", 0),
                "room_finalization_owner_required": finalization.get("owner_required", 0),
            },
            # Direct callers still work without the registry: reported, not required.
            "registry": {
                "ok": not settings.registry_enabled
                or (age is not None and age <= 3 * settings.registry_ttl_seconds),
                "required": False,
                "enabled": settings.registry_enabled,
                "age_seconds": age,
                "consecutive_failures": registry_health.failures,
                "last_error": registry_health.last_error,
            },
        }

    def is_ready(c: dict) -> bool:
        return all(v["ok"] for v in c.values() if v.get("required", True))

    @app.get("/readyz")
    async def readyz(request: Request, response: Response) -> dict:
        c = checks(request)
        response.status_code = 200 if is_ready(c) else 503
        return {"ready": is_ready(c), "checks": c}

    @app.get("/metrics")
    async def metrics(request: Request) -> Response:
        c = checks(request)
        s = c["sessions"]
        gauges = {
            "ready": is_ready(c),
            "schema_version": c["schema"]["version"],
            "db_bytes": settings.db_path.stat().st_size,
            "disk_free_bytes": c["storage"]["free_bytes"],
            "jwks_age_seconds": c["jwks"]["age_seconds"],
            "jwks_fetch_failures": c["jwks"]["consecutive_failures"],
            "jwks_failing_closed": c["jwks"]["failing_closed"],
            "registry_age_seconds": c["registry"]["age_seconds"],
            "registry_failures": c["registry"]["consecutive_failures"],
            "active_sessions": s["active"],
            "max_sessions": settings.max_sessions,
            "orphaned_sessions": s["orphaned"],
            "room_finalization_pending": s["room_finalization_pending"],
            "room_finalization_owner_required": s["room_finalization_owner_required"],
        }
        return Response(ops.prometheus("agentd", gauges), media_type="text/plain; version=0.0.4")

    @app.get("/v1/instance")
    async def instance(request: Request, agent: Agent) -> InstanceInfo:
        return InstanceInfo(
            instance_id=settings.instance_id,
            base_url=settings.base_url,
            worker_types=sorted(settings.worker_types),
            profiles=sorted(settings.profiles),
            max_sessions=settings.max_sessions,
            active_sessions=sup(request).active_count(),
            registry_enabled=settings.registry_enabled,
            capabilities=capabilities(settings, sup(request).policies.lookup(agent)),
        )

    @app.post("/v1/sessions", status_code=status.HTTP_201_CREATED)
    async def spawn(
        req: SpawnRequest, request: Request, response: Response, agent: Agent
    ) -> SpawnResponse:
        assert_identity(agent, req.requester.agent)
        replayed = False
        try:
            row = await sup(request).spawn(req, agent)
        except SpawnReplay as r:
            row, replayed = r.row, True
            response.status_code = status.HTTP_200_OK
        except SpawnError as e:
            raise HTTPException(e.status_code, e.detail) from e
        return SpawnResponse(
            session_id=row["id"],
            instance_id=row["instance_id"],
            status=row["status"],
            events_url=events_url(row["id"]),
            operation_id=row["operation_id"],
            replayed=replayed,
        )

    @app.get("/v1/sessions/by-operation/{operation_id}")
    async def session_by_operation(operation_id: str, request: Request, agent: Agent) -> Session:
        """Reconcile an ambiguous spawn: did the request with this operation_id start a
        session? 404 means this gateway has no record of it (docs#13)."""
        row = sup(request).by_operation(agent, operation_id)
        if row is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "no session for that operation_id")
        return session_from_row(row)

    @app.get("/v1/sessions")
    async def list_sessions(
        request: Request,
        agent: Agent,
        active: bool = False,
        limit: Annotated[int, Query(ge=1, le=500)] = 50,
    ) -> list[Session]:
        sql = "select * from sessions where requester_agent = ?"
        params: list = [agent]
        if active:
            sql += f" and status in ({','.join('?' * len(ACTIVE_STATUSES))})"
            params += ACTIVE_STATUSES
        rows = request.app.state.conn.execute(
            sql + " order by created_at desc limit ?", [*params, limit]
        ).fetchall()
        return [session_from_row(r) for r in rows]

    @app.get("/v1/sessions/{session_id}")
    async def get_session(session_id: str, request: Request, agent: Agent) -> Session:
        return session_from_row(own_session(request, session_id, agent))

    @app.post("/v1/sessions/{session_id}/messages")
    async def send_message(
        session_id: str, req: MessageRequest, request: Request, agent: Agent
    ) -> MessageAccepted:
        assert_identity(agent, req.sender)
        own_session(request, session_id, agent)
        if refused := await sup(request).send_message(session_id, req.message, agent):
            row = sup(request).get_row(session_id)
            raise HTTPException(status.HTTP_409_CONFLICT, f"{refused} ({row['status']})")
        return MessageAccepted(accepted=True, status=sup(request).get_row(session_id)["status"])

    @app.post("/v1/sessions/{session_id}/stop")
    async def stop(
        session_id: str, request: Request, agent: Agent, req: StopRequest | None = None
    ) -> Session:
        own_session(request, session_id, agent)
        reason = req.reason if req else "caller_cancelled"
        await sup(request).stop(session_id, reason)
        return session_from_row(sup(request).get_row(session_id))

    @app.get("/v1/sessions/{session_id}/events")
    async def events(
        session_id: str,
        request: Request,
        agent: Agent,
        after_id: Annotated[int, Query(ge=0)] = 0,
        stream: bool = True,
        last_event_id: Annotated[int | None, Header()] = None,
    ):
        """Server-Sent Events: replays history after `after_id` (or Last-Event-ID), then
        follows live, and closes once the session is terminal. `stream=false` returns the
        backlog as a JSON list instead, for polling."""
        own_session(request, session_id, agent)
        store = sup(request).events
        start = last_event_id if last_event_id is not None else after_id
        if not stream:
            return store.since(session_id, start)

        async def gen() -> AsyncIterator[str]:
            cursor = start
            while True:
                if await request.is_disconnected():
                    return
                batch = store.since(session_id, cursor)
                for ev in batch:
                    cursor = ev["id"]
                    yield f"id: {ev['id']}\nevent: {ev['type']}\ndata: {json.dumps(ev)}\n\n"
                if batch:
                    continue
                # No awaits between the empty since() above and store.wait(), so an
                # append can't be missed.
                if sup(request).get_row(session_id)["status"] in TERMINAL_STATUSES:
                    return
                if not await store.wait(session_id, SSE_KEEPALIVE_SECONDS):
                    yield ": keepalive\n\n"

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app
