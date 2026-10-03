import asyncio
import contextlib
import json
import logging
import sqlite3
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from agentd import db, lobby_client
from agentd.config import Settings
from agentd.events import EventStore
from agentd.models import (
    ACTIVE_STATUSES,
    InstanceInfo,
    MessageAccepted,
    MessageRequest,
    Requester,
    Session,
    SpawnRequest,
    SpawnResponse,
    StopRequest,
)
from agentd.supervisor import SpawnError, Supervisor
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
    )


def create_app(settings: Settings | None = None, verifier: TokenVerifier | None = None) -> FastAPI:
    settings = settings or Settings.load()
    verifier = verifier or TokenVerifier(
        issuer=settings.lobbyd_url,
        domain=settings.lobbyd_domain,
        audience=settings.base_url,
        jwks_url=settings.lobbyd_jwks_url,
    )

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        db.init_db(settings.db_path)
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
                        settings, supervisor.active_count, supervisor.capacity_changed
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
        )

    @app.post("/v1/sessions", status_code=status.HTTP_201_CREATED)
    async def spawn(req: SpawnRequest, request: Request, agent: Agent) -> SpawnResponse:
        assert_identity(agent, req.requester.agent)
        try:
            row = await sup(request).spawn(req, agent)
        except SpawnError as e:
            raise HTTPException(e.status_code, e.detail) from e
        return SpawnResponse(
            session_id=row["id"],
            instance_id=row["instance_id"],
            status=row["status"],
            events_url=events_url(row["id"]),
        )

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
