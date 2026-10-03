from typing import Literal

from pydantic import BaseModel, Field, JsonValue

SessionStatus = Literal["starting", "running", "completed", "failed", "stopped", "expired"]
ACTIVE_STATUSES = ("starting", "running")


class Requester(BaseModel):
    agent: str | None = Field(default=None, description="must match the bearer token")
    surface: str | None = Field(default=None, max_length=64)
    conversation_id: str | None = Field(default=None, max_length=256)


class WorkspaceRequest(BaseModel):
    mode: Literal["mount"] = "mount"
    path: str


class RoomRef(BaseModel):
    """A roomsd room the worker should join, with the invite token minted for it."""

    url: str | None = Field(default=None, description="roomsd base URL; defaults to config")
    room_id: str = Field(max_length=64)
    token: str = Field(repr=False)


class SpawnRequest(BaseModel):
    task: str = Field(min_length=1, max_length=64 * 1024)
    requester: Requester = Field(default_factory=Requester)
    profile: str
    worker_type: str
    workspace: WorkspaceRequest | None = None
    timeout_seconds: float | None = Field(default=None, gt=0)
    idle_timeout_seconds: float | None = Field(default=None, gt=0)
    parent_session_id: str | None = None
    room: RoomRef | None = None
    metadata: dict[str, JsonValue] | None = None


class SpawnResponse(BaseModel):
    session_id: str
    instance_id: str
    status: SessionStatus
    events_url: str


class Session(BaseModel):
    session_id: str
    instance_id: str
    status: SessionStatus
    requester: Requester
    parent_session_id: str | None
    profile: str
    worker_type: str
    task: str
    workspace_path: str | None
    room_id: str | None
    created_at: str
    last_activity_at: str
    expires_at: str
    stopped_at: str | None
    stop_reason: str | None
    exit_code: int | None
    summary: str | None
    events_url: str


class MessageRequest(BaseModel):
    message: str = Field(min_length=1, max_length=64 * 1024)
    sender: str | None = Field(default=None, description="must match the bearer token")


class MessageAccepted(BaseModel):
    accepted: bool
    status: SessionStatus


class StopRequest(BaseModel):
    reason: str = Field(default="caller_cancelled", max_length=200)


class InstanceInfo(BaseModel):
    instance_id: str
    base_url: str | None
    worker_types: list[str]
    profiles: list[str]
    max_sessions: int
    active_sessions: int
    registry_enabled: bool
