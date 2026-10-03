import os
import re
import socket
import sys
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

NAME_RE = r"^[a-z0-9][a-z0-9_.-]{0,63}$"


class Profile(BaseModel):
    """A server-side permission allowlist. Callers choose one by name and never pass raw
    permissions. Unknown keys (e.g. discord_send) are kept and handed to the worker."""

    model_config = ConfigDict(extra="allow", frozen=True)

    max_runtime_minutes: float = Field(gt=0)
    workspace_mount: Literal["none", "read", "read_write"] = "none"
    network: bool = True
    external_actions: bool | Literal["approval_required"] = False
    worker_types: list[str] | None = Field(
        default=None, description="worker types allowed with this profile; null means any"
    )


class WorkerType(BaseModel):
    """How to launch a worker. The command speaks the AGENT_EVENT protocol on stdout."""

    model_config = ConfigDict(frozen=True)

    command: list[str] = Field(min_length=1)
    env: dict[str, str] = Field(default_factory=dict)


DEFAULT_PROFILES = {
    "read_only_research": Profile(
        max_runtime_minutes=30, workspace_mount="none", filesystem="read"
    ),
    "workspace_coder": Profile(
        max_runtime_minutes=120, workspace_mount="read_write", filesystem="workspace"
    ),
    "discord_helper": Profile(
        max_runtime_minutes=45, filesystem="scratch", discord_send=False, discord_read=True
    ),
    "dangerous_needs_approval": Profile(
        max_runtime_minutes=60,
        workspace_mount="read_write",
        filesystem="workspace",
        external_actions="approval_required",
    ),
}

DEFAULT_WORKER_TYPES = {
    "fake": WorkerType(command=[sys.executable, "-m", "agentd.workers.fake"]),
}


def default_instance_id() -> str:
    host = re.sub(r"[^a-z0-9_.-]", "-", socket.gethostname().lower()).strip("-.")
    return f"agentd-{host}"[:64]


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True)

    instance_id: str = Field(default_factory=default_instance_id, pattern=NAME_RE)
    data_dir: Path = Path("/var/lib/agentd")
    base_url: str | None = Field(
        default=None, description="URL other agents use to reach this instance"
    )

    max_sessions: int = Field(default=4, ge=1, description="concurrent active sessions")
    workspace_roots: list[Path] = Field(
        default_factory=list, description="workspace paths must resolve under one of these"
    )
    env_allowlist: list[str] = Field(
        default_factory=lambda: ["PATH", "HOME", "LANG", "LC_ALL", "TZ"],
        description="host env vars a worker may inherit; everything else is dropped",
    )

    default_idle_timeout_seconds: float = 600
    ready_timeout_seconds: float = 60
    stop_grace_seconds: float = 10
    cleanup_interval_seconds: float = 30
    max_line_bytes: int = 64 * 1024
    max_log_bytes: int = 8 * 1024 * 1024

    profiles: dict[str, Profile] = Field(default_factory=lambda: dict(DEFAULT_PROFILES))
    worker_types: dict[str, WorkerType] = Field(default_factory=lambda: dict(DEFAULT_WORKER_TYPES))

    roomsd_url: str | None = None
    roomsd_token: str | None = Field(default=None, repr=False)
    registry_ttl_seconds: int = Field(default=60, ge=5, le=600)

    @property
    def db_path(self) -> Path:
        return self.data_dir / "agentd.sqlite"

    @property
    def sessions_dir(self) -> Path:
        return self.data_dir / "sessions"

    @property
    def registry_enabled(self) -> bool:
        return bool(self.roomsd_url and self.roomsd_token and self.base_url)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Settings":
        """YAML config (path or $AGENTD_CONFIG), then AGENTD_* env overrides."""
        path = path or os.environ.get("AGENTD_CONFIG")
        data = (yaml.safe_load(Path(path).read_text()) or {}) if path else {}
        for key in ("instance_id", "data_dir", "base_url", "roomsd_url", "roomsd_token"):
            if value := os.environ.get(f"AGENTD_{key.upper()}"):
                data[key] = value
        return cls(**data)
