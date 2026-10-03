"""Caller authorization for hosted work (room-o-matic/docs#9).

Authenticating with lobbyd proves who a caller is. Whether they may run work on this
gateway is the operator's decision, recorded here: which profiles, worker types and
workspaces, how many concurrent sessions and how much spend, and whether they're trusted
enough to run without isolation. Default deny.
"""

import threading
from pathlib import Path

import yaml

from agentd.config import CallerPolicy, Settings


class PolicyError(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class CallerPolicies:
    """settings.callers, overlaid by callers_file (re-read whenever its mtime changes, so
    removing a caller from the file revokes them without a restart)."""

    def __init__(self, settings: Settings):
        self._static = dict(settings.callers)
        self._file = settings.callers_file
        self._mtime: float | None = None
        self._from_file: dict[str, CallerPolicy] = {}
        self._lock = threading.Lock()

    def _refresh(self) -> None:
        if self._file is None:
            return
        try:
            mtime = Path(self._file).stat().st_mtime
        except FileNotFoundError:
            mtime = None
        with self._lock:
            if mtime == self._mtime:
                return
            if mtime is None:
                self._from_file = {}  # file removed: everything it granted is revoked
            else:
                raw = yaml.safe_load(Path(self._file).read_text()) or {}
                self._from_file = {k: CallerPolicy(**(v or {})) for k, v in raw.items()}
            self._mtime = mtime

    def lookup(self, principal: str) -> CallerPolicy | None:
        self._refresh()
        merged = {**self._static, **self._from_file}
        return merged.get(principal) or merged.get("*")


def authorize(
    policy: CallerPolicy | None,
    *,
    principal: str,
    profile_name: str,
    worker_type: str,
    backend: str,
) -> CallerPolicy:
    if policy is None:
        raise PolicyError(
            403, f"{principal} is not authorized to run work on this gateway (no caller policy)"
        )
    if not policy.allows(policy.profiles, profile_name):
        raise PolicyError(403, f"{principal} may not use profile {profile_name!r}")
    if not policy.allows(policy.worker_types, worker_type):
        raise PolicyError(403, f"{principal} may not use worker type {worker_type!r}")
    if backend == "process" and policy.trust != "trusted":
        raise PolicyError(
            403,
            "this gateway runs workers without isolation (process backend) and only accepts "
            "trusted callers; untrusted work needs backend: sandbox",
        )
    return policy


def caller_workspace_ok(policy: CallerPolicy, workspace: Path) -> bool:
    if policy.workspace_roots is None:
        return True
    return any(workspace.is_relative_to(r.resolve()) for r in policy.workspace_roots)
