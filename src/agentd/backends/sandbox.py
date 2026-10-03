"""Sandbox backend: each worker runs inside bubblewrap (room-o-matic/docs#9).

What the worker gets:
- its own user, PID, IPC, UTS and cgroup namespaces, plus a network namespace with no
  interfaces unless the profile allows network (egress is all or nothing here);
- a read-only view of the system (/usr, /etc, …) and of the runtime it needs (the Python
  environment and agentd package, plus `sandbox.ro_paths`), and nothing else of the host:
  no home directories, no other sessions, no agentd data dir;
- a private tmpfs /tmp and HOME, its session's scratch and artifacts dirs read-write, and
  its workspace read-only or read-write as the profile says;
- the gateway's allowlisted environment only, with HOME and TMPDIR pointed inside;
- rlimits from `sandbox` settings (address space, file size, CPU time, open files).

Because the worker is in its own PID namespace, everything it starts dies when it exits,
including processes that called setsid() to leave its process group. bwrap missing or
failing means the launch fails: this backend never falls back to running unisolated.

Not provided: cgroup memory/PID accounting (rlimits cover per-process limits only) and
per-destination egress rules. Docker/cgroup backends can add those later.
"""

import asyncio
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import agentd
from agentd.backends.process import ProcessBackend, ProcessHandle
from agentd.config import SandboxSettings

SYSTEM_RO = ("/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc")


@dataclass(frozen=True)
class LaunchSpec:
    scratch_dir: Path
    artifacts_dir: Path
    workspace: Path | None = None
    workspace_mode: str = "none"
    network: bool = True


def runtime_paths() -> list[Path]:
    """What a worker launched with this interpreter needs to read."""
    paths = {Path(sys.prefix), Path(sys.base_prefix), Path(agentd.__file__).resolve().parent.parent}
    exe = Path(sys.executable)
    paths.add(exe.parent)
    paths.add(exe.resolve().parent.parent)
    return sorted(paths)


def bwrap_command(
    settings: SandboxSettings, spec: LaunchSpec, command: list[str], cwd: Path
) -> list[str]:
    args = [
        settings.bwrap,
        "--die-with-parent",
        "--new-session",
        "--unshare-user",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
        "--unshare-cgroup-try",
    ]
    if not spec.network:
        args.append("--unshare-net")
    for path in SYSTEM_RO:
        args += ["--ro-bind-try", path, path]
    for path in [*runtime_paths(), *settings.ro_paths]:
        args += ["--ro-bind-try", str(path), str(path)]
    args += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--dir", "/tmp/home"]
    for path in (spec.scratch_dir, spec.artifacts_dir):
        args += ["--bind", str(path), str(path)]
    if spec.workspace is not None and spec.workspace_mode in ("read", "read_write"):
        flag = "--bind" if spec.workspace_mode == "read_write" else "--ro-bind"
        args += [flag, str(spec.workspace), str(spec.workspace)]
    args += ["--chdir", str(cwd)]
    limits = [
        "prlimit",
        f"--as={settings.memory_bytes}",
        f"--fsize={settings.file_size_bytes}",
        f"--cpu={settings.cpu_seconds}",
        f"--nofile={settings.open_files}",
        "--",
    ]
    return [*args, "--", *limits, *command]


class SandboxBackend(ProcessBackend):
    def __init__(self, settings: SandboxSettings, **kw):
        super().__init__(**kw)
        self.sandbox = settings

    def check(self) -> None:
        """Fail closed at startup if isolation isn't available."""
        if shutil.which(self.sandbox.bwrap) is None:
            raise RuntimeError(f"sandbox backend: {self.sandbox.bwrap!r} not found")

    async def start(
        self,
        command: list[str],
        env: dict[str, str],
        cwd: Path,
        spec: LaunchSpec | None = None,
    ) -> ProcessHandle:
        if spec is None:
            raise RuntimeError("sandbox backend needs a launch spec")
        env = {**env, "HOME": "/tmp/home", "TMPDIR": "/tmp"}
        proc = await asyncio.create_subprocess_exec(
            *bwrap_command(self.sandbox, spec, command, cwd),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            start_new_session=True,
        )
        return ProcessHandle(
            proc, max_pending_bytes=self.max_pending_bytes, send_timeout=self.send_timeout
        )
