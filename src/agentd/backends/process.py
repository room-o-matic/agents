"""Process backend: each worker is a local subprocess in its own process group.

This gives lifecycle control but no isolation: the worker runs as the agentd user and
profile filesystem/network limits are advisory. The Docker backend (milestone 4) is where
those get enforced.
"""

import asyncio
import os
import signal
from collections.abc import AsyncIterator
from pathlib import Path

from agentd.protocol import encode


class ProcessHandle:
    def __init__(self, proc: asyncio.subprocess.Process):
        self._proc = proc

    @property
    def pid(self) -> int:
        return self._proc.pid

    @property
    def stdout(self) -> asyncio.StreamReader:
        return self._proc.stdout

    @property
    def stderr(self) -> asyncio.StreamReader:
        return self._proc.stderr

    async def send(self, obj: dict) -> None:
        stdin = self._proc.stdin
        if stdin is None or stdin.is_closing():
            raise BrokenPipeError("worker stdin is closed")
        stdin.write(encode(obj))
        await stdin.drain()

    def close_stdin(self) -> None:
        if self._proc.stdin and not self._proc.stdin.is_closing():
            self._proc.stdin.close()

    async def wait(self) -> int:
        return await self._proc.wait()

    def terminate(self) -> None:
        self._signal(signal.SIGTERM)

    def kill(self) -> None:
        self._signal(signal.SIGKILL)

    def _signal(self, sig: int) -> None:
        try:
            os.killpg(self._proc.pid, sig)
        except ProcessLookupError:
            pass


class ProcessBackend:
    async def start(self, command: list[str], env: dict[str, str], cwd: Path) -> ProcessHandle:
        proc = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=cwd,
            start_new_session=True,
        )
        return ProcessHandle(proc)


async def iter_lines(reader: asyncio.StreamReader, max_bytes: int) -> AsyncIterator[str]:
    """Yield decoded lines, truncating any line longer than max_bytes instead of failing."""
    buf = b""
    discarding = False
    while chunk := await reader.read(64 * 1024):
        buf += chunk
        while (nl := buf.find(b"\n")) != -1:
            line, buf = buf[:nl], buf[nl + 1 :]
            if discarding:
                discarding = False
                continue
            yield _truncate(line, max_bytes)
        if len(buf) > max_bytes:
            if not discarding:
                yield _truncate(buf, max_bytes)
            discarding = True
            buf = b""
    if buf and not discarding:
        yield _truncate(buf, max_bytes)


def _truncate(line: bytes, max_bytes: int) -> str:
    if len(line) <= max_bytes:
        return line.decode(errors="replace")
    return line[:max_bytes].decode(errors="replace") + " …[truncated]"


def find_orphan(pid: int, session_id: str) -> bool:
    """True if `pid` is still the worker for `session_id` (guards against PID reuse)."""
    try:
        environ = Path(f"/proc/{pid}/environ").read_bytes()
    except OSError:
        return False
    return f"AGENTD_SESSION_ID={session_id}".encode() in environ.split(b"\0")
