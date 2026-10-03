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


class WorkerNotReading(Exception):
    """The worker isn't consuming stdin and its pending input is at the limit."""


class ProcessHandle:
    def __init__(
        self,
        proc: asyncio.subprocess.Process,
        *,
        max_pending_bytes: int = 1024 * 1024,
        send_timeout: float = 5.0,
    ):
        self._proc = proc
        self._max_pending = max_pending_bytes
        self._send_timeout = send_timeout

    @property
    def pid(self) -> int:
        return self._proc.pid

    @property
    def stdout(self) -> asyncio.StreamReader:
        return self._proc.stdout

    @property
    def stderr(self) -> asyncio.StreamReader:
        return self._proc.stderr

    @property
    def pgid(self) -> int:
        # start_new_session=True makes the worker the leader of its own process group.
        return self._proc.pid

    async def send(self, obj: dict) -> None:
        """Queue a message for the worker without ever blocking for long.

        Pending input is capped at max_pending_bytes (WorkerNotReading beyond that), and
        drain() waits at most send_timeout: a message still buffered after that stays
        queued and is delivered if the worker resumes reading.
        """
        stdin = self._proc.stdin
        if stdin is None or stdin.is_closing():
            raise BrokenPipeError("worker stdin is closed")
        data = encode(obj)
        if stdin.transport.get_write_buffer_size() + len(data) > self._max_pending:
            raise WorkerNotReading(f"worker has over {self._max_pending} bytes of unread input")
        stdin.write(data)
        try:
            await asyncio.wait_for(stdin.drain(), self._send_timeout)
        except TimeoutError:
            pass

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
            os.killpg(self.pgid, sig)
        except ProcessLookupError:
            pass

    def group_alive(self) -> bool:
        """Whether any process (leader or descendant) remains in the worker's group.

        A descendant that calls setsid() leaves the group and escapes this; only a
        container/cgroup backend can own those (see docs#3 and the Docker milestone).
        """
        try:
            os.killpg(self.pgid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True


class ProcessBackend:
    def __init__(self, *, max_pending_bytes: int = 1024 * 1024, send_timeout: float = 5.0):
        self.max_pending_bytes = max_pending_bytes
        self.send_timeout = send_timeout

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
        return ProcessHandle(
            proc, max_pending_bytes=self.max_pending_bytes, send_timeout=self.send_timeout
        )


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
