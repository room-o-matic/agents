import os
import time
from datetime import UTC, datetime, timedelta

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def ulid() -> str:
    """26-char Crockford base32 ULID: 48-bit ms timestamp + 80 random bits."""
    n = (int(time.time() * 1000) << 80) | int.from_bytes(os.urandom(10))
    return "".join(_CROCKFORD[(n >> (5 * i)) & 31] for i in reversed(range(26)))


def new_id(prefix: str) -> str:
    return f"{prefix}_{ulid()}"


def now_iso() -> str:
    return _iso(datetime.now(UTC))


def iso_in(seconds: int) -> str:
    return _iso(datetime.now(UTC) + timedelta(seconds=seconds))


def _iso(t: datetime) -> str:
    # Fixed-width UTC timestamps compare correctly as strings in SQL.
    return t.isoformat(timespec="milliseconds").replace("+00:00", "Z")
