import hashlib
import re
import secrets
import sqlite3

from agentd.ids import now_iso

TOKEN_PREFIX = "agtd_"
AGENT_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def create_token(conn: sqlite3.Connection, agent: str) -> str:
    """Issue a bearer token for a caller (Boostie, Missy, …). Only the hash is stored."""
    if not AGENT_NAME_RE.match(agent):
        raise ValueError(f"invalid agent name {agent!r}: use lowercase [a-z0-9_.-], max 64")
    token = TOKEN_PREFIX + secrets.token_urlsafe(32)
    with conn:
        conn.execute(
            "insert into tokens (token_hash, agent, created_at) values (?, ?, ?)",
            (hash_token(token), agent, now_iso()),
        )
    return token


def revoke_tokens(conn: sqlite3.Connection, agent: str) -> int:
    with conn:
        cur = conn.execute(
            "update tokens set revoked_at = ? where agent = ? and revoked_at is null",
            (now_iso(), agent),
        )
    return cur.rowcount


def agent_for_token(conn: sqlite3.Connection, token: str) -> str | None:
    row = conn.execute(
        "select agent from tokens where token_hash = ? and revoked_at is null",
        (hash_token(token),),
    ).fetchone()
    return row["agent"] if row else None
