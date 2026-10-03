import sqlite3
from pathlib import Path

from agentd import ops

SCHEMA = """
create table if not exists sessions (
  id text primary key,
  instance_id text not null,
  requester_agent text not null,
  requester_surface text,
  requester_conversation_id text,
  parent_session_id text,
  profile text not null,
  worker_type text not null,
  status text not null,
  task text not null,
  workspace_path text,
  room_url text,
  pid integer,
  exit_code integer,
  idle_timeout_seconds real not null,
  created_at text not null,
  last_activity_at text not null,
  expires_at text not null,
  stopped_at text,
  stop_reason text,
  summary text,
  metadata_json text,
  -- docs#13: caller-scoped idempotency. Retrying a spawn with the same operation_id
  -- returns the original session instead of starting a second worker.
  operation_id text,
  payload_hash text,
  -- docs#17: room finalization, tracked separately from the session's terminal status.
  room_invite_id text,
  room_finalization text,
  room_finalization_error text,
  room_finalization_attempts integer not null default 0
);

create index if not exists sessions_status on sessions(status);
create index if not exists sessions_requester on sessions(requester_agent, created_at);
create unique index if not exists sessions_operation
  on sessions(requester_agent, operation_id) where operation_id is not null;

create table if not exists events (
  id integer primary key autoincrement,
  session_id text not null references sessions(id),
  type text not null,
  time text not null,
  payload_json text not null
);

create index if not exists events_session_id_id on events(session_id, id);

"""


def connect(path: Path) -> sqlite3.Connection:
    # One connection, used only from the event loop thread (all routes are async).
    conn = sqlite3.connect(path, timeout=10, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("pragma foreign_keys = on")
    conn.execute("pragma busy_timeout = 10000")
    # WAL + NORMAL: commits don't fsync (only checkpoints do). The database stays
    # consistent and an agentd crash loses nothing; an OS crash or power loss can drop the
    # last commits, which recover() treats like any restart. FULL would put one disk sync
    # per worker event on the event loop, so a slow disk would stall every session.
    conn.execute("pragma synchronous = NORMAL")
    return conn


def connect_loop(path: Path) -> sqlite3.Connection:
    """The event loop's connection: never checkpoints inline. Supervisor.checkpoint() does
    it from a worker thread instead, so WAL syncs never block the loop."""
    conn = connect(path)
    conn.execute("pragma wal_autocheckpoint = 0")
    return conn


def checkpoint(path: Path) -> tuple[int, int, int]:
    """Copy the WAL back into the database (PASSIVE: never blocks writers). Runs off the
    event loop. Returns sqlite's (busy, wal_frames, checkpointed_frames)."""
    conn = connect(path)
    try:
        return tuple(conn.execute("pragma wal_checkpoint(PASSIVE)").fetchone())
    finally:
        conn.close()


# docs#24: bump SCHEMA_VERSION with every schema change and add MIGRATIONS[old] to take a
# database from `old` to `old + 1` (in one transaction, see ops.apply_schema). Keep SCHEMA
# the full current schema for fresh databases. Version 1 is the unversioned baseline.
SCHEMA_VERSION = 1
MIGRATIONS: dict[int, ops.Migration] = {}


def init_db(path: Path, backup_dir: Path | None = None) -> dict:
    """Create or upgrade the database; raises ops.SchemaError for unsupported versions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Version check first: a refused database must be left exactly as it was.
    result = ops.apply_schema(
        path,
        service="agentd",
        schema=SCHEMA,
        version=SCHEMA_VERSION,
        migrations=MIGRATIONS,
        backup_dir=backup_dir,
    )
    conn = connect(path)
    try:
        conn.execute("pragma journal_mode = wal")
    finally:
        conn.close()
    return result
