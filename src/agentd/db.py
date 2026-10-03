import sqlite3
from pathlib import Path

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
  payload_hash text
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
    return conn


def init_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = connect(path)
    try:
        conn.execute("pragma journal_mode = wal")
        conn.executescript(SCHEMA)
    finally:
        conn.close()
