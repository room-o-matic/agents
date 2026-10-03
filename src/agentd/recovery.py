"""Backup and restore for agentd (room-o-matic/docs#24). See ops.py for the mechanics.

A backup holds the database and the sessions directory (events.jsonl and artifacts).
A restored snapshot is older than the gateway it replaces. Before serving, a restore:

- fails every session the snapshot shows as starting/running/stopping, and forgets its
  pid: those workers belonged to another process (or host), and a recycled pid must
  never be signalled. Nothing restarts them; callers reconcile by operation_id (docs#13)
  and decide whether to spawn again;
- hands every owed room close-out (room_finalization 'pending') to the inviter as
  'owner_required': the invite tokens were only ever in the old process's memory;
- moves event IDs past `id_gap`, so `after_id` cursors held by callers are never reused.

Caller grants and profiles live in config (callers_file), not in the backup; restore
them from configuration management, where revocations made since are already reflected.
"""

import sqlite3
from pathlib import Path

from agentd import db, ops
from agentd.config import Settings
from agentd.ids import now_iso
from agentd.models import ACTIVE_STATUSES

DEFAULT_ID_GAP = 1_000_000


def backup(settings: Settings, dest: Path) -> dict:
    return ops.backup(
        settings.db_path,
        dest,
        service="agentd",
        schema_version=db.SCHEMA_VERSION,
        extra_dirs={"sessions": settings.sessions_dir},
    )


def post_restore(conn: sqlite3.Connection, id_gap: int) -> dict:
    now = now_iso()
    marks = ",".join("?" * len(ACTIVE_STATUSES))
    summary: dict = {}
    with conn:
        summary["sessions_failed"] = conn.execute(
            f"update sessions set status = 'failed', stop_reason = 'restored_from_backup',"
            f" stopped_at = ?, pid = null where status in ({marks})",
            (now, *ACTIVE_STATUSES),
        ).rowcount
        summary["room_finalizations_to_owner"] = conn.execute(
            "update sessions set room_finalization = 'owner_required',"
            " room_finalization_error = 'restored from backup; revoke the invite by"
            " room_invite_id' where room_finalization = 'pending'"
        ).rowcount
        ops.bump_sequences(conn, ["events"], id_gap)
        summary["id_gap"] = id_gap
    return summary


def restore(settings: Settings, src: Path, *, force: bool = False, id_gap: int = DEFAULT_ID_GAP):
    report = ops.restore(
        src,
        settings.data_dir,
        service="agentd",
        max_schema_version=db.SCHEMA_VERSION,
        extra_dirs={"sessions": settings.sessions_dir},
        force=force,
    )
    report["schema"] = db.init_db(settings.db_path, backup_dir=settings.backup_dir)
    conn = db.connect(settings.db_path)
    try:
        report["invalidated"] = post_restore(conn, id_gap)
    finally:
        conn.close()
    report["report_path"] = str(ops.write_report(settings.data_dir, report))
    return report
