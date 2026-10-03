"""Operations support shared by lobbyd, roomsd and agentd (room-o-matic/docs#24).

Kept identical in all three repos (like verify.py); change it in one, copy it to the
others. Standard library only.

Schema versions
    The schema version lives in SQLite's `user_version`. A fresh database gets the current
    schema and version. Version 1 is the baseline: the unversioned schema every service
    shipped before versioning, adopted in place when its tables and columns match. Upgrades
    run `migrations[v]` (v -> v+1) in one transaction after an automatic pre-upgrade backup;
    a failed step rolls back and the database stays at its old version. A database newer
    than the running code, an unversioned database that doesn't match the baseline, or a
    gap in the migration chain is refused at startup, before any request is served. There
    is no downgrade: roll back by restoring the pre-upgrade backup.

Backups
    `backup` takes a consistent snapshot with SQLite's online backup API (safe under WAL and
    concurrent writers), copies any extra directories, and writes manifest.json with a
    SHA-256 per file. Backup directories are 0700 and files 0600; they hold credentials
    metadata (and, for lobbyd, private signing keys), so encrypt them before they leave the
    host. `verify_backup` re-checks every hash and runs `pragma integrity_check`. `restore`
    verifies first, moves anything it would overwrite aside instead of deleting it, and
    reports how long it took.

Revocation journal
    Revocations (and other access removals) are also appended to a small JSON-lines journal
    outside the database. A snapshot predates the revocations made after it; after a restore
    each service replays the journal so an old snapshot can't revive revoked access. Keep
    the journal on a different volume (or ship it off-host) to survive losing the data dir.
"""

import hashlib
import json
import os
import shutil
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

BACKUP_FORMAT = 1
Migration = Callable[[sqlite3.Connection], None]


class SchemaError(RuntimeError):
    pass


class BackupError(RuntimeError):
    pass


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")


# ----- schema versions -----------------------------------------------------------------


def _columns(conn: sqlite3.Connection) -> dict[str, set[str]]:
    tables = [
        r[0]
        for r in conn.execute(
            "select name from sqlite_master where type = 'table' and name not like 'sqlite_%'"
        )
    ]
    return {t: {r[1] for r in conn.execute(f'pragma table_info("{t}")')} for t in tables}


def baseline_gaps(conn: sqlite3.Connection, schema: str) -> list[str]:
    """Columns the baseline schema has that this (unversioned) database lacks. Tables
    that are missing entirely are fine: the schema script creates them."""
    ref = sqlite3.connect(":memory:")
    try:
        ref.executescript(schema)
        want = _columns(ref)
    finally:
        ref.close()
    have = _columns(conn)
    return [f"{t}.{c}" for t, cols in want.items() if t in have for c in sorted(cols - have[t])]


def apply_schema(
    path: Path,
    *,
    service: str,
    schema: str,
    version: int,
    migrations: dict[int, Migration],
    baseline_schema: str | None = None,
    backup_dir: Path | None = None,
) -> dict:
    """Create or upgrade the database at `path` to `version`, or raise SchemaError.

    `schema` is the current schema (idempotent DDL). `baseline_schema` is the version-1
    schema that unversioned databases are checked against (defaults to `schema`, which is
    right while version is 1). Returns {"from": old, "to": version, "backup": dir|None}.
    """
    conn = sqlite3.connect(path, isolation_level=None)
    try:
        current = conn.execute("pragma user_version").fetchone()[0]
        has_tables = bool(_columns(conn))
        result = {"from": current, "to": version, "backup": None}
        if not has_tables:
            conn.executescript(schema)
            conn.execute(f"pragma user_version = {version}")
            result["from"] = None
            return result
        if current == 0:
            gaps = baseline_gaps(conn, baseline_schema or schema)
            if gaps:
                raise SchemaError(
                    f"{path} is an unversioned {service} database that predates the supported"
                    f" baseline (missing {', '.join(gaps[:8])}). Restore a backup made by a"
                    " supported release instead; this database is left untouched."
                )
            conn.execute("pragma user_version = 1")
            current = 1
        if current > version:
            raise SchemaError(
                f"{path} has {service} schema v{current}, newer than this release (v{version})."
                " Run a release that supports it, or restore the backup taken before the"
                " upgrade; downgrades are not supported."
            )
        if current < version:
            missing = [v for v in range(current, version) if v not in migrations]
            if missing:
                raise SchemaError(
                    f"no {service} migration from schema v{missing[0]}; upgrade through an"
                    " intermediate release"
                )
            if backup_dir is not None:
                conn.close()
                dest = backup_dir / f"pre-upgrade-v{current}-to-v{version}-{_stamp()}"
                backup(path, dest, service=service, schema_version=current)
                result["backup"] = str(dest)
                conn = sqlite3.connect(path, isolation_level=None)
            conn.execute("begin immediate")
            try:
                for v in range(current, version):
                    migrations[v](conn)
                conn.execute(f"pragma user_version = {version}")
                conn.execute("commit")
            except Exception as e:
                conn.execute("rollback")
                raise SchemaError(
                    f"{service} migration to v{version} failed and was rolled back"
                    f" (still v{current}): {e}"
                ) from e
        conn.executescript(schema)  # idempotent: indexes and tables of the current schema
        return result
    finally:
        conn.close()


def read_schema_version(conn: sqlite3.Connection) -> int:
    return conn.execute("pragma user_version").fetchone()[0]


# ----- backups -------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _lock_down(root: Path) -> None:
    os.chmod(root, 0o700)
    for dirpath, dirnames, filenames in os.walk(root):
        for d in dirnames:
            os.chmod(Path(dirpath, d), 0o700)
        for f in filenames:
            os.chmod(Path(dirpath, f), 0o600)


def _integrity(db: Path) -> str:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return conn.execute("pragma integrity_check").fetchone()[0]
    finally:
        conn.close()


def backup(
    db_path: Path,
    dest: Path,
    *,
    service: str,
    schema_version: int | None = None,
    extra_dirs: dict[str, Path] | None = None,
) -> dict:
    """Snapshot `db_path` (and `extra_dirs`, by name) into the new directory `dest`."""
    if dest.exists() and any(dest.iterdir()):
        raise BackupError(f"{dest} exists and is not empty")
    old_umask = os.umask(0o077)
    try:
        dest.mkdir(parents=True, exist_ok=True, mode=0o700)
        started = time.monotonic()
        started_at = now()  # changes journaled from here on may be missing from the snapshot
        target = dest / db_path.name
        src = sqlite3.connect(db_path)
        dst = sqlite3.connect(target)
        try:
            src.backup(dst)
            version = schema_version if schema_version is not None else read_schema_version(dst)
        finally:
            dst.close()
            src.close()
        for name, directory in (extra_dirs or {}).items():
            if directory.exists():
                shutil.copytree(directory, dest / name, symlinks=True)
    finally:
        os.umask(old_umask)
    integrity = _integrity(target)
    if integrity != "ok":
        raise BackupError(f"snapshot failed integrity_check: {integrity}")
    _lock_down(dest)
    files = {
        str(p.relative_to(dest)): _sha256(p)
        for p in sorted(dest.rglob("*"))
        if p.is_file() and p.name != "manifest.json"
    }
    manifest = {
        "format": BACKUP_FORMAT,
        "service": service,
        "database": db_path.name,
        "schema_version": version,
        "snapshot_started_at": started_at,
        "created_at": now(),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "integrity": integrity,
        "files": files,
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    os.chmod(dest / "manifest.json", 0o600)
    return manifest


def verify_backup(src: Path) -> dict:
    """Check a backup's manifest, every file hash, and the database's integrity."""
    try:
        manifest = json.loads((src / "manifest.json").read_text())
    except (OSError, ValueError) as e:
        raise BackupError(f"{src}: unreadable manifest: {e}") from e
    if manifest.get("format") != BACKUP_FORMAT:
        raise BackupError(f"{src}: unsupported backup format {manifest.get('format')!r}")
    for rel, digest in manifest["files"].items():
        p = src / rel
        if not p.is_file():
            raise BackupError(f"{src}: missing {rel}")
        if _sha256(p) != digest:
            raise BackupError(f"{src}: {rel} does not match its checksum")
    extra = {
        str(p.relative_to(src)) for p in src.rglob("*") if p.is_file() and p.name != "manifest.json"
    } - set(manifest["files"])
    if extra:
        raise BackupError(f"{src}: files not in the manifest: {sorted(extra)[:5]}")
    integrity = _integrity(src / manifest["database"])
    if integrity != "ok":
        raise BackupError(f"{src}: integrity_check failed: {integrity}")
    return manifest


def restore(
    src: Path,
    data_dir: Path,
    *,
    service: str,
    max_schema_version: int,
    extra_dirs: dict[str, Path] | None = None,
    force: bool = False,
) -> dict:
    """Verify the backup at `src`, then put it in place under `data_dir`.

    Anything already there (database, WAL files, extra dirs) is moved aside to
    data_dir/pre-restore-<stamp>/, never deleted; without `force` an existing database is
    refused. The caller then opens the database (which applies migrations) and runs its
    service's post-restore rules before serving.
    """
    started = time.monotonic()
    manifest = verify_backup(src)
    if manifest["service"] != service:
        raise BackupError(f"{src} is a {manifest['service']} backup, not {service}")
    if manifest["schema_version"] > max_schema_version:
        raise BackupError(
            f"{src} has schema v{manifest['schema_version']}; this release reads up to"
            f" v{max_schema_version}"
        )
    db_path = data_dir / manifest["database"]
    targets = [db_path, Path(f"{db_path}-wal"), Path(f"{db_path}-shm")]
    targets += list((extra_dirs or {}).values())
    present = [p for p in targets if p.exists()]
    if db_path.exists() and not force:
        raise BackupError(f"{db_path} exists; pass --force to move it aside and restore")
    moved_aside = None
    if present:
        moved_aside = data_dir / f"pre-restore-{_stamp()}"
        moved_aside.mkdir(parents=True, mode=0o700)
        for p in present:
            shutil.move(str(p), moved_aside / p.name)
    old_umask = os.umask(0o077)
    try:
        data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        shutil.copy2(src / manifest["database"], db_path)
        for name, directory in (extra_dirs or {}).items():
            if (src / name).is_dir():
                shutil.copytree(src / name, directory, symlinks=True)
    finally:
        os.umask(old_umask)
    return {
        "service": service,
        "restored_from": str(src),
        "snapshot_started_at": manifest["snapshot_started_at"],
        "snapshot_created_at": manifest["created_at"],
        "snapshot_schema_version": manifest["schema_version"],
        "restored_at": now(),
        "integrity": _integrity(db_path),
        "files": len(manifest["files"]),
        "moved_aside": str(moved_aside) if moved_aside else None,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def write_report(data_dir: Path, report: dict) -> Path:
    """Keep evidence of a restore (or drill): timing, integrity, what was invalidated."""
    out = data_dir / "restore-reports" / f"{_stamp()}.json"
    out.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    out.write_text(json.dumps(report, indent=2) + "\n")
    os.chmod(out, 0o600)
    return out


def bump_sequences(conn: sqlite3.Connection, tables: list[str], gap: int) -> None:
    """After a restore, move AUTOINCREMENT counters past anything issued after the
    snapshot, so IDs and cursors clients already hold are never reused."""
    for t in tables:
        cur = conn.execute("update sqlite_sequence set seq = seq + ? where name = ?", (gap, t))
        if cur.rowcount == 0:
            conn.execute("insert into sqlite_sequence (name, seq) values (?, ?)", (t, gap))


# ----- revocation journal --------------------------------------------------------------


class Journal:
    """Append-only JSON lines, fsynced, outside the database (see module docstring)."""

    def __init__(self, path: Path | None):
        self.path = path

    def append(self, kind: str, **fields) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        line = json.dumps({"kind": kind, "at": now(), **fields}) + "\n"
        fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            os.write(fd, line.encode())
            os.fsync(fd)
        finally:
            os.close(fd)

    def entries(self, since: str | None = None) -> list[dict]:
        if self.path is None or not self.path.exists():
            return []
        out = []
        for line in self.path.read_text().splitlines():
            try:
                e = json.loads(line)
            except ValueError:
                continue  # a torn final line from a crash
            if since is None or e.get("at", "") >= since:
                out.append(e)
        return out


# ----- health --------------------------------------------------------------------------


@dataclass
class LoopHealth:
    """Outcome of a background loop (registry heartbeat, listing sync, ...)."""

    last_ok: float | None = None
    last_error: str | None = None
    failures: int = 0

    def ok(self) -> None:
        self.last_ok, self.last_error, self.failures = time.time(), None, 0

    def failed(self, error: object) -> None:
        self.last_error, self.failures = str(error), self.failures + 1

    def age(self) -> float | None:
        return None if self.last_ok is None else time.time() - self.last_ok


def db_writable(path: Path) -> str | None:
    """None if the database takes a write lock, else the error."""
    try:
        conn = sqlite3.connect(path, timeout=2, isolation_level=None)
        try:
            conn.execute("begin immediate")
            conn.execute("rollback")
        finally:
            conn.close()
    except sqlite3.Error as e:
        return str(e)
    return None


def prometheus(prefix: str, gauges: dict[str, float | int | bool | None]) -> str:
    lines = []
    for name, value in gauges.items():
        if value is None:
            continue
        lines += [f"# TYPE {prefix}_{name} gauge", f"{prefix}_{name} {float(value):g}"]
    return "\n".join(lines) + "\n"
