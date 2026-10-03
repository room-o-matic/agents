import argparse
import json
import logging
import sys
from pathlib import Path

from agentd.config import Settings


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from agentd.app import create_app

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = Settings.load(args.config)
    uvicorn.run(create_app(settings), host=args.host, port=args.port)
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    """Print the effective config (secrets hidden), to check a YAML file and env."""
    settings = Settings.load(args.config)
    print(settings.model_dump_json(indent=2, exclude={"lobbyd_api_key"}))
    return 0


def cmd_backup(args: argparse.Namespace) -> int:
    from agentd import recovery

    manifest = recovery.backup(Settings.load(args.config), Path(args.out))
    print(json.dumps({k: v for k, v in manifest.items() if k != "files"}, indent=2))
    print(
        f"backed up {len(manifest['files'])} files to {args.out}; encrypt it before it"
        " leaves this host",
        file=sys.stderr,
    )
    return 0


def cmd_verify_backup(args: argparse.Namespace) -> int:
    from agentd import ops

    try:
        manifest = ops.verify_backup(Path(args.path))
    except ops.BackupError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(
        f"ok: {manifest['service']} schema v{manifest['schema_version']},"
        f" {len(manifest['files'])} files, taken {manifest['created_at']}"
    )
    return 0


def cmd_restore(args: argparse.Namespace) -> int:
    from agentd import ops, recovery

    try:
        report = recovery.restore(
            Settings.load(args.config), Path(args.source), force=args.force, id_gap=args.id_gap
        )
    except (ops.BackupError, ops.SchemaError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="agentd", description="On-demand agent gateway")
    p.add_argument("--config", help="YAML config file (default: $AGENTD_CONFIG)")
    sub = p.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the gateway")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.set_defaults(func=cmd_serve)

    config = sub.add_parser("config", help="print the effective config")
    config.set_defaults(func=cmd_config)

    # docs#24: operate on the configured data_dir directly (stop the gateway to restore).
    b = sub.add_parser("backup", help="snapshot the database and sessions dir (safe while serving)")
    b.add_argument("--out", required=True, help="new directory to write the backup into")
    b.set_defaults(func=cmd_backup)
    v = sub.add_parser("verify-backup", help="check a backup's checksums and integrity")
    v.add_argument("path")
    v.set_defaults(func=cmd_verify_backup)
    r = sub.add_parser("restore", help="restore a backup into data_dir (gateway stopped)")
    r.add_argument("source", help="backup directory")
    r.add_argument("--force", action="store_true", help="move existing data aside")
    r.add_argument(
        "--id-gap", type=int, default=1_000_000, help="advance event IDs past the snapshot"
    )
    r.set_defaults(func=cmd_restore)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
