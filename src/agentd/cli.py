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


def cmd_ask(args: argparse.Namespace) -> int:
    import asyncio

    from agentd import local

    def show(event: dict) -> None:
        if args.verbose and event["type"] in ("progress", "error"):
            print(f"  … {event.get('message', '')[:200]}", file=sys.stderr, flush=True)

    try:
        cfg = local.load(args.agents)
        r = asyncio.run(
            local.ask(
                cfg,
                args.agent,
                " ".join(args.question),
                context=args.context,
                timeout=args.timeout,
                on_event=show,
            )
        )
    except (local.AskError, ValueError) as e:
        print(f"agentd: {e}", file=sys.stderr)
        return 1
    print(r["answer"])
    if r["cost_usd"] is not None:
        print(f"(cost ${r['cost_usd']:.4f})", file=sys.stderr)
    return 0


def cmd_agents(args: argparse.Namespace) -> int:
    from agentd import local

    try:
        cfg = local.load(args.agents)
    except (local.AskError, ValueError) as e:
        print(f"agentd: {e}", file=sys.stderr)
        return 1
    for a in local.describe(cfg):
        where = f" [{a['workspace']}]" if a["workspace"] else ""
        print(f"{a['agent']}: {a['worker_type']}/{a['profile']}{where}  {a['description']}")
    return 0


def cmd_mcp(args: argparse.Namespace) -> int:
    from agentd import local

    local.build_server(args.agents).run("stdio")
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

    a = sub.add_parser("ask", help="ask a local agent a question (no services needed)")
    a.add_argument("agent")
    a.add_argument("question", nargs="+")
    a.add_argument("--context", help="earlier answers or facts to pass along")
    a.add_argument("--timeout", type=float, help="seconds (at most the profile's runtime)")
    a.add_argument(
        "--agents",
        type=Path,
        help="agents file (default $AGENTD_AGENTS or ~/.config/agentd/agents.yaml)",
    )
    a.add_argument("-v", "--verbose", action="store_true", help="show the agent's progress")
    a.set_defaults(func=cmd_ask)
    ag = sub.add_parser("agents", help="list the local agents")
    ag.add_argument("--agents", type=Path)
    ag.set_defaults(func=cmd_agents)
    m = sub.add_parser("mcp", help="serve the local agents as MCP tools (stdio)")
    m.add_argument("--agents", type=Path)
    m.set_defaults(func=cmd_mcp)
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
