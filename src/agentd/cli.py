import argparse
import logging
import sys

from agentd import auth, db
from agentd.config import Settings


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from agentd.app import create_app

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = Settings.load(args.config)
    uvicorn.run(create_app(settings), host=args.host, port=args.port)
    return 0


def _with_conn(args: argparse.Namespace):
    settings = Settings.load(args.config)
    db.init_db(settings.db_path)
    return db.connect(settings.db_path)


def cmd_token_create(args: argparse.Namespace) -> int:
    conn = _with_conn(args)
    try:
        token = auth.create_token(conn, args.agent)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    finally:
        conn.close()
    print(token)
    return 0


def cmd_token_revoke(args: argparse.Namespace) -> int:
    conn = _with_conn(args)
    try:
        n = auth.revoke_tokens(conn, args.agent)
    finally:
        conn.close()
    print(f"revoked {n} token(s) for {args.agent}")
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    """Print the effective config (secrets hidden), to check a YAML file and env."""
    settings = Settings.load(args.config)
    print(settings.model_dump_json(indent=2, exclude={"roomsd_token"}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="agentd", description="On-demand agent gateway")
    p.add_argument("--config", help="YAML config file (default: $AGENTD_CONFIG)")
    sub = p.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the gateway")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.set_defaults(func=cmd_serve)

    token = sub.add_parser("token", help="manage caller bearer tokens (local DB access)")
    token_sub = token.add_subparsers(dest="token_command", required=True)
    create = token_sub.add_parser("create", help="issue a token for a caller and print it")
    create.add_argument("agent")
    create.set_defaults(func=cmd_token_create)
    revoke = token_sub.add_parser("revoke", help="revoke all tokens for a caller")
    revoke.add_argument("agent")
    revoke.set_defaults(func=cmd_token_revoke)

    config = sub.add_parser("config", help="print the effective config")
    config.set_defaults(func=cmd_config)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
