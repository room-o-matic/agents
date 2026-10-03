"""Live smoke test of the Claude Code adapter against the real `claude` CLI.

NOT run in CI. It spends real money on your Claude account, so it only runs when
explicitly authorized:

    ROM_LIVE_SMOKE=1 uv run python scripts/live_smoke.py [--budget 0.25] [--model haiku]

It runs one read-only session through agentd (in-process, real worker subprocess, real
claude) and prints the claude version it tested, for the support matrix
(docs: design/support-matrix.md).
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from conftest import BASE_URL, DOMAIN, ISSUER, FakeLobby  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from agentd.app import create_app  # noqa: E402
from agentd.config import CallerPolicy, Settings, WorkerType  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--budget", type=float, default=0.25)
    p.add_argument("--model", default="haiku")
    args = p.parse_args()
    if os.environ.get("ROM_LIVE_SMOKE") != "1":
        print("refusing: set ROM_LIVE_SMOKE=1 to authorize a paid live run", file=sys.stderr)
        return 2
    claude = shutil.which("claude")
    if not claude:
        print("claude CLI not found", file=sys.stderr)
        return 2
    version = subprocess.check_output([claude, "--version"], text=True).strip()
    lobby = FakeLobby()
    data = Path(tempfile.mkdtemp(prefix="agentd-live-"))
    settings = Settings(
        instance_id="agentd-live",
        data_dir=data,
        base_url=BASE_URL,
        lobbyd_url=ISSUER,
        lobbyd_domain=DOMAIN,
        ready_timeout_seconds=90,
        callers={"*": CallerPolicy(trust="trusted", profiles=["*"], worker_types=["*"])},
        worker_types={
            "claude": WorkerType(
                command=[
                    sys.executable,
                    "-m",
                    "agentd.workers.claude_code",
                    "--claude",
                    claude,
                    "--model",
                    args.model,
                    "--max-budget-usd",
                    str(args.budget),
                ]
            )
        },
    )
    h = lobby.headers("smoke")
    try:
        with TestClient(create_app(settings, verifier=lobby.verifier())) as c:
            sid = c.post(
                "/v1/sessions",
                headers=h,
                json={
                    "task": "Reply with exactly the word PONG and nothing else. Use no tools.",
                    "profile": "read_only_research",
                    "worker_type": "claude",
                    "timeout_seconds": 180,
                },
            ).json()["session_id"]
            deadline = time.time() + 180
            while time.time() < deadline:
                s = c.get(f"/v1/sessions/{sid}", headers=h).json()
                if s["status"] in ("completed", "failed", "stopped", "expired"):
                    break
                time.sleep(1)
            final = [
                e
                for e in c.get(
                    f"/v1/sessions/{sid}/events", params={"stream": False}, headers=h
                ).json()
                if e["type"] == "final"
            ]
    finally:
        shutil.rmtree(data, ignore_errors=True)
    ok = s["status"] == "completed" and (s["summary"] or "").strip() == "PONG"
    cost = final[0].get("cost_usd") if final else None
    print(f"claude {version} | status={s['status']} summary={s['summary']!r} cost_usd={cost}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
