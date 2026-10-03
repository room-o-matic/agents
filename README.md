# agentd

**An on-demand agent gateway: ephemeral, sessionful helper workers.** Part of [room-o-matic](https://github.com/room-o-matic/docs).

agentd is an always-on gateway that starts helper workers only when they're asked for. Callers talk to a **session**, not a shell process. agentd owns the whole lifecycle:

- spawn, message and stop
- an SSE event stream
- idle and hard timeouts
- room close-out

Workers speak a small structured protocol: JSON-lines events such as `progress`, `artifact`, `needs_input` and `final`. Raw terminal output is never the source of truth.

> Status: MVP. Backends are `process` (no isolation, trusted callers only) and `sandbox` (bubblewrap). A Docker backend is planned.

## What's in the box

- **Profiles are server-side allowlists:** network, filesystem, workspace mount, runtime and budget. Callers pick a profile by name and never pass raw permissions.
- **Default-deny caller grants:** every caller needs an operator policy entry, and untrusted callers run only on the sandbox backend.
- **Worker types are pure config.** The built-in `fake` worker is for tests. There is a **Claude Code adapter** (`agentd.workers.claude_code`) that runs `claude -p` in stream-json mode, with tool permissions derived from the profile.
- **Room integration:** a session can be invited into a [roomsd](https://github.com/room-o-matic/rooms) room. The worker joins with its own guest identity and gets MCP room tools (`rooms_read`, `rooms_send`, `rooms_note_get`, and a compare-and-set `rooms_note_put`). @-mentions wake it, subject to budgets, and the invite is revoked when the session ends.
- **Bounded by design:** capacity is reserved at admission. Worker stdin and output, and the room wakes per session, are all budgeted, so one chatty session can't block the others.

## Run

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/); the sandbox backend also needs bubblewrap. agentd trusts tokens from [lobbyd](https://github.com/room-o-matic/lobby). The [project quickstart](https://github.com/room-o-matic/docs#quickstart-one-machine) brings up all three services.

```bash
uv sync
cp agentd.example.yaml local.yaml      # edit: instance_id, base_url, callers, worker_types
export AGENTD_CONFIG=local.yaml AGENTD_DATA_DIR=.data
export AGENTD_LOBBYD_API_KEY=...       # optional: agentd-scope key named instance_id; joins the registry
uv run agentd config                   # print the effective config (secrets hidden)
uv run agentd serve --port 8765
```

`AGENTD_<KEY>` environment variables override `instance_id`, `data_dir`, `base_url` and the `lobbyd_*` keys. Every setting is documented in [`agentd.example.yaml`](agentd.example.yaml) and `src/agentd/config.py`.

## API at a glance

Callers present a lobbyd access token issued for this instance's `base_url`.

| | |
|---|---|
| `POST /v1/sessions` | Spawn: `{task, profile, worker_type, workspace?, room?: {room_url, token}, operation_id?}` |
| `GET /v1/sessions/{id}` · `GET /v1/sessions/by-operation/{op}` | Status; reconciliation after an ambiguous failure |
| `GET /v1/sessions/{id}/events` | SSE (or `?stream=false` for JSON), resumable with `after_id` |
| `POST /v1/sessions/{id}/messages` · `POST …/stop` | Talk to the worker; stop it (message, then SIGTERM, then SIGKILL) |
| `GET /v1/instance` | Capabilities: worker types, profiles, what *you* are allowed to run, protocol version |
| `/healthz` · `/readyz` · `/metrics` | Operations |

The protocol version is `room-o-matic.agentd/1`. The [roomomatic client](https://github.com/room-o-matic/client) provides `summon`, which picks an instance, invites, spawns and reconciles.

## Operations

```bash
uv run agentd backup --out /backups/agentd-$(date -u +%F)     # database and sessions/; safe while serving
uv run agentd verify-backup /backups/agentd-…
uv run agentd restore /backups/agentd-… --force               # gateway stopped
```

A restore fails sessions that were live in the snapshot, without signalling any PID, and hands owed room close-outs to the inviter. See the [operations guide](https://github.com/room-o-matic/docs/blob/main/design/operations.md).

## Security notes

- **Process backend:** the `process` backend gives **no isolation**. Use it only for trusted callers on a trusted host.
- **Sandbox backend:** `sandbox` runs each worker in bubblewrap. It gets its own namespaces, a read-only system view, a private HOME and `/tmp`, a cleared environment and rlimits. agentd refuses to start if bubblewrap is unavailable.
- **Worker environment:** workers get only `env_allowlist` plus `AGENTD_*`, and the room variables if invited. Room tools refuse to post values of secret-looking variables.
- **Session visibility:** sessions are visible only to the caller that requested them.

## Development

```bash
uv sync && uv run pytest -q            # spawns real fake-worker subprocesses; sandbox tests need bubblewrap
uv run ruff check . && uv run ruff format --check .
ROM_LIVE_SMOKE=1 uv run python scripts/live_smoke.py   # optional, opt-in: one budget-capped real Claude run
```

Architecture and invariants for contributors are in the docs repo's [CLAUDE.md](https://github.com/room-o-matic/docs/blob/main/CLAUDE.md). Issues are tracked in [room-o-matic/docs](https://github.com/room-o-matic/docs/issues).

## License

[Apache-2.0](LICENSE)
