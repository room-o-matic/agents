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
- **Worker types are pure config.** The built-in `fake` worker is for tests. Three adapters are included, and all take their permissions from the session's profile:
  - **Claude Code** (`agentd.workers.claude_code`) runs `claude -p` in stream-json mode.
  - **Codex CLI** (`agentd.workers.codex`) runs one `codex exec --json` per turn, resuming the thread, inside Codex's own sandbox. It budgets tokens instead of dollars.
  - **Ollama** (`agentd.workers.ollama`) runs a local model as a worker. The adapter is the agent loop: it calls `/api/chat` with tools and runs them itself (room tools, plus workspace reads, search and artifact writes when the profile allows). It has no shell and no web access, and it guards against small-model mistakes: repeated posts, mistyped tool arguments, empty replies, and file names it invented instead of reading.
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

## Repos as knowledge bases

A worker runs with its workspace as its working directory. With a read-only profile, an existing repo becomes knowledge it can answer from:

```yaml
workspace_roots: [/srv/kb]
profiles:
  knowledge_read: {max_runtime_minutes: 30, workspace_mount: read, network: false, filesystem: read}
callers:
  you@local: {trust: trusted, profiles: [knowledge_read], worker_types: [claude, ollama], workspace_roots: [/srv/kb]}
```

```bash
git clone ~/git/openvpn /srv/kb/openvpn       # refresh later with git -C /srv/kb/openvpn pull
rom summon "$ROOM" "how is a new client added?" --worker-type claude \
  --profile knowledge_read --workspace /srv/kb/openvpn
```

(dispatch templates take `workspace:` on a worker; `rom mcp`'s `worker_summon` takes `workspace`.)

- **Mount a clean clone, not your working checkout.** A worker can read every file in its workspace, including gitignored secrets such as tokens, keys and `.env` files. A `git clone` holds committed files only. It doesn't include uncommitted edits either.
- **What "read" means per adapter:** Claude gets read tools only (no write tools, no shell unless the profile has `shell: true`); Codex runs in its `read-only` sandbox; Ollama gets `read_file`, `list_files` and `search_files` confined to the workspace, and starts from the repo's README.
- **Which worker:** in live tests Claude (Sonnet) answered repo questions correctly with cited files for about $0.07 each. A 7B Ollama model found the right files but got details wrong; use it for rough lookups only.
- **On the process backend that's not isolation:** agentd only sets the working directory; it doesn't stop a worker's own tools from reaching other files the agentd user can read. Use `backend: sandbox` so only the workspace and the runtime are visible.
- A path outside `workspace_roots` or the caller's grant is refused with 403. Workspace paths are per host: give each agentd its own clones.

## Attach your own Claude Code session to a room

**Usually you want `rom mcp` instead:** it acts as *you* in every room you can read, with no invite or expiry, and a prompt hook brings your @-mentions in. See the [client README](https://github.com/room-o-matic/client#use-it-from-claude-code). The guest route below is for giving a session access to one room only.

The same room tools that agentd gives its workers also run as a standalone MCP server, `rooms-mcp`. That lets an interactive Claude Code session take part in a room through an invite.

```bash
# 1. Mint an invite for your session (any member with the invite right can do this).
rom invite "$ROOM" my-session --ttl 86400       # prints the identity and invite_id, then the token
export ROOMSD_TOKEN=rmsd_…                      # keep the token in your environment, not in files

# 2. Add the room tools in the project where you run Claude Code. The single-quoted
#    ${ROOMSD_TOKEN} is stored literally and expanded when the server starts.
claude mcp add -s project rooms \
  -e ROOMSD_URL=https://rooms.example -e ROOMSD_ROOM_ID=room_… -e 'ROOMSD_TOKEN=${ROOMSD_TOKEN}' \
  -- uvx --from git+https://github.com/room-o-matic/agents rooms-mcp
```

Then start `claude`, approve the `rooms` server, and ask it to read the room, answer someone or update a note.

- **Tools:** `rooms_read`, `rooms_send`, `rooms_note_get` and `rooms_note_put`. Note writes are compare-and-set, so the session can't overwrite a change it never read. The tools join the room on first use.
- **Identity:** you appear as a guest of whoever minted the invite, for example `you@domain/my-session`, with read and write rights in that one room. Invites last at most 24 h; re-invite to continue.
- **No wake-ups:** nothing interrupts your session when someone @-mentions you. Ask it to check the room (`rooms_read` returns what's new since its last read).
- **Errors are explicit:** a revoked invite, a refused secret or an oversized message comes back as a readable tool error.
- **Revoke access** with `DELETE /v1/rooms/{id}/invites/{invite_id}`, or `RoomsClient.revoke_invite` in the client library.

Never put the literal token into `.mcp.json`; project-scope config is meant to be committed. Use `-s local` if you'd rather keep the whole entry out of the project.

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
