"""Stand-in for `codex exec --json [...] [resume <thread_id>] -`.

Emits the JSONL shapes real Codex CLI 0.158 emits (captured live): thread.started,
turn.started, item.started/item.completed (agent_message, command_execution,
file_change), turn.completed with usage, or turn.failed. The prompt comes on stdin.
Behaviour is chosen by whole words in the prompt:

    run-command     run a command item (exit 0), bad-command for exit 1
    write-artifact  write report.md into AGENTD_ARTIFACTS_DIR (a file_change item)
    fail-turn       end the turn with turn.failed
    die             exit 3 mid-turn
    hang            never finish (until killed)
    no-answer       complete the turn without an agent message
    big-usage       report 50k tokens for the turn

With $FAKE_CODEX_HANG_ON_CLOSE set, the closing-summary turn never finishes. Each
invocation appends {"argv", "prompt"} to $FAKE_CODEX_LOG.
"""

import json
import os
import re
import sys
import time


def has(text: str, word: str) -> bool:
    return re.search(rf"(?<![\w-]){re.escape(word)}(?![\w-])", text) is not None


def out(obj: dict) -> None:
    print(json.dumps(obj), flush=True)


def main() -> int:
    argv = sys.argv[1:]
    prompt = sys.stdin.read()
    if log := os.environ.get("FAKE_CODEX_LOG"):
        with open(log, "a") as f:
            f.write(json.dumps({"argv": argv, "prompt": prompt}) + "\n")
    assert argv[0] == "exec" and argv[-1] == "-", argv
    thread = argv[argv.index("resume") + 1] if "resume" in argv else "thread-fake-1"
    out({"type": "thread.started", "thread_id": thread})
    out({"type": "turn.started"})
    print("not json at all", flush=True)  # the adapter must tolerate stray lines
    if has(prompt, "die"):
        return 3
    if has(prompt, "hang") or (
        os.environ.get("FAKE_CODEX_HANG_ON_CLOSE") and "ending this session" in prompt
    ):
        time.sleep(3600)
    if has(prompt, "run-command") or has(prompt, "bad-command"):
        code = 1 if has(prompt, "bad-command") else 0
        item = {
            "id": "item_1",
            "type": "command_execution",
            "command": "/bin/bash -lc 'ls'",
            "aggregated_output": "",
            "exit_code": None,
            "status": "in_progress",
        }
        out({"type": "item.started", "item": item})
        out(
            {
                "type": "item.completed",
                "item": {
                    **item,
                    "aggregated_output": "x\n",
                    "exit_code": code,
                    "status": "completed" if code == 0 else "failed",
                },
            }
        )
    if has(prompt, "write-artifact"):
        with open(os.path.join(os.environ["AGENTD_ARTIFACTS_DIR"], "report.md"), "w") as f:
            f.write("# Report\n")
        out(
            {
                "type": "item.completed",
                "item": {
                    "id": "item_2",
                    "type": "file_change",
                    "status": "completed",
                    "changes": [{"path": "report.md", "kind": "add"}],
                },
            }
        )
    if has(prompt, "fail-turn"):
        out({"type": "turn.failed", "error": {"message": "it broke"}})
        return 1
    if not has(prompt, "no-answer"):
        last = prompt.strip().splitlines()[-1] if prompt.strip() else ""
        out(
            {
                "type": "item.completed",
                "item": {"id": "item_3", "type": "agent_message", "text": f"done: {last}"},
            }
        )
    tokens = 50_000 if has(prompt, "big-usage") else 100
    out(
        {
            "type": "turn.completed",
            "usage": {"input_tokens": tokens, "cached_input_tokens": 0, "output_tokens": 10},
        }
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
