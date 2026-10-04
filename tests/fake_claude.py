"""Stand-in for `claude -p --input-format stream-json --output-format stream-json`.

Emits the same message shapes as Claude Code (system init, assistant text and tool_use
blocks, result). Behaviour is chosen by whole words in each user turn:

    write-artifact  write report.md into AGENTD_ARTIFACTS_DIR
    fail-turn       end the turn with an error result
    die             exit 3 without a result
    slow            don't finish the turn until another user message arrives
    hang            never finish (until killed)

Writes its argv to $FAKE_CLAUDE_ARGV_FILE when set, so tests can check the flags. With
$FAKE_CLAUDE_HANG_ON_CLOSE set, the closing-summary turn never finishes.
Like Claude Code, it repeats the system init message at the start of every turn.
"""

import json
import os
import re
import sys
import time

SESSION = "11111111-2222-3333-4444-555555555555"


def has(text: str, word: str) -> bool:
    """Whole-word keyword match, so e.g. "change" doesn't trigger "hang"."""
    return re.search(rf"(?<![\w-]){re.escape(word)}(?![\w-])", text) is not None


def out(obj: dict) -> None:
    print(json.dumps(obj), flush=True)


def turns():
    for line in sys.stdin:
        msg = json.loads(line)
        content = msg["message"]["content"]
        yield content if isinstance(content, str) else json.dumps(content)


def main() -> int:
    if path := os.environ.get("FAKE_CLAUDE_ARGV_FILE"):
        with open(path, "w") as f:
            json.dump(sys.argv[1:], f)
    pending = turns()
    n = 0
    for text in pending:
        out(
            {
                "type": "system",
                "subtype": "init",
                "session_id": SESSION,
                "model": "fake-1",
                "tools": ["Read"],
            }
        )
        n += 1
        out(
            {
                "type": "assistant",
                "session_id": SESSION,
                "message": {
                    "content": [
                        {"type": "text", "text": f"working on: {text}"},
                        {"type": "tool_use", "name": "Read", "input": {"file_path": "README.md"}},
                    ]
                },
            }
        )
        print("not json at all", flush=True)  # the adapter must tolerate stray lines
        if has(text, "die"):
            return 3
        if has(text, "hang") or (
            os.environ.get("FAKE_CLAUDE_HANG_ON_CLOSE") and "ending this session" in text
        ):
            time.sleep(3600)
        if has(text, "slow"):
            text += " + " + next(pending)
        if has(text, "write-artifact"):
            with open(os.path.join(os.environ["AGENTD_ARTIFACTS_DIR"], "report.md"), "w") as f:
                f.write("# Report\n")
            out(
                {
                    "type": "assistant",
                    "session_id": SESSION,
                    "message": {
                        "content": [
                            {
                                "type": "tool_use",
                                "name": "Write",
                                "input": {"file_path": "report.md"},
                            },
                        ]
                    },
                }
            )
        if has(text, "fail-turn"):
            out(
                {
                    "type": "result",
                    "subtype": "error_during_execution",
                    "is_error": True,
                    "result": "it broke",
                    "session_id": SESSION,
                    "num_turns": n,
                }
            )
            continue
        out(
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "result": f"done: {text}",
                "session_id": SESSION,
                "num_turns": n,
                "total_cost_usd": 0.0123,
                "duration_ms": 42,
            }
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
