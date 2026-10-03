"""Room wake policy for workers (room-o-matic/docs#16).

Decides which room messages become model turns, so authenticated, well-behaved agents
can't keep each other running indefinitely:

- **Addressing is exact.** A worker wakes for messages whose `to` list names it, or that
  @-mention its short name as a whole word: `@worker` doesn't match `@worker-other`.
  Under policy "all", any message from someone else qualifies.
- **Informational types never wake:** status, handoff, decision, artifact, task_update.
  A typed decision or summary doesn't start another round of acknowledgements.
- **Reply chains have a depth limit:** messages at `hop` >= max_hop don't wake (roomsd
  also refuses chains past the room's max_hops).
- **Every message id wakes at most once** (deduplication across repeated deliveries).
- **While a turn is running, wakes queue** (bounded) and are delivered as **one coalesced
  turn** when it finishes. Overflow is dropped *and counted*.
- **Each session has a wake budget:** after `max_wakes` room-triggered turns, further
  wakes are suppressed until the owner intervenes.
- **Suppression is never silent:** every suppressed wake is reported with a reason.
"""

import re
from collections import OrderedDict
from dataclasses import dataclass, field

NON_WAKING_TYPES = frozenset({"status", "handoff", "decision", "artifact", "task_update"})


@dataclass
class WakeGate:
    identity: str
    policy: str = "mentions"  # mentions | all | none
    max_hop: int = 3
    max_wakes: int = 10
    max_queue: int = 20
    wakes_used: int = 0
    queue: list[dict] = field(default_factory=list)
    suppressed: dict[str, int] = field(default_factory=dict)
    _seen: OrderedDict = field(default_factory=OrderedDict)

    def _suppress(self, reason: str) -> tuple[str, str]:
        self.suppressed[reason] = self.suppressed.get(reason, 0) + 1
        return ("suppress", reason)

    def addressed(self, m: dict) -> bool:
        if self.policy == "all":
            return True
        if self.identity in (m.get("to") or []):
            return True
        short = re.escape(self.identity.rsplit("/", 1)[-1])
        full = re.escape(self.identity)
        # Whole-word only: "@worker" must not match "@worker-other" or "@worker.x", but a
        # sentence-ending "." after the name is fine.
        pattern = rf"(?<![\w.@/-])(?:@{short}|{full})(?![\w-]|\.[\w-])"
        return re.search(pattern, m.get("body", "")) is not None

    def offer(self, m: dict, *, in_turn: bool) -> tuple[str, str | None]:
        """Classify a new room message: ("deliver", None), ("queue", None),
        ("ignore", reason) for messages that were never meant to wake us, or
        ("suppress", reason) for wakes refused by a limit (counted and reported)."""
        if self.policy == "none" or m["from"] == self.identity:
            return ("ignore", "self" if m["from"] == self.identity else "policy")
        if m["id"] in self._seen:
            return ("ignore", "duplicate")
        self._seen[m["id"]] = True
        while len(self._seen) > 10_000:
            self._seen.popitem(last=False)
        if m.get("type") in NON_WAKING_TYPES:
            return ("ignore", "informational")
        if not self.addressed(m):
            return ("ignore", "not addressed")
        if (m.get("hop") or 0) >= self.max_hop:
            return self._suppress("reply_chain_too_deep")
        if self.wakes_used >= self.max_wakes:
            return self._suppress("wake_budget_exhausted")
        if in_turn:
            if len(self.queue) >= self.max_queue:
                return self._suppress("queue_full")
            self.queue.append(m)
            return ("queue", None)
        self.wakes_used += 1
        return ("deliver", None)

    def drain(self) -> list[dict]:
        """After a turn ends: everything queued, to send as one coalesced turn (one wake)."""
        if not self.queue:
            return []
        if self.wakes_used >= self.max_wakes:
            self.suppressed["wake_budget_exhausted"] = self.suppressed.get(
                "wake_budget_exhausted", 0
            ) + len(self.queue)
            self.queue.clear()
            return []
        batch, self.queue = self.queue, []
        self.wakes_used += 1
        return batch
