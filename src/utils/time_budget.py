"""Per-problem wall-clock budget — the competition's hard constraint is time, not tokens.

Why this exists (measured 2026-07-29 against intern-s2-preview-397b):

`intern-s2-preview-397b` is a reasoning model. It returns a separate
`reasoning_content` field and BOTH it and `content` are billed against
`max_tokens`. Latency is ~55 completion tokens/s and depends only on tokens
actually generated — `max_tokens` never speeds a call up, it only truncates it
(`finish_reason="length"`, partial CoT leaking into `content`). So token caps
cannot bound a problem's runtime; only a clock can.

The judge run's single failure (Q3 at 1307s vs the 1200s limit) is not explained
by the sum of its stages (~250s measured). The gap is nested transport retry:
`LLMRetryWrapper(max_retries=3)` wrapping a client that itself retries 3× at a
120s socket timeout is up to 9 requests ≈ 1080s from one node. Bounding retries
by count cannot fix that, because the count is split across two layers we do not
jointly control (the platform injects its own client). Bounding them by a
deadline can.

Every LLM call site therefore consults this budget before spending, and optional
stages degrade to deterministic fallbacks once the reserve is reached. The
reserve keeps enough time to always emit a real answer instead of timing out
with nothing.
"""

from __future__ import annotations

import time
from threading import Lock

from config import CONFIG


class TimeBudget:
    """Monotonic per-problem deadline shared by every node in one solve() call.

    Three horizons:
      * ``soft`` (total - reserve) — deadline for *optional* work. Past it the
        graph stops paying for arbitration/coordination LLM calls and falls back
        to deterministic answer selection.
      * ``fast_path`` threshold — enough time left for one more full LLM call?
        Below it, skip retries and second-chance stages.
      * ``total`` — the platform's hard per-problem limit. Never intentionally
        reached; the reserve is what keeps us clear of it.
    """

    def __init__(
        self,
        total_seconds: float | None = None,
        reserve_seconds: float | None = None,
        fast_path_threshold: float | None = None,
        clock=time.monotonic,
    ) -> None:
        self.total = float(
            total_seconds if total_seconds is not None else CONFIG["problem_time_budget_s"]
        )
        self.reserve = float(
            reserve_seconds if reserve_seconds is not None else CONFIG["time_reserve_s"]
        )
        self.fast_path_threshold = float(
            fast_path_threshold
            if fast_path_threshold is not None
            else CONFIG["time_fast_path_threshold_s"]
        )
        self._clock = clock
        self._start = clock()
        self._lock = Lock()
        self._spend_log: list[dict] = []


    def now(self) -> float:
        """This budget's clock reading.

        Callers that measure a call's duration must use *this* clock, not
        `time.monotonic()` directly: if the two disagree, a measured cost cannot be
        compared against `remaining()` and every deadline check built on it silently
        stops working.
        """
        return self._clock()

    def elapsed(self) -> float:
        return self._clock() - self._start

    def remaining(self) -> float:
        """Seconds left before optional work must stop. May go negative."""
        return self.total - self.reserve - self.elapsed()

    def remaining_hard(self) -> float:
        """Seconds left before the platform's hard limit."""
        return self.total - self.elapsed()

    def expired(self) -> bool:
        return self.remaining() <= 0

    def can_afford(self, seconds: float) -> bool:
        return self.remaining() >= float(seconds)

    def fast_path(self) -> bool:
        """True when there is no longer room for a full extra LLM round trip."""
        return self.remaining() < self.fast_path_threshold

    def timeout_for(self, ceiling: float | None) -> float | None:
        """Clamp a node's configured ceiling to what the deadline actually allows.

        A node may never outlive the problem: a generous per-node ceiling catches
        true hangs, while the deadline caps the real world. Returns a small
        positive floor rather than 0 so an already-late node still gets a chance
        to return the answer it has instead of raising.
        """
        allowed = max(1.0, self.remaining_hard())
        if ceiling is None:
            return allowed
        return min(float(ceiling), allowed)


    def record(self, label: str, seconds: float) -> None:
        with self._lock:
            self._spend_log.append({"label": label, "seconds": round(float(seconds), 2)})

    def spend_log(self) -> list[dict]:
        with self._lock:
            return list(self._spend_log)

    def snapshot(self) -> dict:
        return {
            "elapsed_s": round(self.elapsed(), 2),
            "remaining_s": round(self.remaining(), 2),
            "total_s": self.total,
            "reserve_s": self.reserve,
            "fast_path": self.fast_path(),
        }

