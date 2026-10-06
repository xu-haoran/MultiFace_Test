"""Per-problem wall-clock budget with a normal phase and a closeout phase.

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


# Keep this much of the closeout window for the emergency answer and finalization.
CLOSEOUT_EMERGENCY_BUFFER_S = 90


class TimeBudget:
    """Monotonic per-problem deadline shared by every node in one solve() call.

    Two phases:
      * ``normal`` (total - reserve) — ordinary solving and verification.
      * ``closeout`` (the reserve) — compressed reasoning and emergency answer.
    """

    def __init__(
        self,
        total_seconds: float | None = None,
        reserve_seconds: float | None = None,
        clock=time.monotonic,
        sleeper=time.sleep,
    ) -> None:
        self.total = float(
            total_seconds if total_seconds is not None else CONFIG["problem_time_budget_s"]
        )
        self.reserve = float(
            reserve_seconds if reserve_seconds is not None else CONFIG["time_reserve_s"]
        )
        self._clock = clock
        self._sleep = sleeper
        self._start = clock()
        self._lock = Lock()
        self._spend_log: list[dict] = []
        self._closeout_claims: set[str] = set()


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
        """True once the normal-solving phase ends."""
        return self.remaining() <= 0

    def can_afford(self, seconds: float) -> bool:
        return self.remaining() >= float(seconds)

    def closeout_started(self) -> bool:
        """Whether the final reserved phase has begun."""
        return self.remaining_hard() <= self.reserve

    def wait_until_closeout(self) -> bool:
        """Wait until the reserved final phase; return whether it has begun."""
        delay = max(0.0, self.total - self.reserve - self.elapsed())
        if delay:
            self._sleep(delay)
        return self.closeout_started()

    def claim_closeout_attempt(self, label: str) -> bool:
        """Allow a named closeout action to run at most once per problem."""
        with self._lock:
            if label in self._closeout_claims:
                return False
            self._closeout_claims.add(label)
            return True

    def timeout_for(
        self,
        ceiling: float | None,
        *,
        normal_only: bool = False,
        reserve_after: float = 0.0,
    ) -> float | None:
        """Clamp a node's configured ceiling to what the deadline actually allows.

        A node may not outlive its phase. A zero allowance lets the wrapper move
        directly to that node's deterministic fallback.
        """
        available = self.remaining() if normal_only else self.remaining_hard() - reserve_after
        allowed = max(0.0, available)
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
            "closeout_started": self.closeout_started(),
        }

