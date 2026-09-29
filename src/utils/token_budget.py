from config import CONFIG
from threading import Lock


def estimate_tokens(text: str) -> int:
    return max(1, len(text or "") // 4)

class TokenBudget:
    def __init__(self, max_total_tokens: int = None, warn_ratio: float = None):
        self.max_total = max_total_tokens if max_total_tokens is not None else CONFIG["token_budget_max"]
        self.warn_ratio = warn_ratio if warn_ratio is not None else CONFIG["token_budget_warn_ratio"]
        self.consumed = 0
        self._lock = Lock()

    def can_afford(self, estimated_tokens: int) -> bool:
        with self._lock:
            return self.consumed + estimated_tokens <= self.max_total

    def consume(self, prompt_tokens: int, completion_tokens: int) -> None:
        with self._lock:
            self.consumed += prompt_tokens + completion_tokens

    def is_tight(self) -> bool:
        with self._lock:
            return self.consumed >= self.max_total * self.warn_ratio

    def remaining(self) -> int:
        with self._lock:
            return max(0, self.max_total - self.consumed)
