"""Shared reconciliation-round policy for validation and retry routing.

A reconciliation round re-runs the whole solving subgraph — another reasoning call
plus another Python call. On this model that is the most expensive action available
(a single reasoning call measured 77-116s on a mid-difficulty problem, and hard
problems run longer), so the round limit is gated by both budgets: tokens, and now
wall clock. Buying a rerun we cannot finish is strictly worse than answering with
what we already have.
"""

from config import CONFIG
from utils.deps import get_deps
from utils.retry_affordability import last_attempt_cost


def _deps(config):
    try:
        return get_deps(config)
    except Exception:  # noqa: BLE001 - callers may pass a bare config in tests.
        return None


def reconciliation_round_limit(config) -> int:
    deps = _deps(config)
    budget = getattr(deps, "token_budget", None)
    clock = getattr(deps, "time_budget", None)
    if clock is not None and clock.closeout_started():
        return 1
    if budget and budget.is_tight():
        return 1
    return CONFIG["reconciliation_max_rounds"]


def projected_solving_seconds(time_budget, targets=None) -> float:
    """Estimate one more parallel reasoning/Python pass from observed costs."""
    if time_budget is None:
        return 0.0
    return max(last_attempt_cost(time_budget, target)
               for target in (targets or ["reasoning", "python"]))


def reconciliation_retry_available(state, config) -> bool:
    """Whether entering reconciliation can still produce another solving run."""
    completed_rounds = state.get("reconciliation_round", 0)
    if completed_rounds + 1 >= reconciliation_round_limit(config):
        return False
    deps = _deps(config)
    clock = getattr(deps, "time_budget", None)
    target = (state.get("validation_details") or {}).get("repair_target")
    targets = [target] if target in {"reasoning", "python"} else None
    return clock is None or clock.can_afford(projected_solving_seconds(clock, targets))

