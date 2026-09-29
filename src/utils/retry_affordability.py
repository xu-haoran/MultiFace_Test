"""How much would one more attempt cost? Ask the clock, not an optimistic constant.

An agent node's retry is only worth buying if the deadline can fund a call that costs
what the last one cost. The failure this prevents (olympiad-level problem, 2026-07-29):
reasoning and Python each ran twice at 510-552s per attempt, the second attempt on each
branch was authorised with ~349s of budget left, and the problem overran its deadline by
198s to finish with no answer at all. Both branches had spent everything and delivered
nothing — the worst possible trade.
"""

from __future__ import annotations

DEFAULT_ATTEMPT_COST_S = 300.0


def last_attempt_cost(time_budget, label_prefix: str) -> float:
    """Cost of the most recent call whose label starts with `label_prefix`.

    Falls back to DEFAULT_ATTEMPT_COST_S when this problem has no comparable call yet.
    Failed attempts count: a transport failure that burned 363s will burn it again.
    """
    if time_budget is None:
        return 0.0
    matching = [entry["seconds"] for entry in time_budget.spend_log()
                if str(entry.get("label", "")).startswith(label_prefix)]
    return matching[-1] if matching else DEFAULT_ATTEMPT_COST_S


def can_afford_retry(time_budget, label_prefix: str) -> bool:
    """Whether another attempt of this kind fits in the remaining optional-work window."""
    if time_budget is None:
        return True
    return time_budget.can_afford(last_attempt_cost(time_budget, label_prefix))

