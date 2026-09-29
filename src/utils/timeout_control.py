"""Per-node wall-clock ceilings, clamped to the problem deadline.

Previously dead code: nothing called `run_with_timeout`, so `CONFIG["node_timeouts"]`
was decorative and no node had any upper bound. That is why the 2026-07-29 judge run
could spend 1307s on one problem — a node stuck in nested transport retry had nothing
to stop it.

Two layers now cooperate:
  * the ceiling in CONFIG catches a genuine hang in one node;
  * TimeBudget.timeout_for() shortens that ceiling to whatever the 20-minute
    problem deadline still allows, so a late node cannot overrun the whole run.

A cancelled future's thread cannot actually be killed, so a timed-out call keeps
running in the background until its socket timeout fires. That is acceptable here:
the count is bounded by the number of nodes, and the alternative is missing the
platform deadline entirely. What matters is that the *graph* moves on and still
emits an answer.
"""
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Callable, TypeVar

from config import CONFIG

T = TypeVar("T")


class NodeTimeoutError(TimeoutError):
    pass


def get_node_timeout(node_name: str, default: int | None = None) -> int | None:
    return CONFIG["node_timeouts"].get(node_name, default)


def resolve_timeout(node_name: str, time_budget=None, default: int | None = None):
    """Effective ceiling for a node: configured value clamped by the deadline."""
    ceiling = get_node_timeout(node_name, default)
    if time_budget is None:
        return ceiling
    return time_budget.timeout_for(ceiling)


def run_with_timeout(func: Callable[[], T], timeout: int | float | None) -> T:
    if timeout is None:
        return func()
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(func)
        try:
            return future.result(timeout=timeout)
        except FutureTimeoutError as exc:
            future.cancel()
            raise NodeTimeoutError(f"operation timed out after {timeout:.0f}s") from exc
    finally:
        executor.shutdown(wait=False)

