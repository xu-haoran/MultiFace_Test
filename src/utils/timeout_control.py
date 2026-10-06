"""Node limits that protect the 15-minute solve and 5-minute closeout phases.

Python verification is clamped to normal time. Reasoning may enter closeout but
must leave the emergency-answer margin. A timed-out worker thread cannot be killed;
the graph moves on while the injected client finishes or reaches its own timeout.
"""
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Callable, TypeVar

from config import CONFIG
from utils.time_budget import CLOSEOUT_EMERGENCY_BUFFER_S

T = TypeVar("T")


class NodeTimeoutError(TimeoutError):
    pass


def get_node_timeout(node_name: str, default: int | None = None) -> int | None:
    return CONFIG["node_timeouts"].get(node_name, default)


def resolve_timeout(node_name: str, time_budget=None, default: int | None = None):
    """Clamp ordinary verification to phase one and protect closeout time."""
    ceiling = get_node_timeout(node_name, default)
    if time_budget is None:
        return ceiling
    if node_name in {"python_agent", "cross_validator"}:
        return time_budget.timeout_for(ceiling, normal_only=True)
    if node_name == "reasoning_agent":
        return time_budget.timeout_for(
            ceiling, reserve_after=CLOSEOUT_EMERGENCY_BUFFER_S
        )
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

