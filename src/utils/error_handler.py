import logging
import time
import traceback
from typing import Any, Callable

from langchain_core.runnables import RunnableConfig

from utils.deps import get_deps
from utils.problem_anchor import verify_problem_anchor


def _node_name(node_func: Callable, explicit_name: str | None = None) -> str:
    if explicit_name:
        return explicit_name
    name = getattr(node_func, "__name__", "unknown")
    return name[:-5] if name.endswith("_node") else name


def _error_payload(node: str, exc: Exception) -> dict:
    return {
        "node": node,
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(limit=5),
    }


def _fallback_for_node(node: str, state: dict, error: dict, config: dict | None = None) -> dict:
    base = {"errors": [error]}
    if node == "classifier":
        from nodes.classifier_node import _deterministic_fallback, _decision_payload, _rank_alternate_categories
        try:
            deps = get_deps(config)
            sl = deps.skills_loader
            categories = list(sl.categories)
            problem = state.get("problem", "")
            category = _deterministic_fallback(sl, problem, categories, deps.logger)
            return {**base, **_decision_payload(problem, categories, category, 0.0,
                    _rank_alternate_categories(sl, problem, category, categories), ["timeout_deterministic_fallback"])}
        except Exception:
            pass
        return {
            **base,
            "category": "非基础及进阶课程",
            "category_confidence": 0.0,
            "candidate_categories": [],
            "classification_stages_used": ["error_fallback"],
        }
    if node == "reasoning_agent":
        return {
            **base,
            "reasoning_result": {
                "analysis": "",
                "steps": [],
                "answer": "",
            },
            "reasoning_trace": [{"attempt": 0, "status": "failed", "error": error["message"]}],
            "reasoning_attempts": 0,
        }
    if node == "python_agent":
        from utils.verification_evidence import parse_verification_evidence

        output = parse_verification_evidence(
            {
                "success": False,
                "stdout": "",
                "stderr": error["message"],
                "answer": None,
                "execution_time": 0.0,
            },
            candidate_answer=(state.get("reasoning_result") or {}).get("answer", ""),
            problem=state.get("problem", ""),
        )
        return {
            **base,
            "python_code": "",
            "python_output": output,
            "python_trace": [{"attempt": 0, "status": "failed", "error": error["message"]}],
            "python_attempts": 0,
            "python_evidence_status": output["evidence_status"],
            "python_evidence_summary": output["evidence_summary"],
            "python_contradictions": output["contradictions"],
            "objective_python_independent": False,
        }
    if node == "cross_validator":
        return {
            **base,
            "validation_status": "uncertain",
            "validation_details": {"status": "uncertain", "reason": error["message"]},
            "validated_answer": "",
            "next_node": "coordinator",
        }
    if node == "reconciliation":
        return {
            **base,
            "reconciliation_trace": list(state.get("reconciliation_trace") or [])
            + [{"round": state.get("reconciliation_round", 0), "action": "error_fallback"}],
            "reasoning_retry_hint": None,
            "python_retry_hint": None,
            "next_node": "coordinator",
        }
    if node == "semantic_arbiter":
        from nodes.cross_validator_node import _preferred_answer
        from utils.reconciliation_policy import reconciliation_retry_available

        can_retry = False
        # Arbiter timeout/transport failure supplies no mathematical evidence
        # that justifies regenerating both solvers.
        can_retry = False
        answer = "" if can_retry else _preferred_answer(
            state, state.get("validation_details") or {})
        return {
            **base,
            "semantic_arbiter_status": "error",
            "semantic_arbiter_trace": list(state.get("semantic_arbiter_trace") or [])
            + [{"status": "error", "error": error["message"]}],
            "semantic_arbiter_attempts": state.get("semantic_arbiter_attempts", 0) + 1,
            "validated_answer": answer,
            "answer_locked": False,
            "next_node": "reconciliation" if can_retry else "coordinator",
        }
    if node == "coordinator":
        rr = state.get("reasoning_result") or {}
        answer = state.get("validated_answer") or rr.get("answer", "")
        if not answer:
            raw = (state.get("reasoning_raw_response") or "").strip()
            if raw:
                final = f"未能完成完整推导，以下为已获得的部分结果：\n{raw[-2000:]}"
            else:
                final = "解题过程中出现错误，无法给出完整答案。"
        else:
            ptype = (state.get("validation_details") or {}).get("problem_type", "computation")
            # 2026-08-24：响应只放答案本身，不附加标签（步骤在 trace）。
            final = answer
        return {
            **base,
            "final_response": final,
            "fallback_source": "validated_answer" if answer else (
                "reasoning_raw" if state.get("reasoning_raw_response") else "generic_error"),
        }
    if node == "database_retrieval":
        return {**base, "retrieved_examples": [], "retrieval_candidates": [], "knowledge_cards": []}
    if node == "submission":
        return {**base, "final_response": state.get("final_response") or "解题过程中出现错误，无法给出完整答案。",
                "submission_check": {"status": "check_failed", "complete": False}}
    if node == "knowledge_review":
        return {**base, "knowledge_review": {"status": "inconclusive", "reason": error["message"]}}
    return {**base, "should_terminate": True, "next_node": "coordinator"}


class ErrorHandler:
    @staticmethod
    def safe_execute(func: Callable, *args: Any, **kwargs: Any) -> tuple[bool, Any]:
        try:
            return True, func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - convert any node failure into state.
            return False, {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(limit=5),
            }

    @staticmethod
    def node_wrapper(node_func: Callable, node_name: str | None = None) -> Callable:
        name = _node_name(node_func, node_name)
        logger = logging.getLogger("math_agent")

        def wrapped(state: dict, config: RunnableConfig) -> dict:
            from utils.timeout_control import resolve_timeout, run_with_timeout

            time_budget = None
            try:
                time_budget = getattr(get_deps(config), "time_budget", None)
            except Exception:  # noqa: BLE001 - a missing deps shape must not break routing.
                time_budget = None

            check = verify_problem_anchor(state)
            guarded_state = state
            integrity_events = []
            if check.event:
                integrity_events.append(check.event)
                if check.event.get("action") in {"repair", "anchor_corrupt"}:
                    guarded_state = {**state, "problem": check.problem}

            started = time.monotonic()
            try:
                timeout = resolve_timeout(name, time_budget)
                result = run_with_timeout(lambda: node_func(guarded_state, config), timeout)
            except Exception as exc:  # noqa: BLE001 - graph must continue per M4.
                logger.exception("Node %s failed", name)
                result = _fallback_for_node(name, guarded_state, _error_payload(name, exc), config=config)
            elapsed = time.monotonic() - started
            if time_budget is not None:
                time_budget.record(f"node:{name}", elapsed)
            if isinstance(result, dict):
                # Keep earlier branch attempts when a targeted repair replaces
                # its candidate; untouched branches retain their original trace.
                if name in {"reasoning_agent", "python_agent"}:
                    prefix = "reasoning" if name == "reasoning_agent" else "python"
                    trace_key = f"{prefix}_trace"
                    attempts_key = f"{prefix}_attempts"
                    if trace_key in result:
                        previous = list(guarded_state.get(trace_key) or [])
                        result[trace_key] = previous + list(result[trace_key] or [])
                    if attempts_key in result:
                        result[attempts_key] += guarded_state.get(attempts_key, 0)
                if integrity_events:
                    result.setdefault("problem_integrity_events", []).extend(integrity_events)
                if guarded_state is not state:
                    result.setdefault("problem", check.problem)
                result.setdefault("node_timings", []).append(
                    {"node": name, "elapsed_s": round(elapsed, 2)}
                )
            return result

        wrapped.__name__ = f"{name}_wrapped"
        return wrapped


def node_wrapper(node_func: Callable, node_name: str | None = None) -> Callable:
    return ErrorHandler.node_wrapper(node_func, node_name=node_name)

