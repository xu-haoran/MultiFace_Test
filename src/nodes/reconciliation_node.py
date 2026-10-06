"""Deterministic repair routing using cached references.

Single-branch failures repair only that branch. Mathematical conflicts recheck
both independently; exhausted rounds go to selection among existing candidates.
"""
from utils.reconciliation_policy import reconciliation_round_limit
from utils.problem_profile import is_objective_mode


def _bounded(value, limit=1000):
    text = str(value or "").strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "...[truncated]"


def reconciliation_node(state, config):
    round_num = state.get("reconciliation_round", 0) + 1
    max_rounds = reconciliation_round_limit(config)
    recon_trace = list(state.get("reconciliation_trace") or [])

    po = state.get("python_output") or {}
    rr = state.get("reasoning_result") or {}
    details = state.get("validation_details") or {}
    evidence_summary = (
        details.get("python_evidence_summary")
        or state.get("python_evidence_summary")
        or po.get("evidence_summary")
        or ""
    )
    contradictions = details.get("python_contradictions") or state.get("python_contradictions") \
        or po.get("contradictions") or []
    evidence_context = _bounded(evidence_summary or "；".join(map(str, contradictions)))
    candidate = _bounded(rr.get("answer", ""), 2000)
    objective = is_objective_mode(state.get("question_mode", ""))
    target = details.get("repair_target")

    if round_num >= max_rounds:
        recon_trace.append({"round": round_num, "action": "semantic_arbiter"})
        result = {
            "reconciliation_round": round_num,
            "reconciliation_trace": recon_trace,
            "reasoning_retry_hint": None,
            "python_retry_hint": None,
            "next_node": "semantic_arbiter",
            "retry_targets": [],
            "validation_status": "reconciliation_exhausted_arbitrating",
        }
        if evidence_summary or contradictions or details.get("unresolved_contradiction"):
            result["validation_history"] = [{
                "round": round_num,
                "status": "unresolved_contradiction",
                "evidence_status": details.get(
                    "python_evidence_status",
                    state.get("python_evidence_status", "inconclusive"),
                ),
                "evidence_summary": _bounded(evidence_context),
                "contradictions": list(contradictions)[:10],
            }]
        return result

    rs_hint = None
    py_hint = None
    if target in {"reasoning", "python"}:
        hint = ("上一轮该分支失败或答案不完整。请依据原题重新求解，覆盖全部问项。"
                + (f"执行错误：{_bounded(po.get('stderr'), 500)}" if target == "python" else ""))
        recon_trace.append({"round": round_num, "action": f"repair_{target}",
                            "targets": [target], "references_reused": True})
        return {
            "reconciliation_round": round_num, "reconciliation_trace": recon_trace,
            "reasoning_retry_hint": hint if target == "reasoning" else None,
            "python_retry_hint": hint if target == "python" else None,
            "retry_targets": [target], "next_node": "solving",
        }
    if not po.get("success"):
        err = (po.get("stderr") or "")[:300]
        py_hint = (f"上一次 Python 代码执行失败。错误：{err}。"
                   f"请用 sympy 重新生成正确的 ```python``` 代码，使用固定 solve()/verify() 接口返回 answer、checks、evidence。")
        action = "retry_python"
    elif po.get("answer"):
        # 不把 Python 答案灌给推理侧（Python 可能因 API 误用而错，如 jordan_form 解包
        # 反序——若锚定会把正确推理带偏）；两侧独立复核。
        py_ans = po.get("answer")
        if objective:
            recon_trace.append({"round": round_num, "action": "retry_both_independent_objective"})
            return {
                "reconciliation_round": round_num,
                "reconciliation_trace": recon_trace,
                "reasoning_retry_hint": "客观题答案冲突，请完全依据题面独立逐项核对，不要参考上一轮答案。",
                "python_retry_hint": "客观题答案冲突，请独立重算并重新生成验证代码，不要参考上一轮答案。",
                "next_node": "solving",
                "retry_targets": ["reasoning", "python"],
            }
        rs_hint = ("你之前的推理答案与独立程序验证结果不一致。请重新独立推理，逐步复核每一步"
                   "关键计算，不要参考任何外部答案；严格按原格式输出（含 '## 问题分析'、"
                   "'## 详细解题步骤'、'## 最终答案' 章节），'## 最终答案' 后给出复核后的明确结果。")
        py_hint = (f"上一次代码计算结果为：{py_ans}，与理论推导不一致，代码逻辑可能有误。"
                   "请重点检查：多返回值函数的解包顺序（如 sympy 的 A.jordan_form() 返回 (P, J)，"
                   "Jordan 标准形是第二个返回值；A.diagonalize() 返回 (P, D)）、公式实现是否与题意"
                   "一致。修正后只输出一个 ```python``` 代码块，使用固定 solve()/verify() 接口返回 answer、checks、evidence。")
        if candidate:
            rs_hint += f" 当前待复核推理候选为：{candidate}。"
        if evidence_context:
            py_hint += f" 上一轮验证证据（必须独立核对）：{evidence_context}。"
        py_hint += " 请先打印验证状态: PASS|FAIL|INCONCLUSIVE 和验证证据，再打印最终答案。"
        action = "retry_both"
    else:
        rs_hint = "请重新推理，给出明确的 '## 最终答案'。"
        py_hint = "请重新生成代码，确保最后一行 print(\"最终答案:\", answer)。"
        action = "retry_both"

    recon_trace.append({"round": round_num, "action": action})
    return {
        "reconciliation_round": round_num,
        "reconciliation_trace": recon_trace,
        "reasoning_retry_hint": rs_hint,
        "python_retry_hint": py_hint,
        "next_node": "solving",
        "retry_targets": (["reasoning", "python"]
                          if state.get("python_verification_enabled", True) else ["reasoning"]),
    }

