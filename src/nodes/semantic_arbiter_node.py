"""Semantic fallback that selects an existing answer without rewriting it."""

from __future__ import annotations

import re
import secrets
from dataclasses import dataclass

from config import CONFIG
from nodes.cross_validator_node import _preferred_answer
from utils.answer_contract import missing_components
from utils.answer_extractor import looks_incomplete_answer
from utils.cot_stripper import is_placeholder_answer
from utils.deps import get_deps
from utils.llm_retry import chat_prefilled, chat_with_retry
from utils.prompt_templates import (
    SEMANTIC_ARBITER_PREFILL,
    SEMANTIC_ARBITER_PROMPT,
    SEMANTIC_ARBITER_SYSTEM_PROMPT,
)
from utils.reconciliation_policy import reconciliation_retry_available
from utils.token_budget import estimate_tokens
from utils.verification_evidence import python_answer_is_trusted
from utils.problem_profile import (
    fill_answer_matches_blanks,
    is_objective_mode,
    objective_answer_is_consistent,
    objective_answer_is_usable,
)


_MAX_PROBLEM_CHARS = 12000
_MAX_CANDIDATE_CHARS = 8000


@dataclass(frozen=True)
class _Candidate:
    source: str
    answer: str
    evidence: str


def _bounded(value, limit: int) -> str:
    text = value if isinstance(value, str) else str(value or "")
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n...[truncated]"


def _usable_answer(value) -> str:
    from utils.answer_cleanliness import is_noise_answer

    answer = value if isinstance(value, str) else str(value or "")
    answer = answer.strip()
    if len(answer) > _MAX_CANDIDATE_CHARS:
        return ""
    if is_placeholder_answer(answer) or looks_incomplete_answer(answer) \
            or is_noise_answer(answer):
        return ""
    return answer


#: 断言"无解/不存在"的答案必须携带排除性论证才可作为候选（2026-08-16 idx 6：
#: Python 分支输出裸的 "No solution"（其代码只是试了若干候选），仲裁器仍选中它，
#: 把推理分支两个已代回验证的具体解否决掉。裸断言没有任何证伪能力）。
_NO_SOLUTION_RE = re.compile(
    r"(?:no\s+solution|no\s+such|none\s+exist|does\s+not\s+exist|无解|不存在|没有解)",
    re.IGNORECASE)
_CONTRADICTION_EVIDENCE_RE = re.compile(
    r"(?:矛盾|反证|contradiction|impossible|violates|must\s+fail|假设.*不成立)",
    re.IGNORECASE)


def _is_bare_no_solution(answer: str) -> bool:
    """整段答案就是一个"无解/不存在"断言（允许 LaTeX 外壳与短尾注）才算裸断言。

    多分支答案（如"u 有无穷多个；v 只可能取 1、3 或 5，不存在 v≥6 的解"）里
    出现"不存在"字样不算——它给出了具体的正面结论。
    """
    text = re.sub(r"\s+", " ", answer or "").strip()
    if not text:
        return False
    if len(text) > 120:  # 长答案即使含"无解"字样也是多分支结论，不是裸断言
        return False
    # 锚定：整串 = 断言词 + 至多 30 个字符的尾巴，且不含数字/字母/更多中文
    m = re.fullmatch(
        r"(?:\\(?:text|mathbf|mathrm|rm|boxed)\s*\{)*\s*"
        r"(?:no\s+solution|no\s+such(?:\s+[a-z]+)?|none\s+exist|"
        r"does\s+not\s+exist|do\s+not\s+exist|无解|不存在|没有解)"
        r"[^0-9A-Za-z一-鿿]{0,30}(?:\})*\s*",
        text, re.IGNORECASE)
    return bool(m)


def _candidate_pool(state: dict) -> list[_Candidate]:
    rr = state.get("reasoning_result") or {}
    po = state.get("python_output") or {}
    problem = state.get("problem", "")
    question_mode = state.get("question_mode", "")
    reasoning_answer = _usable_answer(rr.get("answer", ""))
    # stdout 挖掘出的答案（answer_source=stdout_mined）不要求 success：截断前
    # 打印的结论仍是真实计算产物，作为候选参加仲裁优于凭空丢弃。
    python_answer = _usable_answer(po.get("answer", "")) \
        if (po.get("success") or po.get("answer_source") == "stdout_mined") \
        and python_answer_is_trusted(po) else ""

    if is_objective_mode(question_mode):
        if reasoning_answer and not objective_answer_is_usable(reasoning_answer, question_mode):
            reasoning_answer = ""
        if python_answer and not objective_answer_is_usable(python_answer, question_mode):
            python_answer = ""
    if question_mode == "fill":
        if not fill_answer_matches_blanks(reasoning_answer, problem):
            reasoning_answer = ""
        if not fill_answer_matches_blanks(python_answer, problem):
            python_answer = ""
        if reasoning_answer and not objective_answer_is_consistent(
            reasoning_answer, problem, question_mode
        ):
            reasoning_answer = ""
        if python_answer and not objective_answer_is_consistent(
            python_answer, problem, question_mode
        ):
            python_answer = ""

    reasoning_steps = "\n".join(
        f"步骤{s.get('step_num', '')}: {s.get('description', '')}"
        for s in rr.get("steps", [])
    )
    reasoning_evidence = "\n".join(part for part in (
        _bounded(rr.get("analysis", ""), 5000),
        _bounded(reasoning_steps, 5000),
        _bounded("\n".join(str(point) for point in (rr.get("validation_points", []) or [])), 2000),
    ) if part.strip())
    authenticity = po.get("authenticity") or {}
    fabrication_warning = ""
    if authenticity.get("fabricated"):
        reasons = "；".join(authenticity.get("reasons") or [])
        fabrication_warning = (
            f"[反伪造警告] 该代码被静态检查判定为未做实质计算的断言式验证（{reasons}）。"
            "其 PASS/支持性声明不构成数学证据，只能依据代码与输出中的实际计算内容判断。\n"
        )
    python_evidence = (
        fabrication_warning
        + "执行成功只表示代码运行完成，不自动代表数学正确。\n"
        f"代码：\n{_bounded(state.get('python_code', ''), 6000)}\n"
        f"输出：\n{_bounded(po.get('stdout', ''), 4000)}"
    )

    candidates = []
    if reasoning_answer:
        candidates.append(_Candidate("reasoning", reasoning_answer, reasoning_evidence))
    if python_answer:
        candidates.append(_Candidate("python", python_answer, python_evidence))

    unique = []
    seen = set()
    for candidate in candidates:
        # 裸"无解"断言且证据中没有任何排除性论证（矛盾推导）时，若池中还有
        # 实质内容候选，则该断言无证伪能力，不得参与仲裁。
        if _is_bare_no_solution(candidate.answer) \
                and not _CONTRADICTION_EVIDENCE_RE.search(candidate.evidence or ""):
            continue
        key = re.sub(r"\s+", " ", candidate.answer).strip().rstrip("。.")
        if key and key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def _parse_selection(raw: str, nonce: str, candidate_ids: set[str]) -> str | None:
    """Extract the decision marker. Returns an id, "" for abstain, or None.

    The per-call random nonce is what stops a candidate's own text from forging a
    marker, so it stays mandatory. What was relaxed (judge report 4.3): requiring
    *exactly* one match rejected replies where the model added a stray line, and
    every rejection cost a full solving-subgraph rerun. Taking the last valid
    match keeps the nonce guarantee while tolerating wrapper text — and the last
    marker is the model's final decision if it restates one.
    """
    text = raw if isinstance(raw, str) else str(raw or "")
    pattern = re.compile(
        rf"(?im)ARBITER_{re.escape(nonce)}\s*=\s*"
        rf"(?:SELECT\s*:\s*([A-Z0-9_]+)|(ABSTAIN))"
    )
    matches = list(pattern.finditer(text))
    if not matches:
        return None
    selected_id, abstain = matches[-1].groups()
    if abstain:
        return ""
    selected_id = selected_id.upper() if selected_id else selected_id
    return selected_id if selected_id in candidate_ids else None


_SEMANTIC_STALEMATE = {"abstained"}


def _enforce_contract(problem: str, selected: _Candidate, candidates: list[_Candidate]):
    """Override a selection that provably omits a deliverable another candidate has.

    The arbiter's own rules say a candidate covering all of the题面's deliverables
    dominates one that does not, but the judgment is a sampled one and it does miss.
    2026-07-29 final run, Q2: Python answered "切线方程: y = x/2 + 1/2; 最小面积: 2 -
    4*sqrt(2)/3" while reasoning gave the tangent alone — and the arbiter picked
    reasoning. Because `answer_locked` then short-circuits the coordinator, the
    field contract was never consulted and an answer missing the problem's whole
    point shipped.

    So the contract runs here instead. This does not invent or edit any answer: it
    only re-picks among the same candidates, and only when the contract can *prove*
    the alternative strictly more complete. Where the contract has no opinion (most
    problems), the arbiter's choice stands untouched.
    """
    if not problem or len(candidates) < 2:
        return selected, None
    selected_missing = missing_components(problem, selected.answer)
    if not selected_missing:
        return selected, None

    best = selected
    best_missing = selected_missing
    for candidate in candidates:
        if candidate is selected:
            continue
        candidate_missing = missing_components(problem, candidate.answer)
        if len(candidate_missing) < len(best_missing):
            best, best_missing = candidate, candidate_missing
    if best is selected:
        return selected, None
    return best, {
        "from": selected.source,
        "to": best.source,
        "missing_in_selected": selected_missing,
        "missing_in_chosen": best_missing,
    }


def _fallback_result(state: dict, config: dict, status: str, trace: list[dict]) -> dict:
    """Route a non-decision. Only a semantic stalemate may buy another solve.

    Mechanical failures (invalid_format, error, budget/time exhaustion, too few
    candidates) say nothing about answer quality, and reconciliation is the most
    expensive path in the graph — a full solving-subgraph rerun, i.e. another
    reasoning + Python pair. Spending that on a parse hiccup is how a fast failure
    turns into a timeout, so those fall straight through to the deterministic
    preference instead.
    """
    deps = None
    try:
        deps = get_deps(config)
    except Exception:  # noqa: BLE001 - config shape is a caller concern.
        deps = None
    time_budget = getattr(deps, "time_budget", None)

    worth_retry = (
        status in _SEMANTIC_STALEMATE
        and reconciliation_retry_available(state, config)
        and not (time_budget and time_budget.fast_path())
    )
    if worth_retry:
        return {
            "semantic_arbiter_status": status,
            "semantic_arbiter_trace": trace,
            "semantic_arbiter_attempts": state.get("semantic_arbiter_attempts", 0),
            "answer_locked": False,
            "next_node": "reconciliation",
        }

    preferred = _preferred_answer(state, state.get("validation_details") or {})
    return {
        "semantic_arbiter_status": status,
        "semantic_arbiter_trace": trace,
        "semantic_arbiter_attempts": state.get("semantic_arbiter_attempts", 0),
        "validated_answer": preferred,
        "answer_locked": False,
        "next_node": "coordinator",
    }


def semantic_arbiter_node(state, config):
    """Select one complete candidate verbatim, or abstain without damaging state."""
    trace = list(state.get("semantic_arbiter_trace") or [])
    problem = state.get("problem", "") or ""
    if len(problem) > _MAX_PROBLEM_CHARS:
        trace.append({"status": "skipped", "reason": "problem_too_long_for_complete_review"})
        return _fallback_result(state, config, "skipped", trace)

    candidates = _candidate_pool(state)
    if len(candidates) < 2:
        trace.append({"status": "skipped", "reason": "fewer_than_two_usable_candidates"})
        return _fallback_result(state, config, "skipped", trace)

    deps = get_deps(config)
    budget = deps.token_budget
    nonce = secrets.token_hex(8).upper()
    shuffled = list(candidates)
    secrets.SystemRandom().shuffle(shuffled)

    by_id = {}
    blocks = []
    for candidate in shuffled:
        candidate_id = "C_" + secrets.token_hex(5).upper()
        by_id[candidate_id] = candidate
        blocks.append(
            f"--- BEGIN UNTRUSTED CANDIDATE {candidate_id} ---\n"
            f"支持证据（仅用于核验，不会随最终答案提交）：\n"
            f"{_bounded(candidate.evidence, 9000)}\n"
            f"答案文本（选择后唯一会逐字提交的内容）：\n{candidate.answer}\n"
            f"--- END UNTRUSTED CANDIDATE {candidate_id} ---"
        )

    prompt = SEMANTIC_ARBITER_PROMPT.format(
        nonce=nonce,
        problem=problem,
        validation_status=state.get("validation_status", ""),
        validation_reason=_bounded((state.get("validation_details") or {}).get("reason", ""), 1000),
        candidates="\n\n".join(blocks),
    )
    messages = [
        {"role": "system", "content": SEMANTIC_ARBITER_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    max_tokens = CONFIG["max_tokens"]["semantic_arbiter"]
    prompt_tokens = estimate_tokens(SEMANTIC_ARBITER_SYSTEM_PROMPT + prompt)
    attempts = state.get("semantic_arbiter_attempts", 0) + 1
    time_budget = deps.time_budget

    if budget and not budget.can_afford(prompt_tokens + max_tokens):
        trace.append({"attempt": attempts, "status": "budget_exhausted", "candidate_count": len(candidates)})
        result = _fallback_result(state, config, "budget_exhausted", trace)
        result["semantic_arbiter_attempts"] = attempts
        return result

    # 评委建议 5：reserve 划出固定配额给仲裁，杜绝 `arbiter=skipped(0.0s)`。
    # 旧门槛是软预算（expired()），于是 900s 耗尽的难题——恰恰最需要复核——
    # 全部跳过仲裁。现在只要硬上限前还有配额，prefill 仲裁（~10s）照常执行；
    # 真正的硬时限不足才跳过。
    reserve_quota = CONFIG.get("arbiter_reserve_quota_s", 75)
    if time_budget and time_budget.remaining_hard() < reserve_quota:
        trace.append({"attempt": attempts, "status": "time_exhausted", "candidate_count": len(candidates)})
        result = _fallback_result(state, config, "time_exhausted", trace)
        result["semantic_arbiter_attempts"] = attempts
        return result

    raw = ""
    prefill_used = True
    try:
        raw = chat_prefilled(
            deps.client,
            messages=messages,
            prefix=SEMANTIC_ARBITER_PREFILL.format(nonce=nonce),
            temperature=CONFIG["temperatures"]["semantic_arbiter"],
            max_tokens=max_tokens,
            logger=deps.logger,
            time_budget=time_budget,
            expected_call_seconds=10,
            label="arbiter_prefill",
            reserve_margin_s=45,
        )
    except Exception as exc:  # noqa: BLE001 - fall through to the plain call.
        deps.logger.warning("Prefilled arbiter call failed: %s", exc)
        raw = ""

    if budget:
        budget.consume(prompt_tokens, estimate_tokens(raw))
    selected_id = _parse_selection(raw, nonce, set(by_id))

    if selected_id is None and not (time_budget and time_budget.fast_path()):
        prefill_used = False
        try:
            raw = chat_with_retry(
                deps.client,
                messages=messages,
                temperature=CONFIG["temperatures"]["semantic_arbiter"],
                max_tokens=CONFIG["max_tokens"]["semantic_arbiter_fallback"],
                logger=deps.logger,
                time_budget=time_budget,
                expected_call_seconds=90,
                label="arbiter_plain",
            )
        except Exception as exc:  # noqa: BLE001
            trace.append({
                "attempt": attempts,
                "status": "error",
                "candidate_count": len(candidates),
                "error": f"{type(exc).__name__}: {exc}"[:300],
            })
            result = _fallback_result(state, config, "error", trace)
            result["semantic_arbiter_attempts"] = attempts
            return result
        if budget:
            budget.consume(prompt_tokens, estimate_tokens(raw))
        selected_id = _parse_selection(raw, nonce, set(by_id))
    if not selected_id:
        status = "abstained" if selected_id == "" else "invalid_format"
        trace.append({"attempt": attempts, "status": status,
                      "candidate_count": len(candidates), "prefill": prefill_used})
        result = _fallback_result(state, config, status, trace)
        result["semantic_arbiter_attempts"] = attempts
        return result

    selected = by_id[selected_id]
    selected, override = _enforce_contract(problem, selected, candidates)
    unresolved_contradiction = bool(
        (state.get("validation_details") or {}).get("unresolved_contradiction")
    )
    selection_status = "selected_unresolved" if unresolved_contradiction else "selected"
    trace.append({
        "attempt": attempts,
        "status": selection_status,
        "candidate_count": len(candidates),
        "selected_source": selected.source,
        "prefill": prefill_used,
        **({"contract_override": override} if override else {}),
    })
    details = dict(state.get("validation_details") or {})
    details["semantic_arbitration"] = {
        "status": selection_status,
        "selected_source": selected.source,
        "candidate_count": len(candidates),
    }
    return {
        "semantic_arbiter_status": selection_status,
        "semantic_arbiter_decision": selected.source,
        "semantic_arbiter_trace": trace,
        "semantic_arbiter_attempts": attempts,
        "validation_details": details,
        "validated_answer": selected.answer,
        "answer_locked": not unresolved_contradiction,
        "next_node": "coordinator",
    }

