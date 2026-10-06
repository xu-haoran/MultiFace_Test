"""交叉验证：比较推理和 Python 结果并决定后续路径。"""
from utils.answer_matcher import AnswerMatcher
from utils.answer_contract import answer_part_count, missing_components
from utils.answer_extractor import is_multi_part_problem, looks_incomplete_answer
from utils.cot_stripper import is_placeholder_answer
from utils.reconciliation_policy import reconciliation_retry_available
from utils.verification_evidence import (
    parse_verification_evidence,
    python_answer_is_trusted,
)
from utils.problem_profile import (
    classify_question_mode,
    fill_answer_matches_blanks,
    is_objective_mode,
    normalize_objective_answer,
    objective_answer_consistency,
    objective_answer_is_consistent,
    objective_answer_is_usable,
)
from utils.objective_verifier import (
    canonical_objective_answer,
    objective_code_is_independent,
    objective_answers_equal,
)
from config import CONFIG
from utils.flow_policy import candidate_complete


def _clean_answer(answer: str) -> str:
    return "" if is_placeholder_answer(answer) else (answer or "")


def _python_trusted(python_output: dict) -> bool:
    """Python 分支是否真的验证过——判断集中在 verification_evidence 里。

    保留这个名字是因为它已是本模块的既有词汇；实现委托给唯一权威入口，避免
    各消费点再各写一份等价判断而漏掉其中之一。
    """
    return python_answer_is_trusted(python_output)


def _same_scalar_or_text(a: str, b: str) -> bool:
    """两个候选是否本质相同（去空白/标点后一致）。"""
    import re as _re
    ka = _re.sub(r"[\s。．.,，;；]+", "", str(a or ""))
    kb = _re.sub(r"[\s。．.,，;；]+", "", str(b or ""))
    return bool(ka) and ka == kb


def _preferred_answer(state: dict, match_result: dict) -> str:
    """Pick the answer to surface as validated_answer.

    证据优先序（2026-08-09 评委建议 1，最高优先级）：
    ``有枚举支撑且反驳对手的 Python 答案 > evidence=support 的分支 > 推理答案 >
    结构化部分结论``。idx 20 事故——Python 枚举出 C(2k,k)² 且 evidence=contradict
    明确反驳了推理的 "4"，仲裁超时后系统仍输出被自家验证器反驳的 "4"——由第一
    条规则在架构上杜绝：证据指向谁，兜底就选谁。

    Computation + python_success + 答案完整 → Python answer (sympy, deterministic).
    Python 答案为空/碎片（评委报告问题 2：'(Matrix([' 等截断片段曾污染
    final_response）→ 回退 reasoning answer；两者都不完整时取非空者兜底。

    uncertain 时保持推理优先（去锚定），但 2026-07-07 评委报告显示：推理侧漏
    结论/漏问项/漏分配而 Python 侧完整时（265/271/283/93/113/275），残缺答案
    仍然胜出。故当推理答案缺契约字段或漏问项、且 Python 答案更完整时改选 Python。
    """
    rr = state.get("reasoning_result") or {}
    po = state.get("python_output") or {}
    problem = state.get("problem", "")
    ptype = match_result.get("problem_type", "computation")
    question_mode = state.get("question_mode") or ptype
    python_answer = _clean_answer(po.get("answer", ""))
    reasoning_answer = _clean_answer(rr.get("answer", ""))
    if question_mode == "fill":
        # 多空答案不完整时只留作诊断，不参与选择。
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
    # 未验证即不可信（2026-08-19 idx=0）：Python 分支没做实质计算、或把结论抄进
    # print 时，它不是一个"验证过的答案"，不得参与答案选择——无论推理侧是否可用。
    # 该分支仍保留在 trace 与证据字段里，只是失去出厂资格。
    if not _python_trusted(po):
        python_answer = ""
    python_ok = bool(po.get("success")) and bool(python_answer)
    reasoning_ok = bool(reasoning_answer) and not looks_incomplete_answer(reasoning_answer)
    if po.get("evidence_status") == "contradict" and python_ok and candidate_complete(
            problem, python_answer, question_mode) and not _same_scalar_or_text(python_answer, reasoning_answer):
        return normalize_objective_answer(python_answer, question_mode) if is_objective_mode(question_mode) else python_answer
    if is_objective_mode(question_mode):
        # 所有降级路径都保持规范答案格式，不能返回原始推理散文。
        for candidate in (reasoning_answer, python_answer):
            normalized = normalize_objective_answer(candidate, question_mode)
            if objective_answer_is_usable(normalized, question_mode):
                return normalized
        return ""
    # 高相似例题只作为推理参考，不提取或锁定答案；冲突时按验证证据选择。
    # 证据优先的最后防线：Python 证据明确反驳推理候选、且 Python 自己持有一个
    # 不同的、可用的、非伪造的答案时，兜底必须选 Python——被自家验证器反驳的
    # 答案不得再被输出。仅在此条件下打破"推理优先"，避免污染无争议路径。
    evidence_status = po.get("evidence_status", "")
    fabricated = bool((po.get("authenticity") or {}).get("fabricated"))
    if evidence_status == "contradict" and python_answer and not fabricated \
            and not looks_incomplete_answer(python_answer) \
            and not _same_scalar_or_text(python_answer, reasoning_answer):
        return python_answer
    if match_result.get("status") == "match" and reasoning_ok:
        # Python/SymPy 提供验证证据，保留已验证的可读推理答案。
        return reasoning_answer
    if match_result.get("status") == "uncertain" and reasoning_ok:
        if python_ok and not looks_incomplete_answer(python_answer):
            r_missing = missing_components(problem, reasoning_answer)
            p_missing = missing_components(problem, python_answer)
            if len(p_missing) < len(r_missing):
                return python_answer
            if is_multi_part_problem(problem) \
                    and answer_part_count(python_answer) > answer_part_count(reasoning_answer):
                return python_answer
        return reasoning_answer
    if ptype == "computation" and python_ok and not looks_incomplete_answer(python_answer):
        # 反伪造：无实质计算的代码答案不得凭"执行成功"压过一个可用的推理答案。
        if not (fabricated and reasoning_ok):
            return python_answer
    if reasoning_ok:
        return reasoning_answer
    return reasoning_answer or python_answer


def cross_validator_node(state, config):
    # Normalize once per pass; review is an optional check inside validation.
    from nodes.normalize_candidates_node import normalize_candidates_node
    from nodes.knowledge_review_node import knowledge_review_node
    normalized = normalize_candidates_node(state, config)
    state = {**state, **normalized}
    reviewed = knowledge_review_node(state, config)
    state = {**state, **reviewed}
    result = _validate_candidates(state, config)
    return {**normalized, **reviewed, **result}


def _validate_candidates(state, config):
    reasoning_result = state.get("reasoning_result") or {}
    raw_python_output = state.get("python_output") or {}
    question_mode = state.get("question_mode") or classify_question_mode(state.get("problem", ""))
    python_output = raw_python_output if "python_verification_trusted" in raw_python_output else parse_verification_evidence(
        raw_python_output,
        candidate_answer=("" if is_objective_mode(question_mode)
                          else reasoning_result.get("answer", "")),
        code=state.get("python_code", ""),
        problem=state.get("problem", ""),
    )
    if is_objective_mode(question_mode):
        # 客观题走专用共识路径；Python 结果必须通过独立性校验。
        reasoning_candidate = canonical_objective_answer(
            reasoning_result.get("answer", ""), question_mode)
        python_candidate = canonical_objective_answer(
            python_output.get("answer", ""), question_mode)
        # 未通过独立性校验的程序结果只作诊断，不参与共识。
        python_independent = bool(
            python_output.get("objective_independent")
            and python_answer_is_trusted(python_output)
            and objective_code_is_independent(
                state.get("python_code", ""), question_mode)
        )
        if not python_independent:
            python_candidate = ""
        candidates = [value for value in (reasoning_candidate, python_candidate) if value]
        semantic_reason = ""
        valid = []
        for value in candidates:
            consistent, reason = objective_answer_consistency(
                value, state.get("problem", ""), question_mode)
            if consistent:
                valid.append(value)
            elif reason:
                semantic_reason = reason
        blank_gap = any(
            question_mode == "fill" and not fill_answer_matches_blanks(value, state.get("problem", ""))
            for value in valid
        )
        if reasoning_candidate and python_candidate and objective_answers_equal(
                reasoning_candidate, python_candidate, question_mode) and len(valid) == 2:
            match_result = {
                "status": "uncertain" if blank_gap else "match",
                "verdict": None if blank_gap else True,
                "comparison_verdict": None if blank_gap else True,
                "confidence": 0.72 if blank_gap else 0.93,
                "reason": "客观题推理与独立 Python 结果一致"
                          + ("，但填空覆盖可能不完整" if blank_gap else ""),
                "method": "objective_independent_consensus",
                "problem_type": question_mode,
                "text_similarity": 1.0,
                "matched_fields": ["reasoning", "python"],
                "mismatched_fields": [],
                "field_coverage": 0.5 if blank_gap else 1.0,
            }
        elif reasoning_candidate and python_candidate and not objective_answers_equal(
                reasoning_candidate, python_candidate, question_mode):
            match_result = {
                "status": "mismatch", "verdict": False, "comparison_verdict": False,
                "confidence": 0.88,
                "reason": "客观题推理与独立 Python 结果冲突",
                "method": "objective_independent_disagreement",
                "problem_type": question_mode, "text_similarity": 0.0,
                "matched_fields": [], "mismatched_fields": ["answer"],
                "field_coverage": 1.0,
            }
        elif valid:
            # 单一候选保留，但标记为不确定，继续走安全收束流程。
            candidate = valid[0]
            match_result = {
                "status": "uncertain", "verdict": None, "comparison_verdict": None,
                "confidence": 0.55 if python_candidate else 0.48,
                "reason": "客观题只有一个可用候选，缺少独立一致性证据",
                "method": "objective_single_candidate",
                "problem_type": question_mode, "text_similarity": 0.0,
                "matched_fields": ["answer"], "mismatched_fields": [],
                "field_coverage": 0.5,
            }
        else:
            match_result = {
                "status": "uncertain", "verdict": None, "comparison_verdict": None,
                "confidence": 0.0,
                "reason": ("客观题候选未通过语义核验：" + semantic_reason)
                          if semantic_reason else "客观题未提取到可提交的答案",
                "method": "objective_semantic_mismatch" if semantic_reason
                          else "objective_unparsed",
                "problem_type": question_mode, "text_similarity": 0.0,
                "matched_fields": [], "mismatched_fields": [], "field_coverage": 0.0,
            }
    else:
        match_result = dict(AnswerMatcher.match_answers(
            problem=state["problem"],
            reasoning_result=reasoning_result,
            python_result=(python_output if python_answer_is_trusted(python_output)
                           else {**python_output, "answer": "", "success": False}),
        ))
    evidence_status = python_output.get("evidence_status", "inconclusive")
    evidence_summary = python_output.get("evidence_summary", "")
    contradictions = list(python_output.get("contradictions") or [])
    match_result.update({
        "python_evidence_status": evidence_status,
        "python_evidence_summary": evidence_summary,
        "python_contradictions": contradictions,
    })
    status = match_result["status"]
    validated_answer = ""
    next_node = "coordinator"
    problem_type = match_result.get("problem_type", "computation")

    history_entry = {
        "round": state.get("reconciliation_round", 0),
        "status": status,
        "evidence_status": evidence_status,
        "evidence_summary": evidence_summary[:1000],
        "contradictions": contradictions[:10],
    }

    health = state.get("candidate_health") or {}
    review = state.get("knowledge_review") or {}
    if review.get("status") == "ambiguous":
        match_result.update({"routing_reason": "knowledge_ambiguity", "knowledge_review": review})
        return {"validation_status": "uncertain_arbitrating", "validation_details": match_result,
                "next_node": "semantic_arbiter", "validated_answer": "",
                "validation_history": [{**history_entry, "status": "uncertain_arbitrating"}]}
    if evidence_status != "contradict" and review.get("status") == "supported" and candidate_complete(
            state.get("problem", ""), review.get("answer", ""), question_mode):
        match_result.update({"status": "match", "method": "source_based_semantic_review",
                             "knowledge_review": review, "routing_reason": "knowledge_sources_support_candidate"})
        return {"validation_status": "match", "validation_details": match_result,
                "validated_answer": review["answer"], "answer_locked": True,
                "next_node": "coordinator", "validation_history": [
                    {**history_entry, "status": "match", "knowledge_review": review}]}
    semantic = python_output.get("semantic_candidate", "")
    if is_objective_mode(question_mode) and semantic and candidate_complete(
            state.get("problem", ""), semantic, question_mode) and not objective_answers_equal(
                semantic, reasoning_result.get("answer", ""), question_mode):
        match_result.update({"routing_reason": "unverified_knowledge_conflict",
                             "knowledge_review": review})
        return {"validation_status": "uncertain_arbitrating", "validation_details": match_result,
                "next_node": "semantic_arbiter", "validated_answer": "",
                "validation_history": [{**history_entry, "status": "uncertain_arbitrating"}]}
    reasoning_ok = health.get("reasoning", False)
    python_ok = health.get("python", False)
    python_enabled = state.get("python_verification_enabled", True)
    # A deliberately omitted program is not a failed branch. Keep uncertainty
    # explicit; the coordinator may submit a complete sole candidate directly.
    if evidence_status != "contradict" and not python_enabled and reasoning_ok:
        match_result.update({"status": "uncertain", "routing_reason": "python_not_required",
                             "method": "single_reasoning_candidate", "knowledge_review": review})
        history_entry["status"] = "uncertain"
        return {
            "validation_status": "uncertain", "validation_details": match_result,
            "validated_answer": reasoning_result.get("answer", ""),
            "next_node": "coordinator", "validation_history": [history_entry],
        }
    # Repair technical/coverage failure on one side without regenerating the
    # healthy candidate. A counterexample is a mathematical conflict instead.
    target = "python" if reasoning_ok else "reasoning"
    repair_state = {**state, "validation_details": {**match_result, "repair_target": target}}
    if evidence_status != "contradict" and reasoning_ok != python_ok \
            and (python_enabled or not reasoning_ok) \
            and reconciliation_retry_available(repair_state, config):
        match_result.update({"routing_reason": "single_branch_failure", "repair_target": target})
        history_entry["status"] = "branch_repairing"
        return {
            "validation_status": "branch_repairing", "validation_details": match_result,
            "validated_answer": "", "next_node": "reconciliation",
            "retry_targets": [target], "validation_history": [history_entry],
        }
    # Equality cannot certify an answer that omits requested deliverables.
    if status == "match" and not candidate_complete(
            state.get("problem", ""), reasoning_result.get("answer", ""), question_mode):
        status = "uncertain"
        match_result["status"] = status
        match_result["verdict"] = None
        match_result["reason"] = "两分支答案一致，但未覆盖题面全部要求"
        match_result["routing_reason"] = "matched_but_incomplete"

    # 反例证据优先于文本或符号一致；程序成功也可能证明候选错误。
    if evidence_status == "contradict":
        match_result["routing_reason"] = "python_evidence_contradiction"
        repair_state = {**state, "validation_details": {**match_result, "repair_target": "reasoning"}}
        if python_ok and reconciliation_retry_available(repair_state, config):
            match_result["repair_target"] = "reasoning"
            status = "mismatch_reconciling"
            next_node = "reconciliation"
        else:
            status = "mismatch_arbitrating"
            next_node = "semantic_arbiter"
            match_result["unresolved_contradiction"] = True
        history_entry["status"] = status
        return {
            "validation_status": status,
            "validation_details": match_result,
            "validated_answer": validated_answer,
            "next_node": next_node,
            "python_output": python_output,
            "python_evidence_status": evidence_status,
            "python_evidence_summary": evidence_summary,
            "python_contradictions": contradictions,
            "retry_targets": ["reasoning"] if next_node == "reconciliation" else [],
            "validation_history": [history_entry],
        }

    if status == "match":
        if problem_type == "computation" and python_output.get("success") \
                and match_result.get("method") in {"symbolic", "structured_symbolic"}:
            match_result["bypass_reason"] = "python_success_and_symbolic_equivalence"
        match_result["routing_reason"] = "validated_match"
        if match_result.get("method") in {
                "objective_direct", "objective_independent_consensus"}:
            validated_answer = normalize_objective_answer(
                reasoning_result.get("answer", ""), question_mode
            )
            if not validated_answer:
                validated_answer = normalize_objective_answer(
                    python_output.get("answer", ""), question_mode
                )
        else:
            # 传解析后的 python_output：可信度字段由 parse_verification_evidence
            # 写入，state 里的原始输出还没有它，否则可信分支会被误判为不可信。
            validated_answer = _preferred_answer(
                {**state, "python_output": python_output}, match_result)
        next_node = "coordinator"
    elif status == "mismatch":
        if reasoning_ok and python_ok:
            status = "mismatch_arbitrating"
            next_node = "semantic_arbiter"
            match_result["routing_reason"] = "complete_candidates_conflict"
        # 按整题 reconciliation_round 重试，与节点内部重试分开。
        elif reconciliation_retry_available(state, config):
            status = "mismatch_reconciling"
            next_node = "reconciliation"
        else:
            status = "mismatch_arbitrating"
            match_result["routing_reason"] = "mismatch_retry_exhausted_semantic_arbitration"
            next_node = "semantic_arbiter"
    else:  # uncertain
        # A failed verifier creates no competing mathematical candidate. After
        # the affordable targeted repair, submit the intact candidate directly.
        if reasoning_ok != python_ok:
            candidate = reasoning_result.get("answer", "") if reasoning_ok else python_output.get("answer", "")
            match_result.update({"status": "uncertain", "routing_reason": "single_candidate_after_verification",
                                 "method": "single_complete_candidate"})
            return {"validation_status": "uncertain", "validation_details": match_result,
                    "validated_answer": candidate, "next_node": "coordinator",
                    "validation_history": [{**history_entry, "status": "uncertain"}]}
        # 解析不确定时先让语义仲裁选择已有候选，也允许弃权后重跑。
        status = "uncertain_arbitrating"
        match_result["routing_reason"] = "insufficient_equivalence_evidence_semantic_arbitration"
        next_node = "semantic_arbiter"

    history_entry["status"] = status
    return {
        "validation_status": status,
        "validation_details": match_result,
        "validated_answer": validated_answer,
        "next_node": next_node,
        "python_output": python_output,
        "python_evidence_status": evidence_status,
        "python_evidence_summary": evidence_summary,
        "python_contradictions": contradictions,
        "validation_history": [history_entry],
    }
