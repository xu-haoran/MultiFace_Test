"""Short source-based review for conceptual work and unverified terminology."""
import json
import re
from utils.deps import get_deps
from utils.llm_retry import chat_prefilled, completion_metadata
from utils.flow_policy import candidate_complete, knowledge_candidate_eligible
from utils.problem_profile import is_objective_mode, normalize_objective_answer
from utils.shared_constraints import problem_constraints, terminology_ambiguities
from utils.skill_excerpt import select_skill_excerpt, selected_card_ids
from utils.token_budget import estimate_tokens


def knowledge_review_node(state, config):
    mode = state.get("question_mode", "computation")
    po = state.get("python_output") or {}
    task = state.get("verification_task") or {}
    if task.get("kind") != "knowledge" and not (
            knowledge_candidate_eligible(state.get("problem", ""), mode) and po.get("semantic_candidate")):
        return {"knowledge_review": {"status": "not_required"}}
    problem = state.get("problem", "")
    ambiguities = terminology_ambiguities(problem, mode)
    rr = state.get("reasoning_result") or {}
    alternative = po.get("semantic_candidate", "")
    from utils.objective_verifier import objective_answers_equal
    conflict = bool(alternative and rr.get("answer") and not objective_answers_equal(
        alternative, rr["answer"], mode))
    if not ambiguities and not conflict:
        return {"knowledge_review": {"status": "not_required", "reason": "no_conflict_or_ambiguity"}}
    deps = get_deps(config)
    if deps.time_budget and deps.time_budget.closeout_started():
        return {"knowledge_review": {"status": "inconclusive", "reason": "closeout_reserved"}}
    candidates = {"reasoning": rr.get("answer", "")}
    if is_objective_mode(mode) and po.get("semantic_candidate"):
        candidates["python_semantic"] = po["semantic_candidate"]
    candidates = {k: v for k, v in candidates.items() if candidate_complete(problem, v, mode)}
    sources = {}
    qualified_reference_ids = set()
    try:
        excerpt = select_skill_excerpt(deps.skills_loader.get_skill_document(state["category"]), problem, 2200)
        if excerpt:
            sources["skill"] = excerpt
            for card in re.finditer(r"(?ms)^## 知识卡：([^\s|]+)[^\n]*\n.*?<!-- END_CARD \1 -->", excerpt):
                sources[card.group(1)] = card.group(0)
    except Exception:
        pass
    for prefix, records in (("reference", state.get("retrieved_examples") or []),):
        for i, record in enumerate(records[:5]):
            if isinstance(record, dict) and record.get("relation") != "irrelevant":
                sources[f"{prefix}_{i}"] = json.dumps(record, ensure_ascii=False, default=str)
                original = record.get("problem") or record.get("question") or ""
                if prefix == "reference" and record.get("answer_safe") is True and re.sub(
                        r"\s+", "", str(original)) == re.sub(r"\s+", "", problem):
                    qualified_reference_ids.add(f"{prefix}_{i}")
    if not sources or not candidates:
        return {"knowledge_review": {"status": "unavailable", "reason": "missing_sources_or_candidates", "ambiguities": ambiguities}}
    prompt = (
        "根据给定来源短核对知识候选，不运行程序。来源也可能含歧义，"
        "不能把技能模板当官方答案键。检查选项/空位与术语层级，必要时弃权。"
        "只输出 JSON：status 为 supported/ambiguous/inconclusive；selected 为候选 ID 或空串；"
        "citations 为 [{source:来源ID, quote:来源中逐字原文}]；reason 为简短推理。"
        "supported 必须有直接相关原文及解释，单纯重复候选不算支持。"
        "填空题候选术语错误且来源有直接答案时，可 selected=source_derived 并提供 answer；"
        "此时每个空位的完整术语都必须逐字见于引用原文，不能从泛称猜测教材限定。"
        + problem_constraints(problem)
        + "\n题目：" + problem + "\n候选：" + json.dumps(candidates, ensure_ascii=False)
        + "\n来源：" + json.dumps(sources, ensure_ascii=False)
    )
    try:
        raw = chat_prefilled(deps.client, messages=[{"role": "user", "content": prompt}], prefix="{",
                              temperature=0, max_tokens=1000, logger=deps.logger,
                              time_budget=deps.time_budget, expected_call_seconds=60, label="knowledge_review")
        if deps.token_budget:
            deps.token_budget.consume(estimate_tokens(prompt), estimate_tokens(raw))
        review = json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", str(raw).strip()))
        review["raw_status"] = review.get("status")
        review["raw_selected"] = review.get("selected")
        citations = review.get("citations") or []
        valid = bool(citations) and all(
            isinstance(c, dict) and c.get("source") in sources and len(str(c.get("quote") or "")) >= 6
            and str(c["quote"]) in sources[c["source"]] for c in citations)
        selected = review.get("selected")
        if selected == "source_derived" and mode == "fill" and valid:
            derived = normalize_objective_answer(str(review.get("answer") or ""), mode)
            quoted = "\n".join(c["quote"] for c in citations)
            terms = [t.strip() for t in re.split(r"[；;、\n]", derived) if t.strip()]
            if terms and all(term in quoted for term in terms) and candidate_complete(problem, derived, mode):
                candidates["source_derived"] = derived
        if review.get("status") != "supported" or selected not in candidates or not valid or not review.get("reason"):
            review["validation_rejections"] = [reason for failed, reason in (
                (not valid, "invalid_citation"), (selected not in candidates, "unknown_candidate"),
                (not review.get("reason"), "missing_reason")) if failed]
            review["status"] = "ambiguous" if review.get("status") == "ambiguous" else "inconclusive"
            review["selected"] = ""
        else:
            review["answer"] = normalize_objective_answer(candidates[selected], mode) if is_objective_mode(mode) else candidates[selected]
        review["kind"] = "source_based_semantic_review"
        review["skill_source"] = "resources/skills"
        review["skill_cards"] = selected_card_ids(sources.get("skill", ""))
        review["ambiguities"] = ambiguities
        if ambiguities and not any(c.get("source") in qualified_reference_ids for c in citations):
            review["status"] = "ambiguous"
            review["selected"] = ""
            review.pop("answer", None)
        review["python_verified"] = False
        review["completion_metadata"] = completion_metadata(raw)
        return {"knowledge_review": review}
    except Exception as exc:
        return {"knowledge_review": {"status": "inconclusive", "reason": str(exc)[:200], "ambiguities": ambiguities}}
