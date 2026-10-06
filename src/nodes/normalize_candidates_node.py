"""Standardize both branches once before each validation pass."""
from utils.flow_policy import candidate_complete, knowledge_candidate_eligible
from utils.problem_profile import is_objective_mode, normalize_objective_answer
from utils.verification_evidence import parse_verification_evidence, python_answer_is_trusted


def normalize_candidates_node(state, config):
    mode = state.get("question_mode", "computation")
    problem = state.get("problem", "")
    rr = dict(state.get("reasoning_result") or {})
    po = dict(state.get("python_output") or {})
    if is_objective_mode(mode):
        rr["answer"] = normalize_objective_answer(rr.get("answer", ""), mode)
    if "python_verification_trusted" not in po:
        po = parse_verification_evidence(
            po, candidate_answer="" if is_objective_mode(mode) else rr.get("answer", ""),
            code=state.get("python_code", ""), problem=problem,
        )
    if is_objective_mode(mode):
        po["answer"] = normalize_objective_answer(po.get("answer", ""), mode)
    if not knowledge_candidate_eligible(problem, mode):
        po["semantic_candidate"] = ""
    elif not python_answer_is_trusted(po):
        po["semantic_candidate"] = po.get("answer", "")
    reasoning_ok = candidate_complete(problem, rr.get("answer", ""), mode)
    python_ok = bool(po.get("success") and python_answer_is_trusted(po)
                     and candidate_complete(problem, po.get("answer", ""), mode))
    if is_objective_mode(mode):
        python_ok = python_ok and bool(po.get("objective_independent"))
    return {
        "reasoning_result": rr, "python_output": po,
        "candidate_health": {"reasoning": reasoning_ok, "python": python_ok},
        "python_evidence_status": po.get("evidence_status", "inconclusive"),
        "python_evidence_summary": po.get("evidence_summary", ""),
        "python_contradictions": po.get("contradictions", []),
        "validated_answer": "", "answer_locked": False,
        "semantic_arbiter_decision": "", "next_node": None,
    }
