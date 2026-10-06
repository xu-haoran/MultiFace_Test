"""Final deterministic contract check, shared by every graph output path."""
from utils.answer_formatter import _strip_answer_label, build_proof_body, format_fill_submission
from utils.flow_policy import candidate_complete, proof_process_present
from utils.problem_profile import is_objective_mode, normalize_objective_answer


def submission_node(state, config):
    mode = state.get("question_mode", "computation")
    problem = state.get("problem", "")
    answer = _strip_answer_label(str(state.get("final_response") or "").strip())
    if is_objective_mode(mode):
        answer = normalize_objective_answer(answer, mode)
        if mode == "fill" and answer:
            answer = format_fill_submission(answer)
    complete = candidate_complete(problem, answer, mode)
    if not complete:
        # Prefer an intact existing answer over malformed coordinator output.
        candidates = [state.get("validated_answer")]
        if state.get("python_evidence_status") != "contradict" or state.get("semantic_arbiter_decision") == "reasoning":
            candidates.append((state.get("reasoning_result") or {}).get("answer"))
        for candidate in candidates:
            if candidate_complete(problem, candidate, mode):
                answer = _strip_answer_label(str(candidate))
                if is_objective_mode(mode):
                    answer = normalize_objective_answer(answer, mode)
                    if mode == "fill":
                        answer = format_fill_submission(answer)
                complete = True
                break
    if mode == "proof" and complete:
        complete = proof_process_present(answer, state.get("reasoning_result") or {})
        answer = build_proof_body(answer, state.get("reasoning_result") or {})
    if not answer or (is_objective_mode(mode) and not complete):
        answer = "解题过程中出现错误，无法给出完整答案。"
    return {"final_response": answer,
            "submission_check": {"complete": complete, "mode": mode,
                                 "support": ("supported" if state.get("validation_status") == "match" else
                                             "unresolved_conflict" if state.get("python_evidence_status") == "contradict"
                                             and not state.get("answer_locked") else "uncertain"),
                                 "status": "ready" if complete else "partial_or_error"}}
