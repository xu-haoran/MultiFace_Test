from utils.problem_profile import classify_question_mode, mode_from_metadata


def input_node(state, config):
    problem = state.get("problem", "")
    metadata = state.get("metadata") if isinstance(state.get("metadata"), dict) else {}
    return {
        "idx": state.get("idx", metadata.get("idx", -1)),
        "question_mode": state.get("question_mode")
        or mode_from_metadata(metadata)
        or classify_question_mode(problem),
    }
