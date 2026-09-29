from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from state.math_agent_state import MathAgentState
from nodes import reasoning_agent_node, python_agent_node, cross_validator_node, database_retrieval_node
from utils.error_handler import node_wrapper
from utils.objective_verifier import objective_needs_independent_python

def build_solving_subgraph():
    sub = StateGraph(MathAgentState)
    sub.add_node("database_retrieval", node_wrapper(database_retrieval_node, "database_retrieval"))
    sub.add_node("reasoning_agent", node_wrapper(reasoning_agent_node, "reasoning_agent"))
    sub.add_node("python_agent", node_wrapper(python_agent_node, "python_agent"))
    sub.add_node("cross_validator", node_wrapper(cross_validator_node, "cross_validator"))

    # 先检索题库
    sub.add_edge(START, "database_retrieval")

    def fan_out(state, config):
        # 高相似题只作为完整例题参考，仍走正常推理和验证流程。
        reasoning_send = Send(
            "reasoning_agent", {**state, "branch_hint": state.get("reasoning_retry_hint")}
        )
        # 已有完整例题时，避免无关的 Python 启发式结果覆盖推理结论。
        if any(isinstance(item, dict) and item.get("relation") == "high_similarity"
               for item in (state.get("retrieved_examples") or [])):
            return [reasoning_send]
        if objective_needs_independent_python(
                state.get("problem", ""), state.get("question_mode", "computation")):
            # 客观题增加独立 Python 尝试，避免只凭格式接受答案。
            return [
                reasoning_send,
                Send("python_agent", {
                    **state,
                    "branch_hint": state.get("python_retry_hint"),
                    "objective_python_enabled": True,
                }),
            ]
        return [
            reasoning_send,
            Send("python_agent", {**state, "branch_hint": state.get("python_retry_hint")}),
        ]
    sub.add_conditional_edges(
        "database_retrieval", fan_out,
        ["reasoning_agent", "python_agent"])
    sub.add_edge("reasoning_agent", "cross_validator")
    sub.add_edge("python_agent", "cross_validator")
    sub.add_edge("cross_validator", END)
    return sub.compile()
