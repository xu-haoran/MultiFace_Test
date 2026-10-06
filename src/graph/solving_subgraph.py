"""Solve cached references; retries dispatch only requested branches."""
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from state.math_agent_state import MathAgentState
from nodes import reasoning_agent_node, python_agent_node, cross_validator_node
from utils.error_handler import node_wrapper
from utils.flow_policy import python_verification_needed, verification_task
from utils.problem_profile import is_objective_mode


def dispatch_solvers(state, config):
    mode = state.get("question_mode", "computation")
    enabled = state.get("python_verification_enabled", python_verification_needed(state.get("problem", ""), mode))
    targets = state.get("retry_targets") or (["reasoning", "python"] if enabled else ["reasoning"])
    targets = [target for target in targets if target != "python" or enabled] or ["reasoning"]
    sends = []
    for target in targets:
        if target == "reasoning":
            sends.append(Send("reasoning_agent", {
                **state, "branch_hint": state.get("reasoning_retry_hint"),
            }))
        elif target == "python":
            sends.append(Send("python_agent", {
                **state, "branch_hint": state.get("python_retry_hint"),
                "objective_python_enabled": is_objective_mode(mode),
            }))
    return sends


def build_solving_subgraph():
    sub = StateGraph(MathAgentState)
    def prepare(state, config):
        task = state.get("verification_task") or verification_task(state.get("problem", ""), state.get("question_mode", "computation"))
        return {"python_verification_enabled": task["python_required"], "verification_task": task}
    sub.add_node("dispatch", prepare)
    sub.add_node("reasoning_agent", node_wrapper(reasoning_agent_node, "reasoning_agent"))
    sub.add_node("python_agent", node_wrapper(python_agent_node, "python_agent"))
    sub.add_node("cross_validator", node_wrapper(cross_validator_node, "cross_validator"))
    sub.add_edge(START, "dispatch")
    sub.add_conditional_edges("dispatch", dispatch_solvers, ["reasoning_agent", "python_agent"])
    sub.add_edge("reasoning_agent", "cross_validator")
    sub.add_edge("python_agent", "cross_validator")
    sub.add_edge("cross_validator", END)
    return sub.compile()
