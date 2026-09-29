from langgraph.graph import StateGraph, START, END
from state.math_agent_state import MathAgentState
from graph.solving_subgraph import build_solving_subgraph
from nodes import (
    input_node, classifier_node, reconciliation_node, semantic_arbiter_node,
    coordinator_node,
)
from utils.error_handler import node_wrapper
from utils.database_client import DatabaseClient

def build_math_agent_graph():
    g = StateGraph(MathAgentState)
    g.add_node("input", node_wrapper(input_node, "input"))
    g.add_node("classifier", node_wrapper(classifier_node, "classifier"))
    g.add_node("solving", build_solving_subgraph())
    g.add_node("reconciliation", node_wrapper(reconciliation_node, "reconciliation"))
    g.add_node("semantic_arbiter", node_wrapper(semantic_arbiter_node, "semantic_arbiter"))
    g.add_node("coordinator", node_wrapper(coordinator_node, "coordinator"))

    g.add_edge(START, "input")
    g.add_edge("input", "classifier")
    g.add_edge("classifier", "solving")

    def route_after_solving(state, config):
        if state.get("should_terminate", False):
            return "coordinator"
        requested = state.get("next_node")
        if requested in {"reconciliation", "semantic_arbiter"}:
            return requested
        return "coordinator"
    g.add_conditional_edges("solving", route_after_solving,
                            {"coordinator": "coordinator", "reconciliation": "reconciliation",
                             "semantic_arbiter": "semantic_arbiter"})

    def route_after_reconciliation(state, config):
        requested = state.get("next_node")
        if requested in {"solving", "semantic_arbiter"}:
            return requested
        return "coordinator"
    g.add_conditional_edges("reconciliation", route_after_reconciliation,
                            {"solving": "solving", "semantic_arbiter": "semantic_arbiter",
                             "coordinator": "coordinator"})

    def route_after_semantic_arbiter(state, config):
        return "reconciliation" if state.get("next_node") == "reconciliation" else "coordinator"
    g.add_conditional_edges("semantic_arbiter", route_after_semantic_arbiter,
                            {"reconciliation": "reconciliation", "coordinator": "coordinator"})

    g.add_edge("coordinator", END)
    return g.compile()

class MathAgentGraph:
    def __init__(self, client, skills_loader=None, mcp_client=None):
        self.app = build_math_agent_graph()
        self.client = client
        self.skills_loader = skills_loader
        self.mcp_client = mcp_client
        # 从项目目录定位数据库，并在复用 Agent 时保留已初始化的客户端。
        self.db_client = DatabaseClient(device="cpu")

    def run(self, initial_state, token_budget=None, time_budget=None):
        from utils.deps import Deps
        from utils.token_budget import TokenBudget
        from utils.time_budget import TimeBudget
        tb = token_budget or TokenBudget()
        # 每题只启动一只时钟，所有节点共用同一墙钟预算。
        clock = time_budget or TimeBudget()
        deps = Deps(client=self.client, skills_loader=self.skills_loader,
                    mcp_client=self.mcp_client, token_budget=tb, time_budget=clock,
                    db_client=self.db_client)
        final_state = self.app.invoke(initial_state, config={"configurable": {"deps": deps}})
        if isinstance(final_state, dict):
            final_state["_time_budget"] = clock.snapshot()
            final_state["_llm_spend_log"] = clock.spend_log()
        return final_state
