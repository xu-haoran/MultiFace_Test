from typing import Any, Dict

from agent.langgraph_math_agent import MathAgentGraph
from config import CONFIG
from utils.skills_loader import SkillsLoader
from utils.python_mcp_client import PythonMCPClient
from utils.token_budget import TokenBudget
from utils.logger import get_logger
from state.math_agent_state import create_initial_state

# ==================== 参与者代码区开始 ====================


class ReasoningAgent:
    """挑战杯数学推理智能体（LangGraph 多代理架构）。"""

    def __init__(self, client, *args, **kwargs):
        self.client = client
        self.logger = get_logger("ReasoningAgent")
        # Client 由平台管理，不在此修改模型、限流、重试或超时参数。
        self.skills_loader = SkillsLoader()
        self.mcp_client = PythonMCPClient()
        self.graph = MathAgentGraph(
            client=client, skills_loader=self.skills_loader, mcp_client=self.mcp_client)
        self.logger.info("ReasoningAgent initialized")

    def solve(self, problem: str, metadata: dict) -> dict:
        # 在入口统一处理参数，避免异常越过平台调用边界。
        raw_meta = metadata if isinstance(metadata, dict) else {}
        # 本地样例可能带标准答案；入口删除答案字段，正式评测只使用题面。
        _answer_keys = {
            "answer", "standard_answer", "reference_answer", "gold",
            "gold_answer", "solution", "reference_solution",
        }
        meta = {key: value for key, value in raw_meta.items()
                if key not in _answer_keys}
        idx = meta.get("idx", -1)
        try:
            self.logger.info(f"Solving problem {idx}")
            initial = create_initial_state(str(problem or ""), meta)
            final_state = self.graph.run(initial, token_budget=TokenBudget())
            result = {
                "final_response": final_state.get("final_response", "无法生成答案"),
                "trace": self._build_trace(final_state),
            }
            self._validate_output(result)
            return result
        except Exception as e:
            self.logger.error(f"Error solving problem {idx}: {e}")
            return {
                "final_response": "解题过程中出现错误，无法给出完整答案。",
                "trace": [{"step": "error", "content": str(e), "idx": idx}],
            }

    def _build_trace(self, state: dict) -> list:
        trace = []
        if state.get("category"):
            trace.append({
                "step": "classification",
                "source": "src/nodes/classifier_node.py::classifier_node",
                "category": state.get("category"),
                "question_mode": state.get("question_mode", "computation"),
                "confidence": state.get("category_confidence", 0.0),
                "candidates": state.get("candidate_categories", []),
                "tools": [
                    {
                        "name": "category_candidates",
                        "stages": state.get("classification_stages_used", []),
                        "selected": state.get("category"),
                    }
                ],
            })
        # 题库检索：检索到什么、是否真的写进了两个子代理的提示词，都要留证。
        # 2026-08-19 之前这一步完全不在 trace 里，只能从 node_timings 的
        # database_retrieval 耗时反推"检索确实跑过"，检索内容无从核对。
        examples = state.get("retrieved_examples")
        candidates = state.get("retrieval_candidates")
        if not isinstance(candidates, list):
            candidates = examples if isinstance(examples, list) else []
        if examples is not None:
            injected_items = {
                id(item) for item in (examples or []) if isinstance(item, dict)
            }
            reasoning_chars = int(state.get("reasoning_reference_chars", 0) or 0)
            python_chars = int(state.get("python_reference_chars", 0) or 0)
            high_threshold = float(CONFIG.get("db_high_similarity", 0.85))
            related_threshold = float(CONFIG.get("db_related_similarity", 0.60))
            related_limit = int(CONFIG.get("db_related_inject_k", 3) or 3)
            trace.append({
                "step": "database_retrieval",
                "source": "src/nodes/database_retrieval_node.py::database_retrieval_node",
                "count": len(candidates),
                "injected_count": len(examples or []),
                "injected_into": [name for name, chars in
                                  (("reasoning_agent", reasoning_chars
                                    + int(state.get("reasoning_knowledge_card_chars", 0) or 0)),
                                   ("python_agent", python_chars
                                    + int(state.get("python_knowledge_card_chars", 0) or 0)))
                                  if chars > 0],
                "thinking": {
                    "purpose": "解题前用原题检索竞赛题库，把最相似的题目与解答作为参考"
                               "示例注入推理与验证两个子代理。",
                    "relation_thresholds": {
                        "high_similarity": {
                            "condition": f"similarity > {high_threshold:.2f}",
                            "injection": "完整题面和参考解答",
                            "max_injected": 1,
                        },
                        "related": {
                            "condition": (
                                f"{related_threshold:.2f} < similarity <= "
                                f"{high_threshold:.2f}"
                            ),
                            "injection": "方法参考，隐藏具体结论/答案/数值",
                            "max_injected": related_limit,
                        },
                        "irrelevant": {
                            "condition": f"similarity <= {related_threshold:.2f}",
                            "injection": "不注入，仅保留检索轨迹",
                            "max_injected": 0,
                        },
                    },
                    "selection_policy": (
                        "有 high_similarity 时只注入最高的一条；否则注入最高的 related 记录"
                        f"（最多 {related_limit} 条）"
                    ),
                    "reference_block_chars": {
                        "reasoning_agent": reasoning_chars,
                        "python_agent": python_chars,
                    },
                    "knowledge_card_chars": {
                        "reasoning_agent": int(
                            state.get("reasoning_knowledge_card_chars", 0) or 0),
                        "python_agent": int(
                            state.get("python_knowledge_card_chars", 0) or 0),
                    },
                    "injection_verified": {
                        "reasoning_agent": state.get("reasoning_reference_chars") is not None,
                        "python_agent": state.get("python_reference_chars") is not None,
                    },
                },
                "tools": [{
                    "name": "vector_database_query",
                    "top_k": len(candidates),
                    "injected_top_k": len(examples or []),
                    "results": [
                        {
                            "rank": i,
                            "similarity": ex.get("similarity", 0.0),
                            "relation": ex.get("relation", ""),
                            "relation_reason": ex.get("relation_reason", ""),
                            "injected": id(ex) in injected_items,
                            "changed_parameters": ex.get("changed_parameters", []),
                            "problem_chars": len(str(ex.get("problem") or "")),
                            "solution_chars": len(str(ex.get("solution") or "")),
                        }
                        for i, ex in enumerate(candidates, 1)
                        if isinstance(ex, dict)
                    ],
                    "relation_summary": state.get("retrieval_relation_summary", {}),
                }],
            })
        if state.get("reasoning_result"):
            rr = state["reasoning_result"]
            trace.append({
                "step": "reasoning",
                "source": "src/nodes/reasoning_agent_node.py::reasoning_agent_node",
                "source_fields": [
                    "reasoning_result", "reasoning_trace", "reasoning_attempts",
                    "reasoning_reference_chars", "reasoning_knowledge_card_chars",
                ],
                "attempts": state.get("reasoning_attempts", 0),
                "steps_count": len(rr.get("steps", [])),
                "thinking": {
                    "validation_points_count": len(rr.get("validation_points", [])),
                    "answer_source": rr.get("answer_source", "parsed_sections"),
                },
                # 记录解析后的推理输出，便于本地核对。
                "output": self._structured_reasoning_output(rr),
                "tools": [
                    {
                        "name": "skill_document",
                        "category": state.get("category", ""),
                        "purpose": "加载对应数学领域 skill 文档作为解题参考。",
                    },
                    {
                        "name": "database_reference_examples",
                        "count": len(state.get("retrieved_examples") or []),
                        "chars_injected": int(state.get("reasoning_reference_chars", 0) or 0),
                        "knowledge_card_chars": int(
                            state.get("reasoning_knowledge_card_chars", 0) or 0),
                        "purpose": "题库相似题与解答，随 skill 文档一并进入推理提示词。",
                        "source_fields": [
                            "retrieved_examples", "knowledge_cards",
                            "reasoning_reference_chars", "reasoning_knowledge_card_chars",
                        ],
                    },
                    {
                        "name": "reasoning_llm",
                        "attempts": self._structured_attempts(
                            state.get("reasoning_trace", [])),
                        "max_attempts_used": state.get("reasoning_attempts", 0),
                        "output_schema": ["analysis", "steps", "answer", "validation_points"],
                        "output": self._structured_reasoning_output(rr),
                    },
                ],
            })
        if state.get("python_output"):
            po = state["python_output"]
            python_code = state.get("python_code", "")
            trace.append({
                "step": "python_verification",
                "source": "src/nodes/python_agent_node.py::python_agent_node",
                "source_fields": [
                    "python_output", "python_trace", "python_attempts",
                    "python_reference_chars", "python_knowledge_card_chars",
                ],
                "attempts": state.get("python_attempts", 0),
                "success": po.get("success", False),
                "thinking": {
                    "purpose": "用生成的 Python/SymPy 代码独立复核答案。",
                    "expected_output_marker": "最终答案:",
                    "evidence_status": po.get("evidence_status",
                                               state.get("python_evidence_status", "")),
                    # 未验证即不可信：trusted=False 时该答案已被排除出候选池，
                    # 评审需要看到判定理由，否则"为什么没用 Python 答案"不可解释。
                    "verification_trusted": po.get("python_verification_trusted", True),
                    "authenticity_reasons": (po.get("authenticity") or {}).get("reasons", []),
                    "objective_independent": bool(
                        po.get("objective_independent",
                               state.get("objective_python_independent", False))),
                },
                "output": self._structured_python_output(po),
                "tools": [
                    {
                        "name": "database_reference_examples",
                        "count": len(state.get("retrieved_examples") or []),
                        "chars_injected": int(state.get("python_reference_chars", 0) or 0),
                        "knowledge_card_chars": int(
                            state.get("python_knowledge_card_chars", 0) or 0),
                        "purpose": "题库相似题与解答，随验证脚本一并进入 Python 提示词。",
                        "source_fields": [
                            "retrieved_examples", "knowledge_cards",
                            "python_reference_chars", "python_knowledge_card_chars",
                        ],
                    },
                    {
                        "name": "python_code_generation",
                        "attempts": self._structured_attempts(
                            state.get("python_trace", [])),
                        "code_length": len(python_code),
                    },
                    {
                        "name": "python_executor",
                        "backend": po.get("execution_backend", ""),
                        "success": po.get("success", False),
                        "execution_time": po.get("execution_time", 0.0),
                },
                ],
            })
        if state.get("validation_details"):
            trace.append({
                "step": "validation",
                "source": "src/nodes/cross_validator_node.py::cross_validator_node",
                "status": state.get("validation_status"),
                "thinking": {
                    "problem_type": state["validation_details"].get("problem_type", ""),
                    "method": state["validation_details"].get("method", ""),
                    "confidence": state["validation_details"].get("confidence", 0.0),
                    "python_evidence_status": state["validation_details"].get(
                        "python_evidence_status", state.get("python_evidence_status", "")),
                },
                "tools": [
                    {
                        "name": "answer_matcher",
                        "status": state.get("validation_status", ""),
                    }
                ],
            })
        if state.get("validation_history"):
            trace.append({
                "step": "validation_history",
                "source": "src/nodes/cross_validator_node.py::cross_validator_node",
                "count": len(state["validation_history"]),
                "tools": [{
                    "name": "validation_history",
                    "rounds": [
                        {
                            "round": item.get("round", index),
                            "status": item.get("status", ""),
                            "evidence_status": item.get("evidence_status", ""),
                            "contradiction_count": len(item.get("contradictions") or []),
                        }
                        for index, item in enumerate(state["validation_history"])
                        if isinstance(item, dict)
                    ],
                }],
            })
        if state.get("problem_anchor") or state.get("problem_integrity_events"):
            anchor = state.get("problem_anchor") or {}
            trace.append({
                "step": "problem_integrity",
                "source": "src/utils/problem_anchor.py::make_problem_anchor",
                "sha256_prefix": str(anchor.get("sha256", ""))[:16],
                "events": state.get("problem_integrity_events", []),
            })
        if state.get("semantic_arbiter_trace"):
            trace.append({
                "step": "semantic_arbitration",
                "source": "src/nodes/semantic_arbiter_node.py::semantic_arbiter_node",
                "status": state.get("semantic_arbiter_status", ""),
                "decision": state.get("semantic_arbiter_decision", ""),
                "answer_locked": state.get("answer_locked", False),
                "thinking": {
                    "purpose": "在不依赖题型关键词的情况下，从既有答案候选中选择完整正确答案。",
                    "policy": "select_existing_candidate_or_abstain",
                },
                "tools": [
                    {
                        "name": "semantic_arbiter_llm",
                        "attempts": state.get("semantic_arbiter_attempts", 0),
                    "rounds": self._structured_attempts(
                        state.get("semantic_arbiter_trace", [])),
                    }
                ],
            })
        if state.get("reconciliation_trace"):
            trace.append({
                "step": "reconciliation",
                "source": "src/nodes/reconciliation_node.py::reconciliation_node",
                "count": len(state["reconciliation_trace"]),
                "thinking": {
                    "purpose": "当推理答案与 Python 验证不一致或不确定时，决定是否重跑子图。",
                },
                "tools": [
                    {
                        "name": "reconciliation_policy",
                    "rounds": self._structured_attempts(
                        state["reconciliation_trace"]),
                    }
                ],
            })
        if state.get("final_response"):
            trace.append({
                "step": "coordination",
                "source": "src/nodes/coordinator_node.py::coordinator_node",
                "response_length": len(state["final_response"]),
                "thinking": {
                    "purpose": "将推理步骤、验证结果与最终答案整理为面向读者的输出。",
                },
                "tools": [
                    {
                        "name": "coordinator_llm",
                        "content_length": len(state.get("coordination_detail", "")),
                        "skipped_for_locked_answer": bool(state.get("answer_locked")),
                    },
                    {
                        "name": "answer_formatter",
                        "final_response_length": len(state.get("final_response", "")),
                    },
                ],
                "fallback_source": state.get("fallback_source", ""),
            })
        # 记录各阶段耗时，便于定位超时。
        if state.get("node_timings") or state.get("_time_budget"):
            trace.append({
                "step": "timing",
                "thinking": {
                    "purpose": "记录各阶段实测耗时与时间预算消耗，用于定位超时瓶颈。",
                },
                "tools": [
                    {
                        "name": "time_budget",
                        "budget": state.get("_time_budget", {}),
                        "node_elapsed": state.get("node_timings", []),
                        "llm_calls": state.get("_llm_spend_log", []),
                    }
                ],
            })
        return trace

    @staticmethod
    def _truncate(value: str, limit: int) -> str:
        if not isinstance(value, str):
            value = str(value or "")
        if len(value) <= limit:
            return value
        return value[:limit].rstrip() + "...[truncated]"

    @staticmethod
    def _structured_attempts(items: Any) -> list:
        """保留各阶段的完整尝试记录，便于本地诊断。"""
        if not isinstance(items, list):
            return []
        return items

    def _structured_reasoning_output(self, result: Any) -> dict:
        """保留解析后的推理内容，便于核对 Agent 实际输出。"""
        if not isinstance(result, dict):
            return {}
        return {
            "analysis": result.get("analysis", ""),
            "steps": result.get("steps", []),
            "answer": result.get("answer", ""),
            "validation_points": result.get("validation_points", []),
            "answer_source": result.get("answer_source", "parsed_sections"),
            "raw_response_excerpt": self._excerpt(
                result.get("raw_response", "")),
        }

    def _structured_python_output(self, output: Any) -> dict:
        """保留 Python 验证结果，便于核对独立求解过程。"""
        if not isinstance(output, dict):
            return {}
        return dict(output)

    @staticmethod
    def _excerpt(value: str, head: int = 600, tail: int = 1400) -> str:
        """头 + 尾节选：截断响应的结论区在尾部，只截头会丢掉最有证据价值的部分。"""
        if not isinstance(value, str):
            value = str(value or "")
        if len(value) <= head + tail + 40:
            return value
        omitted = len(value) - head - tail
        return f"{value[:head]}\n……[中略 {omitted} 字符]……\n{value[-tail:]}"

    def _validate_output(self, result: dict) -> None:
        import json
        assert isinstance(result, dict), "返回值必须是dict"
        assert "final_response" in result, "必须包含final_response"
        assert isinstance(result["final_response"], str), "final_response必须是字符串"
        assert result["final_response"].strip(), "final_response不能为空"
        json.dumps(result, ensure_ascii=False)  # 检查可序列化


# ==================== 参与者代码区结束 ====================
