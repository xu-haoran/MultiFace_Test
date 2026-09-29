from typing import Annotated, Any, Dict, List, Optional
import operator
from typing_extensions import NotRequired, TypedDict

from utils.problem_anchor import make_problem_anchor
from utils.problem_profile import classify_question_mode, mode_from_metadata

class MathAgentState(TypedDict):
    # 输入阶段（必需）
    problem: str
    metadata: Dict[str, Any]
    idx: int
    problem_anchor: NotRequired[Dict[str, Any]]
    problem_integrity_events: NotRequired[Annotated[List[Dict], operator.add]]
    reasoning_attempts: int
    python_attempts: int
    errors: Annotated[List[Dict], operator.add]
    # 分类阶段
    question_mode: NotRequired[str]
    category: NotRequired[str]
    category_confidence: NotRequired[float]
    candidate_categories: NotRequired[List[str]]
    classification_stages_used: NotRequired[List[str]]
    # 分类器保留少量备选学科，低置信度题目可跨学科取知识卡/技能片段。
    alternate_categories: NotRequired[List[str]]
    # 数据库检索阶段
    retrieved_examples: NotRequired[List[Dict[str, Any]]]
    knowledge_cards: NotRequired[List[Dict[str, Any]]]
    retrieval_candidates: NotRequired[List[Dict[str, Any]]]
    retrieval_relation_summary: NotRequired[Dict[str, Any]]
    # 注入证据：两个子代理各自记录参考示例区块实际写入提示词的字符数。
    # 必须分键——两个代理并行写 state，同键并发写会被 LangGraph 拒绝。
    reasoning_reference_chars: NotRequired[int]
    reasoning_knowledge_card_chars: NotRequired[int]
    python_reference_chars: NotRequired[int]
    python_knowledge_card_chars: NotRequired[int]
    # 推理阶段
    reasoning_result: NotRequired[Optional[Dict[str, Any]]]
    reasoning_trace: NotRequired[List[Dict]]
    reasoning_retry_hint: NotRequired[Optional[str]]
    reasoning_raw_response: NotRequired[str]
    # Python 验证阶段
    python_code: NotRequired[str]
    python_output: NotRequired[Optional[Dict[str, Any]]]
    python_trace: NotRequired[List[Dict]]
    python_retry_hint: NotRequired[Optional[str]]
    python_evidence_status: NotRequired[str]
    python_evidence_summary: NotRequired[str]
    python_contradictions: NotRequired[List[str]]
    objective_python_independent: NotRequired[bool]
    objective_python_enabled: NotRequired[bool]
    # 验证阶段
    validation_status: NotRequired[str]
    validation_details: NotRequired[Dict[str, Any]]
    validated_answer: NotRequired[str]
    validation_history: NotRequired[Annotated[List[Dict], operator.add]]
    # 协调阶段
    reconciliation_trace: NotRequired[List[Dict]]
    reconciliation_round: NotRequired[int]
    # 语义仲裁阶段（只选择既有候选，不生成新数学答案）
    semantic_arbiter_status: NotRequired[str]
    semantic_arbiter_decision: NotRequired[str]
    semantic_arbiter_trace: NotRequired[List[Dict]]
    semantic_arbiter_attempts: NotRequired[int]
    answer_locked: NotRequired[bool]
    # 输出阶段
    final_response: NotRequired[str]
    coordination_detail: NotRequired[Optional[str]]  # coordinator 原始解题说明，记入 trace（计算题 final_response 已收敛为简洁答案）
    fallback_source: NotRequired[str]
    # 内部控制
    branch_hint: NotRequired[Optional[str]]  # Send() 扇出携带的重试提示（solving 子图内）
    next_node: NotRequired[Optional[str]]
    should_terminate: NotRequired[bool]
    token_budget_consumed: NotRequired[int]
    # 各节点实测耗时。必须是 reducer 字段：solving 子图内 reasoning_agent 与
    # python_agent 并行扇出，两条分支会同时写入。
    node_timings: NotRequired[Annotated[List[Dict], operator.add]]

def create_initial_state(problem: str, metadata: Dict[str, Any]) -> MathAgentState:
    metadata = metadata if isinstance(metadata, dict) else {}
    question_mode = mode_from_metadata(metadata) or classify_question_mode(problem)
    return MathAgentState(
        problem=problem, metadata=metadata, idx=metadata.get("idx", -1),
        problem_anchor=make_problem_anchor(problem, metadata.get("idx", -1)),
        problem_integrity_events=[],
        question_mode=question_mode,
        reasoning_attempts=0, python_attempts=0, semantic_arbiter_attempts=0,
        answer_locked=False, errors=[], validation_history=[],
    )
