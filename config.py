"""MultiFace 运行配置。
"""

CONFIG = {
    # 模型与重试
    "model": "intern-s2",
    "max_retries_per_node": 2,
    "llm_max_retries": 3,
    "backoff_factor": 2.0,

    # 各节点的最长等待时间（秒）
    "node_timeouts": {
        "input": 5,
        "classifier": 10,
        "reasoning_agent": 1100,
        "python_agent": 300,
        "python_mcp_execute": 60,
        "cross_validator": 90,
        "submission": 5,
        "reconciliation": 10,
        "semantic_arbiter": 10,
        "coordinator": 60,
    },
    # Per-call ceilings prevent optional verification from consuming the entire
    # normal-solving window. Injected clients may continue in the background.
    "llm_call_timeouts": {
        "classifier_prefill": 8,
        "reasoning": 750,
        "python": 120,
        "python_repair": 45,
        "python_compressed": 60,
        "knowledge_review": 60,
    },

    # 向量检索与参考题注入
    "db_retrieval_top_k": 1,
    "db_retrieval_recall_k": 5,
    # 关系分档：>0.88 高相似直供；0.75<sim≤0.88 相关方法；≤0.75 不注入。
    "db_high_similarity": 0.88,
    "db_related_similarity": 0.75,
    "db_related_inject_k": 1,
    "high_similarity_reference_enabled": True,
    "database_answer_fallback_enabled": False,
    "high_similarity_worked_review_enabled": False,  # 不再执行不影响输出的在线复核

    # 题目与解答的混合检索；知识仅从 resources/skills 读取
    "hybrid_retrieval_enabled": True,
    "hybrid_dense_recall_k": 15,
    "hybrid_sparse_recall_k": 15,
    "hybrid_final_k": 5,

    # 分类与验证
    "classifier_top_k": 3,
    "classifier_confidence_threshold": 0.85,

    # 各模型阶段的采样参数。确定性选择阶段关闭额外采样裁剪。
    "temperatures": {
        "classifier": 0.0,
        "reasoning": 0.2,
        "objective_reasoning": 0.1,
        "python": 0.1,
        "semantic_arbiter": 0.0,
        "coordinator": 0.1,
        "emergency_answer": 0.1,
        "high_similarity_worked_review": 0.0,
    },
    "top_p": {
        "classifier": 1.0,
        "reasoning": 0.95,
        "objective_reasoning": 0.95,
        "python": 0.95,
        "semantic_arbiter": 1.0,
        "coordinator": 0.9,
        "emergency_answer": 0.95,
        "high_similarity_worked_review": 1.0,
    },
    "top_k": {
        "classifier": -1,
        "reasoning": -1,
        "objective_reasoning": -1,
        "python": -1,
        "semantic_arbiter": -1,
        "coordinator": -1,
        "emergency_answer": -1,
        "high_similarity_worked_review": -1,
    },

    # 各阶段输出长度；预填充阶段较短，普通推理保留完整空间
    "max_tokens": {
        "classifier": 96,
        "classifier_fallback": 8192,
        "objective_reasoning": 8192,
        "reasoning": 28672,
        "python": 8192,
        "reasoning_compressed": 8192,
        "python_compressed": 4096,
        "reconciliation": 16384,
        "semantic_arbiter": 96,
        "semantic_arbiter_fallback": 4096,
        "coordinator": 4096,
        "emergency_answer": 2048,
    },
    "reconciliation_max_rounds": 2,
    "token_budget_max": 256000,
    "token_budget_warn_ratio": 0.95,

    # 单题最多 20 分钟：前 15 分钟正常求解，最后 5 分钟仅用于收束与应急作答。
    "problem_time_budget_s": 1200,
    "time_reserve_s": 300,

    # 日志
    "log_level": "INFO",
    "log_dir": "logs",
}
