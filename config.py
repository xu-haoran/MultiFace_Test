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
        "input": 60,
        "classifier": 200,
        "solving": 1150,
        "reasoning_agent": 1080,
        "python_agent": 1080,
        "python_mcp_execute": 120,
        "cross_validator": 120,
        "reconciliation": 120,
        "semantic_arbiter": 120,
        "coordinator": 180,
    },

    # 向量检索与参考题注入
    "db_retrieval_top_k": 1,
    "db_retrieval_recall_k": 5,
    # 关系分档：>0.88 高相似直供；0.82<sim≤0.88 相关方法；≤0.82 不注入。
    "db_high_similarity": 0.88,
    "db_related_similarity": 0.82,
    "db_related_inject_k": 1,
    "high_similarity_reference_enabled": True,
    "database_answer_fallback_enabled": False,
    "high_similarity_worked_review_enabled": True,

    # 混合检索和大学数学知识卡
    "hybrid_retrieval_enabled": True,
    "hybrid_dense_recall_k": 15,
    "hybrid_sparse_recall_k": 15,
    "hybrid_final_k": 5,
    "knowledge_cards_enabled": True,
    "knowledge_card_limit": 3,

    # 分类与验证
    "classifier_top_k": 3,
    "classifier_confidence_threshold": 0.85,

    # 各阶段采样温度：数学推理保持低随机性
    "temperatures": {
        "classifier": 0.0,
        "reasoning": 0.1,
        "objective_reasoning": 0.1,
        "python": 0.1,
        "reconciliation": 0.1,
        "semantic_arbiter": 0.0,
        "coordinator": 0.1,
        "emergency_answer": 0.1,
    },

    # 各阶段输出长度；预填充阶段较短，普通推理保留完整空间
    "max_tokens": {
        "classifier": 96,
        "classifier_fallback": 8192,
        "objective_reasoning": 8192,
        "reasoning": 32768,
        "python": 16384,
        "reasoning_compressed": 16384,
        "python_compressed": 8192,
        "reconciliation": 32768,
        "semantic_arbiter": 96,
        "semantic_arbiter_fallback": 8192,
        "coordinator": 2048,
        "emergency_answer": 2048,
    },
    "reconciliation_max_rounds": 2,
    "token_budget_max": 256000,
    "token_budget_warn_ratio": 0.95,

    # 单题墙钟预算。预留时间用于收束、仲裁和应急答题
    "problem_time_budget_s": 1200,
    "time_reserve_s": 180,
    "time_fast_path_threshold_s": 180,
    "arbiter_reserve_quota_s": 75,
    "emergency_reserve_quota_s": 90,
    "compressed_reserve_margin_s": 180,

    # 协调器最多独立生成的候选答案数；失败或无效输出不补足
    "coordinator_vote_attempts": 3,

    # 日志
    "log_level": "INFO",
    "log_dir": "logs",
}
