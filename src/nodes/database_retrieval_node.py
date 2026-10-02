"""题库检索节点：取得 top-k 相似题及解答作为参考。"""

import os
import threading
from pathlib import Path
from typing import Any, Dict

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from config import CONFIG
from utils.deps import get_deps
from utils.logger import get_logger
from utils.retrieval_relation import (
    annotate_candidates,
    dense_similarity_score,
    select_for_injection,
)
from utils.hybrid_retriever import HybridRetriever
from utils.knowledge_card_retriever import KnowledgeCardRetriever

_RETRIEVER_INIT_LOCK = threading.Lock()
# 多个评测线程可能并发访问客户端；保护一次完整检索事务。
_RETRIEVAL_DB_LOCK = threading.RLock()


def database_retrieval_node(state: Dict[str, Any], config: Dict) -> Dict[str, Any]:
    """Query ChromaDB for the top-k similar problems+solutions as reference examples.

    The retriever recalls a larger candidate set, then applies relation bands
    using dense similarity only. A high-similarity label does not itself permit
    answer reuse: the normalized problem must match exactly and an answer must
    be present. Related records and sparse-only lexical matches are method-only
    references and cannot enter the dense-similarity bands.

    Returns:
        Dict with 'retrieved_examples': List[Dict] containing:
            - problem: original problem text
            - solution: solution text
            - dense_similarity: cosine similarity, when dense retrieval found it
            - sparse_similarity: lexical score, when sparse retrieval found it
            - hybrid_score: fused ranking score, never used as a similarity
            - source: source file path
    """
    # 检索是纯增益节点：任何失败都必须降级为"无参考示例"，绝不能打断求解子图。
    # get_deps 曾在 try 之外——缺 deps 的配置会让整条子图记下 KeyError（2026-08-19
    # 由 test_solving_subgraph 暴露）。
    try:
        deps = get_deps(config)
        logger = deps.logger or get_logger()
    except Exception:  # noqa: BLE001 - 配置形状是调用方的事，检索不为此失败。
        get_logger().warning("[db_retrieval] deps unavailable, skip")
        return {"retrieved_examples": []}

    problem = state.get("problem", "").strip()
    if not problem:
        logger.warning("[db_retrieval] empty problem, skip")
        return {"retrieved_examples": []}

    try:
        db_client = deps.db_client
        if db_client is None:
            logger.warning("[db_retrieval] db_client not initialized, skip")
            return {"retrieved_examples": []}

        legacy_top_k = int(CONFIG.get("db_retrieval_top_k", 2) or 2)
        recall_k = int(CONFIG.get("db_retrieval_recall_k", 10) or 10)
        hybrid_used = False
        with _RETRIEVAL_DB_LOCK:
            if bool(CONFIG.get("hybrid_retrieval_enabled", True)):
                # 缓存稀疏索引；失败时回退到兼容的稠密检索。
                try:
                    retriever = getattr(db_client, "_hybrid_retriever", None)
                    if retriever is None:
                        with _RETRIEVER_INIT_LOCK:
                            retriever = getattr(db_client, "_hybrid_retriever", None)
                            if retriever is None:
                                retriever = HybridRetriever(db_client)
                                db_client._hybrid_retriever = retriever
                    raw_results = retriever.query(
                        problem,
                        subject=str(state.get("category") or ""),
                        dense_k=int(CONFIG.get("hybrid_dense_recall_k", 36) or 36),
                        sparse_k=int(CONFIG.get("hybrid_sparse_recall_k", 36) or 36),
                        final_k=max(recall_k, int(CONFIG.get("hybrid_final_k", 14) or 14)),
                    )
                    hybrid_used = True
                except Exception as hybrid_error:  # noqa: BLE001 - retain baseline.
                    logger.warning(f"[db_retrieval] hybrid unavailable, dense fallback: {hybrid_error!r}")
                    raw_results = db_client.query(problem, top_k=max(legacy_top_k, recall_k))
            else:
                raw_results = db_client.query(problem, top_k=max(legacy_top_k, recall_k))
        candidates = annotate_candidates(
            problem,
            raw_results,
            high_similarity_min=float(CONFIG.get("db_high_similarity", 0.85)),
            related_min_similarity=float(CONFIG.get("db_related_similarity", 0.70)),
        )
        examples = select_for_injection(
            candidates,
            related_limit=int(CONFIG.get("db_related_inject_k", 3) or 3),
        )
        for i, r in enumerate(candidates):
            dense_score = dense_similarity_score(r)
            sparse_score = r.get("sparse_similarity")
            logger.info(
                f"[db_retrieval] recall#{i+1} dense="
                f"{f'{dense_score:.3f}' if dense_score is not None else 'n/a'} "
                f"sparse={f'{float(sparse_score):.3f}' if sparse_score is not None else 'n/a'} "
                f"hybrid={r.get('hybrid_score') if r.get('hybrid_score') is not None else 'n/a'} "
                f"relation={r.get('relation', '')} dataset={r.get('source_dataset', '')} "
                f"src={r.get('source', '')[:60]}"
            )

        # 不锁定答案；关系标签只用于安全筛选和 trace。
        relation_counts = {
            name: sum(1 for item in candidates if item.get("relation") == name)
            for name in ("high_similarity", "related", "sparse_reference", "irrelevant")
        }
        logger.info(
            f"[db_retrieval] recalled={len(candidates)} injected={len(examples)} "
            f"relations={relation_counts}"
        )
        result = {
            "retrieved_examples": examples,
            "retrieval_candidates": candidates,
            "retrieval_relation_summary": {
                "recalled": len(candidates),
                "injected": len(examples),
                "answer_safe": sum(1 for item in candidates if item.get("answer_safe") is True),
                "counts": relation_counts,
                "fallback": False,
                "hybrid": hybrid_used,
            },
        }
        if bool(CONFIG.get("knowledge_cards_enabled", True)) and deps.skills_loader is not None:
            try:
                cards = getattr(deps.skills_loader, "_knowledge_card_retriever", None)
                if cards is None:
                    cards = KnowledgeCardRetriever(deps.skills_loader, db_client)
                    deps.skills_loader._knowledge_card_retriever = cards
                primary = str(state.get("category") or "")
                alternate = [str(name) for name in (state.get("alternate_categories") or [])
                             if str(name)]
                subjects = [primary] + [name for name in alternate if name != primary]
                with _RETRIEVAL_DB_LOCK:
                    result["knowledge_cards"] = cards.query(
                        problem,
                        subject=primary,
                        subjects=subjects[:3],
                        limit=int(CONFIG.get("knowledge_card_limit", 3) or 3),
                    )
            except Exception as cards_error:  # noqa: BLE001 - cards are additive only.
                logger.warning(f"[db_retrieval] knowledge cards skipped: {cards_error!r}")
                result["knowledge_cards"] = []
        return result

    except Exception as e:
        # 异常回退仍经过同一关系分档和答案安全门槛。
        logger.error(f"[db_retrieval] relation-aware retrieval error: {e!r}")
        try:
            with _RETRIEVAL_DB_LOCK:
                raw_fallback = deps.db_client.query(
                    problem, top_k=int(CONFIG.get("db_retrieval_top_k", 2) or 2))
            fallback_candidates = annotate_candidates(
                problem,
                raw_fallback,
                high_similarity_min=float(CONFIG.get("db_high_similarity", 0.85)),
                related_min_similarity=float(CONFIG.get("db_related_similarity", 0.70)),
            )
            fallback_examples = select_for_injection(
                fallback_candidates,
                related_limit=int(CONFIG.get("db_related_inject_k", 3) or 3),
            )
            return {
                "retrieved_examples": fallback_examples,
                "retrieval_candidates": fallback_candidates,
                "retrieval_relation_summary": {
                    "recalled": len(fallback_candidates), "injected": len(fallback_examples),
                    "answer_safe": sum(1 for item in fallback_candidates
                                       if item.get("answer_safe") is True),
                    "counts": {
                        name: sum(1 for item in fallback_candidates
                                  if item.get("relation") == name)
                        for name in ("high_similarity", "related", "sparse_reference", "irrelevant")
                    },
                    "fallback": True,
                },
            }
        except Exception as fallback_error:
            logger.error(f"[db_retrieval] legacy fallback error: {fallback_error!r}")
            return {
                "retrieved_examples": [],
                "retrieval_candidates": [],
                "retrieval_relation_summary": {
                    "recalled": 0, "injected": 0, "counts": {}, "fallback": True,
                },
            }



