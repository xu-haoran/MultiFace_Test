"""Offline two-stage retrieval for the bundled mathematics collection.

The Chroma index remains the authoritative dense retriever.  This module adds a
lazy, process-local sparse index over *problem statements only*, fuses dense and
lexical recall with reciprocal-rank fusion, and performs a deterministic
math-aware rerank.  It deliberately does not download a new model or bypass the
relation checks in :mod:`utils.retrieval_relation`.
"""

from __future__ import annotations

import math
import re
import threading
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

_LATEX_RE = re.compile(r"\\[A-Za-z]+")
_WORD_RE = re.compile(r"[A-Za-z]+(?:[-_][A-Za-z]+)*|\d+(?:\.\d+)?")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")


def math_tokens(text: str) -> list[str]:
    """Tokenise English, LaTeX, numbers and Chinese character bigrams.

    The bigrams make the sparse layer useful for Chinese terminology without a
    runtime word-segmentation dependency.  Formula commands are retained because
    they often carry more retrieval signal than surrounding prose.
    """
    value = str(text or "").casefold()
    tokens = _LATEX_RE.findall(value)
    tokens.extend(_WORD_RE.findall(value))
    for run in _CJK_RE.findall(value):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[i:i + 2] for i in range(len(run) - 1))
    return tokens


def _problem_only(document: str) -> str:
    marker = re.search(r"## Problem\s*(.*?)(?=## Solution|\Z)", document or "", re.S | re.I)
    return marker.group(1).strip() if marker else str(document or "")


def _query_views(problem: str) -> list[str]:
    """Create compact head/tail/formula views for long mathematical prompts."""
    text = str(problem or "").strip()
    if len(text) <= 1800:
        return [text]
    views = [text[:1800], text[-1800:]]
    equations = re.findall(r"[^\n]{0,80}(?:=|≤|≥|<|>|\\frac|\\sum|\\int)[^\n]{0,180}", text)
    if equations:
        views.append("\n".join(equations[:12])[:1800])
    question_match = re.findall(r"(?:求|证明|判断|问|determine|find|prove|show)[^\n]{0,260}", text, re.I)
    if question_match:
        views.append(question_match[-1][:1800])
    unique = []
    for view in views:
        view = view.strip()
        if view and view not in unique:
            unique.append(view)
    return unique[:4]


@dataclass
class _SparseIndex:
    ids: list[str]
    problems: list[str]
    vectorizer: Any
    matrix: Any


class HybridRetriever:
    """Dense + sparse candidate recall and local deterministic reranking.

    The sparse index is built only on the first query and stays in memory for
    that process.  If sklearn or Chroma's bulk read is unavailable, callers get
    the original dense results unchanged.
    """

    def __init__(self, db_client: Any):
        self.db_client = db_client
        self._sparse_index: _SparseIndex | None = None
        self._index_lock = threading.Lock()

    def _ensure_sparse_index(self) -> _SparseIndex:
        if self._sparse_index is not None:
            return self._sparse_index
        with self._index_lock:
            if self._sparse_index is not None:
                return self._sparse_index
            from sklearn.feature_extraction.text import TfidfVectorizer

            self.db_client._load_collection()
            raw = self.db_client._collection.get(include=["documents"])
            ids = [str(value) for value in raw.get("ids", [])]
            problems = [_problem_only(value) for value in raw.get("documents", [])]
            if not ids or len(ids) != len(problems):
                raise RuntimeError("empty or malformed local Chroma collection")
            vectorizer = TfidfVectorizer(
                tokenizer=math_tokens,
                token_pattern=None,
                lowercase=False,
                dtype=np.float32,
                sublinear_tf=True,
                norm="l2",
                min_df=1,
            )
            matrix = vectorizer.fit_transform(problems)
            self._sparse_index = _SparseIndex(ids, problems, vectorizer, matrix)
            return self._sparse_index

    def _sparse_recall(self, problem: str, top_k: int) -> list[tuple[str, float]]:
        index = self._ensure_sparse_index()
        query = index.vectorizer.transform([problem])
        scores = (index.matrix @ query.T).toarray().ravel()
        if not len(scores):
            return []
        ranked = scores.argsort()[::-1][:max(1, min(top_k, len(scores)))]
        return [(index.ids[int(pos)], float(scores[int(pos)])) for pos in ranked if scores[int(pos)] > 0]

    @staticmethod
    def _overlap(problem: str, candidate: str) -> float:
        left, right = set(math_tokens(problem)), set(math_tokens(candidate))
        if not left or not right:
            return 0.0
        return len(left & right) / math.sqrt(len(left) * len(right))

    def query(
        self,
        problem: str,
        *,
        subject: str = "",
        dense_k: int = 36,
        sparse_k: int = 36,
        final_k: int = 14,
    ) -> list[Dict[str, Any]]:
        """Return a fused candidate pool, ordered for relation-aware filtering."""
        views = _query_views(problem)
        per_view_dense = max(8, math.ceil(max(1, dense_k) / max(1, len(views))))
        per_view_sparse = max(8, math.ceil(max(1, sparse_k) / max(1, len(views))))
        dense_by_id: dict[str, Dict[str, Any]] = {}
        dense_rank_by_id: dict[str, int] = {}
        for view_index, view in enumerate(views):
            for rank, item in enumerate(self.db_client.query(
                    view, top_k=per_view_dense), 1):
                record = dict(item)
                record_id = str(record.get("record_id") or f"dense:{view_index}:{rank}")
                score = float(record.get("similarity") or 0.0)
                if record_id not in dense_by_id or score > float(
                        dense_by_id[record_id].get("similarity") or 0.0):
                    record["record_id"] = record_id
                    record["query_view"] = view_index
                    dense_by_id[record_id] = record
                    dense_rank_by_id[record_id] = rank + view_index * per_view_dense
        dense = sorted(
            dense_by_id.values(),
            key=lambda item: float(item.get("similarity") or 0.0),
            reverse=True,
        )
        for item in dense:
            item["dense_rank"] = dense_rank_by_id.get(str(item.get("record_id")), 1)

        sparse_by_id: dict[str, float] = {}
        for view in views:
            for record_id, score in self._sparse_recall(view, per_view_sparse):
                sparse_by_id[record_id] = max(float(score), sparse_by_id.get(record_id, 0.0))
        sparse = sorted(sparse_by_id.items(), key=lambda pair: pair[1], reverse=True)[:max(1, sparse_k)]

        merged: dict[str, Dict[str, Any]] = {}
        for rank, item in enumerate(dense, 1):
            record = dict(item)
            record_id = str(record.get("record_id") or f"dense:{rank}")
            record["record_id"] = record_id
            record["dense_rank"] = int(record.get("dense_rank") or rank)
            merged[record_id] = record

        sparse_ids = [record_id for record_id, _ in sparse if record_id not in merged]
        if sparse_ids:
            try:
                fetched = self.db_client.fetch_by_ids(
                    sparse_ids, problem,
                    lexical_scores={record_id: score for record_id, score in sparse},
                )
            except Exception:
                fetched = []
            for item in fetched:
                merged[str(item.get("record_id"))] = dict(item)

        for rank, (record_id, score) in enumerate(sparse, 1):
            if record_id in merged:
                merged[record_id]["sparse_rank"] = rank
                merged[record_id]["sparse_similarity"] = score

        for item in merged.values():
            rrf = 0.0
            if item.get("dense_rank"):
                rrf += 1.0 / (60.0 + float(item["dense_rank"]))
            if item.get("sparse_rank"):
                rrf += 1.0 / (60.0 + float(item["sparse_rank"]))
            lexical = float(item.get("sparse_similarity") or 0.0)
            overlap = self._overlap(problem, str(item.get("problem") or ""))
            subject_bonus = 0.08 if subject and subject == item.get("subject") else 0.0
            item["hybrid_score"] = round(rrf + 0.22 * lexical + 0.10 * overlap + subject_bonus, 6)
            item["rerank_evidence"] = {
                "dense_rank": item.get("dense_rank"),
                "sparse_rank": item.get("sparse_rank"),
                "lexical_similarity": round(lexical, 4),
                "token_overlap": round(overlap, 4),
                "subject_match": bool(subject_bonus),
            }

        return sorted(
            merged.values(), key=lambda item: float(item.get("hybrid_score") or 0.0), reverse=True
        )[:max(1, final_k)]

