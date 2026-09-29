"""Retrieve compact, traceable undergraduate-mathematics knowledge cards.

Cards are derived solely from the repository's existing 18 subject handbooks
and verification notes.  They are method/constraint aids, never answer keys,
so they complement rather than contaminate the public problem collection.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List

from utils.hybrid_retriever import math_tokens
from utils.skill_excerpt import sanitize_answer_anchors


def _section(text: str, anchors: tuple[str, ...], limit: int) -> str:
    """Take a short section after the first matching heading or phrase."""
    for anchor in anchors:
        start = text.find(anchor)
        if start >= 0:
            tail = text[start:start + limit * 2]
            boundary = re.search(r"\n---\s*\n|\n##\s+", tail[80:])
            if boundary:
                tail = tail[:80 + boundary.start()]
            return tail.strip()[:limit]
    return text.strip()[:limit]


class KnowledgeCardRetriever:
    """In-memory index of routing, method and verification cards."""

    def __init__(self, skills_loader: Any, db_client: Any = None):
        self.skills_loader = skills_loader
        self.db_client = db_client
        self._cards: List[Dict[str, Any]] | None = None

    def _build_cards(self) -> List[Dict[str, Any]]:
        cards: List[Dict[str, Any]] = []
        for subject in self.skills_loader.categories:
            handbook = self.skills_loader.get_skill_document(subject)
            verification = self.skills_loader.get_validation_script(subject)
            source_root = f"resources/skills/{subject}"
            cards.extend([
                {
                    "card_id": f"{subject}:routing",
                    "subject": subject,
                    "card_type": "constraint",
                    "title": f"{subject}：适用边界与易错条件",
                    "content": sanitize_answer_anchors(
                        _section(handbook, ("领域边界与路由约定", "高频错误"), 720)),
                    "source": f"{source_root}/{subject}skill.md",
                },
                {
                    "card_id": f"{subject}:method",
                    "subject": subject,
                    "card_type": "method",
                    "title": f"{subject}：方法与定理索引",
                    "content": sanitize_answer_anchors(
                        _section(handbook, ("知识点体系", "第一部分：知识模块", "概述"), 980)),
                    "source": f"{source_root}/{subject}skill.md",
                },
                {
                    "card_id": f"{subject}:verification",
                    "subject": subject,
                    "card_type": "verification",
                    "title": f"{subject}：验证约束与脚本索引",
                    "content": _section(verification, ("使用规则", "验证提示词", "知识点"), 680),
                    "source": f"{source_root}/{subject}验证示例.py",
                },
            ])
        return cards

    def _all_cards(self) -> List[Dict[str, Any]]:
        if self._cards is None:
            self._cards = self._build_cards()
        return self._cards

    def _query_vector_collection(self, problem: str, subject: str,
                                 limit: int, subjects: list[str] | None = None) -> List[Dict[str, Any]]:
        """Use the separately persisted card collection when it is available.

        The collection is deliberately separate from answer-bearing problems, so
        its records can never enter the high-similarity/template answer path.
        """
        if self.db_client is None:
            return []
        self.db_client._load_model()
        self.db_client._load_collection()
        collection = self.db_client._client.get_collection("undergrad_knowledge_cards")
        vector = self.db_client._model.encode(
            [problem], normalize_embeddings=True, convert_to_numpy=True
        )[0].tolist()
        raw = collection.query(
            query_embeddings=[vector], n_results=max(6, limit * 2),
            include=["documents", "metadatas", "distances"],
        )
        allowed_subjects = set(subjects or ([] if not subject else [subject]))
        candidates: List[Dict[str, Any]] = []
        for document, metadata, distance in zip(
            raw.get("documents", [[]])[0], raw.get("metadatas", [[]])[0],
            raw.get("distances", [[]])[0],
        ):
            if allowed_subjects and metadata.get("subject") not in allowed_subjects:
                continue
            parts = str(document or "").split("## Solution", 1)
            title = parts[0].replace("## Problem", "").strip()
            content = parts[1].strip() if len(parts) == 2 else str(document or "")
            candidates.append({
                "card_id": "vector:" + str(metadata.get("subject") or "") + ":" + str(metadata.get("card_type") or ""),
                "subject": metadata.get("subject", ""),
                "card_type": metadata.get("card_type", "method"),
                "title": title,
                "content": sanitize_answer_anchors(content),
                "source": metadata.get("source", "resources/skills"),
                "knowledge_score": round(1 - float(distance), 4),
                "retrieval_source": "undergrad_knowledge_cards",
            })
        return candidates[:max(1, limit)]

    def query(self, problem: str, *, subject: str = "", limit: int = 3,
              subjects: list[str] | None = None) -> List[Dict[str, Any]]:
        """Return a compact, diverse set of method cards for the active domain."""
        try:
            vector_cards = self._query_vector_collection(problem, subject, limit, subjects)
            if vector_cards:
                return vector_cards
        except Exception:
            pass
        candidates = [name for name, score in self.skills_loader.find_candidate_categories(problem, top_k=3) if score > 0]
        if subjects:
            candidates = list(subjects) + [name for name in candidates if name not in subjects]
        if subject and subject in self.skills_loader.categories:
            candidates = [subject] + [name for name in candidates if name != subject]
        if not candidates:
            return []

        wanted = set(candidates[:3])
        query_terms = set(math_tokens(problem))
        scored = []
        for raw in self._all_cards():
            if raw["subject"] not in wanted:
                continue
            terms = set(math_tokens(raw["title"] + "\n" + raw["content"]))
            overlap = len(query_terms & terms) / max(1, len(query_terms))
            primary = 0.12 if raw["subject"] == subject else 0.0
            score = overlap + primary
            scored.append((score, raw))
        scored.sort(key=lambda pair: (pair[0], pair[1]["subject"] == subject), reverse=True)

        selected: List[Dict[str, Any]] = []
        seen_types: set[str] = set()
        for score, raw in scored:
            if raw["card_type"] in seen_types and len(selected) < limit - 1:
                continue
            item = dict(raw)
            item["knowledge_score"] = round(float(score), 4)
            selected.append(item)
            seen_types.add(raw["card_type"])
            if len(selected) >= max(1, limit):
                break
        return selected

