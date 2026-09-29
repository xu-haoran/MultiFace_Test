"""Relation labels for retrieved mathematics problems.

All relation tiers are determined by the retrieval score supplied by the
database layer.  A strict ``>0.88`` gate marks a high-similarity record whose
full problem and solution may be shown to the solver; lower score bands are
template or irrelevant. Text normalization is retained only for
diagnostic parameter/qualifier annotations, never for tier selection.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, Iterable


_NUMBER_RE = re.compile(
    r"(?<![A-Za-z\\])(?:[-+]?\d+(?:\.\d+)?(?:/\d+)?)(?![A-Za-z])"
)
_SPACE_RE = re.compile(r"\s+")
_LATEX_SPACE_RE = re.compile(r"\\(?:,|;|!|quad\b|qquad\b)")
_PUNCT_TRANSLATION = str.maketrans({
    "，": ",", "。": ".", "；": ";", "：": ":", "（": "(", "）": ")",
    "【": "[", "】": "]", "“": '"', "”": '"', "‘": "'", "’": "'",
    "−": "-", "–": "-", "—": "-", "×": "*", "÷": "/",
})

# 会改变答案的条件只作诊断，不覆盖相似度关系分档。
_CRITICAL_GROUPS = {
    "domain_real": ("实数", "real"),
    "domain_integer": ("整数", "integer"),
    "domain_complex": ("复数", "complex"),
    "domain_positive": ("正数", "positive"),
    "all": ("所有", "任意", "for all", "every"),
    "exists": ("存在", "there exists"),
    "unique": ("唯一", "unique"),
    "maximum": ("最大", "maximum", "max"),
    "minimum": ("最小", "minimum", "min"),
    "strict_lt": ("<",),
    "weak_lt": ("<=", "≤", r"\le", r"\leq"),
    "strict_gt": (">",),
    "weak_gt": (">=", "≥", r"\ge", r"\geq"),
    # 这些条件无法由数字替换覆盖，差异会降级候选并抑制照抄结论。
    "parity_even": ("偶数", "even"),
    "parity_odd": ("奇数", "odd"),
    "distinct": ("互不相同", "两两不同", "各不相同", "distinct", "pairwise different"),
    "nonnegative": ("非负", "nonnegative", "non-negative"),
    "nonzero": ("非零", "不为零", "nonzero", "non-zero"),
    "divisibility": ("整除", "可被", "divisible", "divides"),
    "congruence": ("同余", "余数", "模", "congruent", "modulo"),
    "at_least": ("至少", "不少于", "at least", "no fewer than"),
    "at_most": ("至多", "不超过", "at most", "no more than"),
    "connected": ("连通", "connected"),
    "independent": ("独立", "independent"),
    "directed": ("有向", "directed"),
    "undirected": ("无向", "undirected"),
}


def normalize_problem(text: str) -> str:
    """Normalize harmless typography while retaining mathematical conditions."""
    value = unicodedata.normalize("NFKC", str(text or "")).translate(_PUNCT_TRANSLATION)
    value = re.sub(r"(?m)^\s*[-*•◦]\s+(?=[A-Za-z\\(])", "", value)
    value = re.sub(r"\\(?:\(|\)|\[|\])", "", value)
    value = value.replace(r"\leqslant", "≤").replace(r"\leq", "≤").replace(r"\le", "≤")
    value = value.replace(r"\geqslant", "≥").replace(r"\geq", "≥").replace(r"\ge", "≥")
    value = re.sub(r"_\{([^{}]+)\}", r"_\1", value)
    value = re.sub(r"_(\d+)", r"\1", value)
    value = value.replace("_", "")
    value = value.replace(r"\left", "").replace(r"\right", "")
    value = value.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    value = _LATEX_SPACE_RE.sub("", value)
    value = _SPACE_RE.sub("", value).casefold()
    return value.strip(".,;:!?。；：！？")


def number_literals(text: str) -> list[str]:
    return _NUMBER_RE.findall(normalize_problem(text))


def template_signature(text: str) -> str:
    return _NUMBER_RE.sub("<number>", normalize_problem(text))


def _critical_signature(text: str) -> frozenset[str]:
    normalized = normalize_problem(text)
    found = set()
    for name, terms in _CRITICAL_GROUPS.items():
        if any(term.casefold() in normalized for term in terms):
            found.add(name)
    return frozenset(found)


def _changed_parameters(problem: str, candidate: str) -> list[str]:
    ours = number_literals(problem)
    theirs = number_literals(candidate)
    changes = []
    for i, (left, right) in enumerate(zip(ours, theirs), 1):
        if left != right:
            changes.append(f"参数{i}: 参考题 {right} → 本题 {left}")
    if len(ours) != len(theirs):
        changes.append(f"数值项数量: 参考题 {len(theirs)} → 本题 {len(ours)}")
    return changes[:12]


def classify_relation(
    problem: str,
    candidate_problem: str,
    similarity: float,
    *,
    high_similarity_min: float = 0.88,
    template_min_similarity: float = 0.70,
) -> Dict[str, Any]:
    """Return high_similarity/template/irrelevant with evidence.

    Every relation decision uses only ``similarity``.  Text-normalized
    parameter/qualifier annotations are diagnostic context and do not override
    the score band.
    """
    score = float(similarity or 0.0)

    # 高置信档只依据检索相似度，不使用文本相似度。
    if score > float(high_similarity_min):
        return {
            "relation": "high_similarity",
            "relation_reason": "retrieval_similarity_above_threshold",
            "changed_parameters": [],
            "changed_qualifiers": [],
        }

    changed = _changed_parameters(problem, candidate_problem)
    critical_mismatch = sorted(_critical_signature(problem) ^ _critical_signature(candidate_problem))
    if score > template_min_similarity:
        return {
            "relation": "template",
            "relation_reason": "retrieval_similarity_in_template_band",
            "changed_parameters": changed,
            "changed_qualifiers": critical_mismatch,
        }

    return {
        "relation": "irrelevant",
        "relation_reason": "retrieval_similarity_at_or_below_template_threshold",
        "changed_parameters": changed,
        "changed_qualifiers": critical_mismatch,
    }


def annotate_candidates(
    problem: str,
    candidates: Iterable[Dict[str, Any]],
    *,
    high_similarity_min: float = 0.88,
    template_min_similarity: float = 0.70,
) -> list[Dict[str, Any]]:
    annotated = []
    for raw in candidates:
        item = dict(raw or {})
        item.update(classify_relation(
            problem,
            item.get("problem", ""),
            item.get("similarity", 0.0),
            high_similarity_min=high_similarity_min,
            template_min_similarity=template_min_similarity,
        ))
        # 只有高相似记录可提供解答正文；模板题仅作方法提示。
        item["answer_safe"] = item.get("relation") == "high_similarity"
        annotated.append(item)
    return annotated


def select_for_injection(
    candidates: Iterable[Dict[str, Any]],
    *,
    template_limit: int = 3,
) -> list[Dict[str, Any]]:
    """Keep a small, relation-aware context after broad retrieval."""
    values = list(candidates)
    high_similarity = [item for item in values if item.get("relation") == "high_similarity"]
    if high_similarity:
        return high_similarity[:1]
    template = [item for item in values if item.get("relation") == "template"]
    if template and template_limit > 0:
        return template[:max(1, template_limit)]
    return []

