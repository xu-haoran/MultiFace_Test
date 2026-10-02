"""Dense-similarity relation labels and a separate answer-reuse safety gate."""

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


def _answer_identity(text: str) -> str:
    """Normalize only Unicode and whitespace for conservative answer reuse.

    Unlike the diagnostic problem normalizer, this deliberately preserves case,
    punctuation, LaTeX commands, operators, subscripts, and numeric literals.
    False negatives merely disable answer reuse; false positives could leak a
    different problem's answer.
    """
    value = unicodedata.normalize("NFC", str(text or ""))
    return _SPACE_RE.sub(" ", value).strip()


def dense_similarity_score(item: Dict[str, Any]) -> float | None:
    """Read only a dense score; legacy `similarity` is accepted in isolation."""
    value = item.get("dense_similarity")
    if value is None:
        # Compatibility for callers that still provide only a dense `similarity`.
        # If sparse or fused scores exist, the old field is ambiguous and rejected.
        if item.get("sparse_similarity") is not None or item.get("hybrid_score") is not None:
            return None
        value = item.get("similarity")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def answer_reuse_safety(
    problem: str,
    candidate: Dict[str, Any],
    *,
    high_similarity_min: float = 0.85,
) -> tuple[bool, str]:
    """Allow answer-bearing reuse only for an exact normalized prompt match."""
    if candidate.get("relation") != "high_similarity":
        return False, "not_high_similarity"
    score = dense_similarity_score(candidate)
    if score is None or score <= float(high_similarity_min):
        return False, "no_dense_high_similarity_evidence"
    candidate_problem = str(candidate.get("problem") or "")
    if not problem or not candidate_problem or _answer_identity(problem) != _answer_identity(candidate_problem):
        return False, "problem_not_identical_after_safe_normalization"
    if not (str(candidate.get("solution") or "").strip()
            or str(candidate.get("reference_answer") or "").strip()):
        return False, "reference_answer_missing"
    return True, "exact_problem_with_dense_match_and_reference_answer"


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
    high_similarity_min: float = 0.85,
    related_min_similarity: float = 0.70,
) -> Dict[str, Any]:
    """Return high_similarity/related/irrelevant with evidence.

    Every relation decision uses only dense similarity. Text-normalized
    parameter/qualifier annotations are diagnostic context and do not override
    the score band.
    """
    score = float(similarity or 0.0)

    # 高置信档只依据检索相似度，不使用文本相似度。
    if score > float(high_similarity_min):
        return {
            "relation": "high_similarity",
            "relation_reason": "dense_similarity_above_threshold",
            "changed_parameters": [],
            "changed_qualifiers": [],
        }

    changed = _changed_parameters(problem, candidate_problem)
    critical_mismatch = sorted(_critical_signature(problem) ^ _critical_signature(candidate_problem))
    if score > float(related_min_similarity):
        return {
            "relation": "related",
            "relation_reason": "dense_similarity_in_related_band",
            "changed_parameters": changed,
            "changed_qualifiers": critical_mismatch,
        }

    return {
        "relation": "irrelevant",
        "relation_reason": "dense_similarity_at_or_below_related_threshold",
        "changed_parameters": changed,
        "changed_qualifiers": critical_mismatch,
    }


def annotate_candidates(
    problem: str,
    candidates: Iterable[Dict[str, Any]],
    *,
    high_similarity_min: float = 0.85,
    related_min_similarity: float = 0.70,
) -> list[Dict[str, Any]]:
    annotated = []
    for raw in candidates:
        item = dict(raw or {})
        item["high_similarity_min"] = float(high_similarity_min)
        item["related_min_similarity"] = float(related_min_similarity)
        dense_similarity = dense_similarity_score(item)
        item["dense_similarity"] = dense_similarity
        # Keep old readers working without allowing sparse/fused values here.
        item["similarity"] = dense_similarity
        item.update(classify_relation(
            problem,
            item.get("problem", ""),
            dense_similarity or 0.0,
            high_similarity_min=high_similarity_min,
            related_min_similarity=related_min_similarity,
        ))
        sparse_similarity = item.get("sparse_similarity")
        if dense_similarity is None and sparse_similarity is not None:
            try:
                has_lexical_evidence = float(sparse_similarity) > 0.0
            except (TypeError, ValueError):
                has_lexical_evidence = False
            if has_lexical_evidence:
                # Lexical-only evidence may suggest a method, but cannot enter
                # the dense-similarity bands or authorize answer reuse.
                item["relation"] = "sparse_reference"
                item["relation_reason"] = "sparse_only_method_reference"
        safe, reason = answer_reuse_safety(
            problem, item, high_similarity_min=high_similarity_min
        )
        item["answer_safe"] = safe
        item["answer_safety_reason"] = reason
        annotated.append(item)
    return annotated


def select_for_injection(
    candidates: Iterable[Dict[str, Any]],
    *,
    related_limit: int = 3,
) -> list[Dict[str, Any]]:
    """Keep a small relation-aware context; high-similarity gets one full reference."""
    values = list(candidates)
    high_similarity = [item for item in values
                       if item.get("relation") == "high_similarity"]
    if high_similarity:
        # The dense-similarity band controls visibility of one complete example;
        # answer_reuse_safety remains a separate gate for adopting its answer.
        return high_similarity[:1]
    method_only = [item for item in values
                   if item.get("relation") in {"related", "sparse_reference"}]
    if method_only and related_limit > 0:
        return method_only[:related_limit]
    return []

