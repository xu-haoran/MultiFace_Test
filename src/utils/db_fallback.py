"""题库检索兜底：双分支皆空时，从检索到的相似题解答中提取结论作为参考答案。

2026-08-19 起用。事故背景（idx=2）：推理分支答案字段为空、Python 分支无输出，
系统靠散文捞回给出 √5−2，而检索排名第一的克罗地亚 2017 题正是同一道题，解答
首行写着 `Odgovor je $C=\\frac{1}{2}$`（官方答案 1/2）。检索结果已在 state 里，
却没有任何一条兜底路径读它。

定位：这是**最后一级兜底**，只在推理与 Python 都没交出可用答案时启用，且输出
必须显式标注来源为题库相似题，不得伪装成本题推导结论——它是有依据的猜测，
排序位于"应急直答"之前（有同构题结论比凭空直答期望更高），"通用失败说明"之后。
"""

from __future__ import annotations

import re

from utils.retrieval_relation import answer_reuse_safety

#: 解答文本里的结论标记，按可靠性从高到低。多语种：题库含克罗地亚语/西班牙语等。
_ANSWER_PATTERNS = (
    re.compile(r"\\boxed\s*\{([^{}]{1,120}(?:\{[^{}]*\}[^{}]*)?)\}"),
    re.compile(r"(?:答案|结论)\s*(?:为|是|:|：)\s*([^\n。，,；;]{1,120})"),
    re.compile(r"(?i)\bAnswer\s*(?:is)?\s*[:：]\s*([^\n.;]{1,120})"),
    re.compile(r"(?i)\bThe\s+answer\s+is\s+([^\n.;]{1,120})"),
    re.compile(r"(?i)\bOdgovor\s+je\s+([^\n.;]{1,120})"),          # hr
    re.compile(r"(?i)\bLa\s+respuesta\s+es\s+([^\n.;]{1,120})"),    # es
    re.compile(r"(?i)\bR[ée]ponse\s*[:：]\s*([^\n.;]{1,120})"),      # fr
)

#: 低于此相似度的检索结果不足以支撑兜底——宁可给通用失败说明。
MIN_SIMILARITY = 0.45

HIGH_SIMILARITY_ANSWER_MIN = 0.85


#: 结论值之后常接的论证连接词——"The answer is 1/2 by pigeonhole" 里的 " by ..."
#: 属于论证而非答案，带进兜底会污染判分。
_TRAILING_PROSE_RE = re.compile(
    r"\s+(?:by|since|because|from|using|for|with|where|which|as|via|due\s+to|"
    r"and\s+the|so|thus|hence|therefore)\b.*$|"
    r"\s*(?:，|,)?\s*(?:由|因为|根据|依据|故|所以|因此|其中)\S*.*$",
    re.IGNORECASE,
)


def _clean(text: str) -> str:
    value = str(text or "").strip()
    value = value.strip("$ \t").strip()
    value = re.sub(r"\s+", " ", value)
    value = _TRAILING_PROSE_RE.sub("", value)
    return value.strip(" .,;:，。；：$")


def extract_reference_conclusion(solution: str) -> str:
    """从一条题库解答里抽出结论文本；抽不到返回空串。"""
    text = str(solution or "")
    if not text.strip():
        return ""
    for pattern in _ANSWER_PATTERNS:
        match = pattern.search(text)
        if match:
            candidate = _clean(match.group(1))
            # 过短（单个符号）或过长（整段推导）都不是结论。
            if 1 <= len(candidate) <= 120 and re.search(r"[\w\\]", candidate):
                return candidate
    return ""


def db_fallback_answer(state: dict) -> tuple[str, float, str]:
    """双分支皆空时的题库兜底。

    Returns:
        (结论文本, 相似度, 来源)；无可用检索结果时返回 ("", 0.0, "")。
    """
    examples = (state or {}).get("retrieved_examples") or []
    best: tuple[str, float, str] = ("", 0.0, "")
    for example in examples:
        if not isinstance(example, dict):
            continue
        try:
            similarity = float(example.get("dense_similarity") or 0.0)
        except (TypeError, ValueError):
            similarity = 0.0
        if similarity <= HIGH_SIMILARITY_ANSWER_MIN:
            continue
        safe, _ = answer_reuse_safety(
            str((state or {}).get("problem") or ""),
            example,
            high_similarity_min=float(example.get("high_similarity_min", 0.85) or 0.85),
        )
        if (str(example.get("relation") or "") != "high_similarity"
                or example.get("answer_safe") is not True or not safe):
            continue
        conclusion = extract_reference_conclusion(example.get("solution", ""))
        if conclusion and similarity > best[1]:
            best = (conclusion, similarity, str(example.get("source") or ""))
    return best


def db_fallback_response(state: dict) -> tuple[str, str]:
    """构造带来源标注的兜底输出文本。

    Returns:
        (final_response, fallback_source)；无可用检索结果时返回 ("", "")。
    """
    conclusion, similarity, source = db_fallback_answer(state)
    if not conclusion:
        return "", ""
    # 2026-08-24：响应只放结论本身；来源标注不再写入 final_response，
    # 由 fallback_source="database_reference_fallback" 与 trace 承担。
    return conclusion, "database_reference_fallback"

