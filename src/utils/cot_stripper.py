import re


_COT_PREFIX_RE = re.compile(
    r"^\s*(?:Thinking Process|Thought Process|Thinking|Reasoning|Let me think|思考过程|推理过程|分析过程)\s*[：:]",
    re.IGNORECASE,
)

_CONTENT_START_RE = re.compile(
    r"(?m)^\s*(?:"
    r"\{"
    r"|```(?:python)?"
    r"|##\s+"
    r"|(?:\d+[.、]\s*)?问题理解"
    r"|(?:\d+[.、]\s*)?解题思路"
    r"|(?:\d+[.、]\s*)?详细步骤"
    r"|(?:\d+[.、]\s*)?答案验证"
    r"|(?:\d+[.、]\s*)?最终答案"
    r"|结论[：:]"
    r"|解[：:]"
    r"|证明[：:]"
    r")"
)

_VISIBLE_CHARS_RE = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]")
_PUNCT_ONLY_RE = re.compile(r"^[\s\W_]+$", re.UNICODE)

#: \u63d0\u793a\u8bcd\u6a21\u677f\u9aa8\u67b6\u7684\u4e2d\u62ec\u53f7\u6bb5\uff082026-08-09 \u5192\u70df idx 66\uff1a"- [\u9a8c\u8bc1\u70b91\uff1a\u5982\u4f55\u9a8c\u8bc1\u8fd9\u4e2a
#: \u7b54\u6848\u662f\u6b63\u786e\u7684]"\u3001"\u6b65\u9aa41\uff1a[\u63cf\u8ff0]\uff08[\u5177\u4f53\u516c\u5f0f]\uff09" \u66fe\u4ee5 validated answer \u51fa\u5382\uff09\u3002
_TEMPLATE_TOKEN_RE = re.compile(
    r"[\[\u3010]\s*(?:\u63cf\u8ff0|\u5177\u4f53\u516c\u5f0f|\u9a8c\u8bc1\u70b9|\u7ed3\u8bba|\u7b54\u6848|\u65b9\u6cd5|\u6b65\u9aa4|\u8fc7\u7a0b|\u8bf4\u660e|\u6570\u503c|\u8868\u8fbe\u5f0f|"
    r"\u7ed3\u679c|\u586b\u5199|\u6b64\u5904|\u4f60\u7684)[^\]\u3011]*[\]\u3011]"
)

_PLACEHOLDERS = {
    "result",
    "answer",
    "finalanswer",
    "final_answer",
    "placeholder",
    "todo",
    "none",
    "null",
    "n/a",
    "na",
    "明确的最终结果",
    "最终结果",
    "待求",
    "占位",
    "无法确定",
    "不能确定",
    "未知",
}


def strip_cot_prefix(text: str) -> str:
    """Remove a leading CoT preamble when a real content marker follows it."""
    value = text or ""
    if not value.strip():
        return ""
    if not _COT_PREFIX_RE.search(value[:300]):
        return value.strip()

    prefix = _COT_PREFIX_RE.search(value[:300])
    prefix_end = prefix.end() if prefix else 0
    for match in _CONTENT_START_RE.finditer(value):
        if match.start() >= prefix_end:
            return value[match.start():].strip()
    return ""


def contains_cot_marker(text: str) -> bool:
    return bool(_COT_PREFIX_RE.search((text or "")[:300]))


def _normalize_placeholder_text(text: str) -> str:
    value = strip_cot_prefix(text or "")
    value = value.strip().strip("`").strip("$").strip()
    value = value.strip("[]【】()（）{}<>")
    value = re.sub(r"\s+", "", value).lower()
    return value.replace("-", "").replace("_", "")


def is_placeholder_answer(text: str) -> bool:
    value = (text or "").strip()
    if not value:
        return True
    normalized = _normalize_placeholder_text(value)
    if not normalized:
        return True
    if normalized in {item.replace("_", "").replace("-", "") for item in _PLACEHOLDERS}:
        return True
    if _PUNCT_ONLY_RE.match(value) and not _VISIBLE_CHARS_RE.search(value):
        return True
    # 模板骨架回显：剥掉全部模板中括号段后不再有信息负载（数字/等式/公式/成词中文）
    # → 是骨架不是答案。带真实内容的混合行（"[结论] x=3"）不受影响。
    if _TEMPLATE_TOKEN_RE.search(value):
        residue = _TEMPLATE_TOKEN_RE.sub("", value)
        # 结构性标签（步骤N/关键步骤/验证点N）不是信息负载，一并剥掉再判。
        residue = re.sub(r"(?:关键)?步骤\s*\d*|验证点\s*\d*|[：:（()）\[\]【】\s\-•|]+", " ", residue)
        if not re.search(r"[0-9=]|\\[a-zA-Z]{2,}|[一-鿿]{2,}", residue):
            return True
    return False
