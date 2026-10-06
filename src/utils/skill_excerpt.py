"""Select complete, relevant local knowledge cards within a prompt budget."""

from __future__ import annotations

import re

#: Module boundary in these documents: the legacy ``## 模块...``/``### 模块...``
#: heading, the numbered ``## 1. ...`` style, or the CJK-numbered ``## 一、...``
#:
#: 2026-08-20：漏掉中文序号是一个静默的大洞。高等代数的 ``## 一、多项式理论`` …
#: ``## 十、习题索引`` 十个正文章节都不算边界，于是整册 60% 的正文（15292/25101
#: 字符）全被并进 preamble；preamble 超预算后走头部封顶分支，正文永远只露出开头
#: 1200 字符，真正对口的章节一个字也到不了模型手里。数学分析同理（52%）。
#: idx 6/7/17（高等代数）与 8/27（数学分析）正落在这两册里。
#:
#: 2026-08-22：无 API 全量观测发现 8/18 领域文档把模块写成三级标题 ``### 模块N：``
#: （复分析/常微分方程完全不可见，抽象代数/概率论/泛函分析/线性回归/统计推断/
#: 随机过程部分不可见），``##`` 前缀不再匹配，整册退回 doc[:limit] 头部截断。
#: 模块分支放宽为 ``#{2,3}``；数字/中文序号分支保持 ``##`` 不变——``### 1. 题型识别``
#: 这类三级数字子标题不应成为边界（2026-08-22 复分析文档实测）。
#:
#: 同日再补两种变体：微分几何整册用 ``## 1 曲线弧长``（数字+空格+标题，无标点，
#: 20 个知识模块因此全部不可见，preamble 21663 字符）——新增 ``\d+\s+标题``
#: 分支（含 ``## 3 Frenet标架``/``## 19 Gauss-Bonnet定理`` 这类拉丁开头的标题）；
#: 其余 12 个文档文末的 ``## 通用解题方法论``/``## sympy验证技巧``/``## 习题索引``/
#: ``## 常见错误与陷阱``/``## 关键公式速查表``/``## 参考资源`` 等通用章节被并入最后
#: 一个真模块（复分析模块9 因此膨胀到 24201 字符，其词表成为全集、对所有题霸榜），
#: 文档侧已统一改题为 ``## 模块速查：…``（与离散数学/高等代数惯例一致）。
_MODULE_RE = re.compile(
    r"^(?:#{2,3}\s*(?:模块|知识卡：)|##\s*(?:\d+\s*[.)、:：-]|\d+\s+(?:[一-鿿]|[A-Za-z])|"
    r"[一二三四五六七八九十百]+\s*[、.)：:-]))",
    re.MULTILINE,
)
_UNIVERSAL_HEADING_RE = re.compile(r"^##\s*(?:跨学科|通用|附录)", re.MULTILINE)
_TERM_RE = re.compile(r"[一-鿿]{2,}|[A-Za-z][A-Za-z\-]{2,}")
_ANSWER_ANCHOR_RE = re.compile(
    r"(?i)(标准答案|最终答案|答案为|答案是|正确选项|作答模板|判分口径|"
    r"the\s+answer\s+is|final\s+answer|correct\s+option)"
)
_BOXED_RE = re.compile(r"\\boxed\s*\{([^{}]{1,160})\}")
_ANSWER_TAIL_RE = re.compile(
    r"(?i)(?:最终答案|标准答案|答案为|答案是|correct\s+answer|final\s+answer)"
    r"\s*[:：=]?\s*[^\n。；;]{0,180}"
)


def sanitize_answer_anchors(text: str) -> str:
    """Hide case-specific answer keys while preserving method/condition prose."""
    cleaned = []
    for line in str(text or "").splitlines():
        if _ANSWER_ANCHOR_RE.search(line):
            # Headings such as ``### 作答模板`` should not expose the template's
            if line.lstrip().startswith("#"):
                prefix = line[:len(line) - len(line.lstrip())]
                cleaned.append(prefix + "### 方法提示（具体案例结论已隐藏）")
            else:
                cleaned.append("（具体案例结论已隐藏；请按本题条件独立推导。）")
            continue
        line = _BOXED_RE.sub(r"\\boxed{<本题独立结论>}", line)
        line = _ANSWER_TAIL_RE.sub("结论（需按本题独立计算）", line)
        cleaned.append(line)
    return "\n".join(cleaned)

#: 游戏/必胜/策略 and never the word 博弈论, so pure term overlap scored the module that
_TOPIC_CUES = {
    "博弈": ("游戏", "必胜", "策略", "先手", "后手", "轮流", "获胜", "对手", "取石子",
             "双人", "玩家", "甲", "乙",
             # 英文竞赛题面同类线索（term 交集打分吃不到，此处补齐）。
             "game", "player", "players", "opponent", "wins", "winning",
             "turns", "alternately", "strategy"),
    "数论": ("整除", "同余", "素数", "质数", "互素", "模", "gcd", "最大公约", "幂次",
             "正整数", "因子", "倍数",
             # 只收判别力强的英文词——integer/number 之类的泛词会把数论模块
             # 错误地拉进一半的英文题面。
             "prime", "divisible", "divides", "modulo", "coprime"),
}


def _cue_bonus(module: str, problem: str) -> int:
    """Weight a module by domain cues present in the problem statement.

    单模块封顶 9 分：博弈类速查条目共享"博弈"标题，线索加成不封顶时同族条目
    会集体霸榜，把真正对口但标题不带关键词的条目挤出预算（2026-08-09 idx 48）。
    """
    heading = module.split("\n", 1)[0]
    text = (problem or "").casefold()
    bonus = 0
    for key, cues in _TOPIC_CUES.items():
        if key in heading:
            bonus += 3 * sum(1 for cue in cues if cue.casefold() in text)
    return min(bonus, 9)


#: 速查条目携带的显式检索行。中文题面是长 CJK 连续串，_TERM_RE 的"最大连续段"词元
#: 对它几乎必然失配（"概率不超过" ≠ "概率"），所以检索行按**子串**语义单独记分：
#: 行内每个词元（中文或 ≥4 字符拉丁词）在题面中出现即 +2，短拉丁词按词边界匹配。
#: 上限封顶，避免一行通用词把真正的领域模块挤出预算。
_RETRIEVAL_LINE_RE = re.compile(r"^-\s*检索词[：:](.+)$", re.MULTILINE)

#: 检索词行里的虚词一律不计分。这类词在任何英文题面里都存在，一旦混进检索词，
#: 该模块就会对所有英文题目拿满命中分，把真正对口的模块挤掉——2026-08-20 把
#: 「选数禁邻博弈」的检索词误写成自然语言短语（"...to any of those the player has
#: already chosen..."）后，它立刻劫持了 idx 14 这道与博弈毫无关系的分拆题。
#: 护栏放在打分侧而不是靠人写对：检索词行由人手写，迟早会再混进虚词。
_RETRIEVAL_STOPWORDS = frozenset({
    "the", "a", "an", "of", "to", "is", "are", "be", "in", "on", "at", "by",
    "or", "and", "for", "with", "that", "this", "these", "those", "it", "its",
    "as", "if", "then", "than", "so", "such", "any", "all", "each", "every",
    "has", "have", "had", "was", "were", "not", "no", "we", "you", "he", "she",
    "they", "one", "two", "can", "may", "must", "will", "from", "into", "out",
})


def _retrieval_bonus(module: str, problem: str) -> int:
    match = _RETRIEVAL_LINE_RE.search(module)
    if not match:
        return 0
    text = (problem or "").casefold()
    hits = 0
    for token in match.group(1).split():
        tok = token.casefold().strip()
        if len(tok) < 2 or tok in _RETRIEVAL_STOPWORDS:
            continue
        if re.fullmatch(r"[a-z][a-z\-]*", tok):
            if len(tok) >= 4:
                hit = tok in text
            else:
                hit = bool(re.search(rf"(?<![a-z]){re.escape(tok)}(?![a-z])", text))
        else:
            hit = tok in text
        if hit:
            hits += 1

    # 单词元命中只记 1 分：一个泛词不足以断言主题相关，不得压过正文模块的
    # 实词匹配（同分时文档序在前的正文模块优先）。多词元命中才给满权重。
    return hits if hits < 2 else 2 * hits


def _split_modules(document: str, boundary=_MODULE_RE) -> tuple[str, list[str]]:
    """(preamble, sections). `sections` is empty when no boundary matches."""
    starts = [match.start() for match in boundary.finditer(document)]
    if not starts:
        return document, []
    bounds = starts + [len(document)]
    modules = [document[bounds[i]:bounds[i + 1]] for i in range(len(starts))]
    return document[:starts[0]], modules


def _terms(text: str) -> set[str]:
    return {match.group(0).lower() for match in _TERM_RE.finditer(text or "")}


def _score(module: str, problem_terms: set[str]) -> int:
    """Overlap between a module's vocabulary and the problem's.

    The heading counts double: a module titled 博弈论 is a stronger signal for a game
    problem than one incidental mention of 博弈 deep inside another module.
    """
    heading = module.split("\n", 1)[0]
    module_terms = _terms(module)
    heading_terms = _terms(heading)
    return len(module_terms & problem_terms) + 2 * len(heading_terms & problem_terms)


_SCRIPT_SECTION_RE = re.compile(r"^(?:##\s+|#\s*(?:=====|-----))", re.MULTILINE)


def select_script_excerpt(script: str, problem: str, limit: int) -> str:
    """`select_skill_excerpt` for prompt-style validation references.

    A bare ``[:2000]`` can still hide the relevant knowledge point in a long domain
    prompt, so choose the sections whose vocabulary matches the problem instead.
    """
    return _select(script, problem, limit, _SCRIPT_SECTION_RE, sanitize=False)


def select_skill_excerpt(document: str, problem: str, limit: int) -> str:
    """At most `limit` characters of `document`, biased toward `problem`'s topic.

    Selects relevant complete cards within the budget; oversized units are omitted.
    """
    return _select(document, problem, limit, _MODULE_RE, sanitize=True)


def selected_card_ids(excerpt: str) -> list[str]:
    """IDs of complete cards actually injected, for auditable prompt traces."""
    return re.findall(r"^## 知识卡：([^ |\n]+)", excerpt or "", re.MULTILINE)


def _select(document: str, problem: str, limit: int, boundary, *, sanitize: bool = True) -> str:
    """Select complete units only; never truncate a theorem or executable block."""
    if limit <= 0:
        return ""
    finish = sanitize_answer_anchors if sanitize else (lambda value: value)
    text = document or ""
    preamble, modules = _split_modules(text, boundary)
    if not modules:
        # An unstructured oversized document has no safe truncation boundary.
        return finish(text) if len(finish(text)) <= limit else ""
    head = finish(preamble).strip()
    if len(head) > min(300, limit):
        head = ""
    problem_terms = _terms(problem)
    ranked = sorted(enumerate(modules), key=lambda item: (
        -(_score(item[1], problem_terms) + _cue_bonus(item[1], problem)
          + _retrieval_bonus(item[1], problem)), item[0]))
    pieces = [head] if head else []
    used = len(head)
    for index, module in ranked:
        score = (_score(module, problem_terms) + _cue_bonus(module, problem)
                 + _retrieval_bonus(module, problem))
        if score <= 0:
            continue
        unit = finish(module).strip()
        if module.startswith("## 知识卡："):
            match = re.match(r"## 知识卡：([^ |\n]+)", module)
            if not match or f"<!-- END_CARD {match[1]} -->" not in module:
                continue  # malformed cards must not inject partial claims
            required = ("前提", "公式与结论", "反例与边界", "记号", "验证", "来源", "适用范围")
            if not all(f"- {field}：" in unit for field in required):
                continue
            scope = re.search(r"^- 适用范围：(.+)$", unit, re.MULTILINE)
            if scope and "教材约定" in scope[1]:
                source = re.search(r"^- 来源：(.+)$", unit, re.MULTILINE)
                matches = re.search(r"^- 约定匹配词：(.+)$", unit, re.MULTILINE)
                if (not source or re.search(r"未知|未核实|项目原手册|项目数学推导", source[1])
                        or not matches or not any(token.strip() in problem
                            for token in matches[1].split() if len(token.strip()) >= 3)):
                    continue
        cost = len(unit) + (2 if pieces else 0)
        if used + cost <= limit:
            pieces.append(unit)
            used += cost
    # No relevant complete card is preferable to unrelated handbook openings.
    return "\n\n".join(pieces)
