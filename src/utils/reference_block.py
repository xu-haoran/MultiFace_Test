"""题库参考示例区块的唯一构造入口。

推理节点与 Python 节点各自内联拼装过同一段文本，两份实现的截断长度不同、
措辞也会各自漂移。检索关系和注入数量由检索节点统一决定：高相似度最多 1 条，
没有高相似度时相关题型最多 3 条；本模块只负责把已筛选的记录按安全规则格式化，
调用方只保留各自的截断额度。

2026-08-20：区块必须自带**反锚定**说明。检索是向量近邻，命中的常常是"看起来
几乎一样、答案却不同"的近似题，而这正是评测中最贵的一类失分：

* idx 48（黑板上 1997 个 1 的取数博弈）：题库以 0.773 命中 ISL 2020 的同题型
  （2020 个 1、B 可自由选择），其解法用二进制数字和 S₂(n)。代理照搬得
  S₂(1997)=8，而本题因为**个数是奇数**且**硬币剥夺了 B 的选择权**，正解是 999。
* idx 17（x²+y²+z²=xy³+yz³+zx³=3 的实数解个数）：题库以 0.814 命中 USAMO 1973
  的 x+y+z=x²+y²+z²=x³+y³+z³=3（答案只有对称解 1,1,1），代理据此只数出对称解
  得 2，而本题的非对称轨道使正解为 8。

两次都不是"检索没找到"，而是"找到了近似题并把它的结论当成本题的结论"。因此
区块除了给出示例，还必须（1）显式声明结论不可迁移，(2) 把两边题面的数值差异
直接摆出来，让模型无法忽略参数已经变了这件事。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

_HEADER = "\n\n参考示例（统一数学题库召回，已做结构关系检查）：\n"
_KNOWLEDGE_HEADER = "\n\n本科数学知识卡（项目内学科手册提炼）：\n"

#: 反锚定说明。放在示例**之前**——写在后面时模型往往已经先读完解答并锚定了结论。
_ANTI_ANCHOR_NOTE = """
⚠️ 这些记录由向量检索召回。标为“高相似度直供”的记录表示**检索相似度**严格
大于 0.85，题目与完整解答会直接提供给代理。其余都是近似题，不能直接搬用结论。
使用规则：

以下逐项核对规则只约束标为“相关题型/方法参考”的记录；高相似度直供记录遵循
其独立的使用边界。

1. 先逐项核对差异：参数数值（个数/上界/模数）、**奇偶性**、谁做选择、目标是
   最大化还是最小化、约束方向（≤ 还是 <、"所有解"还是"一个解"）。
2. 对“相关题型/方法参考”记录，只要有一项不同，示例的**结论就不可迁移**——
   此时最多借用它的**方法**，并且必须在本题自己的参数下从头重算，再代回题面验证。
3. 对这些低置信度记录若不能确认两题严格同构，就完全忽略示例，独立求解。宁可自己算，
   也不要抄一个参数不同的结论。
"""

#: 题面里有判别力的整数字面量：跳过 0/1 这类到处都是的数，也跳过 LaTeX 命令里
#: 的上下标数字（\sqrt[3]、x^{2}），只留下真正描述规模的参数。
_INT_RE = re.compile(r"(?<![\\^_{\w])(\d{2,})(?![}\w])")
_ANSWER_LINE_RE = re.compile(
    r"(?i)(?:最终答案|标准答案|答案|结论|answer|final\s+answer|correct\s+option)"
)
_NUMBER_LITERAL_RE = re.compile(r"(?<![A-Za-z_])[-+]?\d+(?:\.\d+)?")


def _scale_numbers(text: str) -> list[int]:
    seen: list[int] = []
    for match in _INT_RE.finditer(str(text or "")):
        value = int(match.group(1))
        if value not in seen:
            seen.append(value)
    return seen


def _numeric_diff_line(problem: str, example_problem: str) -> str:
    """把两边题面的规模参数差异摆出来；无从比较时返回空串。"""
    ours = _scale_numbers(problem)
    theirs = _scale_numbers(example_problem)
    if not ours or not theirs:
        return ""
    only_ours = [n for n in ours if n not in theirs][:6]
    only_theirs = [n for n in theirs if n not in ours][:6]
    if not only_ours and not only_theirs:
        return ""
    parts = []
    if only_ours:
        parity = "奇" if only_ours[0] % 2 else "偶"
        parts.append(f"本题独有 {only_ours}（首个为{parity}数）")
    if only_theirs:
        parity = "奇" if only_theirs[0] % 2 else "偶"
        parts.append(f"示例独有 {only_theirs}（首个为{parity}数）")
    return ("**⚠ 参数差异**：" + "；".join(parts)
            + "。参数不同则结论不同，必须在本题参数下重算。\n")


def _method_only_solution(solution: str, limit: int) -> str:
    """Keep procedural guidance while masking answer-bearing literals.

    Related records are retrieved for ideas, not as answer keys.  In
    addition to the prose warning, remove explicit answer lines and mask numeric
    constants so a model cannot simply copy a nearby result.
    """
    kept = []
    for line in str(solution or "").splitlines():
        if _ANSWER_LINE_RE.search(line):
            continue
        kept.append(_NUMBER_LITERAL_RE.sub("<参数>", line))
    text = "\n".join(kept).strip()
    return text[:limit] if text else "（仅提供题型方法；具体结论已隐藏。）"


def build_reference_block(
    examples: List[Dict[str, Any]] | None,
    problem_chars: int,
    solution_chars: int,
    problem: str = "",
) -> str:
    """把检索到的每一条相似题拼成参考区块；无检索结果时返回空串。

    Args:
        examples: 检索节点已按关系和注入上限筛选的 `retrieved_examples`，按相似度降序。
        problem_chars: 每条题面的截断长度。
        solution_chars: 每条解答的截断长度。
        problem: 本题题面。给出时逐条附上与示例的规模参数差异（反锚定）。
    """
    usable = [ex for ex in (examples or []) if isinstance(ex, dict)]
    if not usable:
        return ""
    parts = [_HEADER, _ANTI_ANCHOR_NOTE]
    for i, example in enumerate(usable, 1):
        try:
            similarity = float(example.get("similarity") or 0.0)
        except (TypeError, ValueError):
            similarity = 0.0
        example_problem = str(example.get("problem") or "")
        relation = str(example.get("relation") or "irrelevant")
        dataset = str(example.get("source_dataset") or "olympiad")
        solution_type = str(example.get("solution_type") or "worked_solution")
        relation_label = {
            "high_similarity": "高相似度直供（检索相似度 >0.85）",
            "related": "相关题型/方法参考（0.60 < 检索相似度 ≤ 0.85）",
        }.get(relation, "关系未确认")
        parts.append(
            f"\n### 示例 {i}（关系: {relation_label}；相似度: {similarity:.3f}；"
            f"来源: {dataset}；资料类型: {solution_type}）\n"
        )
        if relation == "high_similarity":
            parts.append(
                "**使用边界**：这是检索相似度严格大于 0.85 的高置信度直供记录。"
                "题面和原始解答完整提供给代理，直接作为本题参考；此分支不再做相关题的"
                "参数屏蔽或字符串模糊处理。请优先沿用解答中的结论与表达，若题面本身出现"
                "可见冲突，再回到本题条件核对。\n"
            )
        elif relation == "related":
            changes = example.get("changed_parameters") or []
            parts.append(
                "**使用边界**：只学习建模与推导方法。参考题答案绝不能直接作为本题答案，"
                "必须按照下列差异重新计算并验证。\n"
            )
            if changes:
                parts.append("**已识别差异**：" + "；".join(map(str, changes[:8])) + "。\n")
        else:
            parts.append(
                "**使用边界**：该记录不在当前注入等级内，只可借鉴可能相关的方法或定理；"
                "不要迁移题目结论。\n"
            )
        # 高相似记录直接提供完整题面和解答，不添加模糊参数说明。
        diff = _numeric_diff_line(problem, example_problem) if problem and relation != "high_similarity" else ""
        if diff:
            parts.append(diff)
        parts.append(f"**题目：**\n{example_problem[:problem_chars]}\n\n")
        reference_answer = str(example.get("reference_answer") or "").strip()
        solution = str(example.get("solution") or "").strip()
        if relation == "high_similarity" and (solution_type == "short_reference" or (reference_answer and not solution)):
            parts.append(
                "**该参考题的短答案（不是完整题解）：**\n"
                f"{(reference_answer or solution)[:solution_chars]}\n"
            )
        elif relation == "high_similarity":
            parts.append(f"**参考题解：**\n{solution[:solution_chars]}\n")
        else:
            parts.append(
                "**参考题方法摘录（参数与结论已屏蔽）：**\n"
                f"{_method_only_solution(solution or reference_answer, solution_chars)}\n"
            )
        if i < len(usable):
            parts.append("\n---\n")
    return "".join(parts)


def build_knowledge_card_block(
    cards: List[Dict[str, Any]] | None,
    *,
    content_chars: int = 700,
) -> str:
    """Format local method cards without presenting them as retrieved answers."""
    usable = [card for card in (cards or []) if isinstance(card, dict)]
    if not usable:
        return ""
    parts = [_KNOWLEDGE_HEADER]
    parts.append(
        "这些卡片来自项目已有的学科手册与验证说明，不是题库答案。"
        "只用它们核对定理前提、方法和验证条件；所有结论仍须由本题推导或计算得到。\n"
    )
    labels = {"constraint": "条件卡", "method": "方法卡", "verification": "验证卡"}
    for card in usable:
        label = labels.get(str(card.get("card_type") or ""), "知识卡")
        parts.append(
            f"\n### {label}：{card.get('title', '')}\n"
            f"来源：{card.get('source', '项目内学科资料')}\n"
            f"{str(card.get('content') or '')[:content_chars]}\n"
        )
    return "".join(parts)
