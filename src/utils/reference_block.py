"""题库参考示例区块的唯一构造入口。

推理节点与 Python 节点共用同一格式化逻辑。检索关系和注入数量由检索节点统一决定：
高相似度最多 1 条，没有高相似度时相关题型最多 3 条；本模块对选中的题面、解答和
正文不做字符截断，并按安全规则格式化。知识仅从本地 skills 读取。

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

from utils.retrieval_relation import answer_reuse_safety

_HEADER = "\n\n参考示例（统一数学题库召回，已做结构关系检查）：\n"

#: 反锚定说明。放在示例**之前**——写在后面时模型往往已经先读完解答并锚定了结论。
_ANTI_ANCHOR_NOTE = """
⚠️ 这些记录由检索召回，必须区分以下两类：

- **高度一致候选题（high_similarity）**：仅按稠密检索相似度严格大于 0.85 分档，不要求规范化题面完全一致；该等级会展示排名最高的一条完整参考记录。标签不等于确认同题，参考答案仍须结合当前题条件核验。
- **相似题目（related）**：仅表示题型或内容相似，不是已确认的同一道题；只供参考解题过程，不能复用其答案、结论或参数。

答案安全门槛仍独立约束答案复用和数据库兜底，但不阻止 high_similarity 候选的完整参考解进入上下文。未通过门槛时必须把参考解当作近似案例，围绕当前题独立推导。
使用规则：

以下逐项核对规则约束所有未通过答案安全门槛的记录；通过门槛的同题记录仍须确认推导
适用于当前题目。

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
_CONCLUSION_RE = re.compile(
    r"(?i)(?:\btherefore\b|\bthus\b|\bhence\b|\bconsequently\b|"
    r"\bit follows that\b|\bwe conclude\b|\bthis shows that\b|"
    r"\bwhich proves that\b|\bso (?:the|we)\b|故|所以|因此|从而|综上|可得)"
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


def _method_only_solution(solution: str) -> str:
    """Keep procedural guidance while masking answer-bearing literals.

    Related records are retrieved for ideas, not as answer keys.  In
    addition to the prose warning, remove explicit answer lines and mask numeric
    constants so a model cannot simply copy a nearby result.
    """
    kept = []
    for line in str(solution or "").splitlines():
        if _ANSWER_LINE_RE.search(line):
            continue
        conclusion = _CONCLUSION_RE.search(line)
        boxed_answer = re.search(r"\\boxed\s*\{", line)
        cut_points = [match.start() for match in (conclusion, boxed_answer) if match]
        if cut_points:
            line = line[:min(cut_points)]
            if line.strip():
                kept.append(_NUMBER_LITERAL_RE.sub("<参数>", line))
            break
        kept.append(_NUMBER_LITERAL_RE.sub("<参数>", line))
    text = "\n".join(kept).strip()
    return text if text else "（仅提供题型方法；具体结论已隐藏。）"


def build_reference_block(
    examples: List[Dict[str, Any]] | None,
    problem: str = "",
) -> str:
    """把检索到的每一条相似题拼成参考区块；无检索结果时返回空串。

    Args:
        examples: 检索节点已按关系和注入上限筛选的 `retrieved_examples`，按相似度降序。
        problem: 本题题面。给出时逐条附上与示例的规模参数差异（反锚定）。
    """
    usable = [ex for ex in (examples or []) if isinstance(ex, dict)
              and ex.get("relation") in {"high_similarity", "related"}]
    if not usable:
        return ""
    parts = [_HEADER, _ANTI_ANCHOR_NOTE]
    for i, example in enumerate(usable, 1):
        try:
            similarity = float(example.get("dense_similarity") or 0.0)
        except (TypeError, ValueError):
            similarity = 0.0
        example_problem = str(example.get("problem") or "")
        relation = str(example.get("relation") or "irrelevant")
        answer_safe, safety_reason = answer_reuse_safety(
            problem,
            example,
            high_similarity_min=float(example.get("high_similarity_min", 0.85) or 0.85),
        )
        answer_safe = answer_safe and example.get("answer_safe") is True
        full_reference_visible = relation == "high_similarity"
        dataset = str(example.get("source_dataset") or "olympiad")
        solution_type = str(example.get("solution_type") or "worked_solution")
        relation_label = {
            "high_similarity": (
                "高度一致候选题（完整参考解已展示；答案复用门槛通过）" if answer_safe
                else "高度一致候选题（完整参考解已展示；答案复用门槛未通过）"
            ),
            "related": (
                f"相似题目（不是已确认的同一道题；仅供参考解题过程；"
                f"{float(example.get('related_min_similarity', 0.70)):.2f} "
                f"< 稠密相似度 ≤ {float(example.get('high_similarity_min', 0.85)):.2f}）"
            ),
        }.get(relation, "关系未确认")
        parts.append(
            f"\n### 示例 {i}（关系: {relation_label}；稠密相似度: {similarity:.3f}；"
            f"来源: {dataset}；资料类型: {solution_type}）\n"
        )
        if answer_safe:
            parts.append(
                "**使用边界**：稠密相似度超过阈值，且规范化题面完全一致、参考答案存在。"
                "可查看完整参考解，但仍要核对其推导与本题条件一致。\n"
            )
        elif relation == "high_similarity":
            parts.append(
                "**使用边界**：该题因稠密相似度超过阈值而展示完整参考解，无需题面规范化完全一致；"
                "但未通过答案复用安全门槛，不代表已确认同题。仅作高相似案例对照，"
                "不得直接搬用结论，须按当前题条件独立推导和验证。\n"
            )
        elif relation == "related":
            changes = example.get("changed_parameters") or []
            parts.append(
                "**使用边界**：这只是相似题，不是已确认的同题；仅供参考解题过程。"
                "不得复用它的答案、结论或参数，必须按当前题条件独立推导并验证。\n"
            )
            if changes:
                parts.append("**已识别差异**：" + "；".join(map(str, changes[:8])) + "。\n")
        else:
            parts.append(
                "**使用边界**：完整参考答案安全门槛未通过；只可借鉴方法，不要迁移题目结论。"
                f"（门槛状态：{safety_reason}）\n"
            )
        # high_similarity 展示完整题面和参考解；其答案安全状态仍单独标注。
        diff = _numeric_diff_line(problem, example_problem) if problem and not answer_safe else ""
        if diff:
            parts.append(diff)
        parts.append(f"**题目：**\n{example_problem}\n\n")
        reference_answer = str(example.get("reference_answer") or "").strip()
        solution = str(example.get("solution") or "").strip()
        if full_reference_visible and (solution_type == "short_reference" or (reference_answer and not solution)):
            parts.append(
                "**高度一致候选的短答案（不是完整题解；请核验后再用于当前题）：**\n"
                f"{reference_answer or solution}\n"
            )
        elif full_reference_visible:
            parts.append(f"**高度一致候选的完整参考解：**\n{solution}\n")
        else:
            parts.append(
                "**相似题参考解题过程（只借鉴方法，不迁移答案）：**\n"
                f"{_method_only_solution(solution or reference_answer)}\n"
            )
        if i < len(usable):
            parts.append("\n---\n")
    return "".join(parts)


