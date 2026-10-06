"""验证代码反伪造检测（评委建议 2，针对失分模式 C）。

评委报告定性为"认识论上最危险的失败模式"：idx 19、28、45、66 的验证代码不做
任何计算，直接以注释/打印断言结论——"基于已知竞赛结果验证…PASS"、"已知 n=5
存在构造"、"该结论来自组合极值理论"——然后以 evidence=support 通过交叉验证并
被仲裁锁定。验证器退化成了结论的回声。

检测思路（纯静态、保守）：

1. **计算实质度**：代码里是否存在真实计算——循环/推导式/递归、sympy·itertools
   等求解调用、对变量的算术运算。没有计算而打印 PASS/FAIL 的，证据不可信。
2. **权威引用断言**：注释或打印字符串里出现"已知/经分析/竞赛结果/well-known/
   known result"且伴随结论数值——把外部记忆当验证。
3. **无条件 PASS**：``print("验证状态: PASS")`` 的实参是字面量而非依赖任何
   分支/比较的变量，代码里也没有任何 assert/if 比较。

被判伪造的证据一律降级为 inconclusive，且不得作为仲裁锁定依据（调用方负责）。
判定宁可放过：只要代码含有实质计算就不算伪造——误伤真实验证会废掉整条
Python 分支，损失比放过一段回声大得多。
"""

from __future__ import annotations

import ast
import re


#: 实质计算信号：任何一个出现即认为代码"确实算了点什么"。
_COMPUTE_RE = re.compile(
    r"(?m)^\s*(?:for|while)\s|"                      # 循环
    r"\[[^\]]*\bfor\b[^\]]*\]|"                      # 推导式
    r"\b(?:solve|solveset|nsolve|dsolve|simplify|expand|factor|gcd|lcm|"
    r"binomial|factorial|summation|integrate|diff|limit|det|rref|nullspace|"
    r"eig|jordan_form|diagonalize|isprime|factorint|totient|"
    r"permutations|combinations|product|accumulate|"
    r"minimize|maximize|linprog|fsolve|brentq|"
    r"sum|max|min|sorted|range)\s*\(|"               # 求解/枚举/聚合调用
    r"\bdef\s+\w+\s*\([^)]*\)\s*:"                   # 定义了函数（递归/记忆化）
)

#: 对变量的真实算术（排除纯字面量算式 "1+1"、"3 * 20**19"）。
#: 至少一侧操作数必须是标识符——2026-08-19 idx=0 事故：旧式 `[\w)]\s*\*\*\s*\w`
#: 把 `3 * 20**19` 这个纯常量表达式认成"实质计算"，于是"没算就宣称 PASS"的
#: 回声代码逃过反伪造闸门，用 1.57e25 覆盖了推理算出的正确答案 20460。
_VAR_ARITH_RE = re.compile(
    r"[A-Za-z_]\w*\s*(?:\*\*|[*/+\-%])\s*[\w(\[]"      # 名字 OP 某物
    r"|[\w)\]]\s*(?:\*\*|[*/+\-%])\s*[A-Za-z_]\w*"     # 某物 OP 名字
)

#: 数值字面量（含科学计数、浮点），用于判断表达式是否"只有常量"。
_NUMERIC_LITERAL_RE = re.compile(r"\d+\.?\d*(?:[eE][-+]?\d+)?")

#: `最终答案:` 的打印形态，用于检查答案是否由字面量常量直接给出。
_FINAL_ANSWER_PRINT_RE = re.compile(
    r"""print\s*\(\s*["']\s*最终答案\s*[:：]?\s*["']\s*,\s*([^)]*)\)"""
    r"""|print\s*\(\s*f["']\s*最终答案\s*[:：]?\s*\{([^}:!]*)[^}]*\}["']\s*\)"""
)


def _expression_is_literal_only(expr: str) -> bool:
    """表达式里是否不含任何标识符（即纯常量算式，无计算来源）。"""
    text = str(expr or "").strip()
    if not text:
        return False
    residue = _NUMERIC_LITERAL_RE.sub("", text)
    residue = re.sub(r"""["'][^"']*["']""", "", residue)
    return not re.search(r"[A-Za-z_]", residue)

#: 比较/断言：验证结论至少要有一次比较。
_ASSERT_RE = re.compile(r"\bassert\b|==|!=|<=|>=|\bis\s+not\b")

#: 权威引用式断言（注释与字符串里）。
_AUTHORITY_RE = re.compile(
    r"已知(?:结论|结果|竞赛|存在)|经分析|由.{0,12}(?:定理|理论|结果)可?(?:知|得)|"
    r"竞赛结果|标准答案|参考答案|(?i:well[- ]known|known\s+result|it\s+is\s+known|"
    r"by\s+the\s+known|from\s+the\s+literature)"
)

#: PASS 的打印形态：实参是字面量（伪造嫌疑）还是变量（正常）。
#: 冒号可以落在第一个字面量里（`print("验证状态:", "PASS")`），旧式两条分支都
#: 匹配不到这一形态——idx=0 正是这样绕过字面量 PASS 检查的。
_LITERAL_PASS_RE = re.compile(
    r"""print\s*\(\s*(["'])\s*验证状态\s*[:：]?\s*(?:PASS|通过)\s*\1"""
    r"""|print\s*\(\s*(["'])\s*验证状态\s*[:：]?\s*\2\s*,\s*(["'])\s*(?:PASS|通过)\s*\3"""
)
_STATUS_MARKER_RE = re.compile(r"验证状态")

_HIGH_RISK_PROBLEM_RE = re.compile(
    r"(?i)\b(?:count|counting|number\s+of|how\s+many|maximum|minimum|largest|"
    r"smallest|optimal|search|arrangement|permutation|combination|divisible)\b|"
    r"计数|多少|最大|最小|极值|最优|搜索|排列|组合|方案|路径|状态转移|动态规划"
)
_HIGH_RISK_CODE_RE = re.compile(
    r"(?i)\b(?:dp|memo|cache|dfs|bfs|backtrack|search|enumerate|brute|"
    r"itertools|combinations|permutations|for|while)\b"
)


def _baseline_evidence(code: str) -> tuple[bool, list[str]]:
    """Find an independently named small-instance comparison in Python code.

    A PASS print is not a baseline. The code must use an actual comparison in
    an assertion or returned check, and one side must depend on an intentionally
    independent brute/exhaustive/small-case implementation.  AST inspection
    avoids treating comments or output strings as evidence.
    """
    try:
        tree = ast.parse(str(code or ""))
    except SyntaxError:
        return False, []
    names = [
        node.id.lower() for node in ast.walk(tree) if isinstance(node, ast.Name)
    ] + [
        node.name.lower()
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    markers = ("brute", "naive", "enumerate", "exhaustive", "exact_small", "small_case")
    independent_names = sorted({
        name for name in names if any(marker in name for marker in markers)
    })
    assignments = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assignments[target.id] = node.value
    def uses_baseline(node, seen=None):
        seen = set(seen or ())
        if isinstance(node, ast.Name) and node.id in assignments and node.id not in seen:
            return uses_baseline(assignments[node.id], seen | {node.id})
        if isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, 'id', '')
            if name.lower() in independent_names:
                return True
        return any(uses_baseline(child, seen) for child in ast.iter_child_nodes(node))
    # Returned computed checks and assertions carry the same comparison. An
    # unused function whose name happens to contain 'brute' is insufficient.
    has_comparison = any(isinstance(node, ast.Compare) and uses_baseline(node)
                         for node in ast.walk(tree))
    return has_comparison and bool(independent_names), independent_names


def _strip_comments_and_strings(code: str) -> tuple[str, str]:
    """(可执行骨架, 注释+字符串文本)。粗粒度扫描即可，不求完美解析。"""
    comments: list[str] = []

    def _grab_comment(match: re.Match) -> str:
        comments.append(match.group(0))
        return ""

    text = str(code or "")
    # 字符串字面量（含三引号）
    strings = re.findall(r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'|"[^"\n]*"|\'[^\'\n]*\'', text)
    skeleton = re.sub(r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'|"[^"\n]*"|\'[^\'\n]*\'', '""', text)
    skeleton = re.sub(r"#[^\n]*", _grab_comment, skeleton)
    return skeleton, "\n".join(comments + strings)


def assess_verification_authenticity(
    code: str, stdout: str = "", problem: str = ""
) -> dict:
    """静态评估验证代码的可信度。

    返回 {"fabricated": bool, "reasons": [str, ...]}。fabricated=True 表示
    该代码的 PASS/support 证据不可作为数学正确性的依据。
    """
    text = str(code or "")
    if not text.strip():
        return {
            "fabricated": False,
            "reasons": [],
            "verified": False,
            "has_compute": False,
            "has_assert": False,
            "answer_hardcoded": False,
            "requires_baseline": False,
            "has_baseline_check": False,
            "baseline_names": [],
        }

    skeleton, prose = _strip_comments_and_strings(text)
    has_compute = bool(_COMPUTE_RE.search(skeleton)) or bool(_VAR_ARITH_RE.search(skeleton))
    has_assert = bool(_ASSERT_RE.search(skeleton))
    claims_pass = bool(_LITERAL_PASS_RE.search(text)) or (
        "PASS" in str(stdout or "") and _STATUS_MARKER_RE.search(str(stdout or "")) is not None
    )
    cites_authority = bool(_AUTHORITY_RE.search(prose))
    has_baseline_check, baseline_names = _baseline_evidence(text)
    requires_baseline = bool(
        _HIGH_RISK_PROBLEM_RE.search(str(problem or ""))
        and _HIGH_RISK_CODE_RE.search(skeleton)
    )

    # 答案自身是否由纯常量表达式直接打印：`print("最终答案:", 3 * 20**19)` 这类
    # 代码没有推导过程，只是把结论抄进 print，属于"未进行验证"。
    answer_hardcoded = False
    for match in _FINAL_ANSWER_PRINT_RE.finditer(text):
        expr = match.group(1) or match.group(2) or ""
        if _expression_is_literal_only(expr):
            answer_hardcoded = True
            break

    reasons: list[str] = []
    if answer_hardcoded:
        reasons.append("最终答案由字面量常量直接打印，代码未推导出该答案")
    if claims_pass and not has_compute:
        reasons.append("代码未含循环/求解/枚举等实质计算却宣称验证通过")
    if claims_pass and has_compute and not has_assert and cites_authority:
        reasons.append("验证结论依赖注释中的已知结论引用而非代码比较")
    if cites_authority and not has_compute:
        reasons.append("以'已知/竞赛结果'等权威引用代替计算")
    # 字面量 PASS + 无任何比较：结论与代码执行结果无关。
    if _LITERAL_PASS_RE.search(text) and not has_assert and not has_compute:
        if "代码未含循环/求解/枚举等实质计算却宣称验证通过" not in reasons:
            reasons.append("验证状态为硬编码字面量，与计算结果无关")

    strength_reasons: list[str] = []
    if requires_baseline and not has_baseline_check:
        strength_reasons.append(
            "高风险计数/DP/搜索/极值代码缺少独立小规模基准与比较断言"
        )

    # verified：这段代码是否真的"验证"过什么——有实质计算，且答案不是抄进
    # print 的常量。调用方据此判定 Python 分支可否作为答案来源（未验证即不可信）。
    verified = bool(has_compute) and not answer_hardcoded \
        and (not requires_baseline or has_baseline_check)
    return {
        "fabricated": bool(reasons),
        "reasons": reasons,
        "strength_reasons": strength_reasons,
        "verified": verified,
        "has_compute": bool(has_compute),
        "has_assert": has_assert,
        "answer_hardcoded": answer_hardcoded,
        "requires_baseline": requires_baseline,
        "has_baseline_check": has_baseline_check,
        "baseline_names": baseline_names,
    }
