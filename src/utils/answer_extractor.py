import re


class AnswerExtractor:
    @staticmethod
    def normalize(answer: str) -> str:
        if not answer:
            return ""
        s = answer.strip().strip("。.").strip()
        s = re.sub(r"\s+", " ", s)
        return s

    @staticmethod
    def extract_from_text(text: str):
        if not text:
            return None
        m = re.search(r"(?:最终答案|答案|结论)[：:]\s*(.+?)(?:\n|$)", text)
        return m.group(1).strip() if m else None


def extract_boxed_answer(text: str) -> str:
    """Return the last balanced ``\\boxed{...}`` body in *text*.

    A simple non-greedy regex stops at the first nested brace (for example in
    ``\\boxed{\\frac{1}{2}}``).  This small scanner is deliberately independent of
    SymPy: the value is still checked by the caller's fragment/placeholder gates.
    """
    source = str(text or "")
    marker = re.compile(r"\\boxed\s*\{")
    found = []
    for match in marker.finditer(source):
        depth = 1
        escaped = False
        for index in range(match.end(), len(source)):
            char = source[index]
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    value = source[match.end():index].strip()
                    if value:
                        found.append(value)
                    break
    return found[-1] if found else ""


def extract_answer_fallback(text: str) -> str:
    """Recover an answer when the model omitted the Markdown section contract.

    The order is conservative: a final boxed value, an explicit answer/结论 line,
    then the last payload-bearing line.  This function never tries to evaluate a
    mathematical expression or invent a value; it only preserves model text.

    2026-08-09 评委报告模式 B：最后一层"任取带数字的行"曾把思维流碎片
    （"-177. Strict. So equality only when all equal."、"7, etc. We already
    tested that and it worked."）当成答案。该层现在必须通过洁净度白名单。
    """
    return extract_answer_fallback_with_source(text)[0]


def extract_answer_fallback_with_source(text: str) -> tuple:
    """`extract_answer_fallback`，另返回命中层级。

    层级标签供调用方区分置信度（2026-08-10 评委报告 4.2）：``boxed`` 与
    ``labelled`` 是模型显式提交的结论（截断响应里出现也算有效产出）；
    ``payload_line`` 是从散文里扫出的最后信息行，置信度与 salvage 同级——
    耗尽判定必须把它视为"没有产出"，让压缩重试先行。
    """
    from utils.answer_cleanliness import looks_committed_result, looks_derivation_fragment

    source = str(text or "").strip()
    if not source:
        return "", ""
    boxed = extract_boxed_answer(source)
    if boxed:
        return boxed, "boxed"

    labelled = list(re.finditer(
        r"(?im)(?:^|[\n。；;])\s*(?:最终答案|答案|结论)\s*[：:]\s*([^\n]+)",
        source,
    ))
    if labelled:
        return labelled[-1].group(1).strip().strip("`"), "labelled"

    lines = [line.strip().strip("`*_ ") for line in source.splitlines() if line.strip()]
    for line in reversed(lines):
        if len(line) > 600 or re.match(r"^#{1,6}\s", line):
            continue
        if re.search(r"[=<>]|\\(?:frac|sqrt|boxed)|\d|正确|错误|成立|收敛|发散|存在|唯一", line):
            if looks_committed_result(line) and not looks_derivation_fragment(line):
                return line, "payload_line"
    return "", ""


# 明显"话说一半"的结尾：冒号/逗号/运算符/引导语，后面本应还有内容
_DANGLING_TAIL_RE = re.compile(r"(?:[：:，,、;；=+\-*/^]|如下|分别为|分别是|包括|即)\s*$")
_UNUSABLE_ANSWER_RE = re.compile(
    r"^\s*(?:no\s*solution|无解|无法求解|无法确定|不能确定)\s*$|"
    r"(?:示例数据|模拟数据|假设数据|替换.*真实数据|未提供.*数据|缺少.*数据)",
    re.IGNORECASE,
)


def looks_incomplete_answer(answer: str) -> bool:
    """判断答案是否为明显不完整的碎片（截断/截头/引导语/未闭合括号）。

    评委报告主要问题 2/3：'(Matrix([' 与 'Z_18 的所有子群如下：' 这类碎片
    曾直接污染 final_response。2026-07-07 报告新增截头残片形态：'1}^n X_i$'、
    '\\sigma^2$ 且无自相关'、'0.390625$'（来自 LaTeX 内部 '=' 处切断）。
    此检测用于 cross_validator 与 formatter 的回退门。
    """
    s = (answer or "").strip()
    if not s:
        return True
    if _UNUSABLE_ANSWER_RE.search(s):
        return True
    # 括号开闭不等 → 截断（开>闭）或截头（闭>开，如 '1}^n X_i$'）。
    # 开闭跨类型合计，兼容半开区间 [0, 1) 记法。
    opens = sum(s.count(ch) for ch in "([{")
    closes = sum(s.count(ch) for ch in ")]}")
    if opens != closes:
        return True
    if _DANGLING_TAIL_RE.search(s):
        return True
    if looks_like_latex_fragment(s):
        return True
    return False


def looks_like_latex_fragment(text: str) -> bool:
    """判断文本是否为被截头/截尾的 LaTeX 残片（如 '2 \\\\ 0, & n \\neq 2 \\end{cases} $$'）。

    特征：\\end{env} 无配对 \\begin{env}；$$ 或单 $ 未配对（'0.390625$'）；
    花括号开闭不等；对齐符 & 出现在任何环境之外。
    """
    s = (text or "").strip()
    if not s:
        return False
    for env in set(re.findall(r"\\end\{(\w+\*?)\}", s)):
        if ("\\begin{%s}" % env) not in s:
            return True
    for env in set(re.findall(r"\\begin\{(\w+\*?)\}", s)):
        if ("\\end{%s}" % env) not in s:
            return True
    if s.count("$$") % 2 == 1:
        return True
    # 单 $ 不配对：LaTeX 数学模式被切断（评委报告 '0.390625$'、'…(n - p - 1)}$'）
    if s.count("$") % 2 == 1:
        return True
    if s.count("{") != s.count("}"):
        return True
    if "&" in s and "\\begin" not in s and "\\&" not in s:
        return True
    return False


_ENUM_MARK_RE = re.compile(r"[（(]\s*[1-9一二三四五]\s*[）)]|[①②③④⑤]|第[一二三四五]问")
#: 提问动词。"找出/求出/列出" 等此前不在首个动词表内，使 "试找出A，也找出B" 漏判。
_ASK_VERB = r"(?:求出?|计算|写出|给出|确定|找出|列出|求解|讨论|判断|证明|描述|说明|分析|比较|证)"
#: 连接两个问项的词。原表只有 并/及/以及/、并，漏掉 也/同时/再/然后/接着 ——
#: ISL 博弈题 "试找出使甲有必胜策略的所有 λ，也找出使乙有必胜策略的所有 λ" 因此
#: 被判为单问项，而该标志决定 _preferred_answer 是否允许用更完整的候选替换半个答案。
_ASK_CONJUNCTION = r"(?:，?\s*(?:并|及|以及|、并|也|同时|再|然后|接着|另外|此外))"
_MULTI_ASK_RE = re.compile(
    # 第二个问项通常带动词；"求均值，以及标准差" 省略动词，故动词可选，
    # 但连接词后必须仍有实质内容（≥2 字），避免把 "并" 当副词的单问项误判。
    r"%s[^。；;]*?%s\s*[^。；;]*?(?:%s[^。；;]*|[^。；;，,]{2,})"
    % (_ASK_VERB, _ASK_CONJUNCTION, _ASK_VERB)
)


def is_multi_part_problem(problem: str) -> bool:
    """检测题目是否包含多个问项（评委报告问题 4：多问只答一问，如 idx=172）。"""
    p = (problem or "").strip()
    if not p:
        return False
    if len(_ENUM_MARK_RE.findall(p)) >= 2:
        return True
    if "分别" in p and re.search(r"求|计算|写出|给出|证明|讨论", p):
        return True
    if _MULTI_ASK_RE.search(p):
        return True
    return False


# 填空空位：ASCII 下划线 / 全角下划线 / 连续破折号
_BLANK_RE = re.compile(r"_{3,}|＿{2,}|—{3,}")
# 选择题：行首选项标记，或题面以空括号收尾（"（　）"）
_CHOICE_RE = re.compile(r"(?m)^\s*[A-D][．.、)）]\s*\S|[（(]\s*[）)]\s*[。.]?\s*$")
_ANSWER_ONLY_HINT_RE = re.compile(r"只需?给出答案|直接写出|不需要(?:写)?过程|无需过程")


def is_answer_only_problem(problem: str) -> bool:
    """题面是否只要求最终答案（填空、选择、明确声明不需过程）。

    评委.md：填空题可给出最终结果、选择题可直接给出选项，而"其余需要推导的题目
    也应保留支撑结论所需的关键步骤"。这个判定就是二者的分界：True → 收敛为纯答案，
    False → 答案后附关键推导步骤。

    判定宁严勿松：漏判（当成需推导）只是多输出几行步骤，误判（当成填空）会丢掉
    判分需要的推导。
    """
    p = (problem or "").strip()
    if not p:
        return False
    if _BLANK_RE.search(p):
        return True
    if _CHOICE_RE.search(p):
        return True
    if _ANSWER_ONLY_HINT_RE.search(p):
        return True
    return False
