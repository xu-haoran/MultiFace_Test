"""洁净度过滤：阻止思维碎片/元叙述/内部指令成为 final_response。

评委报告（2026-08-09）模式 B：约 19 题的 final_response 是未收敛推理的尾部碎片
（"4."、"s_j."、"Count blanks: 1"、"哪里出错了？啊！"、"7, etc. We already tested
that and it worked."）。这些文本永远得 0 分，还会掩盖 trace 里真正有价值的部分结论。

本模块提供三个门：

* ``is_noise_answer``      —— 疑问/感叹/英文元叙述/内部指令为主体的文本一律拒收；
* ``looks_committed_result`` —— 逐行兜底抽取时，只接受"像已提交结论"的行；
* ``extract_partial_findings`` —— 抽不出合规答案时，从原始推理中收集"已证明的
  部分结论"（带结论标记且通过洁净度门的句子），代替原始思维流。

所有判断都是保守的字符串启发式：宁可放过一个可疑但可判分的答案，也不把一个
干净答案误杀成碎片——误杀会触发更低优先级的兜底，损失严格更大。
"""

from __future__ import annotations

import re

# ---- 元叙述 / 内部指令 ----

#: 英文探索性元叙述。这些短语出现在思考文本里，不出现在提交的数学结论里。
_META_NARRATION_RE = re.compile(
    r"(?i)\b(?:let'?s|let us|but wait|wait[,.]|maybe|perhaps|hmm+|we already|"
    r"we need to|we should|i think|i'?ll|let me|not sure|seems? to be|actually|"
    r"try(?:ing)? to (?:see|construct|find)|attempt to|"
    r"if i (?:write|answer|say)|the user|answer key|might think|i'?m wrong|"
    r"tells? me|the skill manual|i am the teacher|not yet\b|do we\b)\b"
)

#: 提示词内部指令被回显为答案（评委报告 idx 105："Count blanks: 1"）。
_INTERNAL_INSTRUCTION_RE = re.compile(
    r"(?i)count\s+blanks?|数空位|空位数量|按选项(?:逐项)?核对|逐项核对题面|"
    r"先数清|请判断|请问|按顺序列出全部结果"
)

#: 中文自我怀疑/过程叙述（评委报告 idx 36："哪里出错了？啊！"）。
_SELF_DOUBT_RE = re.compile(
    r"哪里出错|出错了|怎么回事|矛盾了|不对啊|再试一次|等一下|让我们|我们先试|重新来"
)

#: 已提交结论的正面信号：等式、boxed、明确结论词、集合/区间记号。
_COMMITTED_MARKER_RE = re.compile(
    r"[=≥≤<>∈]|\\boxed|\\frac|\\sqrt|^\s*[-+]?\d+(?:\.\d+)?\s*$|"
    r"答案|结论|因此|所以|综上|最大|最小|存在|唯一|成立|不成立|正确|错误|拒绝|接受"
)

#: 纯数字/短数学表达式（无叙述词），是合法的答案形态，不得因为短而拒收。
_PURE_MATH_RE = re.compile(
    r"^[\s$]*[-+]?[\d\w\\{}()\[\],.^*/+\- ==≥≤]+[\s$.。]*$"
)

_SENTENCE_SPLIT_RE = re.compile(r"(?:[。；;!?！？\n]|(?<!\d)\.(?!\d))+")

#: 英文探索句的句首引导词。单独一句（"We conclude ..."）是合法结论文体，
#: 连续多句才是解题现场的思维流（2026-08-09 冒烟 idx 43："We need 4 players
_EXPLORATORY_LEAD_RE = re.compile(
    r"(?i)^(?:let'?s?|suppose|assume|now|then|consider|try|maybe|perhaps|say|"
    r"we (?:need|want|have|try|test|check|see|use))\b"
)


def _exploratory_sentences(text: str) -> int:
    return sum(
        1 for sentence in _SENTENCE_SPLIT_RE.split(text or "")
        if _EXPLORATORY_LEAD_RE.match(sentence.strip())
    )


def _narrative_ratio(text: str) -> float:
    """英文叙述词密度：单词里 the/we/so/that 类虚词占比高 → 思维流。"""
    words = re.findall(r"[A-Za-z']+", text or "")
    if len(words) < 4:
        return 0.0
    narrative = {
        "the", "we", "so", "that", "this", "it", "is", "was", "then", "and",
        "but", "if", "to", "of", "a", "an", "for", "not", "can", "be", "have",
        "already", "tested", "worked", "seems", "try", "maybe", "wait",
    }
    hits = sum(1 for w in words if w.lower() in narrative)
    return hits / len(words)


def is_noise_answer(text: str) -> bool:
    """答案主体是否为思维碎片/元叙述/内部指令（True → 不得作为答案输出）。"""
    s = (text or "").strip()
    if not s:
        return True
    if _INTERNAL_INSTRUCTION_RE.search(s):
        return True
    if _SELF_DOUBT_RE.search(s):
        return True
    # 疑问/感叹收尾：结论不以问句结束。（允许内部含 '?' 的合法记号，如 "P(X>1)"）
    if re.search(r"[？?！!]\s*$", s):
        return True
    # 句首疑问句（2026-08-09 冒烟 idx 14："Are there any other ... sequences?
    # The known theorem says..."）：答案不会以问句开场；含 boxed 结论时除外。
    first_stop = re.search(r"[。；;!?！？\n]|(?<!\d)\.(?!\d)", s)
    if "\\boxed" not in s and first_stop and first_stop.group(0) in "？?" \
            and first_stop.start() <= 160:
        return True
    # "数字. 英文评述句" 形态（评委报告 idx 17："-177. Strict. So equality only
    # when all equal."）：裸数值后接英文散句、且全文无等式/boxed 绑定 → 思维流。
    if re.match(r"^[-+]?\d+(?:\.\d+)?\s*[.,;]\s+[A-Za-z]", s) \
            and not re.search(r"\\boxed|=", s):
        return True
    # 裸数 + 评估动词（2026-08-09 冒烟 idx 33/2："3 works"、"1/2 works."）：
    # 结论不带评估口吻；分数/小数同样适用。answer_formatter 的头部修复会剥掉
    # 评述词保留数值本体。
    if re.match(r"(?i)^[-+]?\d+(?:[./]\d+)?\s+(?:works?|holds?|fails?|seems|"
                r"is\s+possible|is\s+enough|suffices)\b", s):
        return True
    # 对局/模拟日志行（2026-08-09 冒烟 idx 48："Move 2: A picks 3 and 1. B
    # chooses min(4,2) = 2. State {2,2} -> 1 cookie."）：过程记录不是结论。
    if re.match(r"(?i)^\s*(?:move|round|turn|step)\s+\d+\s*[:：]", s):
        return True
    # 英文元叙述：出现探索短语，且不是一个带明确等式/boxed 的混合行
    if _META_NARRATION_RE.search(s):
        # "x = 3, let's verify" 仍然包含结论；只有当叙述占主导时才拒收
        if not re.search(r"\\boxed|=\s*[-+]?[\d\\]", s) or _narrative_ratio(s) > 0.35:
            return True
    if _narrative_ratio(s) > 0.5:
        return True
    # 多句英文探索散文：≥2 句以探索引导词开头、篇幅是成段英文、且无 boxed 结论。
    # 单句引导（合法结论文体）与短数学行都不触发。
    if "\\boxed" not in s and len(re.findall(r"[A-Za-z']+", s)) >= 15 \
            and _exploratory_sentences(s) >= 2:
        return True
    return False


def looks_committed_result(line: str) -> bool:
    """逐行兜底抽取专用：这一行像不像"已提交的结论"？

    与 ``is_noise_answer`` 的区别：这里是白名单（必须有正面信号），
    用于 extract_answer_fallback 的最后一层"任取带数字的行"——评委报告里
    "-177. Strict. So equality only when all equal." 正是从这层漏出去的。
    """
    s = (line or "").strip()
    if not s or is_noise_answer(s):
        return False
    if not _COMMITTED_MARKER_RE.search(s):
        return False
    # 数字后面跟英文叙述句（"7, etc. We already tested..."）→ 碎片
    if re.search(r"(?i)\d\s*[,.]?\s*(?:etc|so|then|thus we|which)\b", s) \
            and _narrative_ratio(s) > 0.2:
        return False
    return True


# ---- 捞回专用：推导现场碎片 ----

#: 语篇连接词开头的句子是论证的中间分支，不是自立结论（评委报告 idx 2：
#: "Otherwise, D1 > 1/2..."）。therefore/thus 不在列——它们引出的常是真结论。
_CONNECTIVE_LEAD_RE = re.compile(
    r"(?i)^(?:otherwise|however|similarly|conversely|alternatively|moreover|"
    r"furthermore|meanwhile|instead|likewise|nevertheless|nonetheless|"
    r"on the other hand|in (?:this|that|either) case)\b"
)
#: "N if 条件, M if 条件" 条件对：分类讨论的中间引理形态（评委报告 idx 0：
#: "0 if i odd, 2 if i even." 被当成 Hamilton 路径计数交付）。LaTeX cases
#: 环境是正规分段答案写法，不受此规则约束。
_CONDITIONAL_PAIR_RE = re.compile(
    r"(?i)^[^,，;；]{1,60}\bif\b[^,，;；]{1,60}[,，;；]\s*[^,，;；]{1,60}\bif\b"
)
#: 纯单字母变量的加减乘除串（评委报告 idx 33："a+b+c+d."）：没有数值、没有
#: 定义式的裸符号和，是"设目标为 a+b+c+d"的回声，不是计算结果。
_BARE_SYMBOL_SUM_RE = re.compile(
    r"^[A-Za-z](?:\s*[+\-*/]\s*[A-Za-z])+\s*[.。]?$"
)
#: 以英文 true/false 判定收尾的数学断言（评委报告 idx 77："8. 7≥8 false."）：
#: 这是推导中的试错记录。判断题的合法答案是"正确/错误"，不经此形态。
_VERDICT_TAIL_RE = re.compile(r"(?i)(?:^|[^A-Za-z])(?:true|false)\s*[.。]?\s*$")


def looks_derivation_fragment(text: str) -> bool:
    """捞回候选是否为推导现场碎片（True → 不得作为捞回答案）。

    只用于低置信的捞回通道（salvage_conclusion、extract_answer_fallback 的
    行级层、_clean_noise_head 的头部检查）：这些形态在正规 "## 最终答案"
    章节里也可能合法出现（如英文分段函数），因此不进全局 is_noise_answer。
    """
    s = (text or "").strip()
    if not s:
        return False
    if _CONNECTIVE_LEAD_RE.match(s):
        return True
    if "\\begin{cases}" not in s and _CONDITIONAL_PAIR_RE.match(s):
        return True
    if _BARE_SYMBOL_SUM_RE.match(s):
        return True
    if _VERDICT_TAIL_RE.search(s) and re.search(r"[=<>≥≤∈\d]", s):
        return True
    return False


#: 部分结论句必须带的"已确认"标记——比 salvage 的结论标记更宽，但仍排除探索句。
_FINDING_LEAD_RE = re.compile(
    r"已证|证明了|可以证明|必要性|充分性|上界|下界|不变量|引理|"
    r"因此|所以|综上|故|由此可(?:知|得)|我们得到|必须是|只能是|当且仅当"
)


def extract_partial_findings(analysis: str, limit_chars: int = 900) -> str:
    """从未收敛的推理文本中收集干净的部分结论，作为结构化兜底。

    返回 "；" 连接的若干句（每句都带结论标记、有信息负载、通过洁净度门），
    没有可用内容时返回 ""。这实现了评委建议 3 的后半句："抽不出合规答案时
    输出结构化的部分结论+已证引理，而非原始思维流"。
    """
    text = (analysis or "").strip()
    if not text:
        return ""
    findings: list[str] = []
    seen: set[str] = set()
    sentences = [s.strip().strip("*# 　") for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    # 从后向前：推导末端的断言最接近结论。
    for sentence in reversed(sentences):
        if not (8 <= len(sentence) <= 300):
            continue
        if not _FINDING_LEAD_RE.search(sentence):
            continue
        if not re.search(r"[=<>≥≤]|\d|\\frac|\\sqrt|2 的幂|成立|唯一|存在", sentence):
            continue
        if is_noise_answer(sentence):
            continue
        key = re.sub(r"\s+", "", sentence)[:60]
        if key in seen:
            continue
        seen.add(key)
        findings.append(sentence)
        if sum(len(f) for f in findings) >= limit_chars or len(findings) >= 4:
            break
    if not findings:
        return ""
    findings.reverse()  # 恢复推导顺序
    return "；".join(findings)

