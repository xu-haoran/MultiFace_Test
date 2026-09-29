"""从 Python 执行的 stdout 中二次抽取答案（评委建议 1 的后半部分）。

评委报告模式 D 的具体事故：idx 20 的验证代码枚举出 k=1→4、k=2→36 并打印
"both match binomial(2k,k)^2 formula; 4^k pattern fails at k=2" ——正确公式
已经在 stdout 里，但因为缺少最后的 ``print("最终答案:", ...)`` 行，执行器的
answer 字段为空，`python_ok` 为假，这个被自家程序证实的公式再也进不了候选池，
最终输出了被它反驳的 "4"。

执行器（mcp_servers/python_executor）只认 "最终答案:" 标记，这是对的——它是
高置信通道。本模块是低置信通道：仅当高置信通道为空时，按保守优先级从 stdout
末尾挖掘候选，并全部通过洁净度门。挖到的答案打上 ``stdout_mined`` 来源标记，
下游据此区别对待（可作候选、不可无证据锁定）。
"""

from __future__ import annotations

import re

from utils.answer_cleanliness import looks_committed_result


_MAX_ANSWER_CHARS = 240

#: "match/符合 <公式>" 型断言：程序把枚举结果与某公式对上了。
_MATCH_FORMULA_RE = re.compile(
    r"(?i)(?:both\s+match|matches?|符合|吻合|等于公式|match(?:ed)?\s+formula)\s*[:：]?\s*"
    r"([A-Za-z0-9_^*/+\-(),{}\\ ]{2,120}?)(?:\s*(?:formula|公式)|[;；.。]|$)"
)

#: 命名结果行："f(2015) = 3024"、"answer = 506"、"最小值: 2600"、"阈值 = 0.7071"。
_NAMED_RESULT_RE = re.compile(
    r"(?im)^\s*(?:[A-Za-z_][A-Za-z0-9_]*(?:\([^)]{0,40}\))?|[一-鿿]{1,8})"
    r"\s*[=:：]\s*([^\n]{1,160})\s*$"
)

#: 结尾裸结果行：纯数字/短表达式独占一行。
_BARE_RESULT_RE = re.compile(r"^\s*[-+]?\d[\d_,]*(?:\.\d+)?\s*$")

#: 明显是日志/进度而非结论的行。
_LOG_LINE_RE = re.compile(
    r"(?i)verif|check|test(?:ing)?|iter|step|round|progress|elapsed|->|→|"
    r"验证状态|验证证据|开始|正在|尝试|进度|耗时"
)


def _clean(value: str) -> str:
    text = (value or "").strip().strip("`'\"").strip()
    if len(text) > _MAX_ANSWER_CHARS:
        return ""
    return text


def _looks_like_prose_log(value: str) -> bool:
    """迭代日志/叙述行（"k=6: gcd of b^... for odd b up to 49 is 256"）不是结论。

    2026-07-29 的回归教训：执行器旧回退曾把这类 stdout 日志行提为候选参加仲裁。
    高置信通道（最终答案 标记）永远优先；本挖掘通道必须比它更严，宁漏勿滥。
    """
    text = str(value or "")
    latin_words = re.findall(r"\b[A-Za-z]{2,}\b", text)
    if len(latin_words) >= 4:
        return True
    if len(re.findall(r"[一-鿿]", text)) >= 12:
        return True
    return False


def mine_stdout_answer(stdout: str) -> str:
    """stdout 里最像最终结论的一段文本；找不到返回 ""。

    优先级（每级都要通过 looks_committed_result 洁净度门）：
      1. "matches <公式>" 型断言 —— 程序主动把数据与公式对上，是最强信号；
      2. 末尾的命名结果行（``name = value``），跳过日志行；
      3. 末尾的裸数字行。
    """
    text = str(stdout or "").strip()
    if not text:
        return ""

    matches = list(_MATCH_FORMULA_RE.finditer(text))
    if matches:
        candidate = _clean(matches[-1].group(1))
        # "match(es) <公式>" 的断言语境本身就是承诺证据；公式不必含 '='，
        # 只需通过噪声/日志双重否定门。
        from utils.answer_cleanliness import is_noise_answer

        if candidate and not _looks_like_prose_log(candidate) \
                and not is_noise_answer(candidate):
            return candidate

    lines = [line for line in text.splitlines() if line.strip()]
    for line in reversed(lines[-15:]):
        if _LOG_LINE_RE.search(line) or _looks_like_prose_log(line):
            continue
        named = _NAMED_RESULT_RE.match(line)
        if named:
            candidate = _clean(named.group(1))
            if candidate and not _looks_like_prose_log(candidate) \
                    and looks_committed_result(candidate):
                # 保留左侧名称：多数题的可判分形态是 "f(2015) = 3024" 整行。
                whole = _clean(line)
                return whole if whole else candidate
        bare = _BARE_RESULT_RE.match(line)
        if bare:
            candidate = _clean(line)
            if candidate:
                return candidate
    return ""
