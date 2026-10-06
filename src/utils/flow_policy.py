"""Shared routing and submission checks for the cached solving flow."""
import re

from utils.answer_contract import missing_components
from utils.answer_extractor import looks_incomplete_answer, is_multi_part_problem
from utils.answer_cleanliness import is_noise_answer
from utils.cot_stripper import is_placeholder_answer
from utils.problem_profile import (
    is_objective_mode, normalize_objective_answer, objective_answer_is_usable,
    objective_answer_is_consistent, fill_answer_matches_blanks,
)


def verification_task(problem, mode):
    """Classify executable work, ignoring item labels and names of concepts."""
    text = re.sub(r"(?m)^\s*(?:(?:判断|选择|填空)\s*[:：]\s*)?"
                  r"(?:\d+[.．、](?!\d)|[（(]\d+[）)]|[A-EＡ-Ｅ][.．、:：)）])\s*",
                  "", str(problem or ""))
    action = re.search(r"计算|求值|求解|求出|求.*(?:概率|期望|方差|积分|导数|秩)|"
                       r"calculate|compute|evaluate|solve|find\s+(?:all|the|a)", text, re.I)
    formula = re.search(r"[=<>≤≥∑∫]|\\(?:frac|sum|int|sqrt|partial|begin)|"
                        r"[a-zA-Z]\s*\^|\d\s*[+*/×÷]|\b\d+\.\d+\b", text)
    search = re.search(r"枚举|构造.*反例|找.*反例|多少种|共有.*种|所有.*(?:解|整数)|"
                       r"为多少|求[^。\n]*(?:个数|最大值|最小值)|"
                       r"enumerate|counterexample|how many|number of ways|"
                       r"(?:find|determine|compute|calculate)\s+(?:the\s+)?number\s+of|"
                       r"(?:find|determine)\s+(?:the\s+)?(?:maximum|minimum)", text, re.I)
    algorithm = re.search(r"(?:分解|编码|解码|排序|decomposition|encode|encoded|decode|sort)", text, re.I) \
        and re.search(r"(?:消息|字符串|message|string|序列|sequence|数据|data)", text, re.I) \
        and not re.search(r"定义|definition|概念|concept", text, re.I)
    concrete_math = re.search(r"矩阵|行列式|方程|多项式|积分|导数|matrix|polynomial|equation", text, re.I) \
        and re.search(r"\d", text) and re.search(r"\[|\\|[xyzXYZ]|\d\s*[,，]\s*\d", text)
    if search:
        kind, reason = "search", "explicit_search_or_counterexample"
    elif algorithm:
        kind, reason = "search", "explicit_algorithm_on_supplied_data"
    elif action or re.search(r"求\s*[^。\n]{0,70}(?:算子|表达式|导数|积分|伴随)|derive\s", text, re.I):
        kind, reason = "numerical" if re.search(r"\d", text) else "symbolic", "explicit_computation"
    elif concrete_math and not re.search(r"定义|definition|方法|method", text, re.I) or formula and re.search(
            r"判断\s*[^。\n]{0,40}(?:[=<>≤≥]|是否成立)|check\s+[^.\n]{0,40}(?:equal|identity)|"
            r"(?:秩|rank|roots?)\s*(?:是|为|=|is|\()", text, re.I):
        kind, reason = "symbolic", "mathematical_expression_to_check"
    elif is_objective_mode(mode) or re.search(
            r"定义|概念|术语|参数|指标|方法|性质|说法|描述|definition|concept|terminology", text, re.I):
        kind, reason = "knowledge", "concept_or_terminology_without_executable_task"
    else:
        kind, reason = "knowledge", "no_explicit_executable_target"
    return {"kind": kind, "python_required": kind != "knowledge", "reason": reason}


def python_verification_needed(problem, mode):
    return verification_task(problem, mode)["python_required"]


def knowledge_candidate_eligible(problem, mode):
    return is_objective_mode(mode) and verification_task(problem, mode)["kind"] == "knowledge"


def candidate_complete(problem, answer, mode):
    text = str(answer or "").strip()
    if not text or is_placeholder_answer(text) or looks_incomplete_answer(text):
        return False
    if is_objective_mode(mode):
        normalized = normalize_objective_answer(text, mode)
        return bool(objective_answer_is_usable(normalized, mode)
                    and (mode != "fill" or fill_answer_matches_blanks(normalized, problem)))
    if mode != "proof" and is_noise_answer(text):
        return False
    if missing_components(problem, text):
        return False
    if is_multi_part_problem(problem):
        from utils.answer_contract import answer_part_count
        labels = re.findall(r"[（(]([1-9一二三四五六七八九])[）)]", problem or "")
        if answer_part_count(text) < max(2, len(set(labels))):
            return False
    return True


def proof_process_present(answer, reasoning_result):
    from utils.answer_formatter import build_proof_body
    text = str(answer or "")
    return build_proof_body(text, reasoning_result or {}) != text or bool(
        len(text) >= 160 and re.search(r"因此|所以|由此|故|because|therefore|hence|证明过程", text, re.I)
    )
