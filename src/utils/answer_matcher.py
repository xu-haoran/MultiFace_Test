"""Answer matching: computation (strict) vs proof (loose). Per 项目计划.md §3.5/§6.3.

等价匹配管线（评委报告问题 3：'E*I*pi' vs 'πie'、'2*sqrt(2)*pi' vs '2π√2' 曾被判
uncertain）：unicode 常量归一化（π√×÷²）→ LaTeX 转 sympy 文本（\\frac \\sqrt \\pi）→
隐式乘法解析 → simplify 差为零或数值（含复数）容差比较。
"""
import re

from utils.cot_stripper import is_placeholder_answer
from utils.structured_answer import (
    LABEL_ASSIGNMENT_PREFIX_RE, compare_structured_answers, is_self_conflicting,
)

PROOF_KEYWORDS = ["证明", "prove", "show that", "验证", "说明", "推导", "论证", "判断", "判别", "试证"]

_UNICODE_REPLACEMENTS = [
    ("−", "-"), ("–", "-"), ("—", "-"), ("×", "*"), ("·", "*"), ("⋅", "*"),
    ("÷", "/"), ("²", "**2"), ("³", "**3"), ("½", "(1/2)"), ("∞", "oo"),
]

# 纯"常量乘积"风格短答案（πie、2π√2）：全部字符落在此集合内才走逐字符翻译
_CONST_PRODUCT_CHARS_RE = re.compile(r"^[0-9πiIeE√+\-*/^(). ]{1,40}$")


def _const_product_to_expr(s: str) -> str:
    """把 'πie'/'2π√2' 这类紧凑常量乘积翻译成可解析文本 '(pi)(I)(E)'/'2(pi)sqrt(2)'。"""
    s = re.sub(r"√\s*\(([^()]*)\)", r"#R#(\1)", s)
    s = re.sub(r"√\s*([0-9.]+)", r"#R#(\1)", s)
    out = []
    for part in re.split(r"(#R#)", s):
        if part == "#R#":
            out.append("sqrt")
            continue
        buf = []
        for ch in part:
            if ch == "π":
                buf.append("(pi)")
            elif ch in "iI":
                buf.append("(I)")
            elif ch in "eE":
                buf.append("(E)")
            elif ch == "^":
                buf.append("**")
            else:
                buf.append(ch)
        out.append("".join(buf))
    return "".join(out)


_MATRIX_ENV_RE = re.compile(r"\\begin\{([pbvV]?matrix)\}(.*?)\\end\{\1\}", re.DOTALL)


def _latex_matrix_to_expr(s: str) -> str:
    """\\begin{pmatrix}1&2\\\\3&4\\end{pmatrix} → Matrix([[1,2],[3,4]])。"""
    def conv(m):
        rows = [r.strip() for r in re.split(r"\\\\", m.group(2)) if r.strip()]
        rows_txt = ",".join(
            "[" + ",".join(c.strip() or "0" for c in row.split("&")) + "]" for row in rows)
        return f"Matrix([{rows_txt}])"
    return _MATRIX_ENV_RE.sub(conv, s)


def _read_braced_group(s: str, start: int):
    """Read a balanced {...} group beginning at `start`. Returns (content, end).

    Returns (None, start) when `start` is not '{' or the braces never balance.
    """
    if start >= len(s) or s[start] != "{":
        return None, start
    depth = 0
    for index in range(start, len(s)):
        if s[index] == "{":
            depth += 1
        elif s[index] == "}":
            depth -= 1
            if depth == 0:
                return s[start + 1:index], index + 1
    return None, start


def _convert_frac(s: str) -> str:
    r"""Rewrite every \frac{a}{b} as ((a)/(b)), tolerating nested braces.

    A regex with `[^{}]*` cannot express this: in `\frac{\pi(\sqrt{3}+1)}{12}` the
    numerator itself contains braces, so the pattern failed, the expression never
    parsed, and the pair fell through to string similarity. That is exactly why Q3
    of the 2026-07-29 judge run compared `$\frac{\pi(\sqrt{3}+1)}{12}$` against
    `pi/12 + sqrt(3)*pi/12` at 0.40 similarity and paid for an arbitration round.
    """
    out = []
    index = 0
    while True:
        position = s.find("\\frac", index)
        if position == -1:
            out.append(s[index:])
            return "".join(out)
        out.append(s[index:position])
        cursor = position + len("\\frac")
        while cursor < len(s) and s[cursor].isspace():
            cursor += 1
        numerator, cursor = _read_braced_group(s, cursor)
        if numerator is None:
            out.append(s[position:cursor + 1] or s[position:])
            index = max(cursor, position + len("\\frac"))
            continue
        while cursor < len(s) and s[cursor].isspace():
            cursor += 1
        denominator, cursor = _read_braced_group(s, cursor)
        if denominator is None:
            out.append(s[position:cursor] or s[position:])
            index = max(cursor, position + len("\\frac"))
            continue
        out.append(f"(({_convert_frac(numerator)})/({_convert_frac(denominator)}))")
        index = cursor


def _latex_to_expr_text(s: str) -> str:
    s = _latex_matrix_to_expr(s)
    s = s.replace("$$", " ").replace("$", " ")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = _convert_frac(s)
    s = re.sub(r"\\sqrt\s*\[\s*([^\[\]]+)\s*\]\s*\{([^{}]*)\}", r"((\2)**(1/(\1)))", s)
    for _ in range(4):
        new = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r"(sqrt(\1))", s)
        if new == s:
            break
        s = new
    s = re.sub(r"\\sqrt\s*(\d+)", r"(sqrt(\1))", s)
    for cmd in ("operatorname", "mathrm", "text", "mathbf", "boldsymbol"):
        s = re.sub(r"\\%s\s*\{([^{}]*)\}" % cmd, r"\1", s)
    s = (s.replace("\\pi", " pi ").replace("\\cdot", "*").replace("\\times", "*")
         .replace("\\div", "/").replace("\\infty", "oo"))
    s = re.sub(r"\\(sin|cos|tan|cot|sec|csc|log|ln|exp|sinh|cosh|tanh|arcsin|arccos|arctan)\b", r"\1", s)
    if "\\" not in s:
        # 剩余的花括号只是分组（如 e^{2x} → e**(2x)）
        s = s.replace("{", "(").replace("}", ")")
    return s


def _normalize_expr_text(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"^(?:最终答案|答案|结论)\s*[：:]\s*", "", s)
    s = s.strip().rstrip("。.，,；;")
    for src, dst in _UNICODE_REPLACEMENTS:
        s = s.replace(src, dst)
    s = re.sub(r"√\s*\(([^()]*)\)", r"(sqrt(\1))", s)
    s = re.sub(r"√\s*(\d+(?:\.\d+)?|[a-zA-Z])", r"(sqrt(\1))", s)
    if "\\" in s or "$" in s:
        s = _latex_to_expr_text(s)
    s = s.replace("π", "(pi)")
    return s.strip()


def _try_parse_expr(text: str):
    """尽力把答案文本解析成 sympy 表达式；失败返回 None。"""
    import sympy as sp
    from sympy.parsing.sympy_parser import (
        parse_expr, standard_transformations, implicit_multiplication_application,
        convert_xor)
    raw = (text or "").strip()
    if not raw or len(raw) > 400:
        return None
    # 剥离"J = "、"K = "这类命名标签，比较值本身（评委报告 idx=352：标签后接矩阵）。
    # 复用 structured_answer 的标签定义，使 `\bar y = …`、`\lambda = …` 与纯 ASCII
    # 标签走同一套规则——此处曾另写一份仅认 ASCII 的正则，导致希腊字母标签的答案
    # 整条无法解析（2026-07-29 判分 Q3：`$\bar y=\frac{\pi(\sqrt3+1)}{12}$`）。
    label_stripped = LABEL_ASSIGNMENT_PREFIX_RE.sub("", raw, count=1)
    variants = [raw] if label_stripped == raw else [label_stripped, raw]
    candidates = []
    for v in variants:
        if _CONST_PRODUCT_CHARS_RE.match(v) and re.search(r"[πiIeE√]", v):
            candidates.append(_const_product_to_expr(v))
    for v in variants:
        candidates.append(_normalize_expr_text(v))
    candidates.extend(variants)
    transformations = standard_transformations + (implicit_multiplication_application, convert_xor)
    local_dict = {"pi": sp.pi, "e": sp.E, "E": sp.E, "i": sp.I, "I": sp.I, "oo": sp.oo}
    for cand in candidates:
        cand = (cand or "").strip()
        if not cand:
            continue
        try:
            return sp.sympify(cand)
        except Exception:
            pass
        try:
            return parse_expr(cand, transformations=transformations, local_dict=local_dict)
        except Exception:
            continue
    return None


class AnswerMatcher:
    @staticmethod
    def _python_verification_succeeded(python_result) -> bool:
        result = python_result or {}
        try:
            from utils.verification_evidence import python_answer_is_trusted
            trusted = python_answer_is_trusted(result)
        except Exception:
            trusted = False
        return (
            bool(result.get("success"))
            and bool(result.get("stdout"))
            and str(result.get("evidence_status") or "") == "support"
            and trusted
        )

    @staticmethod
    def detect_problem_type(problem: str) -> str:
        from utils.problem_profile import classify_question_mode

        return "proof" if classify_question_mode(problem) == "proof" else "computation"

    @staticmethod
    def _match_computation_detailed(answer1: str, answer2: str, tolerance: float = 1e-6):
        a1 = (answer1 or "").strip()
        a2 = (answer2 or "").strip()
        if is_placeholder_answer(a1) or is_placeholder_answer(a2):
            return {
                "verdict": None,
                "is_match": False,
                "confidence": 0.0,
                "reason": "占位符或空答案，无法判定匹配",
                "method": "placeholder",
                "text_similarity": 0.0,
                "matched_fields": [],
                "mismatched_fields": [],
                "field_coverage": 0.0,
            }

        structured = compare_structured_answers(
            a1, a2, AnswerMatcher._sympy_equivalent, tolerance=tolerance)
        if structured.applicable:
            #
            if a1 == a2 and structured.verdict is not False \
                    and not is_self_conflicting(
                        a1, AnswerMatcher._sympy_equivalent, tolerance):
                return {
                    "verdict": True,
                    "is_match": True,
                    "confidence": 1.0,
                    "reason": "答案字符串完全匹配",
                    "method": "exact",
                    "text_similarity": 1.0,
                    "matched_fields": list(structured.matched_fields),
                    "mismatched_fields": [],
                    "field_coverage": structured.coverage,
                }
            similarity = AnswerMatcher._string_similarity(a1, a2)
            if structured.verdict is True:
                return {
                    "verdict": True,
                    "is_match": True,
                    "confidence": 0.98,
                    "reason": "复合答案字段经归一化后符号等价",
                    "method": "structured_symbolic",
                    "text_similarity": similarity,
                    "matched_fields": list(structured.matched_fields),
                    "mismatched_fields": [],
                    "field_coverage": structured.coverage,
                }
            if structured.verdict is False:
                return {
                    "verdict": False,
                    "is_match": False,
                    "confidence": 0.95,
                    "reason": "复合答案存在不等价字段",
                    "method": "structured_symbolic",
                    "text_similarity": similarity,
                    "matched_fields": list(structured.matched_fields),
                    "mismatched_fields": list(structured.mismatched_fields),
                    "field_coverage": structured.coverage,
                }
            return {
                "verdict": None,
                "is_match": False,
                "confidence": similarity,
                "reason": structured.reason,
                "method": "structured_indeterminate",
                "text_similarity": similarity,
                "matched_fields": list(structured.matched_fields),
                "mismatched_fields": list(structured.mismatched_fields),
                "field_coverage": structured.coverage,
            }

        if a1 == a2 and a1:
            return {
                "verdict": True,
                "is_match": True,
                "confidence": 1.0,
                "reason": "答案字符串完全匹配",
                "method": "exact",
                "text_similarity": 1.0,
                "matched_fields": [],
                "mismatched_fields": [],
                "field_coverage": 0.0,
            }

        similarity = AnswerMatcher._string_similarity(a1, a2)
        try:
            v1, v2 = float(a1), float(a2)
            diff = abs(v1 - v2)
            if diff < 1e-10:
                verdict, confidence, reason = True, 1.0, "数值完全匹配"
            elif diff < tolerance:
                verdict, confidence, reason = True, 0.95, f"数值近似匹配，误差{diff:.2e}"
            else:
                verdict, confidence, reason = False, 0.8, f"数值差异过大: {diff:.2e}"
            return {
                "verdict": verdict,
                "is_match": verdict,
                "confidence": confidence,
                "reason": reason,
                "method": "numeric",
                "text_similarity": similarity,
                "matched_fields": [],
                "mismatched_fields": [],
                "field_coverage": 0.0,
            }
        except Exception:
            pass

        verdict = AnswerMatcher._sympy_equivalent(a1, a2, tolerance)
        if verdict is True:
            return {
                "verdict": True,
                "is_match": True,
                "confidence": 0.98,
                "reason": "符号/数值等价（归一化后）",
                "method": "symbolic",
                "text_similarity": similarity,
                "matched_fields": [],
                "mismatched_fields": [],
                "field_coverage": 0.0,
            }
        if verdict is False:
            return {
                "verdict": False,
                "is_match": False,
                "confidence": 0.8,
                "reason": "符号表达式不等价",
                "method": "symbolic",
                "text_similarity": similarity,
                "matched_fields": [],
                "mismatched_fields": [],
                "field_coverage": 0.0,
            }
        if similarity > 0.9:
            verdict, is_match, reason = True, True, f"高度相似({similarity:.2f})"
        elif similarity > 0.7:
            verdict, is_match, reason = False, False, f"部分相似({similarity:.2f})"
        else:
            verdict, is_match, reason = None, False, f"明显不同({similarity:.2f})"
        return {
            "verdict": verdict,
            "is_match": is_match,
            "confidence": similarity,
            "reason": reason,
            "method": "text",
            "text_similarity": similarity,
            "matched_fields": [],
            "mismatched_fields": [],
            "field_coverage": 0.0,
        }

    @staticmethod
    def match_computation_answer(answer1: str, answer2: str, tolerance: float = 1e-6):
        detail = AnswerMatcher._match_computation_detailed(answer1, answer2, tolerance)
        return detail["is_match"], detail["confidence"], detail["reason"]

    @staticmethod
    def _numeric_sample_equivalent(e1, e2, tolerance: float):
        """Decide equivalence of symbolic expressions by evaluating at fixed points.

        Closes the gap the 2026-07-29 judge report called the system's most
        significant structural weakness (§4.2): with a free symbol present,
        `simplify(e1 - e2)` often cannot decide, `sp.N` cannot evaluate, and a
        provably-equal pair like Q1's `4*pi*(32-a**5)/5` versus
        `\\frac{4\\pi}{5}(32-a^5)` fell through to string similarity (0.60) and
        paid for an arbitration round.

        Deliberately conservative:
          * only runs when both sides carry the *same* symbols — differing names
            may be the same maths relabelled, which sampling cannot adjudicate;
          * sample points are fixed, not random, so a verdict is reproducible;
          * complex points are included, so pairs that agree on the positive reals
            but differ in general (`sqrt(x**2)` vs `x`) are still separated;
          * a single well-conditioned disagreement proves non-equivalence, while
            agreement everywhere is treated as equivalence only with enough
            successful evaluations — otherwise the answer is None, not a guess.
        """
        import sympy as sp

        symbols = sorted(e1.free_symbols | e2.free_symbols, key=str)
        if not symbols or e1.free_symbols != e2.free_symbols:
            return None
        if len(symbols) > 3:
            return None

        base_points = [
            sp.Rational(37, 100), sp.Rational(153, 100), sp.Rational(271, 100),
            sp.Rational(-89, 100), sp.Rational(457, 100),
            sp.Rational(61, 100) + sp.Rational(43, 100) * sp.I,
        ]

        evaluated = 0
        for point in base_points:
            substitution = {
                symbol: point + sp.Rational(index, 17)
                for index, symbol in enumerate(symbols)
            }
            try:
                v1 = complex(sp.N(e1.subs(substitution), 30, chop=True))
                v2 = complex(sp.N(e2.subs(substitution), 30, chop=True))
            except Exception:
                continue  # pole, branch cut, or unevaluable here — try the next point
            if any(map(lambda value: value != value or abs(value) == float("inf"),
                       (v1.real, v1.imag, v2.real, v2.imag))):
                continue
            scale = max(1.0, abs(v1), abs(v2))
            if abs(v1 - v2) > max(tolerance, 1e-9) * scale:
                return False
            evaluated += 1

        return True if evaluated >= 3 else None

    @staticmethod
    def _sympy_equivalent(a1: str, a2: str, tolerance: float):
        """True=等价 / False=确定不等价 / None=无法判定（走字符串相似度）。"""
        import sympy as sp
        e1, e2 = _try_parse_expr(a1), _try_parse_expr(a2)
        if e1 is None or e2 is None:
            return None
        # 矩阵结构化比较：形状不同或差非零矩阵 → 确定不等价（触发 reconciliation 重试，
        # 而非落入低置信 uncertain 直接采信 Python 答案——评委报告 idx=352 教训）
        is_m1 = isinstance(e1, sp.MatrixBase)
        is_m2 = isinstance(e2, sp.MatrixBase)
        if is_m1 or is_m2:
            if is_m1 and is_m2:
                if e1.shape != e2.shape:
                    return False
                try:
                    return bool(sp.simplify(e1 - e2).is_zero_matrix)
                except Exception:
                    return None
            mat, scalar = (e1, e2) if is_m1 else (e2, e1)
            if mat.shape == (1, 1):
                e1, e2 = mat[0, 0], scalar
            else:
                return False
        diff = None
        try:
            diff = sp.simplify(e1 - e2)
            if diff == 0:
                return True
            zero_test = diff.equals(0)
            if zero_test is False and diff.free_symbols:
                return False
        except Exception:
            diff = None
        try:
            c1 = complex(sp.N(e1, 30, chop=True))
            c2 = complex(sp.N(e2, 30, chop=True))
            scale = max(1.0, abs(c1), abs(c2))
            return abs(c1 - c2) <= max(tolerance, 1e-9 * scale)
        except Exception:
            pass  # 含自由符号，无法整体数值比较——改用采样
        sampled = AnswerMatcher._numeric_sample_equivalent(e1, e2, tolerance)
        if sampled is not None:
            return sampled
        if diff is not None:
            try:
                if not diff.free_symbols and abs(complex(sp.N(diff, 30, chop=True))) > tolerance:
                    return False
            except Exception:
                return None
        return None

    @staticmethod
    def match_proof_answer(reasoning_result, python_result):
        rr = reasoning_result or {}
        po = python_result or {}
        answer = rr.get("answer", "")
        reasoning_complete = (
            bool(rr.get("steps")) and len(rr["steps"]) >= 2
            and bool(answer) and len(answer) > 10 and not is_placeholder_answer(answer)
        )
        python_success = AnswerMatcher._python_verification_succeeded(po)
        if reasoning_complete and python_success:
            return True, 0.85, "证明题：推理完整且Python验证通过"
        if reasoning_complete and not python_success:
            return True, 0.70, "证明题：推理完整，Python验证未通过（证明题可能难以编程验证）"
        if not reasoning_complete and python_success:
            return False, 0.50, "证明题：Python验证通过但推理不完整"
        return False, 0.30, "证明题：推理和验证都不完整"

    @staticmethod
    def match_answers(problem, reasoning_result, python_result):
        ptype = AnswerMatcher.detect_problem_type(problem)
        if ptype == "proof":
            proof_is_match, confidence, reason = AnswerMatcher.match_proof_answer(
                reasoning_result, python_result)
            if proof_is_match and confidence >= 0.8:
                verdict, status = True, "match"
            elif (not proof_is_match) and confidence >= 0.6:
                verdict, status = False, "mismatch"
            else:
                verdict, status = None, "uncertain"
            return {
                "status": status,
                "verdict": verdict,
                "confidence": confidence,
                "reason": reason,
                "problem_type": "proof",
                "method": "proof",
                "text_similarity": 0.0,
                "matched_fields": [],
                "mismatched_fields": [],
                "field_coverage": 0.0,
                "confidence_components": {
                    "symbolic": 0.0,
                    "text_similarity": 0.0,
                    "python_success": AnswerMatcher._python_verification_succeeded(
                        python_result),
                    "reasoning_validation_points": len(
                        (reasoning_result or {}).get("validation_points") or []),
                },
            }

        reasoning_answer = (reasoning_result or {}).get("answer", "")
        python_answer = (python_result or {}).get("answer", "") if python_result else ""
        detail = AnswerMatcher._match_computation_detailed(reasoning_answer, python_answer)
        python_success = bool((python_result or {}).get("success"))
        confidence_components = {
            "symbolic": detail["confidence"]
            if detail["method"] in {"symbolic", "structured_symbolic"} else 0.0,
            "text_similarity": detail["text_similarity"],
            "python_success": python_success,
            "reasoning_validation_points": len(
                (reasoning_result or {}).get("validation_points") or []),
        }
        comparison_verdict = detail["verdict"]
        if comparison_verdict is True and detail["confidence"] >= 0.8:
            verdict = True if python_success else None
            status = "match" if python_success else "uncertain"
        elif comparison_verdict is False:
            verdict = False
            status = "mismatch"
        else:
            verdict = None
            status = "uncertain"
        public_detail = {key: value for key, value in detail.items() if key != "is_match"}
        return {
            **public_detail,
            "verdict": verdict,
            "comparison_verdict": comparison_verdict,
            "status": status,
            "problem_type": "computation",
            "confidence_components": confidence_components,
        }

    @staticmethod
    def _string_similarity(a: str, b: str) -> float:
        try:
            from Levenshtein import ratio
            return ratio(a, b)
        except Exception:
            if not a or not b:
                return 0.0
            s = sum(1 for c in a if c in b)
            return s / max(len(a), len(b))

