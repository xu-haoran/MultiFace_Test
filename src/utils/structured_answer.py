from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable


MAX_ANSWER_LENGTH = 4000
MAX_FIELDS = 16
MAX_EXPRESSION_LENGTH = 400
_MATH_DELIMITERS = (("$$", "$$"), (r"\[", r"\]"), (r"\(", r"\)"), ("$", "$"))
_ASYMMETRIC_CLOSERS = (r"\]", r"\)")
_MAXIMUM_ALIAS_RE = re.compile(
    r"^(?:max\s+v|v\s*_\s*(?:\{\s*max\s*\}|max)|最大值)$", re.IGNORECASE)

_LATEX_LETTERS = (
    "alpha", "beta", "gamma", "delta", "epsilon", "varepsilon", "zeta", "eta",
    "theta", "vartheta", "iota", "kappa", "lambda", "mu", "nu", "xi", "pi",
    "varpi", "rho", "varrho", "sigma", "varsigma", "tau", "upsilon", "phi",
    "varphi", "chi", "psi", "omega",
    "Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi", "Sigma", "Upsilon", "Phi",
    "Psi", "Omega",
)
_LATEX_ACCENTS = ("bar", "hat", "tilde", "vec", "dot", "ddot", "overline", "widehat")

_LATEX_LETTER_ALTERNATION = "|".join(sorted(_LATEX_LETTERS, key=len, reverse=True))
_LATEX_ACCENT_ALTERNATION = "|".join(sorted(_LATEX_ACCENTS, key=len, reverse=True))
_LABEL_ATOM = (
    r"(?:\\(?:%s)\s*(?:\{\s*(?:\\(?:%s)|[a-z])\s*\}|\\(?:%s)|[a-z])"
    r"|\\(?:%s)|[a-z])"
) % (_LATEX_ACCENT_ALTERNATION, _LATEX_LETTER_ALTERNATION,
     _LATEX_LETTER_ALTERNATION, _LATEX_LETTER_ALTERNATION)
_LABEL_RE = re.compile(
    r"^(?:max\s+v|v\s*_\s*(?:\{\s*max\s*\}|max)|最大值|"
    r"%s[a-z0-9]*(?:\s*_\s*(?:\{\s*[a-z0-9]+\s*\}|[a-z0-9]+))?)$" % _LABEL_ATOM,
    re.IGNORECASE,
)
_LATEX_ACCENT_STRIP_RE = re.compile(
    r"\\(?:%s)\s*" % _LATEX_ACCENT_ALTERNATION, re.IGNORECASE)
_LATEX_LETTER_STRIP_RE = re.compile(
    r"\\(%s)" % _LATEX_LETTER_ALTERNATION)
_MAXIMUM_LEAD_RE = re.compile(r"最大值\s*(?:为|是|[：:])\s*")
_UNSUPPORTED_CONCLUSION_RE = re.compile(r"(?:最小值|极大值|极小值)\s*(?:为|是|[：:])")
_COMPARISON_RE = re.compile(r"==|!=|<=|>=")


@dataclass(frozen=True)
class ExtractedFields:
    fields: dict[str, tuple[str, ...]]
    assignment_like_count: int
    unconsumed_segments: tuple[str, ...]
    malformed_delimiters: bool = False


@dataclass(frozen=True)
class StructuredComparison:
    applicable: bool
    verdict: bool | None
    matched_fields: tuple[str, ...]
    mismatched_fields: tuple[str, ...]
    left_fields: tuple[str, ...]
    right_fields: tuple[str, ...]
    coverage: float
    reason: str
    unconsumed_segments: tuple[str, ...] = ()


LABEL_ASSIGNMENT_PREFIX_RE = re.compile(
    r"^\s*\$?\s*(?:%s[a-z0-9]*(?:\s*_\s*(?:\{\s*[a-z0-9]+\s*\}|[a-z0-9]+))?)\s*=\s*"
    % _LABEL_ATOM,
    re.IGNORECASE,
)


def _canonical_label(label: str) -> str:
    """Reduce a label to the key both sides of a comparison will agree on.

    `\\bar{y}` → `y`, `\\beta_0` → `beta0`, `\\sigma` → `sigma`, so a LaTeX reasoning
    answer and a plain-text Python answer name the same field.
    """
    raw = (label or "").strip()
    if _MAXIMUM_ALIAS_RE.fullmatch(raw):
        return "maximum"
    normalized = _LATEX_ACCENT_STRIP_RE.sub("", raw)
    normalized = _LATEX_LETTER_STRIP_RE.sub(r"\1", normalized)
    compact = re.sub(r"[\s_{}\\$]+", "", normalized.lower())
    if compact in {"maxv", "vmax", "maximum"}:
        return ""
    return compact if re.fullmatch(r"[a-z][a-z0-9]*", compact) else ""


def _clean_expression(expression: str) -> str:
    return (expression or "").strip().strip("$").strip().rstrip("。.，,；;").strip()


def _opener_at(text: str, index: int):
    return next(((opening, closing) for opening, closing in _MATH_DELIMITERS
                 if text.startswith(opening, index)), None)


def _split_outside_math(text: str) -> tuple[list[str], bool]:
    segments, current = [], []
    closing = None
    malformed = False
    index = 0
    while index < len(text):
        if closing is not None:
            if text.startswith(closing, index):
                current.append(closing)
                index += len(closing)
                closing = None
            else:
                current.append(text[index])
                index += 1
            continue
        opener = _opener_at(text, index)
        if opener:
            opening, closing = opener
            current.append(opening)
            index += len(opening)
        elif any(text.startswith(candidate, index) for candidate in _ASYMMETRIC_CLOSERS):
            malformed = True
            current.append(text[index:index + 2])
            index += 2
        elif text[index] in ";；\n":
            segment = "".join(current).strip()
            if segment:
                segments.append(segment)
            current = []
            index += 1
        else:
            current.append(text[index])
            index += 1
    segment = "".join(current).strip()
    if segment:
        segments.append(segment)
    return segments, malformed or closing is not None


def _math_spans(text: str) -> tuple[list[str], bool]:
    spans = []
    closing = None
    buffer = []
    malformed = False
    index = 0
    while index < len(text):
        if closing is not None:
            if text.startswith(closing, index):
                spans.append("".join(buffer))
                buffer = []
                index += len(closing)
                closing = None
            else:
                buffer.append(text[index])
                index += 1
            continue
        opener = _opener_at(text, index)
        if opener:
            opening, closing = opener
            index += len(opening)
        elif any(text.startswith(candidate, index) for candidate in _ASYMMETRIC_CLOSERS):
            malformed = True
            index += 2
        else:
            index += 1
    return spans, malformed or closing is not None


def _outside_math_text(text: str) -> tuple[str, bool]:
    outside = []
    closing = None
    buffer = []
    malformed = False
    index = 0
    while index < len(text):
        if closing is not None:
            if text.startswith(closing, index):
                content = "".join(buffer)
                outside.append(" " if "=" in content else f"({content})")
                buffer = []
                index += len(closing)
                closing = None
            else:
                buffer.append(text[index])
                index += 1
            continue
        opener = _opener_at(text, index)
        if opener:
            opening, closing = opener
            buffer = []
            index += len(opening)
        elif any(text.startswith(candidate, index) for candidate in _ASYMMETRIC_CLOSERS):
            malformed = True
            index += 2
        else:
            outside.append(text[index])
            index += 1
    return "".join(outside), malformed or closing is not None


def _top_level_equals(text: str) -> list[int]:
    depth = 0
    positions = []
    for index, char in enumerate(text):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        elif char == "=" and depth == 0:
            positions.append(index)
    return positions


def _parse_assignment(text: str):
    value = (text or "").strip()
    if _COMPARISON_RE.search(value):
        return None
    equals = _top_level_equals(value)
    if len(equals) != 1:
        return None
    position = equals[0]
    label, expression = value[:position].strip(), value[position + 1:].strip()
    if not _LABEL_RE.fullmatch(label):
        return None
    key = _canonical_label(label)
    expression = _clean_expression(expression)
    if not key or not expression or len(expression) > MAX_EXPRESSION_LENGTH:
        return None
    return key, expression


def _record(fields: dict[str, list[str]], key: str, expression: str) -> bool:
    value = _clean_expression(expression)
    if (key != "maximum" and not re.fullmatch(r"[a-z][a-z0-9]*", key or "")) \
            or not value or len(value) > MAX_EXPRESSION_LENGTH:
        return False
    if key not in fields and len(fields) >= MAX_FIELDS:
        return False
    fields.setdefault(key, []).append(value)
    return True


def extract_named_fields(answer: str) -> ExtractedFields:
    raw = answer or ""
    if len(raw) > MAX_ANSWER_LENGTH:
        count = max(2, raw.count("=")) if "=" in raw else 0
        return ExtractedFields({}, count, ("answer exceeds structured length limit",), True)
    segments, malformed = _split_outside_math(raw)
    fields: dict[str, list[str]] = {}
    unconsumed: list[str] = []
    assignment_like_count = 0
    for segment in segments:
        spans, span_malformed = _math_spans(segment)
        residual, residual_malformed = _outside_math_text(segment)
        malformed = malformed or span_malformed or residual_malformed
        for span in spans:
            if "=" not in span:
                continue
            assignment_like_count += 1
            parsed = _parse_assignment(span)
            if parsed is None:
                unconsumed.append(span)
            else:
                key, expression = parsed
                if not _record(fields, key, expression):
                    unconsumed.append(span)

        maximum = _MAXIMUM_LEAD_RE.search(segment)
        if maximum:
            assignment_like_count += 1
            tail = segment[maximum.end():].strip()
            tail_spans, _ = _math_spans(tail)
            expression = tail_spans[0] if tail_spans else tail
            if not _record(fields, "maximum", expression):
                unconsumed.append(segment)

        if "=" in residual:
            assignment_like_count += 1
            parsed = _parse_assignment(residual)
            if parsed is None:
                unconsumed.append(residual.strip() or segment)
            else:
                key, expression = parsed
                if not _record(fields, key, expression):
                    unconsumed.append(residual.strip() or segment)

        if _UNSUPPORTED_CONCLUSION_RE.search(segment):
            assignment_like_count += 1
            unconsumed.append(segment)

    return ExtractedFields(
        fields={key: tuple(values) for key, values in fields.items()},
        assignment_like_count=assignment_like_count,
        unconsumed_segments=tuple(dict.fromkeys(unconsumed)),
        malformed_delimiters=malformed,
    )


def is_self_conflicting(
    answer: str,
    equivalent: Callable[[str, str, float], bool | None],
    tolerance: float = 1e-6,
) -> bool:
    """Does one answer assign two non-equivalent values to the same field?

    `a = 1; a = 2` is wrong on its own terms, so two identical copies of it agreeing
    is not evidence of anything. This is the precise reason an identical pair may not
    take the exact-match shortcut — narrower than "the field parser had leftovers",
    which is also true of merely unparseable prose such as `f(x) = x^2`.
    """
    extracted = extract_named_fields(answer)
    if extracted.malformed_delimiters:
        return True  # truncated/unbalanced delimiters: treat as unusable, not agreeing
    return any(
        any(equivalent(values[0], value, tolerance) is not True for value in values[1:])
        for values in extracted.fields.values()
    )


def compare_structured_answers(
    left: str,
    right: str,
    equivalent: Callable[[str, str, float], bool | None],
    tolerance: float = 1e-6,
) -> StructuredComparison:
    left_extracted = extract_named_fields(left)
    right_extracted = extract_named_fields(right)
    left_names = tuple(sorted(left_extracted.fields))
    right_names = tuple(sorted(right_extracted.fields))
    union = set(left_names) | set(right_names)
    unconsumed = tuple(dict.fromkeys(
        left_extracted.unconsumed_segments + right_extracted.unconsumed_segments))
    applicable = (
        left_extracted.assignment_like_count >= 2
        or right_extracted.assignment_like_count >= 2
        or bool(unconsumed)
        or left_extracted.malformed_delimiters
        or right_extracted.malformed_delimiters
    )
    if not applicable:
        return StructuredComparison(
            applicable=False, verdict=None, matched_fields=(), mismatched_fields=(),
            left_fields=left_names, right_fields=right_names, coverage=0.0,
            reason="structured comparison is not applicable",
        )
    if unconsumed or left_extracted.malformed_delimiters or right_extracted.malformed_delimiters:
        return StructuredComparison(
            applicable=True, verdict=None, matched_fields=(), mismatched_fields=(),
            left_fields=left_names, right_fields=right_names, coverage=0.0,
            reason="malformed or unconsumed structured content",
            unconsumed_segments=unconsumed,
        )

    resolved = {}
    duplicate_problem = []
    for side, extracted in (("left", left_extracted), ("right", right_extracted)):
        side_values = {}
        for name, values in extracted.fields.items():
            base = values[0]
            if any(equivalent(base, value, tolerance) is not True for value in values[1:]):
                duplicate_problem.append(f"{side}:{name}")
            side_values[name] = base
        resolved[side] = side_values
    if duplicate_problem:
        return StructuredComparison(
            applicable=True, verdict=None, matched_fields=(), mismatched_fields=(),
            left_fields=left_names, right_fields=right_names, coverage=0.0,
            reason="duplicate fields conflict or cannot be compared",
            unconsumed_segments=tuple(duplicate_problem),
        )

    matched: list[str] = []
    mismatched: list[str] = []
    undecidable: list[str] = []
    for name in sorted(set(left_names) & set(right_names)):
        verdict = equivalent(resolved["left"][name], resolved["right"][name], tolerance)
        if verdict is True:
            matched.append(name)
        elif verdict is False:
            mismatched.append(name)
        else:
            undecidable.append(name)

    coverage = len(matched) / len(union) if union else 0.0
    if mismatched:
        verdict = False
        reason = "one or more structured fields are not equivalent"
    elif len(left_names) < 2 or set(left_names) != set(right_names):
        verdict = None
        reason = "structured field sets are incomplete or different"
    elif undecidable:
        verdict = None
        reason = "one or more structured fields could not be compared"
    else:
        verdict = True
        reason = "all structured fields are symbolically equivalent"
    return StructuredComparison(
        applicable=True,
        verdict=verdict,
        matched_fields=tuple(matched),
        mismatched_fields=tuple(mismatched),
        left_fields=left_names,
        right_fields=right_names,
        coverage=coverage,
        reason=reason,
        unconsumed_segments=(),
    )

