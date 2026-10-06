"""Small, conservative helpers for independent objective-question checking.

Objective questions used to bypass the Python branch entirely.  This module does
not try to solve arbitrary mathematics deterministically; it decides whether a
Python result looks like an *independent* calculation and provides a canonical
comparison for the two solver branches.  The LLM still has to write the actual
verification program, while this layer prevents a successful ``print('A')`` from
being treated as evidence.
"""

from __future__ import annotations

import ast
import re
from typing import Any

from utils.problem_profile import (
    is_objective_mode,
    normalize_objective_answer,
    objective_answer_is_usable,
)


_MATH_CALL_RE = re.compile(
    r"(?i)\b(?:sympy|import\s+sympy|solve|solveset|simplify|factor|expand|"
    r"subs|limit|diff|integrate|series|det|rank|eigen|polys|nsimplify|"
    r"lambdify|numpy|scipy|statistics|variance|mean|gcd|binomial)\b"
)
_OPTION_RE = re.compile(r"(?i)(?:\b[A-E]\b|[Ａ-Ｅ])")


def objective_needs_independent_python(problem: str, mode: str) -> bool:
    """Route executable objective work; concepts use source-based review."""
    from utils.flow_policy import python_verification_needed
    return bool(problem and is_objective_mode(mode) and python_verification_needed(problem, mode))


def _ast_features(code: str) -> tuple[bool, bool, bool]:
    """Return ``(has_compare, has_call, has_assignment)`` for executable code."""
    try:
        tree = ast.parse(str(code or ""))
    except SyntaxError:
        return False, False, False
    has_compare = any(
        isinstance(node, (ast.Compare, ast.Assert, ast.If, ast.While))
        for node in ast.walk(tree)
    )
    has_call = any(isinstance(node, ast.Call) for node in ast.walk(tree))
    has_assignment = any(
        isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign))
        for node in ast.walk(tree)
    )
    return has_compare, has_call, has_assignment


def objective_code_is_independent(code: str, mode: str) -> bool:
    """Conservative authenticity gate for an objective verifier program.

    A program must contain a real branch/comparison, an assignment or function
    call, and mathematical operations.  This intentionally rejects code that
    merely prints a remembered option; rejecting a few valid but opaque programs
    is safer than allowing an answer anchor into the consensus path.
    """
    from utils.python_contract import DRIVER_MARKER, contract_answer_is_computed
    text = str(code or "").split(DRIVER_MARKER, 1)[0]
    if not contract_answer_is_computed(text):
        return False
    if not text.strip() or not _MATH_CALL_RE.search(text):
        return False
    has_compare, has_call, has_assignment = _ast_features(text)
    if not (has_compare and has_call and has_assignment):
        return False
    if mode == "choice" and not _OPTION_RE.search(text):
        return has_compare
    return True


def canonical_objective_answer(answer: Any, mode: str) -> str:
    value = normalize_objective_answer(str(answer or ""), mode)
    return value if objective_answer_is_usable(value, mode) else ""


def objective_answers_equal(left: Any, right: Any, mode: str) -> bool:
    """Compare two already extracted objective answers without prose effects."""
    a = canonical_objective_answer(left, mode)
    b = canonical_objective_answer(right, mode)
    if not a or not b:
        return False
    return re.sub(r"\s+", "", a).casefold() == re.sub(r"\s+", "", b).casefold()


