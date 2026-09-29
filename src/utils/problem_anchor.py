"""Immutable-at-entry problem snapshot used to protect retry branches."""

from dataclasses import dataclass
from hashlib import sha256


@dataclass(frozen=True)
class AnchorCheck:
    ok: bool
    problem: str
    event: dict | None = None


def _digest(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def make_problem_anchor(problem: str, idx: int = -1) -> dict:
    text = str(problem or "")
    return {
        "text": text,
        "length": len(text),
        "sha256": _digest(text),
        "idx": idx,
    }


def verify_problem_anchor(state: dict) -> AnchorCheck:
    """Check the live problem and return the anchored text for a repair."""
    current = str(state.get("problem") or "")
    anchor = state.get("problem_anchor")
    if not isinstance(anchor, dict) or not isinstance(anchor.get("text"), str):
        return AnchorCheck(
            ok=True,
            problem=current,
            event={"action": "unverified", "reason": "missing_anchor"},
        )

    anchored = anchor["text"]
    expected_digest = _digest(anchored)
    observed_digest = _digest(current)
    anchor_valid = (
        anchor.get("length") == len(anchored)
        and anchor.get("sha256") == expected_digest
    )
    if anchor_valid and current == anchored and observed_digest == anchor["sha256"]:
        return AnchorCheck(ok=True, problem=current)

    event = {
        "action": "repair" if anchor_valid else "anchor_corrupt",
        "expected_sha256": anchor.get("sha256", ""),
        "observed_sha256": observed_digest,
    }
    return AnchorCheck(ok=False, problem=anchored, event=event)
