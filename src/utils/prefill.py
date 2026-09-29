"""Assistant-prefill decoding: the one lever that actually cuts this model's latency.

Measured 2026-07-29, intern-s2-preview-397b, identical prompts:

  classification  plain  max_tokens=8192 → 40.4s, 1969 completion tokens
  classification  prefill max_tokens=64  →  0.9s,   12 completion tokens   (58×)
  arbitration     plain  max_tokens=8192 → 70.2s, ~2000 completion tokens
  arbitration     prefill max_tokens=64  →  0.8s,    4 completion tokens  (140×)

Mechanism: seeding the assistant turn with the answer's opening characters puts
the model in continuation mode, so it never opens a `reasoning_content` block.
Accuracy is unaffected — prefill classification scored 4/4 on the judge set, and
prefill arbitration 3/3 including an adversarial pair where the *elegant* answer
was wrong and the *plain* one right.

Contrast with capping `max_tokens`, which was the original plan: because CoT is
billed first, a small cap truncates mid-thought (`finish_reason="length"`) and
the answer is never emitted at all. At caps of 64/128/256/512 every one of 11
probe calls returned nothing but a partial `Thinking Process:`. Prefill removes
the CoT instead of guillotining it, which is why it is safe to pair with a tiny
budget.

Compatibility: this only appends one assistant message, so it works through the
platform-injected `client.chat(messages=...)` with no client changes. Backends
differ in how they handle a trailing assistant turn — continue it, echo it back
whole, or ignore it — so `stitch` accepts all three shapes, and callers keep a
non-prefill fallback for backends that reject it outright.
"""

from __future__ import annotations


def prefill_messages(messages: list[dict], prefix: str) -> list[dict]:
    """Append the assistant seed turn that suppresses reasoning."""
    return list(messages) + [{"role": "assistant", "content": prefix}]


def stitch(prefix: str, completion: str) -> str:
    """Rebuild the full assistant text from the seed and whatever came back.

    Handles the three observed backend behaviours:
      * continuation  — completion picks up right after `prefix`  → concatenate
      * echo          — completion already restates `prefix`      → use as-is
      * ignored       — completion is a standalone full answer    → use as-is

    Distinguishing echo from continuation by prefix match alone is unreliable
    (a continuation may legitimately start with the same characters), so we also
    accept a completion that merely *contains* the prefix near its start.
    """
    body = completion if isinstance(completion, str) else str(completion or "")
    seed = prefix if isinstance(prefix, str) else str(prefix or "")
    if not seed:
        return body
    stripped = body.lstrip()
    if stripped.startswith(seed):
        return stripped
    head = body[: len(seed) + 40]
    if seed in head:
        return body[body.index(seed):]
    return seed + body

