"""Transport retry, bounded by the problem deadline rather than by attempt count.

Why the count alone is not enough: the platform injects its own client, and
`InternChatClient` already retries 3× internally at a 120s socket timeout. Three
outer attempts over that is up to 9 requests ≈ 1080s spent inside a single node —
enough to blow the 20-minute per-problem limit on its own, which is what the
2026-07-29 judge run's Q3 (1307s) actually spent its time on. We cannot lower the
inner count (it belongs to the platform's client), so instead every attempt must
ask the clock for permission first.
"""

import logging
import time
from typing import Dict, List

from config import CONFIG
from utils.prefill import prefill_messages, stitch


#: 失败耗时达到 socket 超时的这个比例即判定为"传输卡死"而非"偶发错误"。
#: 卡死的调用重试还会卡死；偶发错误（4xx/5xx、JSON 解析失败）几秒内就返回，
#: 远低于这条线，仍然照常重试。
_STALL_THRESHOLD_S = 0.8 * 280


class DeadlineExceeded(RuntimeError):
    """Raised when the problem's time budget cannot fund another attempt."""


class LLMRetryWrapper:
    def __init__(
        self,
        client,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        logger=None,
        time_budget=None,
        expected_call_seconds: float | None = None,
        reserve_margin_s: float | None = None,
    ):
        self.client = client
        self.max_retries = max(1, max_retries)
        self.backoff_factor = backoff_factor
        self.logger = logger or logging.getLogger("math_agent")
        self.time_budget = time_budget
        self.expected_call_seconds = expected_call_seconds
        # reserve_margin_s: 允许本次调用动用 reserve（硬上限前的预留时间），但必须
        # 给后续的答案落盘留出 margin 秒。评委报告：64 道零分题里绝大多数
        # `arbiter=skipped(0.0s)` —— 软预算耗尽后仲裁被整体跳过，恰是最需要复核的
        # 场景。仲裁的 prefill 调用只要 ~10s，reserve(300s) 划出固定配额是净收益。
        self.reserve_margin_s = reserve_margin_s

    def _now(self) -> float:
        """Read the budget's own clock, so measured cost and remaining time agree.

        Mixing `time.monotonic()` here with the budget's clock made the observed
        duration meaningless whenever the two differed, which silently disabled the
        adaptive check that is supposed to stop a doomed retry.
        """
        now = getattr(self.time_budget, "now", None)
        return now() if callable(now) else time.monotonic()

    def _affordable(self, observed: float | None = None) -> bool:
        """Can the clock fund another attempt?

        `observed` is how long the previous attempt actually took. It matters: a
        transport failure that burned 363s (the client's own 3 x 120s retries) will
        almost certainly burn another 363s under the same load, so estimating the
        retry at the optimistic `expected_call_seconds` would let us start a call we
        cannot finish. Measured cost always wins over the estimate.
        """
        if not self.time_budget:
            return True
        estimate = self.expected_call_seconds
        if observed is not None:
            estimate = observed if estimate is None else max(estimate, observed)
        if self.reserve_margin_s is not None:
            # 预留配额模式：软预算已尽也允许，只要硬上限前还能容下
            # 本次调用 + margin。remaining_hard 不足时照样拒绝。
            window = self.time_budget.remaining_hard() - float(self.reserve_margin_s)
            return window >= (estimate if estimate is not None else 1.0)
        if estimate is None:
            return not self.time_budget.expired()
        return self.time_budget.can_afford(estimate)

    def chat(
        self,
        messages: List[Dict],
        temperature: float = 0.2,
        max_tokens: int = 4096,
        label: str = "llm",
    ) -> str:
        if not self._affordable():
            need = (f"~{self.expected_call_seconds:.0f}s"
                    if self.expected_call_seconds is not None else "any time at all")
            raise DeadlineExceeded(
                f"{label}: {self.time_budget.remaining():.0f}s left, need {need}"
            )

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            started = self._now()
            try:
                result = self.client.chat(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                if self.time_budget:
                    self.time_budget.record(label, self._now() - started)
                return result
            except Exception as exc:  # noqa: BLE001 - retry transient transport failures.
                last_error = exc
                observed = self._now() - started
                if self.time_budget:
                    self.time_budget.record(f"{label}:failed", observed)
                if attempt >= self.max_retries:
                    break
                # 传输卡死不重试同一个满长调用（2026-08-20 全量计时统计）：221 次
                # 成功调用最长 245s、中位 54s，而 206 次失败**全部**卡在 socket 超时
                # 上（168 次 360s、38 次 780s）——两者之间没有任何重叠。这说明超过
                # 超时的调用不是"慢"，是不会回来了；在同样的并发负载下重试必然
                # 再卡一次。实测代价：idx 17 连吃两个 360s 才升级到压缩路径，最后
                # 只剩 22s 用来真正解题。此处直接放弃重试，把时间交给调用方的
                # 压缩路径——它在观测到的每一次卡死后都在 22–40s 内成功返回。
                if observed >= _STALL_THRESHOLD_S:
                    self.logger.warning(
                        "%s: transport stalled %.0fs (>=%ss); skipping the full-length "
                        "retry so the caller can escalate to its compressed path (%s)",
                        label, observed, _STALL_THRESHOLD_S, exc,
                    )
                    break
                if not self._affordable(observed):
                    self.logger.warning(
                        "%s: abandoning retry %s/%s after %.0fs, time budget cannot "
                        "fund another attempt (%s)",
                        label, attempt, self.max_retries, observed, exc,
                    )
                    break
                delay = 0 if self.backoff_factor == 0 else self.backoff_factor ** (attempt - 1)
                self.logger.warning(
                    "LLM call failed on attempt %s/%s: %s", attempt, self.max_retries, exc
                )
                if delay > 0:
                    time.sleep(delay)
        raise last_error


def chat_with_retry(
    client,
    messages,
    temperature=0.2,
    max_tokens=4096,
    logger=None,
    time_budget=None,
    expected_call_seconds=None,
    label="llm",
):
    return LLMRetryWrapper(
        client,
        max_retries=CONFIG["llm_max_retries"],
        backoff_factor=CONFIG["backoff_factor"],
        logger=logger,
        time_budget=time_budget,
        expected_call_seconds=expected_call_seconds,
    ).chat(
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        label=label,
    )


def chat_prefilled(
    client,
    messages,
    prefix,
    temperature=0.1,
    max_tokens=96,
    logger=None,
    time_budget=None,
    expected_call_seconds=None,
    label="llm_prefill",
    max_retries=1,
    reserve_margin_s=None,
):
    """One prefilled round trip, returning the stitched assistant text.

    Defaults to a single attempt: a prefilled call is the cheap fast path, and
    every caller has a full-budget non-prefill fallback behind it, so spending
    retries here would buy latency without buying reliability.

    ``reserve_margin_s`` 非 None 时，本调用允许动用 reserve（软预算已尽仍可
    执行），仅要求硬上限前剩余 ≥ 调用估时 + margin。用于仲裁保底与应急直答。
    """
    wrapper = LLMRetryWrapper(
        client,
        max_retries=max_retries,
        backoff_factor=CONFIG["backoff_factor"],
        logger=logger,
        time_budget=time_budget,
        expected_call_seconds=expected_call_seconds,
        reserve_margin_s=reserve_margin_s,
    )
    raw = wrapper.chat(
        messages=prefill_messages(messages, prefix),
        temperature=temperature,
        max_tokens=max_tokens,
        label=label,
    )
    return stitch(prefix, raw)

