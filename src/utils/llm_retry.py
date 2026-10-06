"""Transport retry, bounded by the problem deadline rather than by attempt count.

Why the count alone is not enough: the platform injects its own client, and
`InternChatClient` already retries 3× internally at a 120s socket timeout. Three
outer attempts over that is up to 9 requests ≈ 1080s spent inside a single node —
enough to blow the 20-minute per-problem limit on its own, which is what the
2026-07-29 judge run's Q3 (1307s) actually spent its time on. We cannot lower the
inner count (it belongs to the platform's client), so instead every attempt must
ask the clock for permission first.
"""

import inspect
import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Dict, List

from config import CONFIG
from utils.prefill import prefill_messages, stitch


#: 失败耗时达到 socket 超时的这个比例即判定为"传输卡死"而非"偶发错误"。
#: 卡死的调用重试还会卡死；偶发错误（4xx/5xx、JSON 解析失败）几秒内就返回，
#: 远低于这条线，仍然照常重试。
_STALL_THRESHOLD_S = 0.8 * 280


class DeadlineExceeded(RuntimeError):
    """Raised when the problem's time budget cannot fund another attempt."""


class ChatText(str):
    """String-compatible model output with optional provider completion metadata."""

    def __new__(cls, text, finish_reason=None, completion_tokens=None, usage=None):
        value = super().__new__(cls, text or "")
        value.finish_reason = finish_reason
        value.completion_tokens = completion_tokens
        value.usage = usage
        return value


def _field(value, name, default=None):
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _as_chat_text(result):
    """Keep the existing string API while retaining usage/stop metadata if exposed."""
    choices = _field(result, "choices")
    choice = choices[0] if isinstance(choices, (list, tuple)) and choices else None
    message = _field(choice, "message") if choice is not None else None
    content = (
        _field(message, "content") if message is not None else None
    )
    if content is None and choice is not None:
        content = _field(choice, "text")
    if content is None:
        content = _field(result, "content")
    if content is None:
        content = _field(result, "text")
    if content is None:
        content = _field(result, "output_text")
    if content is None:
        content = _field(result, "response")
    if content is None:
        content = result

    if isinstance(content, list):
        content = "".join(
            str(_field(part, "text", ""))
            for part in content
            if _field(part, "type", "text") == "text"
        )
    text = content if isinstance(content, str) else str(content or "")

    finish_reason = _field(choice, "finish_reason") if choice is not None else None
    if finish_reason is None:
        finish_reason = _field(result, "finish_reason")
    if finish_reason is None:
        finish_reason = _field(result, "stop_reason")
    usage = _field(result, "usage")
    completion_tokens = _field(usage, "completion_tokens")
    if completion_tokens is None:
        completion_tokens = _field(usage, "output_tokens")
    if completion_tokens is None:
        completion_tokens = _field(result, "completion_tokens")
    usage_record = {name: _field(usage, name) for name in (
        "prompt_tokens", "completion_tokens", "input_tokens", "output_tokens")
        if _field(usage, name) is not None}
    details = _field(usage, "completion_tokens_details") or _field(usage, "output_tokens_details")
    if details is not None:
        usage_record["reasoning_tokens"] = _field(details, "reasoning_tokens")
    return ChatText(text, finish_reason=finish_reason,
                    completion_tokens=completion_tokens, usage=usage_record or None)


def completion_metadata(response):
    return {"finish_reason": getattr(response, "finish_reason", None),
            "completion_tokens": getattr(response, "completion_tokens", None),
            "usage": getattr(response, "usage", None)}


def _sampling_kwargs(client, top_p, top_k):
    """只向注入客户端传递其调用接口声明支持的采样参数。"""
    try:
        parameters = inspect.signature(client.chat).parameters
    except (TypeError, ValueError):
        # 无法确认接口能力时不冒险传入可选参数，避免影响评测入口。
        return {}

    accepts_extra = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    supported = {}
    for name, value in (("top_p", top_p), ("top_k", top_k)):
        explicitly_supported = (
            name in parameters
            and parameters[name].kind is not inspect.Parameter.POSITIONAL_ONLY
        )
        if value is not None and (explicitly_supported or accepts_extra):
            supported[name] = value
    return supported


def _unsupported_sampling_parameter(error, kwargs):
    """识别客户端或接口明确拒绝的可选采样参数。"""
    message = str(error)
    response = getattr(error, "response", None)
    if response is not None:
        message += " " + str(getattr(response, "text", ""))
    message = message.lower()
    rejection_markers = (
        "unexpected keyword",
        "unsupported parameter",
        "unknown parameter",
        "unrecognized parameter",
        "not supported",
        "not allowed",
        "不支持",
        "未知参数",
        "不允许",
        "extra inputs are not permitted",
        "extra fields not permitted",
    )
    if not any(marker in message for marker in rejection_markers):
        return None
    return next(
        (name for name in ("top_k", "top_p") if name in kwargs and name in message),
        None,
    )


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
        # 非空 margin 表示收束阶段调用：允许使用预留时间，但必须留下指定余量。
        # 收束调用需同时满足自身估时与后续余量。
        # `arbiter=skipped(0.0s)` —— 软预算耗尽后仲裁被整体跳过，恰是最需要复核的
        # 当前仅压缩推理和应急答题使用该预留窗口。
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
            # 收束调用必须同时为本次请求和后续余量留够时间。
            window = self.time_budget.remaining_hard() - float(self.reserve_margin_s)
            return window >= (estimate if estimate is not None else 1.0)
        if estimate is None:
            return not self.time_budget.expired()
        return self.time_budget.can_afford(estimate)

    def _call_client(self, call_kwargs: dict, label: str):
        """Stop waiting at the active phase boundary, even if the client stalls."""
        if not self.time_budget:
            return self.client.chat(**call_kwargs)

        if self.reserve_margin_s is None:
            available = self.time_budget.remaining()
            phase = "normal-solving"
        else:
            available = self.time_budget.remaining_hard() - float(self.reserve_margin_s)
            phase = "closeout"
        ceiling = CONFIG.get("llm_call_timeouts", {}).get(label)
        if ceiling is not None:
            available = min(available, float(ceiling))
        if available < 1.0:
            raise DeadlineExceeded(f"{label}: {phase} time window is exhausted")
        try:
            if "request_timeout" in inspect.signature(self.client.chat).parameters:
                call_kwargs = {**call_kwargs, "request_timeout": available}
        except (TypeError, ValueError):
            pass

        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(self.client.chat, **call_kwargs)
        try:
            return future.result(timeout=available)
        except FutureTimeoutError as exc:
            # Future.result also re-raises a TimeoutError raised by the client.
            if future.done():
                raise
            future.cancel()
            raise DeadlineExceeded(
                f"{label}: call exceeded the {phase} time window"
            ) from exc
        finally:
            # The injected client may not expose cancellation; never block the graph here.
            executor.shutdown(wait=False)

    def chat(
        self,
        messages: List[Dict],
        temperature: float = 0.2,
        max_tokens: int = 4096,
        label: str = "llm",
        top_p: float | None = None,
        top_k: int | None = None,
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
                call_kwargs = {
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                call_kwargs.update(_sampling_kwargs(self.client, top_p, top_k))
                while True:
                    try:
                        result = self._call_client(call_kwargs, label)
                        break
                    except Exception as exc:  # noqa: BLE001 - only downgrade rejected optional params.
                        rejected = _unsupported_sampling_parameter(exc, call_kwargs)
                        if rejected is None:
                            raise
                        call_kwargs.pop(rejected)
                        self.logger.warning(
                            "%s: injected client rejected optional %s; retrying without it",
                            label, rejected,
                        )
                if self.time_budget:
                    self.time_budget.record(label, self._now() - started)
                return _as_chat_text(result)
            except Exception as exc:  # noqa: BLE001 - retry transient transport failures.
                last_error = exc
                observed = self._now() - started
                if self.time_budget:
                    self.time_budget.record(f"{label}:failed", observed)
                if attempt >= self.max_retries:
                    break
                if isinstance(exc, DeadlineExceeded):
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
    top_p=None,
    top_k=None,
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
        top_p=top_p,
        top_k=top_k,
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
    top_p=None,
    top_k=None,
):
    """One prefilled round trip, returning the stitched assistant text.

    Defaults to a single attempt: a prefilled call is the cheap fast path, and
    every caller has a full-budget non-prefill fallback behind it, so spending
    retries here would buy latency without buying reliability.

    ``reserve_margin_s`` 非 None 时，本调用可使用收束窗口，但会为后续阶段保留
    相应余量；普通阶段调用则受前 15 分钟的边界限制。
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
        top_p=top_p,
        top_k=top_k,
        label=label,
    )
    return ChatText(stitch(prefix, raw), finish_reason=getattr(raw, "finish_reason", None),
                    completion_tokens=getattr(raw, "completion_tokens", None),
                    usage=getattr(raw, "usage", None))

