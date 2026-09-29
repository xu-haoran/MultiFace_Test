"""Normalize Python executor output into evidence separate from process status.

2026-08-09 评委报告接线（建议 1/2）：
* 高置信答案通道（"最终答案:" 标记）为空时，从 stdout 保守挖掘候选答案并
  打上 ``answer_source="stdout_mined"``——正确公式已打印却因缺标记被丢弃的
  idx 20/76 类事故由此堵住；
* 携带 ``code`` 时做反伪造静态检查：无实质计算却宣称 PASS 的证据一律降级
  inconclusive，不得再被仲裁当作支持性证据锁定（idx 19/28/45/66）。
"""

from __future__ import annotations

import re

from utils.stdout_miner import mine_stdout_answer
from utils.verification_authenticity import assess_verification_authenticity


_STATUS_RE = re.compile(
    r"验证状态\s*[:：]\s*(PASS|FAIL|INCONCLUSIVE)", re.IGNORECASE
)
_EVIDENCE_RE = re.compile(r"验证证据\s*[:：]\s*(.+)", re.IGNORECASE)
_MAX_RATIO_RE = re.compile(
    r"(?i)(?:maximum|max(?:imum)?|最大(?:值|比值)?|ratio|f\s*\(\s*n\s*\)\s*/\s*n)"
    r"[^:\n]{0,80}[:：=]\s*([-+]?\d+(?:\.\d+)?)"
)
_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
_CONTRADICTION_RE = re.compile(
    r"(?:反例|矛盾|不一致|失败断言|counterexample|contradict|FAIL)",
    re.IGNORECASE,
)


def _scalar(value: str):
    """Return a simple scalar from an answer, or None for compound answers."""
    text = str(value or "")
    numbers = _NUMBER_RE.findall(text)
    if len(numbers) != 1:
        return None
    try:
        return float(numbers[0])
    except ValueError:
        return None


def _bounded(text: str, limit: int = 1000) -> str:
    text = str(text or "").strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "...[truncated]"


def python_answer_is_trusted(python_output: dict | None) -> bool:
    """Python 分支是否真的验证过——未验证的答案一律不可信。

    这是 ``python_verification_trusted`` 的**唯一**读取入口。该字段由
    ``parse_verification_evidence`` 写入，此前每个消费点各写一份等价判断，结果
    2026-08-19 的闸门漏了第三个消费点（协调器 ``_evidence_override``），idx 4 与
    idx 77 的正确推理答案被未验证的 Python 答案覆盖出厂。读取集中到一处，新增
    消费点只需 import，不会再各自漏判。

    缺字段时退回 ``authenticity.verified``；两者皆缺（未提供 code、夹具与空分支）
    时不做有罪推定，沿用旧语义返回 True。
    """
    po = python_output or {}
    if "python_verification_trusted" in po:
        return bool(po.get("python_verification_trusted"))
    authenticity = po.get("authenticity")
    if isinstance(authenticity, dict) and "verified" in authenticity:
        return bool(authenticity["verified"])
    return True


def parse_verification_evidence(
    output: dict | None,
    candidate_answer: str = "",
    code: str = "",
    problem: str = "",
) -> dict:
    """Add conservative evidence fields without changing executor success semantics."""
    result = dict(output or {})
    stdout = str(result.get("stdout") or "")

    # 低置信答案通道：执行器只认 "最终答案:" 标记；标记缺失但 stdout 明确给出
    # 结论时（"f(2015)=3024"、"both match binomial(2k,k)^2"），把它挖出来作候选。
    # 只在高置信通道为空时启用，且不改变 success 语义。
    if not str(result.get("answer") or "").strip() and stdout:
        mined = mine_stdout_answer(stdout)
        if mined:
            result["answer"] = mined
            result["answer_source"] = "stdout_mined"

    marker = _STATUS_RE.search(stdout)
    marker_status = marker.group(1).upper() if marker else ""
    evidence_match = _EVIDENCE_RE.search(stdout)
    evidence_summary = evidence_match.group(1).strip() if evidence_match else ""

    contradictions = []
    for line in stdout.splitlines():
        if _STATUS_RE.search(line):
            continue
        if _CONTRADICTION_RE.search(line):
            text = line.strip()
            if text and text not in contradictions:
                contradictions.append(text)

    if marker_status == "FAIL":
        if evidence_summary and evidence_summary not in contradictions:
            contradictions.insert(0, evidence_summary)
        elif not contradictions:
            contradictions.append("验证状态: FAIL")

    candidate_value = _scalar(candidate_answer or result.get("answer", ""))
    if candidate_value is not None:
        for match in _MAX_RATIO_RE.finditer(stdout):
            observed = float(match.group(1))
            if observed > candidate_value + 1e-12:
                text = match.group(0).strip()
                if text not in contradictions:
                    contradictions.append(text)

    if contradictions:
        status = "contradict"
        if not evidence_summary:
            evidence_summary = contradictions[0]
    elif marker_status == "PASS":
        status = "support" if result.get("success") is not False else "inconclusive"
    else:
        status = "inconclusive"
        if not evidence_summary and marker_status == "INCONCLUSIVE":
            evidence_summary = "程序明确声明无法判定。"

    # 反伪造：不做计算却宣称 PASS 的代码，其 support 不构成数学证据。
    # 只降级 support（contradict 的反例本身就是计算产物，不受权威引用污染）。
    authenticity = assess_verification_authenticity(code, stdout, problem) if code else \
        {
            "fabricated": False,
            "reasons": [],
            "strength_reasons": [],
            "verified": False,
            "requires_baseline": False,
            "has_baseline_check": False,
            "baseline_names": [],
        }
    if authenticity["fabricated"] and status == "support":
        status = "inconclusive"
        warning = "；".join(authenticity["reasons"])
        evidence_summary = f"[反伪造] {warning}。原声明不作为验证证据。"

    # 未验证即不可信（2026-08-19）：代码里没有任何实质计算、或答案是抄进 print
    # 的常量时，这条分支没有"验证"过任何东西，其答案不得充当验证结论。
    # trusted=False 的 Python 答案在下游一律不参与答案选择。
    # 没有 code 可查时不做有罪推定：判定权交回旧语义（真实运行中 python_output
    # 带答案必然伴随 code，无 code 只出现在夹具与空分支）。
    trusted = bool(authenticity.get("verified")) if code else True
    if status == "support" and not trusted:
        status = "inconclusive"
        strength_reasons = authenticity.get("strength_reasons") or []
        if strength_reasons:
            evidence_summary = (
                "[验证强度不足] " + "；".join(strength_reasons)
                + (f"。原声明不作为验证证据。" if evidence_summary else "")
            )
    result["python_verification_trusted"] = trusted

    result.update({
        "evidence_status": status,
        "evidence_summary": _bounded(evidence_summary or stdout),
        "contradictions": contradictions,
        "evidence_marker": bool(marker),
        "authenticity": authenticity,
    })
    return result

