"""Python 验证节点：生成、执行并按证据自检验证代码。"""
import ast
import re
from utils.deps import get_deps
from utils.llm_retry import chat_prefilled, chat_with_retry
from utils.prompt_templates import PYTHON_PROMPT
from utils.token_budget import estimate_tokens
from utils.cot_stripper import strip_cot_prefix
from utils.retry_affordability import can_afford_retry, last_attempt_cost
from utils.reference_block import build_knowledge_card_block, build_reference_block
from utils.skill_excerpt import (
    sanitize_answer_anchors,
    select_script_excerpt,
    select_skill_excerpt,
)
from utils.verification_evidence import parse_verification_evidence
from utils.problem_profile import is_objective_mode, structure_instruction
from utils.objective_verifier import objective_code_is_independent
from config import CONFIG


def _reference_examples_block(examples, problem: str = "") -> str:
    """Python 提示词里的题库参考区块（题面 600 / 解答 800 字符）。

    额度小于推理侧：这里的示例只用于提示"该用什么工具算"，验证脚本本身才是主体。
    仍需传入本题题面——验证脚本按近似题的参数写出来，会得出一个自洽但答非本题的
    "验证结论"，比推理侧照抄更难识破。
    """
    return build_reference_block(examples, problem_chars=600, solution_chars=800,
                                 problem=problem)


def _knowledge_cards_block(cards) -> str:
    """Expose only concise local method/verification constraints to Python."""
    return build_knowledge_card_block(cards, content_chars=460)


def _extract_code(response: str) -> str:
    """Extract Python code from an LLM response.

    1. strict fence (```python on its own line, newline required) — avoids
       matching '```python code block' written mid-sentence in CoT prose.
    2. loose fence, only if the content looks like code.
    3. content-based: first 'import'/'from' line to end (stop at closing ```).
    4. return '' if nothing code-like — do NOT execute CoT as code.
    """
    response = strip_cot_prefix(response)
    m = re.search(r"```python[ \t]*\n(.*?)\n```", response, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```python\s+(.*?)\s+```", response, re.DOTALL)
    if m:
        cand = m.group(1).strip()
        if re.match(r"^\s*(import |from |#|def |class |if |for |while |try |with |@[a-zA-Z]|[a-zA-Z_]\w*\s*=)", cand):
            return cand
    m = re.search(r"```python[ \t]*\n(.*)\Z", response, re.DOTALL)
    if m:
        cand = m.group(1).strip()
        if re.match(r"^\s*(import |from |#|def |class |if |for |while |try |with |"
                    r"@[a-zA-Z]|[a-zA-Z_]\w*\s*=|print\s*\()", cand):
            return cand
    m = re.search(r"```\s+(.*?)\s+```", response, re.DOTALL)
    if m:
        cand = m.group(1).strip()
        if re.match(r"^\s*(import |from |#|def )", cand):
            return cand
    lines = response.splitlines()
    start = None
    for i, ln in enumerate(lines):
        if re.match(r"^\s*(import |from )", ln):
            start = i
            break
    if start is not None:
        block = []
        for ln in lines[start:]:
            if ln.strip() == "```":
                break
            block.append(ln)
        return "\n".join(block).strip()
    return ""


_ANSWER_PRINT_RE = re.compile(r"print\s*\(\s*[\"'][^\"']*最终答案")

_EXHAUSTION_MIN_CHARS = 2000


_EVIDENCE_RETRY_CONTRACT = (
    "\n无论本轮是否成功，代码都必须在最终答案前打印两行："
    'print("验证状态:", status)，其中 status 必须是 "PASS"、"FAIL" 或 '
    '"INCONCLUSIVE"；以及 print("验证证据:", evidence)。'
    'FAIL 必须包含反例或失败断言，无法判断时使用 INCONCLUSIVE。'
    '最后一行必须是 print("最终答案:", answer)，不得省略这三类输出。'
)


def _retry_prompt(instruction: str, problem: str, candidate_answer: str = "") -> str:
    """Keep the evidence contract and candidate context in every short retry."""
    candidate_hint = ""
    if candidate_answer:
        candidate_hint = (
            "\n待核验候选（不可信，仅用于独立核对）：\n"
            + candidate_answer[:2000]
            + "\n请不要盲从候选，必须根据独立计算填写验证状态和验证证据。"
        )
    return instruction + _EVIDENCE_RETRY_CONTRACT + candidate_hint + "\n问题：" + problem


def _prints_answer(code: str) -> bool:
    """Does the code contain the mandated answer print at all?

    A cheap truncation detector: `max_tokens` cutting a generation mid-file leaves
    valid-looking but unfinished code. Matching the print call (rather than merely the
    marker text) avoids counting a comment that mentions 最终答案.
    """
    return bool(_ANSWER_PRINT_RE.search(code or ""))


_TRUNCATED_SYNTAX_MARKERS = (
    "was never closed", "unterminated string literal", "unterminated triple-quoted",
    "unexpected eof", "eof while scanning", "expected an indented block",
)


def _incomplete_code_reason(response: str, code: str) -> str:
    """Identify strong end-of-generation truncation without rejecting useful code.

    Missing ``print(\"最终答案\")`` alone is not truncation: Submission_5 can execute
    such code and deterministically append a function call.  We only divert to the
    compressed generator when a long/unclosed response also fails parsing at its
    tail with an EOF-like signature.
    """
    source = code or ""
    if not source.strip():
        return ""
    text = strip_cot_prefix(response or "")
    python_fences = len(re.findall(r"```python\b", text, re.IGNORECASE))
    all_fences = len(re.findall(r"```", text))
    unclosed_fence = python_fences > 0 and all_fences % 2 == 1
    if len(text) < _EXHAUSTION_MIN_CHARS and not unclosed_fence:
        return ""
    try:
        ast.parse(source)
        return ""  # syntactically complete code remains executable, fence or not
    except SyntaxError as exc:
        message = str(exc.msg or "").lower()
        lines = source.splitlines()
        at_tail = (exc.lineno or 0) >= max(1, len(lines) - 1)
        if at_tail and any(marker in message for marker in _TRUNCATED_SYNTAX_MARKERS):
            return "token_budget_exhausted_in_python_code"
    return ""


_DEF_RE = re.compile(r"(?m)^def\s+([A-Za-z_]\w*)\s*\(([^)]*)\)\s*(?:->[^:\n]+)?:")
_MAIN_NAME_RE = re.compile(r"(?i)main|solve|count|compute|calc|answer|result|final")


def _append_answer_print(code: str) -> str:
    """定义了函数却没有产出结论的代码，确定性补一个主调用 + 打印。

    2026-08-10 评委报告 idx 0：DFS 建模方向完全正确，但只定义
    count_hamiltonian_paths() 未调用，stdout 空转 0.058s；补打印的 LLM 短重试又被
    预算门禁拒绝。改写是确定性的（不花 LLM 时间）：挑一个可无参调用的函数追加
    print("最终答案:", fn())，执行仍受 60s 执行器时限保护。挑不出目标时返回 ""。
    """
    source = code or ""
    if not source.strip():
        return ""
    callable_names = []
    for name, params in _DEF_RE.findall(source):
        params_clean = params.strip()
        if params_clean and not all("=" in part for part in params_clean.split(",")):
            continue  # 需要实参的函数不能盲调
        callable_names.append(name)
    if not callable_names:
        return ""
    preferred = [n for n in callable_names if _MAIN_NAME_RE.search(n)]
    target = (preferred or callable_names)[-1]
    return source + f'\n\nprint("最终答案:", {target}())\n'


#: 压缩重生成的估时（8192 token @ ~50 tok/s + 余量）。
_COMPRESSED_CALL_ESTIMATE_S = 200

#: 与推理分支同一条 reserve_margin 定价规则（2026-08-10 评委建议 2）：软预算
#: 已尽也放行压缩重生成，只要求硬上限前剩余 ≥ 估时 + 收尾余量。
_COMPRESSED_RESERVE_MARGIN_S = CONFIG.get("compressed_reserve_margin_s", 150)

#: assistant 种子：以代码围栏开头接管助手轮，抑制 reasoning_content（机制同
#: 分类器/仲裁器 prefill），8192 token 全部用于代码本身。
_COMPRESSED_PREFILL = "```python\n"


def _compressed_python_call(deps, problem, candidate_answer,
                            failure="因长度超限在代码之前被截断"):
    """CoT 耗尽/传输超时后的熔断重生成（评委建议 4；2026-08-10 建议 2/3）。

    普通重试会以同样方式再截断一次；prefill 压缩调用直接从代码围栏开始续写。
    失败返回 (None, 原因)。
    """
    clock = deps.time_budget
    if clock and clock.remaining_hard() - _COMPRESSED_RESERVE_MARGIN_S \
            < _COMPRESSED_CALL_ESTIMATE_S:
        return None, "compressed_retry_unaffordable"
    instruction = _retry_prompt(
        f"上一次生成{failure}。现在禁止任何解释文字：直接给出一个"
        "**尽可能短**的完整 Python 程序。优先做小规模精确计算（枚举/DP/sympy），"
        "省略探索性打印。高风险计数/DP/搜索/极值题必须定义独立的 brute/naive/"
        "enumerate/exact_small 基准函数，并对至少两个不同的小规模输入使用 assert 与主算法比较；"
        "不能只把候选公式代回自身。",
        problem, candidate_answer)
    try:
        resp = chat_prefilled(
            deps.client,
            messages=[{"role": "user", "content": instruction}],
            prefix=_COMPRESSED_PREFILL,
            temperature=CONFIG["temperatures"]["python"],
            max_tokens=CONFIG["max_tokens"]["python_compressed"],
            logger=deps.logger,
            time_budget=clock,
            expected_call_seconds=_COMPRESSED_CALL_ESTIMATE_S,
            label="python_compressed",
            reserve_margin_s=_COMPRESSED_RESERVE_MARGIN_S,
        )
    except Exception as exc:  # noqa: BLE001
        deps.logger.warning("Compressed python retry failed: %s", exc)
        return None, f"error: {str(exc)[:160]}"
    if deps.token_budget:
        deps.token_budget.consume(estimate_tokens(instruction), estimate_tokens(resp))
    return resp, ""


def python_agent_node(state, config):
    deps = get_deps(config)
    sl = deps.skills_loader
    client = deps.client
    mcp_client = deps.mcp_client
    budget = deps.token_budget
    max_attempts = 1 if budget and budget.is_tight() else CONFIG["max_retries_per_node"]
    problem, category = state["problem"], state["category"]
    question_mode = state.get("question_mode", "computation")
    if is_objective_mode(question_mode) and not state.get("objective_python_enabled"):
        return {
            "python_code": "",
            "python_output": {
                "success": False,
                "skipped": True,
                "stdout": "",
                "stderr": "objective question: python verification skipped",
                "answer": "",
                "execution_time": 0.0,
                "evidence_status": "inconclusive",
                "evidence_summary": "客观题采用快速推理路径，未生成程序。",
                "contradictions": [],
            },
            "python_trace": [{"attempt": 0, "status": "skipped", "reason": "objective_mode"}],
            "python_attempts": 0,
            "python_evidence_status": "inconclusive",
            "python_evidence_summary": "客观题采用快速推理路径，未生成程序。",
            "python_contradictions": [],
            "objective_python_independent": False,
            "python_reference_chars": 0,
            "python_knowledge_card_chars": 0,
        }
    reasoning_result = state.get("reasoning_result") or {}
    candidate_answer = str(reasoning_result.get("answer") or "").strip()
    prompt_candidate_answer = "" if is_objective_mode(question_mode) else candidate_answer
    try:
        validation_script = sl.get_validation_script(category)
    except Exception:
        validation_script = ""
    alternate_scripts = []
    for alternate in (state.get("alternate_categories") or [])[:2]:
        if alternate == category or alternate not in sl.categories:
            continue
        try:
            alt_script = sanitize_answer_anchors(
                select_script_excerpt(sl.get_validation_script(alternate), problem, 700))
        except Exception:
            alt_script = ""
        if alt_script:
            alternate_scripts.append(f"\n[备选验证脚本：{alternate}]\n{alt_script}")

    # 题库参考示例：与推理节点收到的是同一批（同一次检索的全部结果）。
    examples_text = _reference_examples_block(state.get("retrieved_examples"), problem)
    cards_text = _knowledge_cards_block(state.get("knowledge_cards"))

    base_prompt = PYTHON_PROMPT.format(
        validation_script=(sanitize_answer_anchors(
                               select_script_excerpt(validation_script, problem, 2000))
                           + "".join(alternate_scripts)
                           + examples_text + cards_text),
        problem=problem, category=category)
    base_prompt += structure_instruction(problem)
    # 技能手册的速查条目（题型方法论）与验证示例互补：前者决定"算什么/怎么建模"，
    # 后者给"怎么写验证代码"。同一素材源推理分支已在读，这里用小额度共享。
    try:
        skill_doc = sl.get_skill_document(category)
    except Exception:
        skill_doc = ""
    knowledge_ref = select_skill_excerpt(skill_doc, problem, 1600)
    if knowledge_ref:
        base_prompt += "\n\n[技能知识点参考]\n" + knowledge_ref
    if prompt_candidate_answer:
        base_prompt += (
            "\n\n待核验候选（不可信，仅用于独立核对）：\n"
            + prompt_candidate_answer[:2000]
            + "\n请不要盲从该候选；根据代码计算结果填写验证状态和验证证据。"
        )
    if is_objective_mode(question_mode):
        base_prompt += (
            "\n\n[客观题独立核验模式]\n"
            "不要读取或猜测推理分支的答案，必须从题面独立计算。选择题请逐项"
            "检查 A-E 选项并打印全部正确选项；判断题打印‘正确’或‘错误’；"
            "填空题按空位顺序打印全部结果。代码必须把比较、代入、枚举或符号"
            "化简的结果映射到最终答案，不能只 print 一个记忆中的字母。"
        )
    hint = state.get("branch_hint")
    prompt = f"{base_prompt}\n\n[复核提示] {hint}" if hint else base_prompt

    trace = []
    attempts = 0
    last_output = None
    last_code = ""
    clock = deps.time_budget

    def finalize(output):
        enriched = parse_verification_evidence(
            output,
            candidate_answer=prompt_candidate_answer,
            code=last_code,
            problem=problem,
        )
        if is_objective_mode(question_mode):
            enriched["objective_independent"] = bool(
                enriched.get("success")
                and str(enriched.get("answer") or "").strip()
                and objective_code_is_independent(last_code, question_mode)
            )
        return {
            "python_code": last_code,
            "python_output": enriched,
            "python_trace": trace,
            "python_attempts": attempts,
            "python_evidence_status": enriched.get("evidence_status", "inconclusive"),
            "python_evidence_summary": enriched.get("evidence_summary", ""),
            "python_contradictions": enriched.get("contradictions", []),
            "objective_python_independent": bool(enriched.get("objective_independent")),
            "python_reference_chars": len(examples_text),
            "python_knowledge_card_chars": len(cards_text),
        }
    if clock and clock.fast_path():
        max_attempts = 1
    compressed_next = False
    compressed_used = False
    compressed_failure = "因长度超限在代码之前被截断"
    while True:
        if compressed_next:
            # 熔断重生成不占普通重试名额，也不按上一轮 450s 的实测成本计价——
            # 它的成本上界就是压缩额度本身（~200s），reserve_margin 检查在调用内。
            compressed_next = False
            compressed_used = True
            attempts += 1
            resp, comp_fail = _compressed_python_call(deps, problem, prompt_candidate_answer,
                                                      failure=compressed_failure)
            if resp is None:
                trace.append({"attempt": attempts, "status": "skipped",
                              "reason": comp_fail or "compressed_retry_failed"})
                break
        else:
            if attempts >= max_attempts:
                break
            # 2026-07-29.
            if attempts and not can_afford_retry(clock, "python"):
                deps.logger.warning(
                    "Skipping Python retry: %.0fs left, last attempt cost %.0fs",
                    clock.remaining(), last_attempt_cost(clock, "python"))
                trace.append({"attempt": attempts + 1, "status": "skipped",
                              "reason": "retry_unaffordable"})
                break
            attempts += 1
            try:
                resp = chat_with_retry(
                    client,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=CONFIG["temperatures"]["python"],
                    max_tokens=CONFIG["max_tokens"]["python"],
                    logger=deps.logger,
                    time_budget=deps.time_budget,
                    label="python",
                )
            except Exception as exc:  # noqa: BLE001
                deps.logger.warning("Python attempt %s failed: %s", attempts, exc)
                last_output = last_output or {
                    "success": False, "stdout": "", "stderr": str(exc)[:300],
                    "answer": None, "execution_time": 0.0,
                }
                trace.append({"attempt": attempts, "status": "failed", "error": str(exc)[:200]})
                # 传输失败同样先试压缩挽救（2026-08-10 评委建议 3，与推理分支
                # 同规则）：压缩调用输出短，在并发拥堵下更可能按时返回。
                if not compressed_used:
                    compressed_failure = "因网络传输故障未能返回"
                    compressed_next = True
                    continue
                break
            if budget:
                budget.consume(estimate_tokens(prompt), estimate_tokens(resp))
        code = _extract_code(resp)
        if code:
            last_code = code
        incomplete_reason = _incomplete_code_reason(resp, code)
        if incomplete_reason:
            last_output = {
                "success": False, "stdout": "",
                "stderr": "code generation ended with incomplete Python syntax",
                "answer": None, "execution_time": 0.0,
            }
            trace.append({"attempt": attempts, "status": "failed",
                          "error": incomplete_reason,
                          "response_chars": len(resp or "")})
            if compressed_used:
                break
            compressed_failure = "因长度超限在 Python 程序中途被截断"
            compressed_next = True
            continue
        if not code:
            # 评委报告模式 A 的 Python 侧（大量 "Python 无代码/无产出"）。改发
            # prefill 压缩重生成；压缩后仍无代码则放弃本分支。
            exhausted = len(resp or "") >= _EXHAUSTION_MIN_CHARS
            last_output = {"success": False, "stdout": "",
                           "stderr": "code generation truncated before any code block"
                                     if exhausted else "no code block extracted",
                           "answer": None, "execution_time": 0.0}
            trace.append({"attempt": attempts, "status": "failed",
                          "error": "truncated before code" if exhausted else "no code block",
                          "response_chars": len(resp or "")})
            if compressed_used:
                break
            if exhausted:
                compressed_next = True
                continue
            prompt = _retry_prompt(
                "你上一次没有输出 ```python``` 代码块。请只输出一个 ```python``` 代码块，"
                "不要任何其他文字。用 sympy 计算，最后一行 print(\"最终答案:\", answer)。",
                problem, prompt_candidate_answer)
            continue
        if not _prints_answer(code):
            # 缺少最终答案打印的代码**仍然执行**：stdout 里往往已有结论
            # （评委报告 idx 38：正确公式在代码里，旧逻辑拒绝执行去换一次
            # 450s 的重生成，最终一无所获）。执行后由 stdout 二次抽取兜底。
            trace.append({"attempt": attempts, "status": "missing_answer_print",
                          "note": "executing anyway; stdout miner may recover"})
        if mcp_client is None:
            output = {"success": False, "stdout": "", "stderr": "MCP client unavailable",
                      "answer": None, "execution_time": 0.0}
        else:
            output = mcp_client.execute(code, timeout=CONFIG["node_timeouts"]["python_mcp_execute"])
        last_output = output
        trace.append({
            "attempt": attempts,
            "status": "success" if output.get("success") else "failed",
            "error": (output.get("stderr") or "")[:200],
        })
        if output.get("success"):
            probe = parse_verification_evidence(
                output,
                candidate_answer=prompt_candidate_answer,
                code=code,
                problem=problem,
            )
            if str(probe.get("answer") or "").strip():
                return finalize(output)
            # 执行成功但连 stdout 挖掘都拿不到结论 → 先做确定性补调用修复
            # （2026-08-10 评委建议 5：idx 0 的代码方向正确、只差一个主调用，
            # 而"补打印"LLM 短重试恰恰会被预算门禁拒绝）。修不动再花 LLM 重试。
            trace.append({"attempt": attempts, "status": "no_extractable_answer"})
            repaired = _append_answer_print(code)
            if repaired and mcp_client is not None:
                repaired_output = mcp_client.execute(
                    repaired, timeout=CONFIG["node_timeouts"]["python_mcp_execute"])
                repaired_probe = parse_verification_evidence(
                    repaired_output,
                    candidate_answer=prompt_candidate_answer,
                    code=repaired,
                    problem=problem,
                )
                answer_text = str(repaired_probe.get("answer") or "").strip()
                if repaired_output.get("success") and answer_text \
                        and answer_text.lower() not in {"none", "null"}:
                    last_code = repaired
                    trace.append({"attempt": attempts,
                                  "status": "deterministic_print_repair"})
                    return finalize(repaired_output)
                trace.append({"attempt": attempts, "status": "print_repair_failed",
                              "error": (repaired_output.get("stderr") or "")[:200]})
            if compressed_used:
                return finalize(output)
            prompt = _retry_prompt(
                "你上一次的代码执行成功但没有输出任何最终结论。请输出修正后的 "
                "```python``` 代码块：保留原计算，务必以 print(\"最终答案:\", answer) 结尾。",
                problem, prompt_candidate_answer)
            continue
        prompt = _retry_prompt(
            f"代码执行失败：\n错误：{(output.get('stderr') or '')[:500]}\n"
            "请只输出修正后的 ```python``` 代码块，用 sympy，结尾 print(\"最终答案:\", answer)。",
            problem, prompt_candidate_answer)
    return finalize(last_output)

