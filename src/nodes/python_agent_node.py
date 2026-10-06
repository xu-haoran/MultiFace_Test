"""Python 验证节点：生成、执行并按证据自检验证代码。"""
import ast
import re
from utils.deps import get_deps
from utils.llm_retry import chat_prefilled, chat_with_retry
from utils.prompt_templates import PYTHON_PROMPT
from utils.token_budget import estimate_tokens
from utils.cot_stripper import strip_cot_prefix
from utils.retry_affordability import can_afford_retry, last_attempt_cost
from utils.reference_block import build_reference_block
from utils.skill_excerpt import (
    sanitize_answer_anchors,
    select_script_excerpt,
    select_skill_excerpt,
    selected_card_ids,
)
from utils.verification_evidence import parse_verification_evidence
from utils.problem_profile import is_objective_mode, structure_instruction
from utils.objective_verifier import objective_code_is_independent
from utils.python_contract import PYTHON_CONTRACT, prepare_verification_program
from utils.shared_constraints import problem_constraints
from config import CONFIG


def _reference_examples_block(examples, problem: str = "") -> str:
    """Python 提示词里的题库参考区块，完整保留入选记录正文。

    仍需传入本题题面——验证脚本按近似题的参数写出来，会得出一个自洽但答非本题的
    "验证结论"，比推理侧照抄更难识破。
    """
    return build_reference_block(examples, problem=problem)


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


_EVIDENCE_RETRY_CONTRACT = PYTHON_CONTRACT


def _retry_prompt(instruction: str, problem: str, candidate_answer: str = "") -> str:
    """Keep the evidence contract and candidate context in every short retry."""
    candidate_hint = ""
    if candidate_answer:
        candidate_hint = (
            "\n待核验候选（不可信，仅用于独立核对）：\n"
            + candidate_answer[:2000]
            + "\n请不要盲从候选，必须根据独立计算填写检查项及其对象、关系、范围和证据。"
        )
    return instruction + _EVIDENCE_RETRY_CONTRACT + candidate_hint + "\n问题：" + problem


_TRUNCATED_SYNTAX_MARKERS = (
    "was never closed", "unterminated string literal", "unterminated triple-quoted",
    "unexpected eof", "eof while scanning", "expected an indented block",
)


def _incomplete_code_reason(response: str, code: str) -> str:
    """Treat EOF syntax as truncation only when provider metadata confirms a cap.

    Otherwise the same syntax error follows the local interface/API repair path.
    """
    source = code or ""
    if not source.strip():
        return ""
    finish_reason = str(getattr(response, "finish_reason", "") or "").lower()
    completion_tokens = getattr(response, "completion_tokens", None)
    limit = CONFIG["max_tokens"]["python"]
    if finish_reason not in {"length", "max_tokens", "token_limit", "max_output_tokens"} \
            and not (isinstance(completion_tokens, int) and completion_tokens >= limit):
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


#: 压缩重生成的估时（8192 token @ ~50 tok/s + 余量）。
_COMPRESSED_CALL_ESTIMATE_S = 200

# Python 压缩重生成属于普通验证，只能使用前 15 分钟的时间。

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
    if clock and clock.closeout_started():
        return None, "closeout_reserved_for_reasoning_and_emergency"
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
            top_p=CONFIG["top_p"]["python"],
            top_k=CONFIG["top_k"]["python"],
            logger=deps.logger,
            time_budget=clock,
            expected_call_seconds=_COMPRESSED_CALL_ESTIMATE_S,
            label="python_compressed",
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
    cards_text = ""  # Knowledge comes exclusively from local resources/skills.

    base_prompt = PYTHON_PROMPT.format(
        validation_script=(sanitize_answer_anchors(
                               select_script_excerpt(validation_script, problem, 2000))
                           + "".join(alternate_scripts)
                           + examples_text + cards_text),
        problem=problem, category=category)
    base_prompt += problem_constraints(problem)
    base_prompt += PYTHON_CONTRACT
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
            + "\n请不要盲从该候选；根据代码计算结果填写结构化检查项和证据。"
        )
    if is_objective_mode(question_mode):
        base_prompt += (
            "\n\n[客观题独立核验模式]\n"
            "不要读取或猜测推理分支的答案，必须从题面独立计算。选择题请逐项"
            "检查 A-E 选项并返回全部正确选项；判断题返回‘正确’或‘错误’；"
            "填空题按空位顺序返回全部结果。代码必须把比较、代入、枚举或符号"
            "化简的结果映射到最终答案，不能只 print 一个记忆中的字母。"
        )
    hint = state.get("branch_hint")
    prompt = f"{base_prompt}\n\n[复核提示] {hint}" if hint else base_prompt
    repair_only = (state.get("validation_details") or {}).get("repair_target") == "python"
    if repair_only:
        max_attempts = 1
        previous_code = str(state.get("python_code") or "")
        if previous_code:
            prompt += ("\n仅修复原代码中导致失败的部分，保留正确计算；不要重新探索解法。\n原代码：\n"
                       + previous_code.split("# MULTIFACE_FIXED_DRIVER", 1)[0])

    trace = []
    attempts = 0
    last_output = None
    last_code = ""
    clock = deps.time_budget

    def finalize(output):
        enriched = output if output and "python_verification_trusted" in output else parse_verification_evidence(
            output,
            candidate_answer=prompt_candidate_answer,
            code=last_code,
            problem=problem,
        )
        if is_objective_mode(question_mode):
            enriched["objective_independent"] = bool(
                enriched.get("success")
                and enriched.get("python_verification_trusted")
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
            "python_knowledge_card_chars": 0,
            "python_skill_cards": selected_card_ids(knowledge_ref),
        }
    if clock and clock.closeout_started():
        trace.append({"attempt": 0, "status": "skipped",
                      "reason": "closeout_reserved_for_reasoning_and_emergency"})
        return finalize({
            "success": False,
            "skipped": True,
            "stdout": "",
            "stderr": "normal Python verification ended at the 15-minute boundary",
            "answer": None,
            "execution_time": 0.0,
            "evidence_status": "inconclusive",
            "evidence_summary": "最后 5 分钟预留给压缩推理与应急作答。",
            "contradictions": [],
        })
    compressed_next = False
    compressed_used = repair_only
    compressed_failure = "因长度超限在代码之前被截断"
    while True:
        if compressed_next:
            # Python 压缩重生成仍属于普通验证，只能在第 15 分钟前运行。
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
                resp = chat_prefilled(
                    client,
                    messages=[{"role": "user", "content": prompt}],
                    prefix="```python\n",
                    temperature=CONFIG["temperatures"]["python"],
                    max_tokens=CONFIG["max_tokens"]["python"],
                    top_p=CONFIG["top_p"]["python"],
                    top_k=CONFIG["top_k"]["python"],
                    logger=deps.logger,
                    time_budget=deps.time_budget,
                    expected_call_seconds=120 if attempts == 1 and not repair_only else 45,
                    label="python" if attempts == 1 and not repair_only else "python_repair",
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
            exhausted = str(getattr(resp, "finish_reason", "") or "").lower() in {
                "length", "max_tokens", "token_limit", "max_output_tokens"}
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
                "不要任何其他文字。用 sympy 计算，以固定接口返回结果。",
                problem, prompt_candidate_answer)
            continue
        try:
            code, repairs = prepare_verification_program(code)
            last_code = code
            trace.append({"attempt": attempts, "status": "contract_prepared", "local_repairs": repairs,
                          "finish_reason": getattr(resp, "finish_reason", None),
                          "usage": getattr(resp, "usage", None)})
        except (SyntaxError, ValueError) as exc:
            last_output = {"success": False, "stdout": "", "stderr": str(exc), "answer": None}
            trace.append({"attempt": attempts, "status": "contract_invalid", "error": str(exc)[:300]})
            prompt = _retry_prompt("仅修复以下代码的语法/API/返回接口，保留数学算法。错误："
                                   + str(exc)[:300] + "\n原代码：\n" + code,
                                   problem, prompt_candidate_answer)
            continue
        if mcp_client is None:
            output = {"success": False, "stdout": "", "stderr": "MCP client unavailable",
                      "answer": None, "execution_time": 0.0}
        else:
            remaining = clock.remaining() if clock else None
            if remaining is not None and remaining < 1:
                output = {"success": False, "skipped": True, "stdout": "",
                          "stderr": "normal-solving time window ended",
                          "answer": None, "execution_time": 0.0}
            else:
                execute_timeout = CONFIG["node_timeouts"]["python_mcp_execute"]
                if remaining is not None:
                    execute_timeout = min(execute_timeout, int(remaining))
                output = mcp_client.execute(code, timeout=execute_timeout)
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
                return finalize(probe)
            trace.append({"attempt": attempts, "status": "no_extractable_answer"})
            if compressed_used:
                return finalize(output)
            prompt = _retry_prompt(
                "你上一次的代码执行成功但没有输出任何最终结论。请输出修正后的 "
                "```python``` 代码块：保留原计算，以固定接口返回结果。",
                problem, prompt_candidate_answer)
            continue
        prompt = _retry_prompt(
            f"代码执行失败：\n错误：{(output.get('stderr') or '')[:500]}\n"
            "仅修复原代码的语法/API/接口错误，保留数学算法，只输出修正后的代码块。\n原代码：\n" + code.split("# MULTIFACE_FIXED_DRIVER", 1)[0],
            problem, prompt_candidate_answer)
    return finalize(last_output)

