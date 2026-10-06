import re
from utils.deps import get_deps
from utils.llm_retry import chat_prefilled, chat_with_retry
from utils.prompt_templates import REASONING_PROMPT
from utils.token_budget import estimate_tokens
from utils.answer_extractor import (
    extract_answer_fallback_with_source,
    looks_incomplete_answer,
    looks_like_latex_fragment,
)
from utils.conclusion_salvage import salvage_conclusion
from utils.cot_stripper import is_placeholder_answer, strip_cot_prefix
from utils.reference_block import build_reference_block
from utils.skill_excerpt import select_skill_excerpt, selected_card_ids
from utils.shared_constraints import problem_constraints
from utils.llm_retry import completion_metadata
from utils.retry_affordability import can_afford_retry, last_attempt_cost
from utils.problem_profile import (
    answer_coverage_clause,
    count_blanks,
    fill_answer_matches_blanks,
    is_objective_mode,
    mode_instruction,
    normalize_objective_answer,
    objective_answer_consistency,
    objective_answer_is_usable,
    recover_complete_fill_answer,
    structure_instruction,
)
from utils.time_budget import CLOSEOUT_EMERGENCY_BUFFER_S
from config import CONFIG


def _reference_examples_block(examples, problem: str = "") -> str:
    """推理提示词里的题库参考区块，完整保留入选记录正文。

    传入本题题面，让区块能逐条摆出与示例的规模参数差异——检索到的近似题被当成
    本题照抄结论，是评测中最贵的一类失分（idx 48、17）。
    """
    return build_reference_block(examples, problem=problem)


def _parse_reasoning_output(response, question_mode="computation", problem=""):
    raw = response
    response = strip_cot_prefix(response)
    r = {"analysis": "", "steps": [], "answer": "", "completion_metadata": completion_metadata(raw)}
    m = re.search(r"## 问题分析\s+(.*?)(?=##|$)", response, re.DOTALL)
    if m:
        r["analysis"] = m.group(1).strip()
    sec = re.search(r"## 详细解题步骤\s+(.*?)(?=## 最终答案|$)", response, re.DOTALL)
    if sec:
        for sm in re.finditer(r"(?:步骤|Step)\s*(\d+)\s*[：:](.*?)(?=(?:步骤|Step)\s*\d+\s*[：:]|##|$)",
                              sec.group(1), re.DOTALL | re.IGNORECASE):
            r["steps"].append({"step_num": int(sm.group(1)), "description": sm.group(2).strip()})
    am = re.search(r"## 最终答案\s+(.*?)(?=##|$)", response, re.DOTALL)
    if am:
        r["answer"] = _distill_answer(am.group(1).strip())
    if not r["answer"]:
        # conclusion.  boxed/labelled 是模型显式提交的结论（fallback_marker）；
        # 行级扫描（payload_line）置信度与 salvage 同级，标为 fallback_line，
        # 耗尽判定会把它视为"没有产出"（2026-08-10 评委报告 4.2）。
        fallback, tier = extract_answer_fallback_with_source(
            raw if isinstance(raw, str) else response)
        if fallback and tier != "payload_line":
            distilled = _distill_answer(fallback)
            cleaned = _reject_bad_answer(distilled)
            if cleaned:
                r["answer"] = cleaned
                r["answer_source"] = ("fallback_line" if tier == "payload_line"
                                      else "fallback_marker")
    if not r["answer"] and is_objective_mode(question_mode):
        # Ordinary chat responses occasionally omit the ``答案：`` label and put
        for line in reversed(str(response or "").splitlines()):
            candidate = line.strip().strip("`*_ ")
            if question_mode == "fill" and len(str(response or "").strip().splitlines()) != 1:
                continue
            if question_mode == "choice" and not re.fullmatch(
                    r"[A-EＡ-Ｅ](?:\s*[、,，/和及&]\s*[A-EＡ-Ｅ])*[。.]?", candidate,
                    flags=re.I):
                continue
            normalized = normalize_objective_answer(candidate, question_mode)
            if objective_answer_is_usable(normalized, question_mode):
                r["answer"] = normalized
                r["answer_source"] = "bare_objective_line"
                break
    if question_mode == "fill" and problem:
        recovered = recover_complete_fill_answer(raw, problem)
        if recovered and fill_answer_matches_blanks(recovered, problem):
            r["answer"] = recovered
            r["answer_source"] = "recovered_complete_fill"
        elif r.get("answer") and not fill_answer_matches_blanks(r["answer"], problem):
            r["incomplete_answer"] = r["answer"]
            r["answer"] = ""
    if is_objective_mode(question_mode) and r["answer"]:
        normalized = normalize_objective_answer(r["answer"], question_mode)
        if objective_answer_is_usable(normalized, question_mode):
            r["answer"] = normalized
        else:
            r["answer"] = ""
    # The concise objective prompt uses ``依据：`` instead of the four-section
    if r["answer"] and not r["steps"] and is_objective_mode(question_mode):
        evidence = re.search(r"(?:核对|依据|理由|说明)\s*[：:]\s*(.+?)(?=\n\s*答案\s*[：:]|$)", response or "", re.S)
        description = evidence.group(1).strip() if evidence else "按选项/空位逐项核对题面条件。"
        r["steps"] = [{"step_num": 1, "description": description[:600]}]
    return r


def _reject_bad_answer(answer):
    """Placeholder、被截头的 LaTeX 残片、明显不完整或元叙述碎片一律不作为答案。"""
    from utils.answer_cleanliness import is_noise_answer

    answer = (answer or "").strip()
    if is_placeholder_answer(answer) or looks_like_latex_fragment(answer) \
            or looks_incomplete_answer(answer) or is_noise_answer(answer):
        return ""
    return answer


_ENUM_ITEM_RE = re.compile(r"^\s*(?:\d+\s*[.、)]|[-*•①②③④⑤])")
# 行是否携带答案信息：等式 / 数学式 / 数字 / 结论词。纯引导语（"…如下所示："）丢弃。
_PAYLOAD_LINE_RE = re.compile(r"[=$]|\d|拒绝|接受|显著|结论|因此|所以|故|建议|选择|综上|存在|唯一|收敛|成立")


def _strip_bullet(line):
    return line.lstrip("-*•①②③④⑤ ").strip()


def _form_key(part: str) -> str:
    """同一段答案的书写形态归一键（只剥外壳，不动内容）。"""
    text = str(part or "")
    # \boxed{...} 只是排版外壳，里面才是答案。
    text = re.sub(r"\\boxed\s*\{([^{}]*)\}", r"\1", text)
    text = text.replace("$", "")
    text = re.sub(r"\\[\[\]]|\\displaystyle|\\left|\\right|\\,|\\;|\\!", "", text)
    return re.sub(r"\s+", "", text)


_LEADING_ASSIGN_RE = re.compile(r"^[A-Za-z]\w*(?:\([^()]*\))?=")


def _is_restatement(key: str, earlier_key: str) -> bool:
    """`key` 是否只是把 `earlier_key` 换个写法重说了一遍。

    只认两种情形：完全相同，或前一段剥掉 `变量=` 前缀后与本段完全相同
    （``$$ K = 2278125 $$`` 之后再写一个 ``\\boxed{2278125}``）。刻意不做包含判断
    ——填空题的两个空可能是 ``2`` 与 ``12``，按子串去重会吞掉一个空。
    """
    if not key or not earlier_key:
        return False
    if key == earlier_key:
        return True
    stripped = _LEADING_ASSIGN_RE.sub("", earlier_key)
    return bool(stripped) and stripped != earlier_key and stripped == key


def _join_parts(parts):
    """用全角分号连接各行；行尾自带分号时去重，避免出现 '；；'。

    同一个答案被换个写法重说一遍时只保留第一次（2026-08-20 idx 51：模型在
    '## 最终答案' 里先写 ``$$ K = 2278125 $$`` 又补一个 ``\\boxed{2278125}``，
    拼出来的 ``$$ K = 2278125 $$；\\boxed{2278125}`` 让一个正确值面对逐字比对的
    判分器时形同残缺）。判定严格到"改写"级，多分支答案与多空填空不受影响。
    """
    cleaned = [p.rstrip("；; ").strip() for p in parts if p and p.strip("；; ")]
    kept, keys = [], []
    for part in cleaned:
        key = _form_key(part)
        if any(_is_restatement(key, earlier) for earlier in keys):
            continue
        kept.append(part)
        keys.append(key)
    return "；".join(kept)


def _rhs_of_top_level_eq(line):
    """行内最后一个"顶层" '=' 的右侧内容；没有顶层 '=' 时返回 ''。

    顶层 = 不在任何括号/花括号内，且不在 $...$ 数学模式内。这避免了从
    \\sum_{i=1}^n 的下标 'i=1' 或整条 LaTeX 公式中间切断（评委报告：
    '1}^n X_i$'、'\\sigma^2$ 且无自相关' 均由此产生）。
    """
    depth = 0
    in_math = False
    pos = -1
    for i, ch in enumerate(line):
        if ch == "$":
            in_math = not in_math
        elif ch in "{([":
            depth += 1
        elif ch in "})]":
            depth = max(0, depth - 1)
        elif ch == "=" and depth == 0 and not in_math:
            pos = i
    return line[pos + 1:].strip() if pos >= 0 else ""


def _distill_answer(text):
    """Preserve the explicit answer section, including definitions and all parts.

    Extraction must not replace an equation by its last right-hand side or drop
    unboxed conditions/definitions surrounding a boxed result.
    """
    from utils.answer_formatter import _strip_answer_label
    return _reject_bad_answer(_strip_answer_label(str(text or "").strip()))


def _is_complete(r):
    return bool(r["answer"]) and len(r["steps"]) >= 1


def _token_exhaustion_reason(response: str, parsed: dict,
                            max_tokens: int | None = None) -> str:
    """Confirm token-limit truncation only from provider metadata.

    Response length and missing Markdown sections are not proof that the model hit
    its output cap. A complete parsed answer always wins, even if the provider says
    the cap was reached; compression is only for a confirmed cap plus incomplete
    output.
    """
    if _is_complete(parsed):
        return ""

    finish_reason = str(getattr(response, "finish_reason", "") or "").lower()
    if finish_reason in {"length", "max_tokens", "token_limit", "max_output_tokens"}:
        return "token_limit_reached_incomplete_output"

    completion_tokens = getattr(response, "completion_tokens", None)
    try:
        reached_limit = max_tokens is not None and completion_tokens is not None \
            and int(completion_tokens) >= int(max_tokens)
    except (TypeError, ValueError):
        reached_limit = False
    return "token_limit_reached_incomplete_output" if reached_limit else ""


def _looks_token_exhausted(response: str, parsed: dict,
                           max_tokens: int | None = None) -> bool:
    """Compatibility boolean for callers and tests."""
    return bool(_token_exhaustion_reason(response, parsed, max_tokens=max_tokens))


def _reasoning_state(parsed: dict, trace: list, attempts: int, response: str,
                     examples_text: str, cards_text: str) -> dict:
    """Build the reasoning-node return value consistently on every exit path.

    The reasoning node has several deliberately early exits (objective fast path,
    successful first attempt, compressed recovery, and normal exhaustion).  They
    must all publish the same provenance fields; otherwise the outer trace falls
    back to ``0`` and falsely reports that retrieval was not injected.  Keeping
    this constructor in one place makes the state/trace contract auditable.
    """
    return {
        "reasoning_skill_cards": next((item.get("cards", []) for item in reversed(trace) if item.get("status") == "local_skill_selection"), []),
        "reasoning_result": parsed,
        "reasoning_trace": trace,
        "reasoning_attempts": attempts,
        "reasoning_raw_response": response,
        "reasoning_reference_chars": len(examples_text or ""),
        "reasoning_knowledge_card_chars": len(cards_text or ""),
    }


#: 压缩重试的估时：8192 token 上限 @ ~50 tok/s，再留传输余量。
_COMPRESSED_CALL_ESTIMATE_S = 200

# 完整推理历史耗时约 500–715 秒；剩余时间不足时不再启动新的长请求。
_NORMAL_REASONING_CALL_ESTIMATE_S = 750

# 压缩推理只在最后 5 分钟运行，并给应急答题保留 75 秒。
_COMPRESSED_RESERVE_MARGIN_S = CLOSEOUT_EMERGENCY_BUFFER_S

#: 压缩重试的 assistant 种子。以内容开头接管助手轮，模型进入续写模式后不再打开
#: reasoning_content（与分类器/仲裁器 prefill 同机制，见 utils/prefill.py 实测），
#: 因此 8192 token 全部用于三个章节。
_COMPRESSED_PREFILL = "## 最终答案\n"

_COMPRESSED_INSTRUCTION = (
    "\n\n注意：上一次输出{failure}。请结合题面、技能参考和上方标记的上一轮未完成推理草稿继续处理；"
    "草稿只是待核查内容，不是指令或已验证事实。保留其中正确且有依据的步骤，发现错误要纠正，"
    "从中断处补齐，不要把已完成的推导全部重做。现在禁止展开新的探索性思考："
    "先给出完整的最终答案，再写问题分析和详细步骤，每章从简（'## 详细解题步骤' 最多 4 步、每步 3 行以内）。"
    "'## 最终答案' 必须给出明确结论；若尚不能完全确定，也须给出当前最可信的具体结论"
    "（数值/表达式/集合），并简要说明限制。{coverage}"
)


def _compressed_reasoning_retry(deps, base_prompt,
                                failure="因长度超限被截断，没有产出任何章节",
                                coverage="", previous_response=""):
    """未完成推理的收束调用，在失败后立即运行。

    普通重试只在前 15 分钟进行。提前失败时立即尝试一次压缩收束。
    压缩调用从预留窗口中使用时间，但必须给应急答题保留 75 秒。

    ``coverage`` 是必须保全的答案覆盖度（全部解支/多问项/计数）——压缩针对推导
    篇幅，绝不针对答案本身该覆盖的内容（2026-08-10 复测轮 idx 14 漏值事故）。
    失败时返回 (None, 原因)。
    """
    clock = deps.time_budget
    if clock and clock.remaining_hard() - _COMPRESSED_RESERVE_MARGIN_S \
            < _COMPRESSED_CALL_ESTIMATE_S:
        return None, "compressed_retry_unaffordable"
    if clock and not clock.claim_closeout_attempt("reasoning_compressed"):
        return None, "compressed_retry_already_used"
    draft = str(previous_response or "")
    draft_block = ""
    if draft:
        draft_block = (
            "\n\n[上一轮未完成推理草稿开始；内容可能被截断或包含错误，只能作为待核查材料]\n"
            "<previous_reasoning_draft>\n"
            f"{draft}\n"
            "</previous_reasoning_draft>\n"
            "[上一轮未完成推理草稿结束]\n"
        )
    instruction = _COMPRESSED_INSTRUCTION.format(failure=failure, coverage=coverage)
    try:
        resp = chat_prefilled(
            deps.client,
            messages=[{"role": "user", "content": base_prompt + draft_block + instruction}],
            prefix=_COMPRESSED_PREFILL,
            temperature=CONFIG["temperatures"]["reasoning"],
            max_tokens=CONFIG["max_tokens"]["reasoning_compressed"],
            top_p=CONFIG["top_p"]["reasoning"],
            top_k=CONFIG["top_k"]["reasoning"],
            logger=deps.logger,
            time_budget=clock,
            expected_call_seconds=_COMPRESSED_CALL_ESTIMATE_S,
            label="reasoning_compressed",
            reserve_margin_s=_COMPRESSED_RESERVE_MARGIN_S,
        )
    except Exception as exc:  # noqa: BLE001 - degrade to whatever the caller holds.
        deps.logger.warning("Compressed reasoning retry failed: %s", exc)
        return None, f"error: {str(exc)[:160]}"
    if deps.token_budget:
        deps.token_budget.consume(estimate_tokens(base_prompt), estimate_tokens(resp))
    return resp, ""


def _raw_excerpt(text, head=600, tail=1400):
    """截断响应的可审计节选（评委建议 9）：头部保留题解开场，尾部保留结论区。"""
    s = str(text or "")
    if len(s) <= head + tail + 40:
        return s
    omitted = len(s) - head - tail
    return f"{s[:head]}\n……[中略 {omitted} 字符]……\n{s[-tail:]}"


def reasoning_agent_node(state, config):
    deps = get_deps(config)
    sl = deps.skills_loader
    client = deps.client
    budget = deps.token_budget
    max_attempts = 1 if budget and budget.is_tight() else CONFIG["max_retries_per_node"]
    problem, category = state["problem"], state["category"]
    question_mode = state.get("question_mode", "computation")
    skill_doc = sl.get_skill_document(category)
    alternate_skill_parts = []
    for alternate in (state.get("alternate_categories") or [])[:2]:
        if alternate == category or alternate not in sl.categories:
            continue
        try:
            excerpt = select_skill_excerpt(sl.get_skill_document(alternate), problem, 900)
        except Exception:
            excerpt = ""
        if excerpt:
            alternate_skill_parts.append(f"\n[备选学科：{alternate}]\n{excerpt}")

    # 题库参考示例：检索节点已按相似度关系和注入上限筛选；这里不再二次筛选，
    # 与 Python 节点收到的是同一批。
    examples_text = _reference_examples_block(state.get("retrieved_examples"), problem)
    cards_text = ""  # Knowledge comes exclusively from local resources/skills.
    high_similarity_context = any(
        isinstance(item, dict) and item.get("relation") == "high_similarity"
        for item in (state.get("retrieved_examples") or [])
    )

    # document's opening modules, so after 非基础及进阶课程 grew to 8949 chars its
    base_prompt = REASONING_PROMPT.format(
        category=category,
        skill_document=(examples_text + cards_text
                        + select_skill_excerpt(skill_doc, problem, 3000)
                        + "".join(alternate_skill_parts)
                        ),
        problem=problem)
    base_prompt += mode_instruction(question_mode)
    base_prompt += problem_constraints(problem)
    if high_similarity_context:
        base_prompt += (
            "\n\n[高相似候选完整参考解] 检索区块因稠密相似度超过 0.85 展示了一条完整参考解，"
            "不要求规范化题面完全一致。它不一定是同一道题；仅在该条另行标明答案复用门槛通过时，"
            "才可考虑复用答案，其他情况须针对当前题独立推导并核验。\n"
        )
    # 压缩重试要压的是推导篇幅，不是答案该覆盖的分支（评委复测 idx 14）。
    coverage_clause = answer_coverage_clause(problem)

    if is_objective_mode(question_mode):
        domain_guard = ""
        if category == "抽象代数" and re.search(
            r"分裂域|Galois|伽罗瓦|域扩张|扩张次数|单位根", problem, flags=re.I
        ):
            domain_guard = (
                "[域扩张核验] 定义每个新增生成元，列出全部候选根并代回原多项式；"
                "不要混用正实根与所选复根的记号。对 x^n−a 型，可先选 α 满足 α^n=a，"
                "再以 αζ_n^k 列出根，ζ_n 为原始 n 次单位根；核对所写域是否包含全部根。"
                "用最小多项式与塔定理说明扩张次数，各层须考虑已有生成元之间的关系，"
                "不能无依据把次数机械相乘。Galois 判断核对正规性与可分性。"
                "最终答案里写出生成元的定义、次数和所有要求的判断。"
            )
        # 空位数由代码侧数好写进提示（评委报告 idx 105：提示里的"先数空位"元指令
        # 被模型原样回显成了答案 "Count blanks: 1"；idx 86：三空只答零空未被拦截）。
        from utils.problem_profile import count_blanks, fill_answer_matches_blanks

        blank_count = count_blanks(problem) if question_mode == "fill" else 0
        if question_mode == "fill":
            blank_note = (
                f"题面共 {blank_count} 个空位。" if blank_count >= 2 else ""
            )
            answer_shape = (
                f"{blank_note}答案行按空位顺序写全部结果，多空用分号分隔"
                "（如：空1: <结果>；空2: <结果>）。每空必须填教材术语/数值/明确方向，"
                "禁止填'不确定'，禁止输出任何操作说明文字"
            )
        elif question_mode == "true_false":
            answer_shape = "答案行只写：正确 或 错误"
        else:
            answer_shape = (
                "先判断单选/多选（题面注明多选或多个选项独立成立才多选），"
                "答案行列出全部正确选项字母；措辞不精确或概念被替换的选项一律不选"
            )
        # 两行契约把"答案"排在"依据"前面，模型先承诺后核对——对需要实算的客观题
        # （域扩张次数、维数、阶、收敛半径等）这个顺序恰好是反的：2026-08-10 复测轮
        # idx 86 就是先写下 Q(∜5,i);8 再补依据。检出实算标记时改为"先核对后作答"，
        # 输出契约不变（解析取最后一个"答案："标签），成本只多几行核对文字。
        needs_derivation = bool(re.search(
            r"分裂域|扩张次数|\[E\s*[:：]\s*(?:Q|\\mathbb\{Q\})\]|Galois|伽罗瓦|"
            r"维数|秩|阶数|特征值|收敛半径|次数是|degree\b|dimension\b",
            problem, flags=re.I))
        if needs_derivation:
            contract = (
                "先算后答：必须严格按此顺序输出三行，不要输出 Thinking Process：\n"
                "核对：<逐层写出关键计算（列全部根/逐层次数/定义核对），不超过五行>\n"
                f"答案：<{answer_shape}>\n"
                "依据：<一句话说明答案与上面核对的一致性>"
            )
        else:
            contract = (
                "不要展开长篇推导，不要输出 Thinking Process。先核对，再提交：\n"
                "核对：<按选项或空位检查定义、定理前提及边界，不超过五句；"
                "明确哪些前提来自题面、哪些来自记号的通常定义，避免漏掉有界性等必要条件>\n"
                f"答案：<{answer_shape}>"
            )
        contract += (
            "\n若题目问某个记号的定义或通常参数，先采用该课程的标准记号约定；"
            "不能因另一种参数化或另一种记号也合法，就把它作为同一记号下的额外选项。"
            "题面明确要求多选或问哪些陈述成立时逐项选择；单一术语/定义题按最直接的标准定义作答。"
            "若确实无法唯一确定口径，在核对中说明，不凭空声称教材或官方规定。"
        )
        objective_prompt = (
            "你是数学、统计学与计量经济学教师。严格按题库教材定义逐项核对题面选项/空位，"
            "特别检查相近概念的边界（非参数回归不等于非线性回归，标准差与方差按题面参数化区分，来源未限定时保留歧义）；"
            f"{domain_guard}"
            "\n先阅读下面的普通检索参考；它们是工作解/方法提示，不是答案锁定。若检索相似度很高，"
            "复核其推导后尽量保持同一答案术语和语言；若相似度较低，只借方法并按本题重算。\n"
            f"{examples_text}{cards_text}\n"
            + ("通过答案安全门槛的同题参考记录存在时，可参考其原始题解；仍须核对本题条件。\n"
               if high_similarity_context else "")
            + f"{contract}\n\n"
            f"题型：{question_mode}\n学科：{category}\n技能参考：\n"
            f"{select_skill_excerpt(skill_doc, problem, 2200)}\n\n"
            "只有明确通过答案安全门槛的同题参考，才可参考其完整解答；其他近似题只能借方法。"
            "仍须在本题选项/条件上独立核对，再输出本题答案，不要迁移近似题结论。\n\n"
            f"题目：\n{problem}"
        )
        objective_prompt += problem_constraints(problem)
        if state.get("branch_hint"):
            objective_prompt += "\n\n[独立复核提示] " + str(state["branch_hint"])
        trace = [{"status": "local_skill_selection", "source": "resources/skills",
              "cards": selected_card_ids(select_skill_excerpt(skill_doc, problem, 2200)),
              "objective_cards": selected_card_ids(select_skill_excerpt(skill_doc, problem, 2200)),
              "alternate_cards": [selected_card_ids(part) for part in alternate_skill_parts]}]
        response = ""
        attempts = 0

        def _semantic_check(candidate: dict) -> bool:
            if not candidate.get("answer"):
                return False
            okay, reason = objective_answer_consistency(
                candidate.get("answer", ""), problem, question_mode)
            if okay:
                return True
            candidate["invalid_answer"] = candidate.get("answer", "")
            candidate["semantic_invalid_reason"] = reason
            candidate["answer"] = ""
            return False

        try:
            attempts = 1
            response = chat_with_retry(
                client,
                messages=[{"role": "user", "content": objective_prompt}],
                temperature=CONFIG["temperatures"].get(
                    "objective_reasoning", CONFIG["temperatures"]["reasoning"]),
                max_tokens=CONFIG["max_tokens"]["objective_reasoning"],
                top_p=CONFIG["top_p"]["objective_reasoning"],
                top_k=CONFIG["top_k"]["objective_reasoning"],
                logger=deps.logger,
                time_budget=deps.time_budget,
                expected_call_seconds=45,
                label="objective_reasoning_plain",
            )
            if budget:
                budget.consume(estimate_tokens(objective_prompt), estimate_tokens(response))
        except Exception as exc:  # noqa: BLE001 - degrade to the normal parser below.
            deps.logger.warning("Objective reasoning failed: %s", exc)
            trace.append({"attempt": attempts, "status": "failed", "error": str(exc)[:200]})
        parsed = _parse_reasoning_output(
            response, question_mode=question_mode, problem=problem)
        semantic_ok = _semantic_check(parsed)
        fill_complete = question_mode != "fill" or fill_answer_matches_blanks(
            parsed.get("answer", ""), problem)
        if parsed.get("answer") and fill_complete and semantic_ok:
            trace.append({"attempt": attempts, "status": "objective_plain",
                          "response_chars": len(response or "")})
            return _reasoning_state(parsed, trace, attempts, response,
                                    examples_text, cards_text)
        if not parsed.get("answer"):
            # 解析失败的原始响应必须留证（评委建议 9：75-212 字符的失败响应
            # 只记了字符数，评审无法区分"模型答错"与"解析器漏掉"）。
            trace.append({"attempt": attempts, "status": "unparsed",
                          "response_chars": len(response or ""),
                          "raw_excerpt": _raw_excerpt(response, head=300, tail=500)})
        if parsed.get("answer") and not fill_complete:
            # 空位覆盖不足（评委报告 idx 86：三空只答零空仍以 0.78 置信放行）。
            # 保留这个不完整候选，但先花一次种子重试争取全空覆盖。
            trace.append({"attempt": attempts, "status": "fill_blank_mismatch",
                          "blanks": count_blanks(problem),
                          "response_chars": len(response or "")})
        if parsed.get("semantic_invalid_reason"):
            trace.append({"attempt": attempts, "status": "semantic_mismatch",
                          "reason": parsed["semantic_invalid_reason"],
                          "response_chars": len(response or "")})
        incomplete_first = parsed if (
            parsed.get("answer") or parsed.get("incomplete_answer")
        ) else None
        if not (deps.time_budget and deps.time_budget.closeout_started()):
            try:
                attempts += 1
                response = chat_prefilled(
                    client,
                    messages=[{"role": "user", "content": objective_prompt}],
                    prefix="核对：",
                    temperature=CONFIG["temperatures"].get(
                        "objective_reasoning", CONFIG["temperatures"]["reasoning"]),
                    max_tokens=CONFIG["max_tokens"]["objective_reasoning"],
                    top_p=CONFIG["top_p"]["objective_reasoning"],
                    top_k=CONFIG["top_k"]["objective_reasoning"],
                    logger=deps.logger,
                    time_budget=deps.time_budget,
                    expected_call_seconds=60,
                    label="objective_reasoning_prefill",
                )
                if budget:
                    budget.consume(estimate_tokens(objective_prompt), estimate_tokens(response))
            except Exception as exc:  # noqa: BLE001
                deps.logger.warning("Objective reasoning retry failed: %s", exc)
                trace.append({"attempt": attempts, "status": "failed", "error": str(exc)[:200]})
            retry_parsed = _parse_reasoning_output(
                response, question_mode=question_mode, problem=problem)
            retry_semantic_ok = _semantic_check(retry_parsed)
            retry_fill_complete = question_mode != "fill" or fill_answer_matches_blanks(
                retry_parsed.get("answer", ""), problem)
            # 重试答案只有在"有答案且不比首轮更残缺"时才替换首轮候选。
            if retry_parsed.get("answer") and retry_semantic_ok \
                    and (retry_fill_complete or incomplete_first is None):
                parsed = retry_parsed
            elif incomplete_first is not None:
                parsed = incomplete_first
            if retry_parsed.get("semantic_invalid_reason"):
                trace.append({"attempt": attempts, "status": "semantic_mismatch",
                              "reason": retry_parsed["semantic_invalid_reason"],
                              "response_chars": len(response or "")})
        parsed_fill_complete = question_mode != "fill" or fill_answer_matches_blanks(
            parsed.get("answer", ""), problem)
        if (not parsed.get("answer") or not parsed_fill_complete
                or parsed.get("semantic_invalid_reason")):
            # 客观题格式/语义失败不代表长推理触及 token 上限，不走压缩推理。
            trace.append({"status": "compressed_skipped",
                          "reason": "not_a_confirmed_reasoning_token_limit"})
        final_entry = {"attempt": attempts,
                       "status": "success" if parsed.get("answer") else "failed",
                       "objective": True, "response_chars": len(response or "")}
        if not parsed.get("answer"):
            final_entry["raw_excerpt"] = _raw_excerpt(response, head=300, tail=500)
        trace.append(final_entry)

        return _reasoning_state(parsed, trace, attempts, response,
                                examples_text, cards_text)

    hint = state.get("branch_hint")
    # 复核重试也保留 skill 文档与字段契约（旧实现只发 hint+题面，丢失了学科口径）
    prompt = f"{base_prompt}\n\n[复核提示] {hint}" if hint else base_prompt
    hinted_base = prompt
    trace = [{"status": "local_skill_selection", "source": "resources/skills",
              "cards": selected_card_ids(select_skill_excerpt(skill_doc, problem, 3000)),
              "objective_cards": selected_card_ids(select_skill_excerpt(skill_doc, problem, 2200)),
              "alternate_cards": [selected_card_ids(part) for part in alternate_skill_parts]}]
    attempts = 0
    resp = ""
    parsed = {"analysis": "", "steps": [], "answer": ""}
    clock = deps.time_budget
    if clock and clock.closeout_started():
        max_attempts = 1
    for _ in range(max_attempts):
        if attempts and not can_afford_retry(clock, "reasoning"):
            deps.logger.warning(
                "Skipping reasoning retry: %.0fs left, last attempt cost %.0fs",
                clock.remaining(), last_attempt_cost(clock, "reasoning"))
            trace.append({"attempt": attempts + 1, "status": "skipped",
                          "reason": "retry_unaffordable"})
            break
        attempts += 1
        try:
            resp = chat_with_retry(
                client,
                messages=[{"role": "user", "content": prompt}],
                temperature=CONFIG["temperatures"]["reasoning"],
                max_tokens=CONFIG["max_tokens"]["reasoning"],
                top_p=CONFIG["top_p"]["reasoning"],
                top_k=CONFIG["top_k"]["reasoning"],
                logger=deps.logger,
                time_budget=deps.time_budget,
                expected_call_seconds=_NORMAL_REASONING_CALL_ESTIMATE_S,
                label="reasoning",
            )
        except Exception as exc:  # noqa: BLE001 - keep whatever we already parsed.
            deps.logger.warning("Reasoning attempt %s failed: %s", attempts, exc)
            trace.append({"attempt": attempts, "status": "failed", "error": str(exc)[:200]})
            break
        if budget:
            budget.consume(estimate_tokens(prompt), estimate_tokens(resp))
        parsed = _parse_reasoning_output(
            resp, question_mode=question_mode, problem=problem)
        # 已有可用答案和步骤就直接结束；即使服务端报告 length，也不压缩覆盖。
        if _is_complete(parsed):
            complete_entry = {"attempt": attempts, "status": "success",
                              "response_chars": len(resp or "")}
            finish_reason = getattr(resp, "finish_reason", None)
            if finish_reason is not None:
                complete_entry["finish_reason"] = str(finish_reason)
            trace.append(complete_entry)
            return _reasoning_state(parsed, trace, attempts, resp,
                                    examples_text, cards_text)

        exhaustion_reason = _token_exhaustion_reason(
            resp, parsed, max_tokens=CONFIG["max_tokens"]["reasoning"])
        exhausted = bool(exhaustion_reason)
        entry = {"attempt": attempts, "status": "failed",
                 "response_chars": len(resp or "")}
        if exhausted:
            entry.update({"reason": exhaustion_reason,
                          "raw_excerpt": _raw_excerpt(resp)})
        else:
            entry["reason"] = "incomplete_but_token_limit_not_confirmed"
            finish_reason = getattr(resp, "finish_reason", None)
            completion_tokens = getattr(resp, "completion_tokens", None)
            if finish_reason is not None:
                entry["finish_reason"] = str(finish_reason)
            if completion_tokens is not None:
                entry["completion_tokens"] = completion_tokens
        trace.append(entry)
        # Missing deliverables justify completion independently of provider
        # metadata. Reuse the draft instead of buying another full exploration.
        if exhausted or not parsed.get("answer"):
            attempts += 1
            compressed_resp, fail_reason = _compressed_reasoning_retry(
                deps, hinted_base,
                failure=("已达到长推理输出 token 上限，但回答被截断且结构不完整" if exhausted
                         else "上一轮有推导但缺少可提交的最终答案"),
                coverage=coverage_clause, previous_response=resp)
            if compressed_resp is None:
                trace.append({"attempt": attempts, "status": "skipped",
                              "reason": fail_reason or "compressed_retry_failed"})
                break
            resp = compressed_resp
            compressed_parsed = _parse_reasoning_output(
                resp, question_mode=question_mode, problem=problem)
            # 压缩重试产出的任何章节都严格优于上一轮的空产出。
            if compressed_parsed.get("answer") or compressed_parsed.get("steps") \
                    or compressed_parsed.get("analysis"):
                if not compressed_parsed.get("answer") and parsed.get("answer"):
                    # 压缩重试补齐章节但没写出结论时，保留此前捞回的低置信答案
                    # （来源标记不变），聊胜于无。
                    compressed_parsed["answer"] = parsed["answer"]
                    compressed_parsed["answer_source"] = parsed.get(
                        "answer_source", "salvaged_prose")
                parsed = compressed_parsed
                parsed["answer_source"] = parsed.get("answer_source") or "compressed_prefill"
            trace.append({"attempt": attempts,
                          "status": "success" if _is_complete(compressed_parsed) else "failed",
                          "reason": "compressed_prefill_after_" + (exhaustion_reason or "missing_answer"),
                          "response_chars": len(resp or "")})
            if _is_complete(compressed_parsed):
                return _reasoning_state(parsed, trace, attempts, resp,
                                        examples_text, cards_text)
            break
        if parsed.get("answer") and question_mode != "proof":
            return _reasoning_state(parsed, trace, attempts, resp, examples_text, cards_text)
        prompt = (base_prompt + "\n\n补齐缺失的证明步骤，保留已有最终答案；交付范围以题面要求为准。")
    return _reasoning_state(parsed, trace, attempts, resp,
                            examples_text, cards_text)

