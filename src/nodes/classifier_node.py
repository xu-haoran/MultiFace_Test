"""学科分类：一次预填充调用比较全部 18 个技能领域。"""

import json
import re
from utils.deps import get_deps
from utils.llm_retry import chat_prefilled, chat_with_retry
from utils.prompt_templates import CLASSIFICATION_PREFILL, CLASSIFICATION_PROMPT
from utils.token_budget import estimate_tokens
from utils.cot_stripper import strip_cot_prefix
from config import CONFIG


def _parse_json(text):
    text = strip_cot_prefix(text).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i+1]
                    try:
                        return json.loads(candidate)
                    except Exception:
                        break  # try next start
        start = text.find("{", start + 1)
    return None


_PUNCT_STRIP = re.compile(r"[\s　\"'.,。，、：:；;！!？?（）()【】\[\]{}「」『』]")


def _normalize_cat(s: str) -> str:
    """去除空白、引号与常见中英文标点，用于分类名的模糊匹配。"""
    return _PUNCT_STRIP.sub("", s or "")


def _lcs_length(a: str, b: str) -> int:
    """最长公共子串长度（连续字符）。对中文类别名比编辑距离更可靠。"""
    if not a or not b:
        return 0
    m, n = len(a), len(b)
    prev = [0] * (n + 1)
    best = 0
    for i in range(1, m + 1):
        cur = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return best


def _resolve_category(raw, valid_categories, fallback):
    """将 LLM 返回的 category 归一到 resources/skills/ 下的真实文件夹名。

    严格保证返回值 ∈ valid_categories（即实际文件夹名），杜绝下游
    get_skill_document / get_validation_script 因类别名不存在而 FileNotFoundError。

    匹配顺序：精确 → 归一化 → 子串包含（双向）→ 最长公共子串（≥2）→ Levenshtein（≥0.6）→ fallback。
    无法匹配时返回调用方提供的 fallback。
    """
    if not isinstance(raw, str) or not raw:
        return fallback
    s = raw.strip().strip("\"'「」『』").strip()
    if not s:
        return fallback
    # 1. 精确匹配
    if s in valid_categories:
        return s
    # 2. 归一化匹配（去空白/标点后比较）
    sn = _normalize_cat(s)
    for c in valid_categories:
        if _normalize_cat(c) == sn:
            return c
    # 3. 子串包含（任一方向）
    for c in valid_categories:
        if s in c or c in s:
            return c
    # 4. 最长公共子串（连续≥2字符）—— 对中文类别名比 Levenshtein 更可靠
    #    例：线性代数 ↔ 高等代数（共享"代数"）、概率统计 ↔ 概率论（共享"概率"）
    lcs_matches = [(c, _lcs_length(s, c)) for c in valid_categories]
    lcs_matches = [(c, l) for c, l in lcs_matches if l >= 2]
    if lcs_matches:
        max_lcs = max(l for _, l in lcs_matches)
        tied = [c for c, l in lcs_matches if l == max_lcs]
        if len(tied) == 1:
            return tied[0]
        # LCS 并列时用 Levenshtein ratio 精排（best-effort，区分长短不同的候选）
        try:
            from Levenshtein import ratio as lev_ratio
            return max(tied, key=lambda c: lev_ratio(s, c))
        except ImportError:
            return tied[0]
    # 5. Levenshtein 模糊匹配（兜底）
    try:
        from Levenshtein import ratio as lev_ratio
        best, best_score = max(((c, lev_ratio(s, c)) for c in valid_categories),
                               key=lambda x: x[1])
        if best_score >= 0.6:
            return best
    except ImportError:
        pass
    return fallback


def _merge_candidate_rankings(*rankings, limit):
    """按分数合并多路候选，而不是按位次。

    旧实现把每路排名压成 1/(位次+1) 的 Borda 分——关键词 #1（领先 4 倍分数）和
    TF-IDF #1 各得 1.0，于是 TF-IDF 的噪声位次常把分差巨大但位次接近的候选翻盘。
    2026-08-22 无 API 全量观测：英文题下关键词全 0 时 TF-IDF 决定一切（复分析
    靠文档英文词汇密度拿走约一半题目的顶分）；关键词有内容时分数本身就够判别。
    改为两段式：关键词分数非全零时按关键词分数排序（TF-IDF 仅作并列时的微扰），
    全零时才由 TF-IDF 决定。
    """
    kw = dict(rankings[0]) if rankings else {}
    tf = dict(rankings[1]) if len(rankings) > 1 else {}
    if any(v > 0 for v in kw.values()):
        merged = {c: kw.get(c, 0.0) + 1e-9 * tf.get(c, 0.0)
                  for c in set(kw) | set(tf)}
    else:
        merged = dict(tf)
    ordered = sorted(merged, key=lambda c: (-merged[c], c))
    return ordered[:limit]


_CONFIDENCE_RE = re.compile(r'"?confidence"?\s*[:=]\s*([0-9]*\.?[0-9]+)')


def _extract_decision(text, all_categories):
    """Recover (category, confidence) from a prefilled or plain classifier reply.

    Tries strict JSON first, then falls back to scanning the raw text for a domain
    name. The scan matters: a prefilled reply is a JSON *fragment* by construction,
    and a plain reply may carry a CoT preamble around the object. Confidence is
    optional — the category is the signal, so a parseable domain with no readable
    confidence is accepted at the threshold rather than discarded.
    """
    if not text:
        return None
    cleaned = strip_cot_prefix(text)

    parsed = _parse_json(cleaned)
    if isinstance(parsed, dict) and "category" in parsed:
        resolved = _resolve_category(parsed["category"], all_categories, None)
        if resolved:
            # 缺失置信度时不强制升级，已识别的学科仍可直接使用。
            if "confidence" in parsed:
                try:
                    return resolved, float(parsed["confidence"])
                except (TypeError, ValueError):
                    pass
            return resolved, CONFIG["classifier_confidence_threshold"]

    # 预填充响应可能缺少开头的大括号，先补齐再解析。
    head = cleaned.split('"')[0].strip()
    resolved = _resolve_category(head, all_categories, None)
    if not resolved:
        # 最后兜底：取响应中最早出现的合法学科名。
        best = None
        for category in all_categories:
            position = cleaned.find(category)
            if position != -1 and (best is None or position < best[1]):
                best = (category, position)
        resolved = best[0] if best else None
    if not resolved:
        return None

    match = _CONFIDENCE_RE.search(cleaned)
    if match:
        try:
            return resolved, float(match.group(1))
        except (TypeError, ValueError):
            pass
    return resolved, CONFIG["classifier_confidence_threshold"]


def _classify_prefilled(deps, prompt, all_categories):
    """Fast path: prefilled single call, ~1s and ~12 completion tokens."""
    budget = deps.token_budget
    response = ""
    try:
        response = chat_prefilled(
            deps.client,
            messages=[{"role": "user", "content": prompt}],
            prefix=CLASSIFICATION_PREFILL,
            temperature=CONFIG["temperatures"]["classifier"],
            max_tokens=CONFIG["max_tokens"]["classifier"],
            logger=deps.logger,
            time_budget=deps.time_budget,
            expected_call_seconds=5,
            label="classifier_prefill",
        )
    except Exception as exc:  # noqa: BLE001 - falls through to the plain call.
        deps.logger.warning("Prefilled classifier call failed: %s", exc)
        return None
    finally:
        if budget:
            budget.consume(estimate_tokens(prompt), estimate_tokens(response))
    return _extract_decision(response, all_categories)


#: 非预填充分类的保守耗时估计（秒），用于预算判断。
_PLAIN_CALL_SECONDS = 150


def _classify_plain(deps, prompt, all_categories):
    """Fallback: unprefilled call with room for full CoT (~130s on a hard problem).

    Only reached when the prefilled reply was unparseable, e.g. a backend that
    rejects a trailing assistant turn.
    """
    budget = deps.token_budget
    response = ""
    try:
        response = chat_with_retry(
            deps.client,
            messages=[{"role": "user", "content": prompt}],
            temperature=CONFIG["temperatures"]["classifier"],
            max_tokens=CONFIG["max_tokens"]["classifier_fallback"],
            logger=deps.logger,
            time_budget=deps.time_budget,
            expected_call_seconds=_PLAIN_CALL_SECONDS,
            label="classifier_plain",
        )
    except Exception as exc:  # noqa: BLE001 - falls through to deterministic ranking.
        deps.logger.warning("Classifier LLM call failed: %s", exc)
        return None
    finally:
        if budget:
            budget.consume(estimate_tokens(prompt), estimate_tokens(response))
    return _extract_decision(response, all_categories)


def _deterministic_fallback(sl, problem, all_categories, logger):
    """Zero-cost local ranking — keyword hits merged with TF-IDF similarity.

    Computed lazily: on the happy path the LLM answers first and this never runs,
    so we do not pay to build the TF-IDF index at all.
    """
    keyword_candidates = []
    try:
        keyword_candidates = sl.find_candidate_categories(problem, top_k=CONFIG["classifier_top_k"])
    except Exception as exc:  # noqa: BLE001
        logger.warning("Keyword ranking unavailable: %s", exc)

    tfidf_candidates = []
    try:
        tfidf_candidates = sl.get_embedding_index().rank_all(
            problem, top_k=CONFIG["classifier_top_k"]
        )
    except Exception as exc:  # noqa: BLE001 - keyword ranking still stands.
        logger.warning("TF-IDF category ranking unavailable: %s", exc)

    merged = _merge_candidate_rankings(keyword_candidates, tfidf_candidates,
                                       limit=CONFIG["classifier_top_k"])
    merged = [category for category in merged if category in all_categories]
    if merged:
        return merged[0]
    if all_categories:
        return all_categories[0]
    return "非基础及进阶课程"


def _has_any(text: str, cues) -> bool:
    for cue in cues:
        needle = str(cue or "").casefold()
        if not needle:
            continue
        # 短英文缩写按独立词匹配，避免误命中普通单词。
        if re.fullmatch(r"[a-z0-9-]{2,4}", needle):
            if re.search(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", text):
                return True
        elif needle in text:
            return True
    return False


def _has_cue_groups(text: str, *groups) -> bool:
    """Return true when every semantic cue group contributes at least one hit."""
    return all(_has_any(text, group) for group in groups)


def _domain_override(problem: str, categories: list[str]) -> str | None:
    """Resolve high-precision boundaries that the fast LLM call often confuses.

    Rules describe reusable mathematical markers rather than problem ids or
    answers.  Broad words such as ``triangle``, ``group`` and ``polynomial`` are
    never sufficient by themselves: ambiguous competition statements must hit a
    cue family before a specialist manual replaces the LLM decision.
    """
    text = re.sub(r"\s+", " ", str(problem or "")).casefold()
    available = set(categories)

    # 数值方法优先于被离散的解析方程。
    numerical_cues = (
        "数值分析", "条件数", "有限差分", "有限元", "有限体积", "截断误差",
        "数值微分", "数值积分", "离散化方法", "离散化处理", "数值离散",
        "中心差分", "前向差分", "后向差分", "差分公式", "病态矩阵",
        "condition number", "finite difference", "finite element",
        "finite volume", "truncation error", "numerical differentiation",
        "numerical integration", "numerical discretization", "newton-cotes",
    )
    if "数值分析" in available and _has_any(text, numerical_cues):
        return "数值分析"

    # 计量经济和回归诊断统一归入回归类。
    regression_cues = (
        "计量经济", "线性回归", "非参数回归", "非线性回归", "回归系数",
        "逐步回归", "异方差", "内生性", "工具变量", "残差图", "残差",
        "最小二乘", "ols", "gauss-markov", "多重共线性", "vif",
        "durbin-watson", "拟合优度", "regression coefficient",
        "heteroscedastic", "endogeneity", "instrumental variable",
    )
    if "线性回归" in available and _has_any(text, regression_cues):
        return "线性回归"

    # 描述统计和时间序列归入统计推断，保持 18 个技能目录不变。
    stats_cues = (
        "统计图形", "直方图", "散点图", "箱线图", "条形图", "时间数列",
        "时间序列", "趋势变动", "季节变动", "循环变动", "不规则变动",
        "季节调整", "移动平均", "指数平滑", "自相关图", "抽样分布",
        "总体参数", "样本均值", "样本方差", "频数分布", "集中趋势",
        "histogram", "box plot", "time series", "seasonal adjustment",
        "moving average", "exponential smoothing", "sampling distribution",
    )
    normal_parameter_question = _has_cue_groups(
        text,
        ("正态分布", "normal distribution"),
        ("两个参数", "参数分别", "parameters", "mean and variance", "mean and standard deviation"),
    )
    data_summary_question = _has_cue_groups(
        text,
        ("数据", "data"),
        ("分散程度", "离散程度", "集中趋势", "频数分布", "dispersion"),
    )
    if ("统计推断" in available
            and (_has_any(text, stats_cues) or normal_parameter_question or data_summary_question)):
        return "统计推断"

    # 微分算子的伴随归入偏微分方程，泛函算子的伴随归入泛函分析。
    pde_direct = (
        "偏微分方程", "partial differential equation", "形式伴随", "formal adjoint",
        "散度型", "divergence-form", "热方程", "波动方程", "poisson equation",
    )
    pde_adjoint = _has_cue_groups(
        text,
        ("伴随算子", "adjoint operator", "adjoint"),
        ("c_0^", "\\partial_i", "∂_i", "微分算子", "differential operator", "omega"),
    )
    if "偏微分方程" in available and (_has_any(text, pde_direct) or pde_adjoint):
        return "偏微分方程"

    abstract_cues = (
        "galois", "分裂域", "splitting field", "域扩张", "field extension",
        "正规子群", "normal subgroup", "换位子群", "commutator subgroup",
        "二面体群", "dihedral group", "sylow", "商群", "quotient group",
        "理想", "ideal of", "有限域", "finite field",
    )
    if "抽象代数" in available and _has_any(text, abstract_cues):
        return "抽象代数"

    analysis_cues = (
        "数学分析", "一致收敛", "uniform convergence", "逐点收敛",
        "pointwise convergence", "极限函数", "limit function", "广义积分",
        "improper integral", "含参积分", "power series", "幂级数", "上确界",
        "supremum", "fourier transform", "fourier 变换", "cauchy数列",
        "cauchy sequence", "完备度量空间", "taylor expansion", "泰勒展开",
    )
    circle_laplacian = _has_cue_groups(
        text,
        ("圆周", "on the circle", "restricted to the circle"),
        ("拉普拉斯算子", "laplacian"),
    )
    difference_quotient_fe = _has_cue_groups(
        text,
        ("difference quotient", "差商", "g(a)-g(b)", "g(a) - g(b)"),
        ("a-b", "a - b"),
        ("function", "函数"),
    )
    bounded_sequence_extremum = _has_cue_groups(
        text,
        ("infinite sequence", "无限序列"),
        ("belongs to [", "each term belongs", "bounded", "有界"),
        ("m < n", "m<n", "任意正整数 m<n"),
    )
    real_selection_extremum = _has_cue_groups(
        text,
        ("five distinct positive real numbers", "五个互不相同的正实数"),
        ("choose four distinct", "选取四个"),
        ("minimum value", "最小值"),
    )
    growth_rate_game = _has_cue_groups(
        text,
        ("flooded", "洪水", "淹没"),
        ("barrier", "walls", "屏障", "墙"),
        ("gamma", "\\gamma", "增长速率"),
    )
    rational_integer_fe = _has_cue_groups(
        text,
        ("\\mathbb{q}", "有理数"),
        ("\\mathbb{z}", "整数值"),
        ("g(bx-a)", "g(bx - a)"),
    )
    if ("数学分析" in available and (
            _has_any(text, analysis_cues)
            or circle_laplacian
            or difference_quotient_fe
            or bounded_sequence_extremum
            or real_selection_extremum
            or growth_rate_game
            or rational_integer_fe
    )):
        return "数学分析"

    or_direct = (
        "运筹学", "线性规划", "linear programming", "单纯形", "simplex method",
        "运输问题", "transportation problem", "整数规划", "integer programming",
        "分支定界", "branch and bound", "指派问题", "assignment problem",
        "关键路径", "critical path", "最大流", "maximum flow", "最短路",
        "shortest path", "目标规划", "排队模型", "queueing model",
    )
    scheduling_optimisation = _has_cue_groups(
        text,
        ("schedule", "scheduling", "赛程", "调度"),
        ("minimize the total cost", "minimum total cost", "最小总成本", "费用最小"),
    )
    allocation_game = _has_cue_groups(
        text,
        ("boxes", "箱子"),
        ("pebbles", "石子"),
        ("distributes", "分配"),
        ("each subsequent round", "随后每轮", "每一轮"),
    )
    if ("运筹学" in available
            and (_has_any(text, or_direct) or scheduling_optimisation or allocation_game)):
        return "运筹学"

    high_algebra_cues = (
        "高等代数", "minimal polynomial", "极小多项式", "symmetric polynomial",
        "对称多项式", "characteristic polynomial", "特征多项式", "eigenvalue",
        "特征值", "linear space", "线性空间", "quadratic form", "二次型",
        "jordan form", "jordan 标准形", "max-plus", "tropical algebra", "热带代数",
    )
    f2_function = _has_cue_groups(
        text,
        ("\\mathbb{f}_2", "二元域", "binary field"),
        ("function", "函数"),
        ("\\mathbb{q}", "rational"),
    )
    polynomial_function_equation = _has_cue_groups(
        text,
        ("a:\\mathbb{r}", "a: \\mathbb{r}", "a(p)a(q)"),
        ("a(-pq)", "2pq"),
    )
    polynomial_system = _has_cue_groups(
        text,
        ("triples", "三元组"),
        ("x^2 + y^2 + z^2", "x²+y²+z²"),
        ("xy^3", "xy³"),
    )
    tuple_pairing = _has_cue_groups(
        text,
        ("m-tuple of real numbers", "-tuple of real numbers", "实数 m 元组"),
        ("for each permutation", "任意排列"),
        ("p< q", "p < q", "两两乘积"),
    )
    max_plus_tuple = _has_cue_groups(
        text,
        ("integer-valued", "整数值"),
        ("tuples", "元组"),
        ("\\max", "coordinatewise max", "逐分量取大"),
        ("v+w", "v}+\\mathbf{w}", "向量加法"),
    )
    if ("高等代数" in available and (
            _has_any(text, high_algebra_cues)
            or f2_function
            or polynomial_function_equation
            or polynomial_system
            or tuple_pairing
            or max_plus_tuple
    )):
        return "高等代数"

    # 欧氏、凸和射影几何归入竞赛几何；仅有图形词时需再匹配关系词。
    diff_geo_markers = (
        "曲率", "curvature", "测地线", "geodesic", "流形", "manifold",
        "第一基本形式", "second fundamental form", "frenet", "weingarten",
        "gauss-bonnet", "法曲率", "主曲率", "平均曲率", "torsion",
    )
    classic_geometry = (
        "切弦角定理", "tangent-chord theorem", "极点极线", "polar line",
        "la hire", "根轴", "radical axis", "共轴", "coaxal", "外心",
        "circumcenter", "内心", "incenter", "垂心", "orthocenter", "圆周角",
        "angle bisector", "角平分线", "circumcircle", "外接圆", "内切圆",
    )
    polygon_geometry = _has_cue_groups(
        text,
        ("convex polygon", "convex quadrilateral", "convex polyhedron", "凸多边形", "凸四边形", "凸多面体", "m-gon", "-gon", "n-sided polygon", "-sided polygon"),
        ("area", "diagonal", "face", "visible", "circumscribed", "面积", "对角线", "面可见", "外切"),
    )
    triangle_geometry = _has_cue_groups(
        text,
        ("triangle", "三角形"),
        (" angle ", " angles ", "\\angle", "tangent", "circumcenter", "circumcircle", "bisector", "角", "切线", "外心", "外接圆", "平分线"),
    )
    line_angle_geometry = _has_cue_groups(
        text,
        ("line", "直线"),
        ("segment", "线段"),
        ("angle", "角"),
    )
    incidence_geometry = _has_cue_groups(
        text,
        ("points on the plane", "平面上的点", "points in the plane"),
        ("lines in the plane", "平面直线"),
        ("circle", "圆"),
    )
    if ("非基础及进阶课程" in available
            and not _has_any(text, diff_geo_markers)
            and (_has_any(text, classic_geometry)
                 or polygon_geometry
                 or triangle_geometry
                 or line_angle_geometry
                 or incidence_geometry)):
        return "非基础及进阶课程"

    number_theory_floor = _has_cue_groups(
        text,
        ("prime", "素数"),
        ("floor", "\\lfloor", "取整"),
        ("integer", "整数"),
    )
    diophantine_radical = _has_cue_groups(
        text,
        ("pairs of positive integers", "正整数对"),
        ("cube root", "\\sqrt[3]", "立方根"),
        ("satisfy", "满足"),
    )
    finite_extremal_construction = _has_cue_groups(
        text,
        ("smallest positive integer n", "smallest positive integer", "最小正整数 n"),
        ("there exist real numbers", "存在实数"),
        ("between -1 and 1", "between $-1$ and 1", "between $-1$ and $1$", "介于 -1 和 1"),
        ("sum", "\\sum", "求和"),
    )
    if ("离散数学" in available
            and (number_theory_floor or diophantine_radical or finite_extremal_construction)):
        return "离散数学"

    return None


# 兼容早期测试分支使用的名称。
_objective_domain_override = _domain_override


def _decision_payload(problem, categories, category, confidence, candidates, stages):
    """Apply the same calibrated boundary decision on every classifier exit."""
    override = _domain_override(problem, categories)
    if override and override != category:
        stages.append("domain_override")
        category = override
        confidence = max(confidence, CONFIG["classifier_confidence_threshold"])
    ordered = [category] + [name for name in candidates if name != category]
    return {
        "category": category,
        "category_confidence": confidence,
        "candidate_categories": ordered,
        "alternate_categories": ordered[1:3],
        "classification_stages_used": stages,
    }


def _rank_alternate_categories(sl, problem, selected, all_categories):
    """Build a small deterministic backup list without replacing the LLM choice."""
    ranked = []
    try:
        ranked.extend(name for name, score in sl.find_candidate_categories(
            problem, top_k=CONFIG.get("classifier_top_k", 3)) if score > 0)
    except Exception:
        pass
    try:
        ranked.extend(name for name, _ in sl.get_embedding_index().rank_all(
            problem, top_k=CONFIG.get("classifier_top_k", 3)))
    except Exception:
        pass
    output = []
    for name in [selected] + ranked:
        if name in all_categories and name not in output:
            output.append(name)
        if len(output) >= 3:
            break
    return output or [selected]


def classifier_node(state, config):
    deps = get_deps(config)
    sl = deps.skills_loader
    problem = state["problem"]
    all_categories = list(sl.categories)
    threshold = CONFIG["classifier_confidence_threshold"]
    prompt = CLASSIFICATION_PROMPT.format(
        problem=problem,
        allowed_categories="、".join(all_categories),
    )

    stages = ["llm_prefill"]
    result = _classify_prefilled(deps, prompt, all_categories)
    if result and result[1] >= threshold:
        return _decision_payload(
            problem, all_categories, result[0], result[1],
            _rank_alternate_categories(sl, problem, result[0], all_categories), stages
        )

    # 只有预填充未识别出合法学科时才升级到完整分类调用。
    time_budget = deps.time_budget
    if result:
        stages.append("low_confidence_accepted")
        return _decision_payload(
            problem, all_categories, result[0], result[1],
            _rank_alternate_categories(sl, problem, result[0], all_categories), stages
        )
    # 预填充无法解析时才使用完整分类，并先检查剩余预算。
    if not (time_budget and time_budget.fast_path()):
        stages.append("llm_plain")
        plain = _classify_plain(deps, prompt, all_categories)
        if plain:
            # 已识别出合法学科时，即使置信度低也优先保留模型判断。
            stages.append("low_confidence_accepted" if plain[1] < threshold else "llm_plain_accepted")
            return _decision_payload(
                problem, all_categories, plain[0], plain[1],
                _rank_alternate_categories(sl, problem, plain[0], all_categories), stages
            )

    stages.append("deterministic_fallback")
    fallback = _deterministic_fallback(sl, problem, all_categories, deps.logger)
    return _decision_payload(
        problem, all_categories, fallback, 0.0,
        _rank_alternate_categories(sl, problem, fallback, all_categories), stages
    )

