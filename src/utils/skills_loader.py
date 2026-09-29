import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

from utils.skill_excerpt import _RETRIEVAL_STOPWORDS


#: 检索词行的英文判别词：确定性分类器（LLM 不可用时的兜底）用它们给英文题面打分。
#: 2026-08-22 无 API 全量观测：英文题下中文关键词命中恒为 0，而 TF-IDF 的
#: char-ngram 余弦对英文题只是文档词汇密度噪声——复分析/抽象代数两本文档英文
#: 最密，于是拿走了约一半题目的顶分（博弈/组合/数论题全被分进复分析）。
#: 各模块 ``- 检索词：`` 行里本来就写好了判别词（game/player/parity/coprime…），
#: 按词边界取出并做 ICF（逆类别频率）加权：minimum/number 这类跨类通用词权近 0，
#: parity/lebesgue/dual 这类判别词权重高。
_RETRIEVAL_LINE_RE = re.compile(r"^-\s*检索词[：:](.+)$", re.MULTILINE)
_LATIN_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z\-]{3,}")


def _latin_tokens_from_retrieval_lines(text: str) -> set:
    tokens = set()
    for m in _RETRIEVAL_LINE_RE.finditer(text):
        for token in m.group(1).split():
            tok = token.casefold()
            if _LATIN_TOKEN_RE.fullmatch(tok) and tok not in _RETRIEVAL_STOPWORDS:
                tokens.add(tok)
    return tokens


DOMAIN_ALIASES = {
    "概率论": [
        "依概率", "依分布", "几乎必然", "随机变量", "分布函数", "概率收敛",
        "中心极限定理", "大数定律", "Slutsky", "贝叶斯", "全概率",
        "二项分布", "泊松分布", "指数分布", "正态分布", "矩母函数", "特征函数",
    ],
    "随机过程": [
        "Brownian", "布朗", "Wiener", "首达时", "停时", "鞅", "Markov链",
        "马尔可夫", "Poisson过程", "泊松过程", "生灭过程", "更新过程", "随机游走",
    ],
    "数学分析": [
        "一致收敛", "函数列", "函数项级数", "逐点收敛", "极限函数", "可导",
        "导数列", "Taylor", "泰勒", "幂级数", "含参积分", "广义积分", "上确界",
        "Fourier变换", "Cauchy数列", "完备度量空间", "差商", "Laplacian on circle",
    ],
    "统计推断": [
        "最大似然", "MLE", "置信区间", "假设检验", "拒绝域", "显著性水平",
        "p值", "p-value", "样本", "估计量", "无偏", "方差估计",
        "极大似然", "矩估计", "Fisher", "C-R", "第二类错误", "功效",
        "似然比", "检验统计量", "临界值", "统计图形", "直方图", "散点图",
        "箱线图", "时间数列", "时间序列", "趋势变动", "季节变动", "移动平均",
        "指数平滑", "自相关图", "样本均值", "样本方差",
    ],
    "复分析": ["留数", "解析", "全纯", "Cauchy", "柯西", "Laurent", "洛朗", "极点"],
    "抽象代数": [
        "有限域", "群", "环", "理想", "同态", "子群", "正规子群", "域扩张",
        "Galois", "分裂域", "二面体群", "换位子群", "Sylow",
    ],
    "高等代数": [
        "矩阵", "特征值", "特征向量", "线性空间", "秩", "行列式", "二次型",
        "极小多项式", "对称多项式", "特征多项式", "二元域", "max-plus", "热带代数",
    ],
    "常微分方程": ["常微分", "ODE", "初值问题", "通解", "特解", "Wronskian"],
    "偏微分方程": [
        "偏微分", "PDE", "热方程", "波动方程", "Laplace方程", "边值问题",
        "形式伴随", "散度型", "formal adjoint", "divergence-form",
    ],
    "泛函分析": ["Banach", "Hilbert", "有界线性算子", "泛函", "弱收敛", "紧算子"],
    "拓扑学": ["拓扑", "开集", "闭集", "紧致", "连通", "同胚", "基本群"],
    "微分几何": ["流形", "曲率", "测地线", "联络", "切空间", "第一基本形式"],
    "数值分析": [
        "插值", "Newton", "迭代", "误差", "数值积分", "Runge-Kutta",
        "中心差分", "数值微分", "复化", "梯形公式", "Simpson", "Romberg",
        "Frobenius", "条件数", "Euler", "稳定区间", "Doolittle", "LU分解",
        "有限差分", "有限元", "有限体积", "截断误差", "finite difference",
        "finite element", "condition number", "numerical discretization",
    ],
    "测度积分": ["测度", "可测", "Lebesgue", "勒贝格", "几乎处处", "支配收敛", "Fatou"],
    "运筹学": [
        "线性规划", "单纯形", "对偶", "运输问题", "排队", "决策树", "指派问题",
        "整数规划", "分支定界", "最短路", "关键路径", "目标规划", "KKT",
        "调度", "资源分配", "linear programming", "simplex", "scheduling",
        "branch and bound", "assignment problem", "maximum flow",
    ],
    "离散数学": [
        "图", "树", "组合", "递推", "生成函数", "布尔", "命题逻辑", "数论",
        "整除", "素数", "丢番图", "同余", "鸽巢", "组合博弈", "拉丁方",
    ],
    "非基础及进阶课程": [
        "欧氏几何", "平面几何", "凸几何", "射影几何", "外心", "内心", "垂心",
        "角平分线", "外接圆", "内切圆", "根轴", "极点极线", "切弦角",
        "circumcenter", "orthocenter", "circumcircle", "radical axis", "polar line",
    ],
    "线性回归": [
        "回归", "最小二乘", "OLS", "残差", "t检验", "F检验", "R方",
        "\\beta", "标准误", "SSE", "SSR", "SST", "S_{xx}", "S_{xy}",
        "ANOVA", "方差分析表", "均值响应", "预测区间", "Gauss-Markov",
        "Durbin-Watson", "BLUE", "偏F", "多重共线性", "VIF", "异方差", "内生性",
        "工具变量", "非参数回归", "非线性回归", "残差图", "自相关",
    ],
}


DOMAIN_PRIORITY_TERMS = {
    "线性回归": [
        "回归", "OLS", "最小二乘", "\\beta", "标准误", "SSE", "SSR",
        "ANOVA", "方差分析表", "均值响应", "预测区间", "Gauss-Markov",
        "Durbin-Watson", "VIF", "BLUE", "偏F",
    ],
    "统计推断": [
        "统计推断", "极大似然", "最大似然", "MLE", "矩估计", "估计量",
        "置信区间", "假设检验", "显著性水平", "拒绝域", "p值", "p-value",
        "第二类错误", "功效", "C-R", "Fisher", "似然比", "直方图", "时间序列",
    ],
    "数值分析": [
        "数值分析", "数值积分", "数值微分", "中心差分", "复化", "梯形公式",
        "Simpson", "Romberg", "Frobenius", "条件数", "稳定区间",
        "显式 Euler", "Euler 法", "Doolittle", "LU分解", "插值", "有限差分",
        "有限元", "有限体积", "截断误差", "condition number", "finite difference",
    ],
    "运筹学": [
        "运筹学", "线性规划", "单纯形", "运输问题", "指派问题", "整数规划",
        "分支定界", "最短路", "关键路径", "排队模型", "目标规划",
    ],
    "偏微分方程": [
        "偏微分方程", "形式伴随", "散度型", "formal adjoint", "divergence-form",
        "热方程", "波动方程", "边值问题",
    ],
    "高等代数": [
        "高等代数", "极小多项式", "对称多项式", "特征多项式", "二元域",
        "max-plus", "热带代数", "特征值", "线性空间", "二次型",
    ],
    "非基础及进阶课程": [
        "欧氏几何", "平面几何", "外心", "内心", "垂心", "角平分线",
        "外接圆", "根轴", "极点极线", "circumcenter", "circumcircle",
    ],
}


def _domain_priority_boost(category: str, problem: str) -> float:
    """Large deterministic boost for domains with report-critical ambiguity."""
    text = problem or ""
    terms = DOMAIN_PRIORITY_TERMS.get(category, [])
    hits = sum(1 for term in terms if term and term in text)
    if not hits:
        return 0.0
    return 100.0 + hits * 5.0


class SkillsLoader:
    DEFAULT_BASE = Path(__file__).resolve().parents[2] / "resources" / "skills"

    def __init__(self, base_path: str = None):
        self.base_path = Path(base_path) if base_path else self.DEFAULT_BASE
        self.categories = self._scan_category_names()
        self.keywords_index = self._build_keywords_index()
        self._retrieval_tokens, self._retrieval_weights = self._build_retrieval_index()
        self._doc_cache: Dict[str, str] = {}
        self._script_cache: Dict[str, str] = {}
        self._embedding_index = None

    def _scan_category_names(self) -> List[str]:
        return sorted(p.name for p in self.base_path.iterdir() if p.is_dir())

    def _build_keywords_index(self) -> Dict[str, List[str]]:
        idx = {}
        for cat in self.categories:
            kw = set()
            kw.add(cat)
            kw.update(DOMAIN_ALIASES.get(cat, []))
            md = self.base_path / cat / f"{cat}skill.md"
            if md.exists():
                text = md.read_text(encoding="utf-8")
                for line in text.splitlines():
                    s = line.strip().lstrip("#").strip()
                    s = re.sub(r"^\d+\.\s*", "", s)
                    if 1 < len(s) <= 12 and not s.startswith("|"):
                        kw.add(s)
            idx[cat] = list(kw)
        return idx

    def _build_retrieval_index(self):
        """类别级英文检索词表 + ICF 权重。

        Returns:
            (tokens_by_category: Dict[str, set[str]], weights: Dict[str, float])
        """
        tokens_by_cat: Dict[str, set] = {}
        for cat in self.categories:
            md = self.base_path / cat / f"{cat}skill.md"
            if md.exists():
                tokens_by_cat[cat] = _latin_tokens_from_retrieval_lines(
                    md.read_text(encoding="utf-8"))
            else:
                tokens_by_cat[cat] = set()
        df: Dict[str, int] = {}
        for tokens in tokens_by_cat.values():
            for tok in tokens:
                df[tok] = df.get(tok, 0) + 1
        n = len(self.categories)
        weights = {
            tok: math.log((n + 1) / (df_count + 1))
            for tok, df_count in df.items()
        }
        return tokens_by_cat, weights

    def get_skill_document(self, category: str) -> str:
        if category not in self._doc_cache:
            f = self.base_path / category / f"{category}skill.md"
            self._doc_cache[category] = f.read_text(encoding="utf-8")
        return self._doc_cache[category]

    def get_validation_script(self, category: str) -> str:
        if category not in self._script_cache:
            f = self.base_path / category / f"{category}验证示例.py"
            self._script_cache[category] = f.read_text(encoding="utf-8")
        return self._script_cache[category]

    def find_candidate_categories(self, problem: str, top_k: int = 5) -> List[Tuple[str, float]]:
        scores = {}
        problem_words = set(
            re.findall(r"[a-z][a-z\-]*", (problem or "").casefold()))
        for cat, kws in self.keywords_index.items():
            hits = [kw for kw in kws if kw and kw in problem]
            alias_hits = [kw for kw in DOMAIN_ALIASES.get(cat, []) if kw and kw in problem]
            category_hit = 1.0 if cat in problem else 0.0
            # 英文判别词：只按词边界命中（cover 不得命中 coverage），ICF 加权。
            latin_score = sum(
                self._retrieval_weights.get(tok, 0.0)
                for tok in self._retrieval_tokens.get(cat, ())
                if tok in problem_words)
            scores[cat] = (len(hits) + len(alias_hits) + category_hit
                           + latin_score + _domain_priority_boost(cat, problem))
        return sorted(scores.items(), key=lambda x: (x[1], x[0]), reverse=True)[:top_k]

    def get_embedding_index(self):
        if self._embedding_index is None:
            from utils.category_embedding_index import CategoryEmbeddingIndex
            self._embedding_index = CategoryEmbeddingIndex(self)
        return self._embedding_index

