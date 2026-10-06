"""Fixed generated-program entry point and machine-readable output."""
import ast
import json
import re

RESULT_MARKER = "MULTIFACE_RESULT:"
DRIVER_MARKER = "# MULTIFACE_FIXED_DRIVER"
PYTHON_CONTRACT = '''
[程序接口，优先于参考脚本中的旧输出格式]
只定义无参数 solve()（独立求解）或 verify()（验证），不要自己调用入口或打印最终答案。
返回 dict，必有 answer、checks、evidence。answer 只含题目所求，诊断写 evidence。
checks 是列表，每项包含 passed（Python bool）、description、target、relation、scope。
target: premise / reasoning_candidate / python_candidate / program。
relation: support / contradict / inconclusive，表示检查结果对 target 的关系。
scope: exact / universal / counterexample / finite_sample / heuristic。
反例推翻题干而支持“错误”时，target=premise、relation=contradict、scope=counterexample；
不能把反驳题干当作反驳推理候选。针对候选时必须附 candidate_answer 原文。
有限样本符合只表示有限样本；不得声明全称证明。无可执行检查时 checks=[]。
不得构造评分权重、硬编码布尔值或术语列表冒充数学验证。
固定驱动负责调用、校验返回结构和打印；所有数学结果必须由本程序计算得出。
最小结构示例（只示范接口，必须替换为本题实际计算）：
def solve():
    value = sum(range(4))
    expected = 3 * 4 // 2
    return {"answer": str(value), "checks": [{"passed": bool(value == expected),
        "description": "独立恒等式核对", "target": "python_candidate",
        "relation": "support", "scope": "exact", "candidate_answer": str(value)}],
        "evidence": {"computed_value": value}}
'''

_DRIVER = '''
# MULTIFACE_FIXED_DRIVER
import json as _mf_json
_mf_result = ENTRYPOINT()
if not isinstance(_mf_result, dict):
    raise TypeError("solve()/verify() must return a dict")
if not {"answer", "checks", "evidence"}.issubset(_mf_result):
    raise ValueError("missing answer/checks/evidence")
if _mf_result["answer"] is None or not str(_mf_result["answer"]).strip():
    raise ValueError("answer must be nonempty")
if not isinstance(_mf_result["checks"], list):
    raise TypeError("checks must be a list")
for _mf_check in _mf_result["checks"]:
    if isinstance(_mf_check, dict):
        _mf_passed = _mf_check.get("passed")
        _mf_type = type(_mf_passed)
        if ((_mf_type.__module__.startswith("numpy") and _mf_type.__name__ == "bool_")
                or (_mf_type.__module__.startswith("sympy") and _mf_type.__name__ in ("BooleanTrue", "BooleanFalse"))):
            _mf_check["passed"] = bool(_mf_passed)
    if not isinstance(_mf_check, dict) or not isinstance(_mf_check.get("passed"), bool):
        raise TypeError("each check must contain a Python bool passed")
    for _mf_key, _mf_allowed in (
        ("target", ("premise", "reasoning_candidate", "python_candidate", "program")),
        ("relation", ("support", "contradict", "inconclusive")),
        ("scope", ("exact", "universal", "counterexample", "finite_sample", "heuristic")),
    ):
        if _mf_check.get(_mf_key) not in _mf_allowed:
            raise ValueError("invalid check " + _mf_key)
    if not str(_mf_check.get("description", "")).strip():
        raise ValueError("check description is required")
_mf_answer = _mf_result["answer"]
if isinstance(_mf_answer, dict):
    _mf_answer = _mf_answer.get("final_answer", _mf_answer.get("full_answer"))
    if _mf_answer is None or isinstance(_mf_answer, dict):
        raise TypeError("answer dict must explicitly contain final_answer or full_answer")
if isinstance(_mf_answer, (list, tuple)):
    _mf_answer = "; ".join(str(part) for part in _mf_answer)
_mf_result["answer"] = str(_mf_answer).strip()
print("MULTIFACE_RESULT:", _mf_json.dumps(_mf_result, ensure_ascii=False, default=str))
print("最终答案:", _mf_result["answer"])
'''


class _GroebnerRepair(ast.NodeTransformer):
    def __init__(self, symbol_names=()):
        self.repairs = []
        self.symbol_names = set(symbol_names)

    def visit_Call(self, node):
        self.generic_visit(node)
        name = node.func.attr if isinstance(node.func, ast.Attribute) else ""
        # Only repair unambiguously separated expression arguments before symbols.
        if name == "groebner" and len(node.args) >= 3:
            count = 0
            for arg in node.args:
                if isinstance(arg, ast.BinOp) or (
                        isinstance(arg, ast.Name) and self.symbol_names and arg.id not in self.symbol_names):
                    count += 1
                else:
                    break
            if count >= 2 and count < len(node.args) and all(
                    isinstance(arg, ast.Name) for arg in node.args[count:]):
                node.args = [ast.List(elts=node.args[:count], ctx=ast.Load())] + node.args[count:]
                self.repairs.append("groebner_polynomials_wrapped_in_list")
        return node


def prepare_verification_program(source):
    """Compile first, apply narrow API repairs, then attach one fixed driver."""
    source = str(source or "").split(DRIVER_MARKER, 1)[0].rstrip()
    tree = ast.parse(source)
    symbol_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            name = node.value.func.attr if isinstance(node.value.func, ast.Attribute) else getattr(node.value.func, "id", "")
            if name in {"symbols", "Symbol"}:
                for target in node.targets:
                    symbol_names.update(n.id for n in ast.walk(target) if isinstance(n, ast.Name))
    repair = _GroebnerRepair(symbol_names)
    tree = ast.fix_missing_locations(repair.visit(tree))
    if repair.repairs:
        source = ast.unparse(tree)
    entries = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {"solve", "verify"}]
    if len(entries) != 1:
        raise ValueError("define exactly one solve() or verify() entry point")
    entry = entries[0]
    if entry.args.posonlyargs or entry.args.args or entry.args.kwonlyargs or entry.args.vararg or entry.args.kwarg:
        raise ValueError("solve()/verify() must take no arguments")
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)):
            continue
        if any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
               and n.func.id == entry.name for n in ast.walk(node)):
            raise ValueError("entry point must only be called by the fixed driver")
    compile(source, "<verification>", "exec")
    return source + "\n" + _DRIVER.replace("ENTRYPOINT", entry.name), repair.repairs


def read_program_result(stdout):
    for line in reversed(str(stdout or "").splitlines()):
        if line.startswith(RESULT_MARKER):
            try:
                result = json.loads(line[len(RESULT_MARKER):].strip())
            except (ValueError, TypeError):
                return None
            if isinstance(result, dict) and isinstance(result.get("checks"), list):
                return result
    return None


def contract_answer_is_computed(source):
    """Conservative dependency check; constant answer dictionaries are not computation."""
    try:
        tree = ast.parse(str(source).split(DRIVER_MARKER, 1)[0])
    except SyntaxError:
        return False
    assignments = {}
    functions = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assignments[target.id] = node.value
    mathematical_calls = re.compile(r"sin|cos|tan|sqrt|solve|simplify|integrate|diff|limit|"
                                    r"det|rank|eigen|sum|prod|comb|factor|expand|roots|groebner|"
                                    r"rref|norm|mean|var|std|dot|count|brute|exact_small|"
                                    r"^(?:sf|cdf|pmf|pdf|ppf|isf|log|exp|loggamma)$", re.I)
    def computed(node, seen=None):
        seen = set(seen or ())
        if isinstance(node, ast.Name) and node.id in assignments and node.id not in seen:
            return computed(assignments[node.id], seen | {node.id})
        if isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            if mathematical_calls.search(name):
                return True
            # A locally named solver may return an actual computed value. Follow
            # its return dependencies, rather than trusting its name or unrelated
            # calculations elsewhere in the function.
            if name in functions and 'function:'+name not in seen:
                return any(computed(part.value, seen | {'function:'+name})
                           for part in ast.walk(functions[name]) if isinstance(part, ast.Return))
        if isinstance(node, (ast.BinOp, ast.Compare)) and any(
                isinstance(n, ast.Name) for n in ast.walk(node)):
            return True
        return any(computed(child, seen) for child in ast.iter_child_nodes(node))
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value == "answer" and computed(value):
                    return True
    return False
