"""The same problem constraints for all independently assembled prompts."""
import re
from utils.problem_profile import structure_instruction


def terminology_ambiguities(problem, mode):
    text = str(problem or "")
    if mode == "fill" and re.search(r"(?:分散|离散)程度", text) and re.search(r"指标|度量", text) \
            and not re.search(r"平方|同量纲|量纲|单位|公式|标准差|方差|变异系数|极差", text):
        return ["unqualified_dispersion_measure: 方差、标准差等均为有效指标，严格答案键与数学合理性应分开记录"]
    return []


def problem_constraints(problem):
    return structure_instruction(problem) + (
        "\n[术语与来源约束]\n区分定义、基础方法、命名程序和教材特定口径。"
        "技能条目不是官方答案键，不得凭题号或无来源的固定模板认定唯一答案。"
        "解释标准记号时先采用其通常定义域和常规课程约定；区分对象本身成立所需的缺省前提"
        "与结论另需的额外假设。不能仅因题面省略缺省前提就改变积分、范数、概率事件等对象的含义。"
        "确有不同合理口径时明确区分，优先回答题面所用记号的通常含义。"
        "最终答案中必须定义新引入的符号，保留参数范围、排除值和全部所求分量；"
        "不要让答案依赖未提交的草稿或隐藏步骤。"
        "概率建模先写出事件及量词：存在至少一个对象发生事件、预先指定对象发生事件、"
        "随机抽取对象发生事件是不同问题；不能把存在陈述擅自改成指定对象的条件概率，"
        "也不能引入题面未给出的随机到达或分配顺序。"
        "专业方法名称可以指算法、程序或传统计算方法，按题面上下文与可靠来源解释；"
        "不能无依据只允许其中某一层级，也不能把可替代对象强称为同义词。"
        "题目宽泛询问可用方法时，实际求解算法也可是一种方法；"
        "只有明确询问估计准则、模型类别等狭窄概念时，才依该限定排除其他层级。"
    )
