"""评测兼容入口。

实际实现位于 ``src/agent/user_agent.py``；保留本文件是为了兼容平台
直接加载仓库根目录 ``user_agent.py`` 的调用方式。
"""

from pathlib import Path
import sys


_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# 评测器可能预先缓存同名模块；清除来源不在本项目 src/ 下的缓存，避免导入冲突。
_INTERNAL_PACKAGES = ("agent", "graph", "nodes", "state", "utils", "mcp_servers")
for _name in _INTERNAL_PACKAGES:
    _module = sys.modules.get(_name)
    _origin = getattr(_module, "__file__", None) if _module is not None else None
    _owned = False
    if _origin:
        try:
            _owned = Path(_origin).resolve().is_relative_to(_SRC.resolve())
        except (OSError, ValueError):
            _owned = False
    if _module is not None and not _owned:
        for _loaded_name in list(sys.modules):
            if _loaded_name == _name or _loaded_name.startswith(_name + "."):
                del sys.modules[_loaded_name]

from agent.user_agent import ReasoningAgent as _ReasoningAgent


class ReasoningAgent(_ReasoningAgent):
    """平台要求的根入口类。"""

    pass

__all__ = ["ReasoningAgent"]
