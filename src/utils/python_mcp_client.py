"""Client for the python_executor MCP server (in-process call) + local fallback."""
import threading

from mcp_servers.python_executor import server as _server


_IN_PROCESS_FALLBACK_LOCK = threading.Lock()


class PythonMCPClient:
    """Calls execute_python (subprocess path = MCP tool) with exec-in-process fallback.

    The MCP server's tool function is called in-process (no MCP transport needed);
    this keeps the call synchronous and concurrency-safe (each call = independent
    subprocess). If the subprocess path raises, fall back to in-process exec.
    """

    def __init__(self, mcp_session=None):
        self.mcp_session = mcp_session

    def execute(self, code: str, timeout: int = 30) -> dict:
        def _fallback(backend: str) -> dict:
            with _IN_PROCESS_FALLBACK_LOCK:
                result = _server._exec_in_process(code, timeout)
            result["execution_backend"] = backend
            return result

        try:
            result = _server.execute_python(code, timeout)
            startup_error = str(result.get("stderr") or "")
            if (not result.get("success")
                    and "Py_HashRandomization_Init" in startup_error):
                return _fallback("local_fallback_hash_seed")
            result["execution_backend"] = "mcp"
            return result
        except Exception:
            return _fallback("local_fallback")

