import logging
from dataclasses import dataclass, field
from typing import Any, Optional

@dataclass
class Deps:
    client: Any
    skills_loader: Any = None
    mcp_client: Any = None
    token_budget: Any = None
    time_budget: Any = None
    db_client: Any = None
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("math_agent"))

def get_deps(config: dict) -> "Deps":
    return config["configurable"]["deps"]

class MockClient:
    """Test client that returns preset responses in order."""
    def __init__(self, responses=None):
        self.responses = list(responses) if responses else []
        self.calls = []
    def chat(self, messages, temperature=0.2, max_tokens=4096):
        self.calls.append({"messages": messages, "temperature": temperature, "max_tokens": max_tokens})
        if self.responses:
            return self.responses.pop(0)
        return "mock response"

