import logging
from pathlib import Path

from config import CONFIG


def _formatter():
    try:
        from pythonjsonlogger.json import JsonFormatter

        return JsonFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    except Exception:
        return logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")


def get_logger(name: str = "math_agent") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, CONFIG.get("log_level", "INFO"), logging.INFO))
    logger.propagate = False
    if logger.handlers:
        return logger

    handler = logging.StreamHandler()
    handler.setFormatter(_formatter())
    logger.addHandler(handler)
    return logger


def get_log_dir() -> Path:
    return Path(CONFIG.get("log_dir", "logs"))
