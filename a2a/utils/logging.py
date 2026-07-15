"""Structured JSON logging for A2A runtime.

Provides a consistent logging format for observability tools.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any


class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if hasattr(record, "event"):
            log_entry["event"] = record.event  # type: ignore[attr-defined]

        for key in ("trace_id", "agent_id", "semantic_label", "latency_ms",
                     "tensor_shape", "tensor_dtype", "plugin_id"):
            if hasattr(record, key):
                log_entry[key] = getattr(record, key)

        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, default=str)


def setup_logging(
    level: str = "INFO",
    stream: Any = sys.stderr,
) -> None:
    """Configure root logger with JSON formatting.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR).
        stream: Output stream (default stderr).
    """
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JSONFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))


def get_logger(name: str) -> logging.Logger:
    """Get a named logger (already configured for JSON output)."""
    return logging.getLogger(name)


__all__ = ["JSONFormatter", "setup_logging", "get_logger"]
