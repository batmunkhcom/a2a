"""Tests for structured JSON logging."""

import json
import logging

from a2a.utils.logging import JSONFormatter, setup_logging


def test_json_formatter_produces_valid_json() -> None:
    fmt = JSONFormatter()
    record = logging.LogRecord(
        name="test.logger",
        level=logging.INFO,
        pathname="test.py",
        lineno=1,
        msg="test message",
        args=(),
        exc_info=None,
    )
    output = fmt.format(record)
    parsed = json.loads(output)
    assert parsed["level"] == "INFO"
    assert parsed["message"] == "test message"
    assert "timestamp" in parsed


def test_json_formatter_with_exception() -> None:
    fmt = JSONFormatter()
    try:
        raise ValueError("boom")
    except ValueError:
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="error occurred",
            args=(),
            exc_info=Exception(),
        )
        # Use a real exception info from sys.exc_info()
        import sys
        record.exc_info = sys.exc_info()
        output = fmt.format(record)
        parsed = json.loads(output)
        assert "exception" in parsed


def test_setup_logging_configures_root() -> None:
    setup_logging("DEBUG")
    logger = logging.getLogger("test_setup")
    assert logger.isEnabledFor(logging.DEBUG)


def test_json_formatter_fields() -> None:
    fmt = JSONFormatter()
    record = logging.LogRecord(
        name="a2a.transport",
        level=logging.INFO,
        pathname="x",
        lineno=1,
        msg="sent tensor",
        args=(),
        exc_info=None,
    )
    output = fmt.format(record)
    parsed = json.loads(output)
    assert parsed["logger"] == "a2a.transport"
    assert parsed["level"] == "INFO"
