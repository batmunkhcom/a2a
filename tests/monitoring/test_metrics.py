"""Tests for Prometheus metrics registry."""

from a2a.monitoring.metrics import MetricsRegistry, get_metrics


def test_counter_increment() -> None:
    reg = MetricsRegistry()
    reg.increment("test_counter")
    assert reg.get_counter("test_counter") == 1
    reg.increment("test_counter", 5)
    assert reg.get_counter("test_counter") == 6


def test_histogram_observe() -> None:
    reg = MetricsRegistry()
    reg.observe("latency", 0.01)
    reg.observe("latency", 0.05)
    reg.observe("latency", 0.02)

    stats = reg.get_histogram("latency")
    assert stats["count"] == 3
    assert stats["min"] == 0.01
    assert stats["max"] == 0.05


def test_gauge_set_get() -> None:
    reg = MetricsRegistry()
    reg.set_gauge("plugins_active", 5.0)
    assert reg.get_gauge("plugins_active") == 5.0


def test_render_text_format() -> None:
    reg = MetricsRegistry()
    reg.increment("test_counter", 42)
    reg.set_gauge("test_gauge", 3.14)

    text = reg.render_text()
    assert "test_counter" in text
    assert "42" in text
    assert "test_gauge" in text
    assert "3.14" in text


def test_record_tensor_sent() -> None:
    reg = MetricsRegistry()
    reg.record_tensor_sent("log-reader", "error_context", 8192)
    sent = reg.get_counter(
        'a2a_tensors_sent_total{agent_id="log-reader",label="error_context"}'
    )
    assert sent == 1


def test_record_latency() -> None:
    reg = MetricsRegistry()
    reg.record_latency("code-fixer", "error_context", 0.015)
    stats = reg.get_histogram(
        'a2a_tensor_latency_seconds{agent_id="code-fixer",label="error_context"}'
    )
    assert stats["count"] == 1
    assert stats["min"] == 0.015


def test_global_registry() -> None:
    reg = get_metrics()
    assert isinstance(reg, MetricsRegistry)
