"""Prometheus metrics for A2A runtime.

Tracks tensor throughput, latency, plugin health, and transport metrics.
"""

from __future__ import annotations

import time
from threading import Lock


class MetricsRegistry:
    """Simple metrics collector compatible with Prometheus exposition format.

    Collects counters, histograms, and gauges for A2A observability.
    """

    def __init__(self) -> None:
        self._counters: dict[str, int] = {}
        self._histograms: dict[str, list[float]] = {}
        self._gauges: dict[str, float] = {}
        self._start_time = time.monotonic()
        self._lock = Lock()

    # ── Counters ────────────────────────────────────────────────

    def increment(self, name: str, value: int = 1) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + value

    def get_counter(self, name: str) -> int:
        return self._counters.get(name, 0)

    # ── Histograms ──────────────────────────────────────────────

    def observe(self, name: str, value: float) -> None:
        with self._lock:
            if name not in self._histograms:
                self._histograms[name] = []
            self._histograms[name].append(value)

    def get_histogram(self, name: str) -> dict[str, float]:
        values = self._histograms.get(name, [])
        if not values:
            return {"count": 0, "sum": 0.0, "avg": 0.0, "min": 0.0, "max": 0.0}
        return {
            "count": len(values),
            "sum": sum(values),
            "avg": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }

    # ── Gauges ──────────────────────────────────────────────────

    def set_gauge(self, name: str, value: float) -> None:
        with self._lock:
            self._gauges[name] = value

    def get_gauge(self, name: str) -> float:
        return self._gauges.get(name, 0.0)

    # ── Prometheus text format ──────────────────────────────────

    def render_text(self) -> str:
        """Render metrics in Prometheus text exposition format."""
        lines: list[str] = []

        # Helpers
        def _line(name: str, value: str, labels: str = "") -> str:
            if labels:
                return f"{name}{{{labels}}} {value}"
            return f"{name} {value}"

        # Counters
        for name, value in sorted(self._counters.items()):
            lines.append(f"# TYPE {name} counter")
            lines.append(_line(name, str(value)))

        # Histograms
        for name in sorted(self._histograms.keys()):
            stats = self.get_histogram(name)
            base = f"{name}"
            lines.append(f"# TYPE {base} histogram")
            lines.append(_line(f"{base}_count", str(stats["count"])))
            lines.append(_line(f"{base}_sum", f"{stats['sum']:.6f}"))
            lines.append(_line(f"{base}_bucket", str(stats["count"]), 'le="+Inf"'))

        # Gauges
        for name, value in sorted(self._gauges.items()):
            lines.append(f"# TYPE {name} gauge")
            lines.append(_line(name, f"{value:.6f}"))

        return "\n".join(lines) + "\n"

    # ── Predefined A2A metrics ──────────────────────────────────

    def record_tensor_sent(self, agent_id: str, label: str, size_bytes: int) -> None:
        self.increment(f"a2a_tensors_sent_total{{agent_id=\"{agent_id}\",label=\"{label}\"}}")

    def record_tensor_received(self, agent_id: str, label: str) -> None:
        self.increment(f"a2a_tensors_received_total{{agent_id=\"{agent_id}\",label=\"{label}\"}}")

    def record_latency(self, agent_id: str, label: str, latency_s: float) -> None:
        key = f"a2a_tensor_latency_seconds{{agent_id=\"{agent_id}\",label=\"{label}\"}}"
        self.observe(key, latency_s)

    def record_plugin_active(self, count: int) -> None:
        self.set_gauge("a2a_plugin_active", float(count))

    def record_rate_limit_hit(self, agent_id: str) -> None:
        self.increment(f"a2a_rate_limit_hits_total{{agent_id=\"{agent_id}\"}}")

    def record_backpressure_event(self, agent_id: str, status: str) -> None:
        self.increment(f"a2a_backpressure_events_total{{agent_id=\"{agent_id}\",status=\"{status}\"}}")


# Global registry instance
_metrics = MetricsRegistry()


def get_metrics() -> MetricsRegistry:
    """Return the global metrics registry."""
    return _metrics


__all__ = ["MetricsRegistry", "get_metrics"]
