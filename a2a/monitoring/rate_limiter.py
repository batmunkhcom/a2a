"""Rate limiter using Token Bucket algorithm.

Per-agent and per-route rate limiting for A2A tensor traffic.
"""

from __future__ import annotations

import time
from threading import Lock


class TokenBucket:
    """Thread-safe token bucket rate limiter.

    Args:
        bucket_size: Maximum burst size (tokens).
        refill_rate: Tokens per second to refill.
    """

    def __init__(self, bucket_size: int = 20, refill_rate: float = 100.0) -> None:
        self._bucket_size = float(bucket_size)
        self._refill_rate = float(refill_rate)
        self._tokens = float(bucket_size)
        self._last_refill = time.monotonic()
        self._lock = Lock()

    def allow(self) -> bool:
        """Check if one action is allowed (consumes 1 token).

        Returns:
            True if token available and consumed.
        """
        return self.consume(1)

    def consume(self, n: int = 1) -> bool:
        """Consume n tokens if available. Refills automatically.

        Returns:
            True if tokens were consumed.
        """
        with self._lock:
            self._refill()
            if self._tokens >= n:
                self._tokens -= n
                return True
            return False

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(
            self._bucket_size,
            self._tokens + elapsed * self._refill_rate,
        )
        self._last_refill = now

    @property
    def tokens(self) -> float:
        with self._lock:
            self._refill()
            return self._tokens

    @property
    def bucket_size(self) -> float:
        return self._bucket_size

    @property
    def refill_rate(self) -> float:
        return self._refill_rate


class RateLimiter:
    """Manages token buckets per agent and per route.

    Args:
        default_bucket_size: Default burst size.
        default_refill_rate: Default tokens/sec.
    """

    def __init__(
        self,
        default_bucket_size: int = 20,
        default_refill_rate: float = 100.0,
    ) -> None:
        self._default_size = default_bucket_size
        self._default_rate = default_refill_rate
        self._agent_buckets: dict[str, TokenBucket] = {}
        self._route_buckets: dict[str, TokenBucket] = {}
        self._lock = Lock()

    def configure_agent(
        self, agent_id: str, bucket_size: int, refill_rate: float
    ) -> None:
        """Set per-agent rate limit."""
        with self._lock:
            self._agent_buckets[agent_id] = TokenBucket(bucket_size, refill_rate)

    def configure_route(
        self, route: str, bucket_size: int, refill_rate: float
    ) -> None:
        """Set per-route rate limit."""
        with self._lock:
            self._route_buckets[route] = TokenBucket(bucket_size, refill_rate)

    def allow(self, agent_id: str | None = None, route: str | None = None) -> bool:
        """Check if request is allowed.

        Checks both agent-specific and route-specific limits.
        Both must allow for the request to pass.

        Args:
            agent_id: Optional agent identifier.
            route: Optional semantic route label.

        Returns:
            True if allowed by all applicable limits.
        """
        # Check agent bucket
        if agent_id:
            bucket = self._get_or_create_agent(agent_id)
            if not bucket.allow():
                return False

        # Check route bucket
        if route:
            bucket = self._get_or_create_route(route)
            if not bucket.allow():
                return False

        return True

    def _get_or_create_agent(self, agent_id: str) -> TokenBucket:
        with self._lock:
            if agent_id not in self._agent_buckets:
                self._agent_buckets[agent_id] = TokenBucket(
                    self._default_size, self._default_rate
                )
            return self._agent_buckets[agent_id]

    def _get_or_create_route(self, route: str) -> TokenBucket:
        with self._lock:
            if route not in self._route_buckets:
                self._route_buckets[route] = TokenBucket(
                    self._default_size, self._default_rate
                )
            return self._route_buckets[route]


class BackpressureController:
    """Flow control: signals sender to slow down when receiver is overloaded.

    Args:
        max_queue_depth: Queue depth that triggers backpressure.
        threshold: Fraction of max depth that triggers SLOW_DOWN signal.
    """

    def __init__(
        self, max_queue_depth: int = 500, threshold: float = 0.8
    ) -> None:
        self._max_depth = max_queue_depth
        self._threshold = threshold
        self._current_depth = 0
        self._status = "NORMAL"
        self._lock = Lock()

    @property
    def status(self) -> str:
        """Current flow control status: NORMAL, SLOW_DOWN, PAUSE, RESUME."""
        return self._status

    @property
    def queue_depth(self) -> int:
        return self._current_depth

    def enqueue(self, count: int = 1) -> str:
        """Register enqueued items and return flow control signal.

        Returns:
            Status string: "NORMAL", "SLOW_DOWN", or "PAUSE".
        """
        with self._lock:
            self._current_depth += count
            if self._current_depth >= self._max_depth:
                self._status = "PAUSE"
            elif self._current_depth >= int(self._max_depth * self._threshold):
                self._status = "SLOW_DOWN"
            else:
                self._status = "NORMAL"
            return self._status

    def dequeue(self, count: int = 1) -> str:
        """Register dequeued items and potentially signal RESUME.

        Returns:
            Status string.
        """
        with self._lock:
            self._current_depth = max(0, self._current_depth - count)
            if self._current_depth == 0:
                self._status = "RESUME"
            elif self._current_depth < int(self._max_depth * self._threshold):
                self._status = "NORMAL"
            return self._status
