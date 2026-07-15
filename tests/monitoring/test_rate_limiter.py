"""Tests for TokenBucket, RateLimiter, and BackpressureController."""

import time

from a2a.monitoring.rate_limiter import (
    BackpressureController,
    RateLimiter,
    TokenBucket,
)


def test_token_bucket_consumes_token() -> None:
    bucket = TokenBucket(bucket_size=5, refill_rate=100.0)
    assert bucket.allow() is True
    assert bucket.tokens < 5.0


def test_token_bucket_refills_over_time() -> None:
    bucket = TokenBucket(bucket_size=5, refill_rate=100.0)
    # Consume all tokens
    for _ in range(5):
        bucket.allow()
    assert bucket.tokens < 1.0

    # Wait a little
    time.sleep(0.05)
    assert bucket.tokens > 0.0


def test_token_bucket_denies_when_empty() -> None:
    bucket = TokenBucket(bucket_size=2, refill_rate=0.1)  # very slow refill
    assert bucket.allow() is True
    assert bucket.allow() is True
    assert bucket.allow() is False


def test_token_bucket_consume_multiple() -> None:
    bucket = TokenBucket(bucket_size=10, refill_rate=1000.0)
    assert bucket.consume(5) is True
    assert bucket.consume(3) is True
    assert bucket.consume(3) is False  # only 2 left


def test_rate_limiter_respects_agent_config() -> None:
    limiter = RateLimiter(default_bucket_size=5, default_refill_rate=10.0)
    limiter.configure_agent("slow-agent", bucket_size=2, refill_rate=1.0)

    # Default agent has 5 tokens
    assert limiter.allow("normal-agent") is True

    # Slow agent only has 2
    limiter.allow("slow-agent")
    limiter.allow("slow-agent")
    assert limiter.allow("slow-agent") is False


def test_rate_limiter_route_limits() -> None:
    limiter = RateLimiter()
    limiter.configure_route("error_context", bucket_size=3, refill_rate=100.0)

    assert limiter.allow(route="error_context") is True
    assert limiter.allow(route="error_context") is True
    assert limiter.allow(route="error_context") is True
    assert limiter.allow(route="error_context") is False


def test_backpressure_normal() -> None:
    bp = BackpressureController(max_queue_depth=100, threshold=0.8)
    assert bp.status == "NORMAL"


def test_backpressure_slow_down() -> None:
    bp = BackpressureController(max_queue_depth=100, threshold=0.5)
    status = bp.enqueue(60)  # Above 50% threshold
    assert status == "SLOW_DOWN"


def test_backpressure_pause() -> None:
    bp = BackpressureController(max_queue_depth=100)
    bp.enqueue(100)
    assert bp.status == "PAUSE"


def test_backpressure_resume() -> None:
    bp = BackpressureController(max_queue_depth=100)
    bp.enqueue(50)
    status = bp.dequeue(50)
    assert status == "RESUME"
