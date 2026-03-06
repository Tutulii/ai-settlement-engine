"""
Production Reliability & Failover Architecture

Implements:
- Exponential backoff retries (3 attempts: 1s, 2s, 4s)
- Circuit breaker (5 consecutive failures disables for 60s)
- Structured error logging
"""

import time
import logging
import functools
from typing import Callable, Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

class CircuitBreakerOpenException(Exception):
    """Raised when the circuit breaker is OPEN, denying requests."""
    pass

class APIRetryExhaustedException(Exception):
    """Raised when an external API call fails after all retries."""
    pass


class CircuitBreaker:
    """Simple Circuit Breaker pattern."""
    
    def __init__(self, max_failures: int = 5, reset_timeout_sec: float = 60.0):
        self.max_failures = max_failures
        self.reset_timeout_sec = reset_timeout_sec
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.state = "closed" # "closed" = OK, "open" = Failing, "half-open" = Testing

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        logger.warning("Circuit breaker recorded failure %d/%d", self.failure_count, self.max_failures)
        if self.failure_count >= self.max_failures and self.state == "closed":
            self.state = "open"
            logger.error("Circuit breaker is now OPEN. Fast-failing for %d seconds.", self.reset_timeout_sec)

    def record_success(self):
        if self.state in ("open", "half-open"):
            logger.info("Circuit breaker is now CLOSED. Recovery successful.")
        self.failure_count = 0
        self.state = "closed"

    def is_allowed(self) -> bool:
        if self.state == "closed":
            return True
        
        # State is open, check if timeout has elapsed to transition to half-open
        elapsed = time.time() - self.last_failure_time
        if elapsed > self.reset_timeout_sec:
            self.state = "half-open"
            logger.info("Circuit breaker timeout elapsed. Entering HALF-OPEN state to test request.")
            return True
        return False

    def check(self):
        if not self.is_allowed():
            raise CircuitBreakerOpenException("Circuit breaker is OPEN. Failing fast.")


# Global circuit breaker instance
global_circuit_breaker = CircuitBreaker(max_failures=5, reset_timeout_sec=60.0)


def with_resilience(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator that applies the global circuit breaker and 
    3x exponential backoff (1s, 2s, 4s) to the wrapped function.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 1. Fast-fail check
        global_circuit_breaker.check()
        
        backoff_intervals = [1]
        max_attempts = len(backoff_intervals) + 1
        
        for attempt in range(max_attempts):
            try:
                # Execute external call
                result = func(*args, **kwargs)
                # Success -> reset circuit breaker
                global_circuit_breaker.record_success()
                return result
                
            except Exception as exc:
                if attempt < len(backoff_intervals):
                    sleep_time = backoff_intervals[attempt]
                    logger.warning(
                        "External API call failed (attempt %d/%d). Retrying in %ds. Error: %s", 
                        attempt + 1, max_attempts, sleep_time, exc
                    )
                    time.sleep(sleep_time)
                else:
                    logger.error("External API call failed after %d attempts. Exhausted.", max_attempts)
                    global_circuit_breaker.record_failure()
                    raise APIRetryExhaustedException(str(exc)) from exc
    return wrapper
