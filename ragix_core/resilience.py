"""
Resilience - Error recovery, retries, and fault tolerance

Provides resilience patterns for robust agent execution.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-25
"""

import asyncio
import functools
import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, List, Optional, Type, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar("T")


class BackoffStrategy(str, Enum):
    """Backoff strategy for retries."""
    CONSTANT = "constant"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    EXPONENTIAL_JITTER = "exponential_jitter"


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL_JITTER
    retry_exceptions: tuple = (Exception,)
    exclude_exceptions: tuple = ()
    on_retry: Optional[Callable[[int, Exception], None]] = None


def calculate_delay(
    attempt: int,
    config: RetryConfig,
) -> float:
    """
    Calculate delay before next retry.

    Args:
        attempt: Current attempt number (1-based)
        config: Retry configuration

    Returns:
        Delay in seconds
    """
    if config.strategy == BackoffStrategy.CONSTANT:
        delay = config.base_delay

    elif config.strategy == BackoffStrategy.LINEAR:
        delay = config.base_delay * attempt

    elif config.strategy == BackoffStrategy.EXPONENTIAL:
        delay = config.base_delay * (2 ** (attempt - 1))

    elif config.strategy == BackoffStrategy.EXPONENTIAL_JITTER:
        base_delay = config.base_delay * (2 ** (attempt - 1))
        # Add random jitter (0.5 to 1.5 of base)
        delay = base_delay * (0.5 + random.random())

    else:
        delay = config.base_delay

    return min(delay, config.max_delay)


def should_retry(
    exception: Exception,
    config: RetryConfig,
) -> bool:
    """
    Check if exception should trigger retry.

    Args:
        exception: The exception that occurred
        config: Retry configuration

    Returns:
        True if should retry
    """
    # Check exclusions first
    if config.exclude_exceptions and isinstance(exception, config.exclude_exceptions):
        return False

    # Check if in retry list
    return isinstance(exception, config.retry_exceptions)


def retry(config: Optional[RetryConfig] = None):
    """
    Decorator for retrying functions.

    Args:
        config: Retry configuration (uses defaults if not provided)

    Example:
        @retry(RetryConfig(max_attempts=5))
        def unstable_operation():
            ...
    """
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None

            for attempt in range(1, config.max_attempts + 1):
                try:
                    return func(*args, **kwargs)

                except Exception as e:
                    last_exception = e

                    if not should_retry(e, config):
                        logger.debug(f"Not retrying {func.__name__}: {e}")
                        raise

                    if attempt >= config.max_attempts:
                        logger.warning(
                            f"Max retries ({config.max_attempts}) exceeded for {func.__name__}"
                        )
                        raise

                    delay = calculate_delay(attempt, config)
                    logger.info(
                        f"Retry {attempt}/{config.max_attempts} for {func.__name__} "
                        f"in {delay:.2f}s: {e}"
                    )

                    if config.on_retry:
                        config.on_retry(attempt, e)

                    time.sleep(delay)

            raise last_exception

        return wrapper

    return decorator


def retry_async(config: Optional[RetryConfig] = None):
    """
    Decorator for retrying async functions.

    Args:
        config: Retry configuration
    """
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exception = None

            for attempt in range(1, config.max_attempts + 1):
                try:
                    return await func(*args, **kwargs)

                except Exception as e:
                    last_exception = e

                    if not should_retry(e, config):
                        raise

                    if attempt >= config.max_attempts:
                        raise

                    delay = calculate_delay(attempt, config)
                    logger.info(
                        f"Retry {attempt}/{config.max_attempts} for {func.__name__} "
                        f"in {delay:.2f}s: {e}"
                    )

                    if config.on_retry:
                        config.on_retry(attempt, e)

                    await asyncio.sleep(delay)

            raise last_exception

        return wrapper

    return decorator


class RetryContext:
    """
    Context manager for retry logic.

    Provides more control over retry behavior.

    Example:
        async with RetryContext(max_attempts=3) as ctx:
            for attempt in ctx:
                try:
                    result = await risky_operation()
                    break
                except Exception as e:
                    ctx.record_failure(e)
    """

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL_JITTER,
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.strategy = strategy
        self.attempt = 0
        self.last_exception: Optional[Exception] = None
        self._config = RetryConfig(
            max_attempts=max_attempts,
            base_delay=base_delay,
            strategy=strategy,
        )

    def __iter__(self):
        return self

    def __next__(self) -> int:
        self.attempt += 1
        if self.attempt > self.max_attempts:
            if self.last_exception:
                raise self.last_exception
            raise StopIteration
        return self.attempt

    def record_failure(self, exception: Exception):
        """Record a failure for potential retry."""
        self.last_exception = exception

    async def delay(self):
        """Wait before next retry."""
        if self.attempt < self.max_attempts:
            delay = calculate_delay(self.attempt, self._config)
            await asyncio.sleep(delay)

    def delay_sync(self):
        """Wait before next retry (sync version)."""
        if self.attempt < self.max_attempts:
            delay = calculate_delay(self.attempt, self._config)
            time.sleep(delay)


@dataclass
class FallbackChain:
    """
    Chain of fallback operations.

    Tries operations in order until one succeeds.

    Example:
        chain = FallbackChain([
            lambda: primary_api(),
            lambda: secondary_api(),
            lambda: cached_result(),
        ])
        result = chain.execute()
    """
    operations: List[Callable[[], Any]]
    on_fallback: Optional[Callable[[int, Exception], None]] = None

    def execute(self) -> Any:
        """Execute operations until one succeeds."""
        last_exception = None

        for i, operation in enumerate(self.operations):
            try:
                return operation()
            except Exception as e:
                last_exception = e
                if self.on_fallback and i < len(self.operations) - 1:
                    self.on_fallback(i, e)
                logger.debug(f"Fallback {i} failed: {e}")

        if last_exception:
            raise last_exception
        raise RuntimeError("No operations in fallback chain")

    async def execute_async(self) -> Any:
        """Execute async operations until one succeeds."""
        last_exception = None

        for i, operation in enumerate(self.operations):
            try:
                result = operation()
                if asyncio.iscoroutine(result):
                    result = await result
                return result
            except Exception as e:
                last_exception = e
                if self.on_fallback and i < len(self.operations) - 1:
                    self.on_fallback(i, e)

        if last_exception:
            raise last_exception
        raise RuntimeError("No operations in fallback chain")


@dataclass
class Timeout:
    """
    Timeout wrapper for operations.

    Example:
        result = await Timeout(5.0).execute(async_operation())
    """
    seconds: float
    error_message: str = "Operation timed out"

    async def execute(self, coro) -> Any:
        """Execute coroutine with timeout."""
        try:
            return await asyncio.wait_for(coro, timeout=self.seconds)
        except asyncio.TimeoutError:
            raise TimeoutError(self.error_message)

    def execute_sync(self, func: Callable[[], T], *args, **kwargs) -> T:
        """Execute function with timeout using threading."""
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, *args, **kwargs)
            try:
                return future.result(timeout=self.seconds)
            except concurrent.futures.TimeoutError:
                raise TimeoutError(self.error_message)


class Bulkhead:
    """
    Bulkhead pattern for limiting concurrent executions.

    Prevents resource exhaustion by limiting concurrency.

    Example:
        bulkhead = Bulkhead(max_concurrent=5)
        async with bulkhead:
            await heavy_operation()
    """

    def __init__(self, max_concurrent: int = 10):
        """
        Initialize bulkhead.

        Args:
            max_concurrent: Maximum concurrent executions
        """
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._active = 0

    async def __aenter__(self):
        await self._semaphore.acquire()
        self._active += 1
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._active -= 1
        self._semaphore.release()

    @property
    def active(self) -> int:
        """Number of active executions."""
        return self._active

    @property
    def available(self) -> int:
        """Number of available slots."""
        return self.max_concurrent - self._active


class GracefulDegradation:
    """
    Graceful degradation handler.

    Provides fallback behavior when primary operation fails.

    Example:
        degradation = GracefulDegradation(
            primary=lambda: call_api(),
            fallback=lambda: cached_response(),
            health_check=lambda: api.is_available(),
        )
        result = await degradation.execute()
    """

    def __init__(
        self,
        primary: Callable,
        fallback: Callable,
        health_check: Optional[Callable[[], bool]] = None,
        failure_threshold: int = 3,
        recovery_time: float = 30.0,
    ):
        """
        Initialize graceful degradation.

        Args:
            primary: Primary operation
            fallback: Fallback operation
            health_check: Optional function to check primary availability
            failure_threshold: Failures before switching to fallback
            recovery_time: Seconds before trying primary again
        """
        self.primary = primary
        self.fallback = fallback
        self.health_check = health_check
        self.failure_threshold = failure_threshold
        self.recovery_time = recovery_time

        self._failures = 0
        self._last_failure: Optional[float] = None
        self._using_fallback = False

    def _should_try_primary(self) -> bool:
        """Check if should try primary operation."""
        if not self._using_fallback:
            return True

        if self._last_failure is None:
            return True

        if time.time() - self._last_failure > self.recovery_time:
            if self.health_check:
                try:
                    return self.health_check()
                except Exception:
                    return False
            return True

        return False

    async def execute(self) -> Any:
        """Execute with graceful degradation."""
        if self._should_try_primary():
            try:
                result = self.primary()
                if asyncio.iscoroutine(result):
                    result = await result

                # Success - reset failure state
                self._failures = 0
                self._using_fallback = False
                return result

            except Exception as e:
                self._failures += 1
                self._last_failure = time.time()

                if self._failures >= self.failure_threshold:
                    self._using_fallback = True
                    logger.warning(f"Switching to fallback after {self._failures} failures")

                logger.debug(f"Primary failed, trying fallback: {e}")

        # Use fallback
        result = self.fallback()
        if asyncio.iscoroutine(result):
            result = await result
        return result


def with_timeout(seconds: float, error_message: str = "Operation timed out"):
    """
    Decorator to add timeout to async functions.

    Args:
        seconds: Timeout in seconds
        error_message: Error message on timeout
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=seconds,
                )
            except asyncio.TimeoutError:
                raise TimeoutError(error_message)
        return wrapper
    return decorator


def with_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL_JITTER,
):
    """
    Convenience decorator for retries with common settings.

    Args:
        max_attempts: Maximum retry attempts
        base_delay: Base delay between retries
        strategy: Backoff strategy
    """
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        strategy=strategy,
    )
    return retry_async(config)


def with_fallback(fallback_fn: Callable):
    """
    Decorator to add fallback behavior.

    Args:
        fallback_fn: Function to call if primary fails
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                if asyncio.iscoroutine(result):
                    result = await result
                return result
            except Exception as e:
                logger.debug(f"Primary failed, using fallback: {e}")
                result = fallback_fn(*args, **kwargs)
                if asyncio.iscoroutine(result):
                    result = await result
                return result
        return wrapper
    return decorator
