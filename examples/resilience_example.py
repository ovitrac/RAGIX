#!/usr/bin/env python3
"""
RAGIX Resilience Example - Error Recovery and Fault Tolerance

This example demonstrates RAGIX v0.7 resilience patterns:
- Retry with exponential backoff
- Circuit breaker
- Rate limiting
- Graceful degradation
- Fallback chains

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-25
"""

import asyncio
import random
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ragix_core import (
    # Retry
    retry_async,
    RetryConfig,
    BackoffStrategy,
    with_retry,
    with_timeout,
    # Circuit Breaker
    CircuitBreaker,
    # Rate Limiting
    RateLimiter,
    # Degradation
    GracefulDegradation,
    FallbackChain,
    Bulkhead,
)


# Simulated unreliable service
class UnreliableService:
    """Simulates a service that sometimes fails."""

    def __init__(self, failure_rate: float = 0.5):
        self.failure_rate = failure_rate
        self.call_count = 0

    async def call(self):
        self.call_count += 1
        await asyncio.sleep(0.1)

        if random.random() < self.failure_rate:
            raise ConnectionError(f"Service unavailable (call {self.call_count})")

        return f"Success (call {self.call_count})"


async def demonstrate_retry():
    """Show retry with backoff."""
    print("=" * 60)
    print("RAGIX v0.7 - Retry with Backoff")
    print("=" * 60)

    service = UnreliableService(failure_rate=0.6)

    @retry_async(RetryConfig(
        max_attempts=5,
        base_delay=0.5,
        strategy=BackoffStrategy.EXPONENTIAL_JITTER,
    ))
    async def call_with_retry():
        return await service.call()

    print("\nCalling unreliable service with retry...")
    print("-" * 40)

    try:
        result = await call_with_retry()
        print(f"  Result: {result}")
    except Exception as e:
        print(f"  Failed after retries: {e}")

    print(f"  Total calls made: {service.call_count}")


async def demonstrate_circuit_breaker():
    """Show circuit breaker pattern."""
    print("\n" + "=" * 60)
    print("RAGIX v0.7 - Circuit Breaker")
    print("=" * 60)

    breaker = CircuitBreaker(
        failure_threshold=3,
        recovery_timeout=2.0,
        half_open_requests=2,
    )

    service = UnreliableService(failure_rate=0.8)  # High failure rate

    print("\nSimulating failures to trip circuit breaker...")
    print("-" * 40)

    for i in range(10):
        state = breaker.state
        print(f"  [{i+1}] Circuit state: {state}", end=" -> ")

        if not breaker.is_allowed():
            print("Request blocked (circuit open)")
            await asyncio.sleep(0.5)
            continue

        try:
            result = await service.call()
            breaker.record_success()
            print(f"Success")
        except Exception as e:
            breaker.record_failure()
            print(f"Failed: {e}")

        await asyncio.sleep(0.3)

    print(f"\n  Final circuit state: {breaker.state}")


async def demonstrate_rate_limiting():
    """Show rate limiting."""
    print("\n" + "=" * 60)
    print("RAGIX v0.7 - Rate Limiting")
    print("=" * 60)

    # 2 requests per second, burst of 3
    limiter = RateLimiter(rate=2.0, burst=3)

    print("\nSending 10 requests with rate limiting (2/sec)...")
    print("-" * 40)

    for i in range(10):
        acquired = await limiter.acquire_async(timeout=0.5)
        if acquired:
            print(f"  [{i+1}] Request allowed (tokens: {limiter.available_tokens:.1f})")
        else:
            print(f"  [{i+1}] Request rate limited")

        await asyncio.sleep(0.2)


async def demonstrate_graceful_degradation():
    """Show graceful degradation."""
    print("\n" + "=" * 60)
    print("RAGIX v0.7 - Graceful Degradation")
    print("=" * 60)

    primary_failures = 0

    async def primary_operation():
        nonlocal primary_failures
        primary_failures += 1
        if primary_failures <= 3:
            raise ConnectionError("Primary service down")
        return "Response from primary"

    async def fallback_operation():
        return "Cached/fallback response"

    degradation = GracefulDegradation(
        primary=primary_operation,
        fallback=fallback_operation,
        failure_threshold=2,
        recovery_time=1.0,
    )

    print("\nSimulating primary service failures...")
    print("-" * 40)

    for i in range(6):
        result = await degradation.execute()
        using = "fallback" if degradation._using_fallback else "primary"
        print(f"  [{i+1}] Using {using}: {result}")
        await asyncio.sleep(0.5)


async def demonstrate_bulkhead():
    """Show bulkhead pattern for concurrency limiting."""
    print("\n" + "=" * 60)
    print("RAGIX v0.7 - Bulkhead (Concurrency Limiting)")
    print("=" * 60)

    bulkhead = Bulkhead(max_concurrent=3)

    async def worker(id: int):
        async with bulkhead:
            print(f"    Worker {id} started (active: {bulkhead.active})")
            await asyncio.sleep(0.5)
            print(f"    Worker {id} finished (active: {bulkhead.active})")

    print("\nRunning 6 workers with max 3 concurrent...")
    print("-" * 40)

    # Start all workers
    await asyncio.gather(*[worker(i) for i in range(1, 7)])

    print(f"\n  All workers completed")


async def demonstrate_fallback_chain():
    """Show fallback chain pattern."""
    print("\n" + "=" * 60)
    print("RAGIX v0.7 - Fallback Chain")
    print("=" * 60)

    def primary_api():
        raise ConnectionError("Primary API down")

    def secondary_api():
        raise ConnectionError("Secondary API down")

    def cached_result():
        return "Cached data from last successful call"

    chain = FallbackChain(
        operations=[primary_api, secondary_api, cached_result],
        on_fallback=lambda i, e: print(f"    Fallback {i} failed: {e}"),
    )

    print("\nTrying fallback chain...")
    print("-" * 40)

    try:
        result = chain.execute()
        print(f"  Result: {result}")
    except Exception as e:
        print(f"  All fallbacks failed: {e}")


async def main():
    """Run all demonstrations."""
    await demonstrate_retry()
    await demonstrate_circuit_breaker()
    await demonstrate_rate_limiting()
    await demonstrate_graceful_degradation()
    await demonstrate_bulkhead()
    await demonstrate_fallback_chain()

    print("\n" + "=" * 60)
    print("Resilience demonstration complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
