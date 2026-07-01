"""Tests for LLM inference retry-with-backoff behavior."""

import pytest
import backoff

from logdetective.server.exceptions import (
    LogDetectiveInferenceError,
    LogDetectiveInferenceTimeout,
    LogDetectiveInferenceRateLimit,
)
from logdetective.server.models import InferenceConfig
from logdetective.server.utils import inference_retry_backoff, inference_retry_giveup


@pytest.mark.asyncio
async def test_retry_on_timeout_succeeds():
    """Transient timeout on first call, success on second."""
    call_count = 0

    @backoff.on_exception(
        backoff.expo,
        (LogDetectiveInferenceTimeout, LogDetectiveInferenceRateLimit),
        max_tries=3,
        max_time=60,
        on_backoff=inference_retry_backoff,
        on_giveup=inference_retry_giveup,
    )
    async def retryable():
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise LogDetectiveInferenceTimeout("temporary timeout")
        return "ok"

    result = await retryable()
    assert result == "ok"
    assert call_count == 2


@pytest.mark.asyncio
async def test_retry_on_rate_limit_succeeds():
    """Transient rate limit on first call, success on second."""
    call_count = 0

    @backoff.on_exception(
        backoff.expo,
        (LogDetectiveInferenceTimeout, LogDetectiveInferenceRateLimit),
        max_tries=3,
        max_time=60,
        on_backoff=inference_retry_backoff,
        on_giveup=inference_retry_giveup,
    )
    async def retryable():
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise LogDetectiveInferenceRateLimit("rate limited")
        return "ok"

    result = await retryable()
    assert result == "ok"
    assert call_count == 2


@pytest.mark.asyncio
async def test_no_retry_on_permanent_error():
    """LogDetectiveInferenceError propagates immediately without retry."""
    call_count = 0

    @backoff.on_exception(
        backoff.expo,
        (LogDetectiveInferenceTimeout, LogDetectiveInferenceRateLimit),
        max_tries=3,
        max_time=60,
        on_backoff=inference_retry_backoff,
        on_giveup=inference_retry_giveup,
    )
    async def retryable():
        nonlocal call_count
        call_count += 1
        raise LogDetectiveInferenceError("permanent error")

    with pytest.raises(LogDetectiveInferenceError, match="permanent error"):
        await retryable()
    assert call_count == 1


@pytest.mark.asyncio
async def test_retry_exhaustion_raises():
    """All retries exhausted — exception propagates."""
    call_count = 0

    @backoff.on_exception(
        backoff.expo,
        (LogDetectiveInferenceTimeout, LogDetectiveInferenceRateLimit),
        max_tries=3,
        max_time=60,
        on_backoff=inference_retry_backoff,
        on_giveup=inference_retry_giveup,
    )
    async def retryable():
        nonlocal call_count
        call_count += 1
        raise LogDetectiveInferenceTimeout("still timing out")

    with pytest.raises(LogDetectiveInferenceTimeout, match="still timing out"):
        await retryable()
    assert call_count == 3


@pytest.mark.asyncio
async def test_backoff_handler_called():
    """inference_retry_backoff is called when a retry occurs."""
    backoff_calls = []
    call_count = 0

    def tracking_backoff(details):
        backoff_calls.append(details)
        inference_retry_backoff(details)

    @backoff.on_exception(
        backoff.expo,
        (LogDetectiveInferenceTimeout,),
        max_tries=2,
        max_time=60,
        on_backoff=tracking_backoff,
    )
    async def retryable():
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise LogDetectiveInferenceTimeout("timeout")
        return "ok"

    await retryable()
    assert len(backoff_calls) == 1
    assert backoff_calls[0]["tries"] == 1


@pytest.mark.asyncio
async def test_giveup_handler_called():
    """inference_retry_giveup is called when retries are exhausted."""
    giveup_calls = []

    def tracking_giveup(details):
        giveup_calls.append(details)
        inference_retry_giveup(details)

    @backoff.on_exception(
        backoff.expo,
        (LogDetectiveInferenceTimeout,),
        max_tries=2,
        max_time=60,
        on_giveup=tracking_giveup,
    )
    async def retryable():
        raise LogDetectiveInferenceTimeout("timeout")

    with pytest.raises(LogDetectiveInferenceTimeout):
        await retryable()
    assert len(giveup_calls) == 1
    assert giveup_calls[0]["tries"] == 2


def test_config_defaults():
    """InferenceConfig has correct default retry values."""
    config = InferenceConfig()
    assert config.retry_max_tries == 3
    assert config.retry_max_time == 120


def test_config_custom_values():
    """InferenceConfig accepts custom retry values."""
    config = InferenceConfig(retry_max_tries=5, retry_max_time=300)
    assert config.retry_max_tries == 5
    assert config.retry_max_time == 300
