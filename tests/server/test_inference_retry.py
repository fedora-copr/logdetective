"""Tests for LLM inference retry-with-backoff behavior."""

from unittest.mock import patch

import pytest
from tenacity import (
    retry,
    RetryCallState,
    retry_if_exception_type,
    stop_after_attempt,
    stop_any,
    stop_before_delay,
    wait_fixed,
    wait_exponential_jitter,
)

from logdetective.server.exceptions import (
    LogDetectiveInferenceError,
    LogDetectiveInferenceTimeout,
    LogDetectiveInferenceRateLimit,
)
from logdetective.server.models import InferenceConfig
from logdetective.server.utils import inference_retry_backoff, inference_retry_giveup


@pytest.mark.asyncio
async def test_stop_before_delay_exceeds_threshold():
    """Check that tenacity's stop_before_delay behavior works as expected."""
    current_time = 0.0
    call_count = 0

    async def fake_sleep(seconds):
        nonlocal current_time
        current_time += seconds

    @retry(
        stop=stop_before_delay(5),
        wait=wait_fixed(2),
        retry=retry_if_exception_type(ValueError),
        retry_error_callback=lambda _: "given_up",
    )
    async def retryable():
        nonlocal call_count
        call_count += 1
        raise ValueError("error")

    with (
        patch("asyncio.sleep", side_effect=fake_sleep),
        patch("time.monotonic", side_effect=lambda: current_time)
    ):
        result = await retryable()

    assert result == "given_up"
    assert call_count == 3  # only the calls at 0s, 2s, 4s happen
    assert int(current_time) == 4  # timer does not get updated after the last call


@pytest.mark.asyncio
async def test_retry_on_timeout_succeeds():
    """Transient timeout on first call, success on second."""
    call_count = 0

    @retry(
        stop=stop_any(stop_after_attempt(3), stop_before_delay(60)),
        wait=wait_exponential_jitter(),
        retry=retry_if_exception_type((
            LogDetectiveInferenceTimeout,
            LogDetectiveInferenceRateLimit,
        )),
        before_sleep=inference_retry_backoff,
        retry_error_callback=inference_retry_giveup,
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

    @retry(
        stop=stop_any(stop_after_attempt(3), stop_before_delay(60)),
        wait=wait_exponential_jitter(),
        retry=retry_if_exception_type((
            LogDetectiveInferenceTimeout,
            LogDetectiveInferenceRateLimit,
        )),
        before_sleep=inference_retry_backoff,
        retry_error_callback=inference_retry_giveup,
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

    @retry(
        stop=stop_any(stop_after_attempt(3), stop_before_delay(60)),
        wait=wait_exponential_jitter(),
        retry=retry_if_exception_type((
            LogDetectiveInferenceTimeout,
            LogDetectiveInferenceRateLimit,
        )),
        before_sleep=inference_retry_backoff,
        retry_error_callback=inference_retry_giveup,
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
    """All retries exhausted - exception propagates."""
    call_count = 0

    @retry(
        stop=stop_any(stop_after_attempt(3), stop_before_delay(60)),
        wait=wait_exponential_jitter(),
        retry=retry_if_exception_type((
            LogDetectiveInferenceTimeout,
            LogDetectiveInferenceRateLimit,
        )),
        before_sleep=inference_retry_backoff,
        retry_error_callback=inference_retry_giveup,
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
    backoff_calls: list[int] = []
    call_count = 0

    def tracking_backoff(retry_state: RetryCallState):
        # retry state is modified during following backoffs
        backoff_calls.append(retry_state.attempt_number)
        inference_retry_backoff(retry_state)

    @retry(
        stop=stop_any(stop_after_attempt(2), stop_before_delay(60)),
        wait=wait_exponential_jitter(),
        retry=retry_if_exception_type(LogDetectiveInferenceTimeout),
        before_sleep=tracking_backoff,
    )
    async def retryable():
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise LogDetectiveInferenceTimeout("timeout")
        return "ok"

    await retryable()
    assert len(backoff_calls) == 1
    assert backoff_calls[0] == 1


@pytest.mark.asyncio
async def test_giveup_handler_called():
    """inference_retry_giveup is called when retries are exhausted."""
    giveup_calls: list[RetryCallState] = []

    def tracking_giveup(retry_state: RetryCallState):
        giveup_calls.append(retry_state)
        inference_retry_giveup(retry_state)

    @retry(
        stop=stop_any(stop_after_attempt(2), stop_before_delay(60)),
        wait=wait_exponential_jitter(),
        retry=retry_if_exception_type(LogDetectiveInferenceTimeout),
        retry_error_callback=tracking_giveup,
    )
    async def retryable():
        raise LogDetectiveInferenceTimeout("timeout")

    with pytest.raises(LogDetectiveInferenceTimeout):
        await retryable()
    assert len(giveup_calls) == 1
    assert giveup_calls[0].attempt_number == 2


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
