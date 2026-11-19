"""Cancellation ownership while joining synchronous work."""

import asyncio
import threading

import pytest

from logdetective.utils import run_blocking


@pytest.mark.asyncio
@pytest.mark.parametrize("trigger", [
    "timeout", "cancel", "cancel_then_expiry", "expiry_then_cancel",
])
@pytest.mark.parametrize("fails", [False, True])
async def test_direct_blocking_cancellation_counts(trigger, fails):
    """Joining sanitization preserves explicit cancellation even if work raises."""
    started, release, finished = threading.Event(), threading.Event(), threading.Event()

    def work():
        started.set()
        try:
            assert release.wait(3)
            if fails:
                raise ValueError("worker failed")
        finally:
            finished.set()

    async def execute():
        async with asyncio.timeout(10 if trigger == "cancel" else 0.05):
            await run_blocking(work)

    operation = asyncio.create_task(execute())
    try:
        async with asyncio.timeout(2):
            while not started.is_set():
                await asyncio.sleep(0.001)
        if trigger in ("cancel", "cancel_then_expiry"):
            operation.cancel()
            await asyncio.sleep(0.001)
            operation.cancel()
        await asyncio.sleep(0.08)
        if trigger == "expiry_then_cancel":
            operation.cancel()
            await asyncio.sleep(0.001)
            operation.cancel()
        assert not operation.done()
    finally:
        release.set()
        with pytest.raises(TimeoutError if trigger == "timeout" else asyncio.CancelledError):
            await operation
    assert finished.is_set()
    assert operation.cancelling() == (0 if trigger == "timeout" else 2)
