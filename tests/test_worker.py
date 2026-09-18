"""Worker process startup behavior."""

from unittest.mock import patch

import pytest

from logdetective import config, utils, worker


@pytest.mark.parametrize("dsn", [None, "https://public@example.com/1"])
@pytest.mark.parametrize("queue", ["analysis", "maintenance"])
def test_worker_initializes_sentry_before_running_jobs(monkeypatch, dsn, queue):
    """Each worker consumes only its queue and initializes shared Sentry."""
    monkeypatch.setattr(config.SERVER_CONFIG.general, "sentry_dsn", dsn)
    monkeypatch.setattr(config.SERVER_CONFIG.task_queue, "concurrency", 3)

    with patch.object(utils.sentry_sdk, "init") as sentry_init:
        with patch.object(worker.app, "run_worker") as run_worker:
            worker.main(queue)

    if dsn is None:
        sentry_init.assert_not_called()
    else:
        sentry_init.assert_called_once_with(dsn=dsn, traces_sample_rate=1.0)
    run_worker.assert_called_once_with(
        queues=[queue],
        concurrency=3 if queue == "analysis" else 1,
        shutdown_graceful_timeout=(
            config.SERVER_CONFIG.task_queue.shutdown_graceful_timeout
        ),
        stalled_worker_timeout=config.SERVER_CONFIG.task_queue.stalled_worker_timeout,
    )
