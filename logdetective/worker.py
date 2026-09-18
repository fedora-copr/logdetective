"""Executable Procrastinate worker entry point."""

from argparse import ArgumentParser

from logdetective.config import SERVER_CONFIG
from logdetective.tasks import app
from logdetective.utils import init_sentry


def main(queue: str) -> None:
    """Initialize Sentry and run one queue with its worker limit.

    The maintenance worker has its own job slot so analysis work cannot occupy
    the capacity used for reconciliation and expiry.

    Args:
        queue: Either ``analysis`` or ``maintenance``, consumed by this worker
            process alone.

    Returns:
        ``None`` after the Procrastinate worker shuts down. During normal service
        operation, the call blocks while jobs are consumed.
    """
    init_sentry()
    app.run_worker(
        queues=[queue],
        concurrency=(SERVER_CONFIG.task_queue.concurrency if queue == "analysis" else 1),
        shutdown_graceful_timeout=(
            SERVER_CONFIG.task_queue.shutdown_graceful_timeout
        ),
        stalled_worker_timeout=SERVER_CONFIG.task_queue.stalled_worker_timeout,
    )


if __name__ == "__main__":
    parser = ArgumentParser(description="Run one Log Detective worker queue")
    parser.add_argument("queue", choices=("analysis", "maintenance"))
    main(parser.parse_args().queue)
