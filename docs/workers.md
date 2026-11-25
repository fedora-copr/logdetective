# Workers

Log Detective executes each analysis as a [Procrastinate](https://github.com/procrastinate-org/procrastinate) task
in a worker pod.

Scale analysis capacity by adding worker pods or by setting `task_queue.concurrency`
in [server/config.yml](../server/config.yml) to control how many jobs each analysis worker runs in parallel.

Long-running analysis is performed by the `worker` service. The
`maintenance-worker` service consumes only reconciliation and expiry jobs with
one reserved worker slot.

Analysis worker concurrency, polling hints, graceful shutdown, stalled-worker
detection, and the default 30-day result retention are configured under
`task_queue` in `server/config.yml`.

## Reconciliation

The periodic reconciler compares application state with queue job status. It
confirms a `CANCELLING` task when its job is missing, succeeded, aborted, or
cancelled. For other active tasks, a succeeded job without a published outcome
reports `result_lost`. Stalled jobs and other missing or unexpectedly terminal
combinations fail the application task and fence it against late publication.

See [Task management](database.md#task-management) for how Log Detective and
Procrastinate store task and job state separately.

## Cancellation and timeouts

Cancellation is cooperative: async work stops at cancellation points, while an already-running
blocking Koji, GitLab, extraction, sanitization, or embedding call is allowed to
finish before the public task becomes `cancelled`.

These blocking calls use bounded client timeouts where network access is involved.
The container's `stop_grace_period` must remain longer than `shutdown_graceful_timeout`.

All analysis tasks have a timeout set to `general.analysis_timeout` (default 1200 seconds).

The worker establishes an absolute event-loop deadline after successful `mark_started()`,
before execution.

Retrieval, sanitization, setup, extraction, inference, retry
backoff, and result assembly consume it. Every retry receives fresh agent and
controller state but the same deadline; completed suite baseline evidence is reused.

Expiry produces a task-level `timeout`, rather than a partial result.

The time allocation doesn't include queue waiting, terminal state publication, and necessary cleanup operations.

Explicit cancellation and HTTP/provider/Koji limits still apply.

`run_blocking()` joins shielded workers through repeated cancellation without
clearing the caller's cancellation counts. The enclosing timeout removes only its
own request: timeout alone reports `timeout`, while overlapping explicit
cancellation remains cancellation and is acknowledged immediately after cleanup.
Database publication fences remain authoritative.

Python cannot forcibly stop a running thread. Cleanup waits for it to return and
can hold resources indefinitely if it never does; the execution deadline is not
a hard cleanup bound. Executor threads may be reused after the call finishes.
