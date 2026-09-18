"""Replace the undeployed task queue with application analysis records.

Revision ID: a090e4500d54
Revises: 8887255bee62
Create Date: 2026-09-17 12:00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "a090e4500d54"
down_revision: str | None = "8887255bee62"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _require_empty() -> None:
    """Verify that the application task table contains no accepted work.

    Returns:
        ``None`` when ``task_analysis`` is empty.

    Raises:
        RuntimeError: If the database cannot report table state or any task record
            would be destroyed by the migration.
    """
    exists = op.get_bind().scalar(
        sa.text("SELECT EXISTS (SELECT 1 FROM task_analysis)")
    )
    if exists is None:
        raise RuntimeError("Unable to determine whether task_analysis is empty")
    if exists:
        raise RuntimeError(
            "task_analysis is not empty; archive or handle its records before migration"
        )


def _require_no_koji_metrics() -> None:
    """Refuse a downgrade that cannot represent accepted Koji metrics."""
    exists = op.get_bind().scalar(
        sa.text(
            "SELECT EXISTS ("
            "SELECT 1 FROM analyze_request_metrics "
            "WHERE endpoint::text = 'analyze_koji_task'"
            ")"
        )
    )
    if exists is None:
        raise RuntimeError("Unable to determine whether Koji metrics exist")
    if exists:
        raise RuntimeError(
            "Koji metrics exist; archive or handle them before downgrading"
        )


def upgrade() -> None:
    """Create the Procrastinate-linked application task schema.

    Returns:
        ``None`` after the task table, enum values, indexes, and constraints are
        installed.

    Raises:
        RuntimeError: If the existing task table contains records.
    """
    _require_empty()

    op.drop_table("task_analysis")

    op.execute("DROP TYPE IF EXISTS tasktype;")
    op.execute("DROP TYPE IF EXISTS analysisstate;")

    analysis_state = postgresql.ENUM(
        "scheduled",
        "in_progress",
        "cancelling",
        "cancelled",
        "done",
        "error",
        name="analysisstate",
        create_type=False,
    )
    analysis_state.create(op.get_bind(), checkfirst=False)

    task_type = postgresql.ENUM(
        "generic", "koji", "gitlab", name="tasktype", create_type=False
    )
    task_type.create(op.get_bind(), checkfirst=False)

    op.create_table(
        "task_analysis",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("task_id", sa.UUID(), nullable=False),
        sa.Column("owner_token_name", sa.String(), nullable=True),
        sa.Column("task_type", task_type, nullable=False),
        sa.Column("source_id", sa.String(), nullable=True),
        sa.Column("request_hash", sa.String(length=64), nullable=False),
        sa.Column("input_payload", sa.JSON(), nullable=True),
        sa.Column("request_size", sa.Integer(), nullable=False),
        sa.Column("procrastinate_job_id", sa.BigInteger(), nullable=False),
        sa.Column("state", analysis_state, nullable=False),
        sa.Column("generation", sa.Integer(), nullable=False),
        sa.Column("request_received_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "cancellation_requested_at", sa.DateTime(timezone=True), nullable=True
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("response_metrics_id", sa.Integer(), nullable=True),
        sa.Column("response", sa.LargeBinary(), nullable=True),
        sa.Column("task_metadata", sa.JSON(), nullable=True),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.Column("error_message", sa.String(length=255), nullable=True),
        sa.ForeignKeyConstraint(
            ["response_metrics_id"], ["analyze_request_metrics.id"]
        ),
        sa.UniqueConstraint("task_id"),
        sa.UniqueConstraint("procrastinate_job_id"),
    )
    for column in (
        "task_id",
        "owner_token_name",
        "source_id",
        "procrastinate_job_id",
        "state",
        "request_received_at",
        "finished_at",
        "expires_at",
    ):
        op.create_index(f"ix_task_analysis_{column}", "task_analysis", [column])
    op.create_index(
        "uix_task_analysis_source",
        "task_analysis",
        ["task_type", "source_id"],
        unique=True,
        postgresql_where=sa.text("source_id IS NOT NULL"),
    )

    # Recreate the type so every existing label is normalized and downgrade can
    # remove the new Koji label without discarding metrics.
    op.execute("ALTER TYPE endpointtype RENAME TO endpointtype_uppercase;")
    endpoint_type = postgresql.ENUM(
        "analyze",
        "analyze_staged",
        "analyze_stream",
        "analyze_gitlab_job",
        "analyze_koji_task",
        name="endpointtype",
        create_type=True,
    )
    endpoint_type.create(op.get_bind(), checkfirst=False)
    op.execute(
        "ALTER TABLE analyze_request_metrics "
        "ALTER COLUMN endpoint TYPE endpointtype "
        "USING lower(endpoint::text)::endpointtype;"
    )
    op.execute("DROP TYPE endpointtype_uppercase;")


def downgrade() -> None:
    """Restore the former task schema without deleting accepted work.

    Returns:
        ``None`` after the previous table, indexes, and constraints are restored.

    Raises:
        RuntimeError: If the current task table or Koji metrics contain records
            that the former schema cannot represent.
    """
    _require_empty()
    _require_no_koji_metrics()
    op.drop_table("task_analysis")

    op.execute("DROP TYPE tasktype;")
    op.execute("DROP TYPE analysisstate;")

    analysis_state = postgresql.ENUM(
        "SCHEDULED", "DONE", "IN_PROGRESS", "ERROR",
        name="analysisstate", create_type=False,
    )
    analysis_state.create(op.get_bind(), checkfirst=False)
    task_type = postgresql.ENUM(
        "GENERIC", "KOJI", name="tasktype", create_type=False
    )
    task_type.create(op.get_bind(), checkfirst=False)
    op.create_table(
        "task_analysis",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("task_id", sa.UUID(), nullable=False),
        sa.Column("request_received_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("response_returned_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("response_metrics_id", sa.Integer(), nullable=True),
        sa.Column("state", analysis_state, nullable=False),
        sa.Column("attempt_count", sa.Integer(), nullable=False),
        sa.Column("task_metadata", sa.JSON(), nullable=True),
        sa.Column("response", sa.LargeBinary(), nullable=True),
        sa.Column("task_type", task_type, nullable=False),
        sa.Column("external_task_id", sa.String(), nullable=True),
        sa.ForeignKeyConstraint(
            ["response_metrics_id"], ["analyze_request_metrics.id"]
        ),
        sa.UniqueConstraint("task_id"),
        sa.UniqueConstraint("external_task_id"),
    )
    for column in (
        "task_id",
        "request_received_at",
        "response_returned_at",
        "state",
        "external_task_id",
    ):
        op.create_index(f"ix_task_analysis_{column}", "task_analysis", [column])

    op.execute("ALTER TYPE endpointtype RENAME TO endpointtype_lowercase;")
    endpoint_type = postgresql.ENUM(
        "ANALYZE",
        "ANALYZE_STAGED",
        "ANALYZE_STREAM",
        "ANALYZE_GITLAB_JOB",
        name="endpointtype",
        create_type=True,
    )
    endpoint_type.create(op.get_bind(), checkfirst=False)
    op.execute(
        "ALTER TABLE analyze_request_metrics "
        "ALTER COLUMN endpoint TYPE endpointtype "
        "USING upper(endpoint::text)::endpointtype;"
    )
    op.execute("DROP TYPE endpointtype_lowercase;")
