"""Add Testing Farm suite tasks and request metrics.

Revision ID: c91b7e46a0d2
Revises: 2e3a14913472
"""

import sqlalchemy as sa
from alembic import op

revision = "c91b7e46a0d2"
down_revision = "2e3a14913472"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Extend fresh task schema for the Testing Farm endpoint and deadline."""
    op.execute("ALTER TYPE tasktype ADD VALUE IF NOT EXISTS 'tft_suite'")
    op.execute("ALTER TYPE endpointtype ADD VALUE IF NOT EXISTS 'analyze_tft'")
    op.add_column("task_analysis", sa.Column("deadline_at", sa.DateTime(timezone=True)))
    op.create_index("ix_task_analysis_deadline_at", "task_analysis", ["deadline_at"])


def downgrade() -> None:
    """Remove labels only when no retained suite tasks or metrics use them."""
    op.drop_index("ix_task_analysis_deadline_at", table_name="task_analysis")
    op.drop_column("task_analysis", "deadline_at")
    bind = op.get_bind()
    in_use = bind.scalar(sa.text(
        "SELECT EXISTS (SELECT 1 FROM task_analysis "
        "WHERE task_type::text = 'tft_suite') "
        "OR EXISTS (SELECT 1 FROM analyze_request_metrics "
        "WHERE endpoint::text = 'analyze_tft')"
    ))
    if in_use is None or in_use:
        raise RuntimeError("Remove Testing Farm suite tasks and metrics before downgrade")

    op.execute("ALTER TYPE tasktype RENAME TO tasktype_with_suite")
    op.execute("CREATE TYPE tasktype AS ENUM ('generic', 'koji', 'gitlab')")
    op.execute(
        "ALTER TABLE task_analysis ALTER COLUMN task_type TYPE tasktype "
        "USING task_type::text::tasktype"
    )
    op.execute("DROP TYPE tasktype_with_suite")
    op.execute("ALTER TYPE endpointtype RENAME TO endpointtype_with_tft")
    op.execute(
        "CREATE TYPE endpointtype AS ENUM "
        "('analyze', 'analyze_staged', 'analyze_stream', "
        "'analyze_gitlab_job', 'analyze_koji_task')"
    )
    op.execute(
        "ALTER TABLE analyze_request_metrics ALTER COLUMN endpoint "
        "TYPE endpointtype USING endpoint::text::endpointtype"
    )
    op.execute("DROP TYPE endpointtype_with_tft")
