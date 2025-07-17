"""Add koji_task_analysis table

Revision ID: ae776dc8160b
Revises: 360aea89443f
Create Date: 2025-07-07 12:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision: str = "ae776dc8160b"
down_revision: Union[str, None] = "360aea89443f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Create the koji_task_analysis table
    op.create_table(
        "koji_task_analysis",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "koji_instance",
            sa.String(255),
            nullable=False,
            comment="The koji instance identifier",
        ),
        sa.Column(
            "task_id",
            sa.BigInteger(),
            nullable=False,
            comment="The koji task ID",
        ),
        sa.Column(
            "log_file_name",
            sa.String(255),
            nullable=False,
            comment="The name of the log file",
        ),
        sa.Column(
            "request_received_at",
            sa.DateTime(),
            nullable=False,
            comment="Timestamp when the request was received",
        ),
        sa.Column(
            "response_id",
            sa.Integer(),
            nullable=True,
            comment="The id of the analyze request metrics for this task",
        ),
        sa.ForeignKeyConstraint(
            ["response_id"],
            ["analyze_request_metrics.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("task_id", name="uix_task_id"),
    )

    # Create indexes
    op.create_index(
        op.f("ix_koji_task_analysis_koji_instance"),
        "koji_task_analysis",
        ["koji_instance"],
        unique=False,
    )
    op.create_index(
        op.f("ix_koji_task_analysis_task_id"),
        "koji_task_analysis",
        ["task_id"],
        unique=True,
    )
    op.create_index(
        op.f("ix_koji_task_analysis_request_received_at"),
        "koji_task_analysis",
        ["request_received_at"],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    # Drop indexes
    op.drop_index(
        op.f("ix_koji_task_analysis_request_received_at"),
        table_name="koji_task_analysis",
    )
    op.drop_index(
        op.f("ix_koji_task_analysis_task_id"),
        table_name="koji_task_analysis",
    )
    op.drop_index(
        op.f("ix_koji_task_analysis_koji_instance"),
        table_name="koji_task_analysis",
    )

    # Drop table
    op.drop_table("koji_task_analysis")
