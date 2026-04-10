"""Removed certainty from the model

Revision ID: 725400e0ffb4
Revises: b4ed423319f9
Create Date: 2026-03-30 14:35:36.947819

"""
from typing import Sequence, Union
import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '725400e0ffb4'
down_revision: Union[str, None] = 'b4ed423319f9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.drop_column("analyze_request_metrics", "response_certainty")


def downgrade() -> None:
    """Downgrade schema."""
    op.add_column(
        "analyze_request_metrics",
        sa.Column(
            'response_certainty',
            sa.DOUBLE_PRECISION(precision=53),
            autoincrement=False,
            nullable=True,
            comment='Certainty for generated response'),
    )
