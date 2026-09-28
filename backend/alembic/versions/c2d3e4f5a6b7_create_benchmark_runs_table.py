"""create benchmark_runs table

Revision ID: c2d3e4f5a6b7
Revises: a8b2c4e5f6d7
Create Date: 2026-09-28 10:55:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'c2d3e4f5a6b7'
down_revision: Union[str, Sequence[str], None] = 'a8b2c4e5f6d7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'benchmark_runs',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('benchmark_id', sa.String(length=64), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('status', sa.String(length=32), server_default='completed', nullable=False),
        sa.Column('preset_name', sa.String(length=64), nullable=True),
        sa.Column('query', sa.Text(), nullable=False),
        sa.Column('document_id', sa.String(length=64), nullable=True),
        sa.Column('expected_answer', sa.Text(), nullable=True),
        sa.Column('selected_versions', sa.JSON(), nullable=False),
        sa.Column('results', sa.JSON(), nullable=False),
        sa.Column('pareto_summary', sa.JSON(), nullable=False),
        sa.Column('created_by', sa.String(length=255), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_benchmark_runs_benchmark_id'), 'benchmark_runs', ['benchmark_id'], unique=True)
    op.create_index(op.f('ix_benchmark_runs_status'), 'benchmark_runs', ['status'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_benchmark_runs_status'), table_name='benchmark_runs')
    op.drop_index(op.f('ix_benchmark_runs_benchmark_id'), table_name='benchmark_runs')
    op.drop_table('benchmark_runs')
