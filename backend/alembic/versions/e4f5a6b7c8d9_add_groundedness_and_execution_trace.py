"""add groundedness and execution trace to workflow_executions

Revision ID: e4f5a6b7c8d9
Revises: d3e4f5a6b7c8
Create Date: 2026-10-22 13:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e4f5a6b7c8d9'
down_revision: Union[str, Sequence[str], None] = 'd3e4f5a6b7c8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)
    cols = [c['name'] for c in insp.get_columns('workflow_executions')]
    
    if 'groundedness' not in cols:
        op.add_column('workflow_executions', sa.Column('groundedness', sa.Float(), nullable=True))
    if 'execution_trace' not in cols:
        op.add_column('workflow_executions', sa.Column('execution_trace', sa.JSON(), nullable=True))


def downgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)
    cols = [c['name'] for c in insp.get_columns('workflow_executions')]
    
    if 'execution_trace' in cols:
        op.drop_column('workflow_executions', 'execution_trace')
    if 'groundedness' in cols:
        op.drop_column('workflow_executions', 'groundedness')
