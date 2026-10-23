"""make execution metrics nullable

Revision ID: f5a6b7c8d9e0
Revises: e4f5a6b7c8d9
Create Date: 2026-10-22 16:30:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'f5a6b7c8d9e0'
down_revision = 'e4f5a6b7c8d9'
branch_labels = None
depends_on = None


def upgrade():
    # Make metric columns nullable so that failed executions can store null instead of fabricated zeros
    op.alter_column('workflow_executions', 'input_tokens', existing_type=sa.Integer(), nullable=True)
    op.alter_column('workflow_executions', 'output_tokens', existing_type=sa.Integer(), nullable=True)
    op.alter_column('workflow_executions', 'total_tokens', existing_type=sa.Integer(), nullable=True)
    op.alter_column('workflow_executions', 'cost', existing_type=sa.Float(), nullable=True)
    op.alter_column('workflow_executions', 'llm_latency_ms', existing_type=sa.Float(), nullable=True)


def downgrade():
    op.alter_column('workflow_executions', 'llm_latency_ms', existing_type=sa.Float(), nullable=False)
    op.alter_column('workflow_executions', 'cost', existing_type=sa.Float(), nullable=False)
    op.alter_column('workflow_executions', 'total_tokens', existing_type=sa.Integer(), nullable=False)
    op.alter_column('workflow_executions', 'output_tokens', existing_type=sa.Integer(), nullable=False)
    op.alter_column('workflow_executions', 'input_tokens', existing_type=sa.Integer(), nullable=False)
