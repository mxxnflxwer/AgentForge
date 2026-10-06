"""create workflow lifecycle tables

Revision ID: d3e4f5a6b7c8
Revises: c2d3e4f5a6b7
Create Date: 2026-10-04 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd3e4f5a6b7c8'
down_revision: Union[str, Sequence[str], None] = 'c2d3e4f5a6b7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Add activated_at column to workflow_versions if not present
    conn = op.get_bind()
    insp = sa.inspect(conn)
    wf_columns = [c['name'] for c in insp.get_columns('workflow_versions')]
    if 'activated_at' not in wf_columns:
        op.add_column('workflow_versions', sa.Column('activated_at', sa.DateTime(timezone=True), nullable=True))

    # 2. Create workflow_executions table
    if not insp.has_table('workflow_executions'):
        op.create_table(
            'workflow_executions',
            sa.Column('id', sa.String(length=36), nullable=False),
            sa.Column('execution_id', sa.String(length=64), nullable=False),
            sa.Column('workflow_version_id', sa.String(length=64), nullable=False),
            sa.Column('user_id', sa.String(length=36), sa.ForeignKey('users.id', ondelete='SET NULL'), nullable=True),
            sa.Column('query', sa.Text(), nullable=False),
            sa.Column('response', sa.Text(), nullable=True),
            sa.Column('retrieved_chunks', sa.JSON(), nullable=True),
            sa.Column('model', sa.String(length=128), nullable=False),
            sa.Column('provider', sa.String(length=64), nullable=False),
            sa.Column('input_tokens', sa.Integer(), server_default='0', nullable=False),
            sa.Column('output_tokens', sa.Integer(), server_default='0', nullable=False),
            sa.Column('total_tokens', sa.Integer(), server_default='0', nullable=False),
            sa.Column('cost', sa.Float(), server_default='0.0', nullable=False),
            sa.Column('llm_latency_ms', sa.Float(), server_default='0.0', nullable=False),
            sa.Column('total_execution_time_ms', sa.Float(), server_default='0.0', nullable=False),
            sa.Column('status', sa.String(length=32), server_default='success', nullable=False),
            sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
            sa.PrimaryKeyConstraint('id')
        )
        op.create_index(op.f('ix_workflow_executions_execution_id'), 'workflow_executions', ['execution_id'], unique=True)
        op.create_index(op.f('ix_workflow_executions_workflow_version_id'), 'workflow_executions', ['workflow_version_id'], unique=False)
        op.create_index(op.f('ix_workflow_executions_user_id'), 'workflow_executions', ['user_id'], unique=False)
        op.create_index(op.f('ix_workflow_executions_status'), 'workflow_executions', ['status'], unique=False)

    # 3. Create workflow_feedback table
    if not insp.has_table('workflow_feedback'):
        op.create_table(
            'workflow_feedback',
            sa.Column('id', sa.String(length=36), nullable=False),
            sa.Column('feedback_id', sa.String(length=64), nullable=False),
            sa.Column('execution_id', sa.String(length=64), nullable=False),
            sa.Column('workflow_version_id', sa.String(length=64), nullable=False),
            sa.Column('user_id', sa.String(length=36), sa.ForeignKey('users.id', ondelete='SET NULL'), nullable=True),
            sa.Column('rating', sa.Integer(), nullable=False),
            sa.Column('feedback_text', sa.Text(), nullable=True),
            sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
            sa.PrimaryKeyConstraint('id')
        )
        op.create_index(op.f('ix_workflow_feedback_feedback_id'), 'workflow_feedback', ['feedback_id'], unique=True)
        op.create_index(op.f('ix_workflow_feedback_execution_id'), 'workflow_feedback', ['execution_id'], unique=False)
        op.create_index(op.f('ix_workflow_feedback_workflow_version_id'), 'workflow_feedback', ['workflow_version_id'], unique=False)
        op.create_index(op.f('ix_workflow_feedback_user_id'), 'workflow_feedback', ['user_id'], unique=False)

    # 4. Create workflow_audit_events table
    if not insp.has_table('workflow_audit_events'):
        op.create_table(
            'workflow_audit_events',
            sa.Column('id', sa.String(length=36), nullable=False),
            sa.Column('event_id', sa.String(length=64), nullable=False),
            sa.Column('workflow_version_id', sa.String(length=64), nullable=False),
            sa.Column('event_type', sa.String(length=32), nullable=False),
            sa.Column('performed_by', sa.String(length=36), nullable=True),
            sa.Column('user_email', sa.String(length=255), nullable=False),
            sa.Column('details', sa.JSON(), nullable=True),
            sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
            sa.PrimaryKeyConstraint('id')
        )
        op.create_index(op.f('ix_workflow_audit_events_event_id'), 'workflow_audit_events', ['event_id'], unique=True)
        op.create_index(op.f('ix_workflow_audit_events_workflow_version_id'), 'workflow_audit_events', ['workflow_version_id'], unique=False)
        op.create_index(op.f('ix_workflow_audit_events_event_type'), 'workflow_audit_events', ['event_type'], unique=False)


def downgrade() -> None:
    op.drop_table('workflow_audit_events')
    op.drop_table('workflow_feedback')
    op.drop_table('workflow_executions')
    op.drop_column('workflow_versions', 'activated_at')
