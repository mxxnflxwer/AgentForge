"""create documents and document_chunks tables

Revision ID: a8b2c4e5f6d7
Revises: 011f8cecc2ec
Create Date: 2026-08-30 22:15:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'a8b2c4e5f6d7'
down_revision: Union[str, Sequence[str], None] = '011f8cecc2ec'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create enum for document processing status
    doc_status_enum = sa.Enum('UPLOADED', 'PROCESSING', 'COMPLETED', 'FAILED', name='document_processing_status')
    doc_status_enum.create(op.get_bind(), checkfirst=True)

    op.create_table('documents',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('owner_id', sa.String(length=36), nullable=False),
        sa.Column('original_filename', sa.String(length=255), nullable=False),
        sa.Column('stored_filename', sa.String(length=255), nullable=False),
        sa.Column('file_type', sa.String(length=50), nullable=False),
        sa.Column('file_size', sa.Integer(), nullable=False),
        sa.Column('mime_type', sa.String(length=100), nullable=False),
        sa.Column('specialty', sa.String(length=100), nullable=True),
        sa.Column('page_count', sa.Integer(), nullable=True),
        sa.Column('extracted_character_count', sa.Integer(), nullable=True),
        sa.Column('chunk_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('processing_status', sa.Enum('UPLOADED', 'PROCESSING', 'COMPLETED', 'FAILED', name='document_processing_status'), nullable=False),
        sa.Column('processing_error', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('processed_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['owner_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_documents_owner_id'), 'documents', ['owner_id'], unique=False)
    op.create_index(op.f('ix_documents_stored_filename'), 'documents', ['stored_filename'], unique=False)

    op.create_table('document_chunks',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('document_id', sa.String(length=36), nullable=False),
        sa.Column('chunk_index', sa.Integer(), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('section_title', sa.String(length=255), nullable=True),
        sa.Column('token_count', sa.Integer(), nullable=True),
        sa.Column('character_count', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_document_chunks_document_id'), 'document_chunks', ['document_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_document_chunks_document_id'), table_name='document_chunks')
    op.drop_table('document_chunks')
    op.drop_index(op.f('ix_documents_stored_filename'), table_name='documents')
    op.drop_index(op.f('ix_documents_owner_id'), table_name='documents')
    op.drop_table('documents')
    
    doc_status_enum = sa.Enum('UPLOADED', 'PROCESSING', 'COMPLETED', 'FAILED', name='document_processing_status')
    doc_status_enum.drop(op.get_bind(), checkfirst=True)
