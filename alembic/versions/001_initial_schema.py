"""Initial schema migration: creates users, voice_samples, and detection_results tables.

Revision ID: 001
Revises: None
Create Date: 2026-09-23 20:55:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# Revision identifiers, used by Alembic.
revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create initial schema tables: users, voice_samples, detection_results.

    Creates tables and associated indexes for user identity, enrolled voice embeddings,
    and real-time inference detection logs.
    """
    # --------------------------------------------------------------------------
    # 1. Create users table
    # --------------------------------------------------------------------------
    op.create_table(
        'users',
        sa.Column('id', postgresql.UUID(as_uuid=True), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('username', sa.String(length=100), nullable=False),
        sa.Column('email', sa.String(length=255), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('username', name='uq_users_username'),
        sa.UniqueConstraint('email', name='uq_users_email')
    )

    # --------------------------------------------------------------------------
    # 2. Create voice_samples table
    # --------------------------------------------------------------------------
    op.create_table(
        'voice_samples',
        sa.Column('id', postgresql.UUID(as_uuid=True), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('file_path', sa.Text(), nullable=False),
        sa.Column('embedding', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('sample_rate', sa.Integer(), server_default='16000', nullable=True),
        sa.Column('duration', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE', name='fk_voice_samples_user_id'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_voice_samples_user_id', 'voice_samples', ['user_id'], unique=False)

    # --------------------------------------------------------------------------
    # 3. Create detection_results table
    # --------------------------------------------------------------------------
    op.create_table(
        'detection_results',
        sa.Column('id', postgresql.UUID(as_uuid=True), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('session_id', sa.String(length=255), nullable=False),
        sa.Column('timestamp', sa.DateTime(), server_default=sa.text('now()'), nullable=True),
        sa.Column('risk_score', sa.Float(), nullable=False),
        sa.Column('verdict', sa.String(length=50), nullable=False),
        sa.Column('risk_level', sa.String(length=20), nullable=False),
        sa.Column('features_json', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('audio_duration', sa.Float(), nullable=True),
        sa.Column('model_used', sa.String(length=100), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='SET NULL', name='fk_detection_results_user_id'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_detection_results_session_id', 'detection_results', ['session_id'], unique=False)
    op.create_index('ix_detection_results_timestamp', 'detection_results', ['timestamp'], unique=False)
    op.create_index('ix_detection_results_risk_level', 'detection_results', ['risk_level'], unique=False)


def downgrade() -> None:
    """Drop detection_results, voice_samples, and users tables and indexes."""
    # --------------------------------------------------------------------------
    # 1. Drop detection_results table and indexes
    # --------------------------------------------------------------------------
    op.drop_index('ix_detection_results_risk_level', table_name='detection_results')
    op.drop_index('ix_detection_results_timestamp', table_name='detection_results')
    op.drop_index('ix_detection_results_session_id', table_name='detection_results')
    op.drop_table('detection_results')

    # --------------------------------------------------------------------------
    # 2. Drop voice_samples table and indexes
    # --------------------------------------------------------------------------
    op.drop_index('ix_voice_samples_user_id', table_name='voice_samples')
    op.drop_table('voice_samples')

    # --------------------------------------------------------------------------
    # 3. Drop users table
    # --------------------------------------------------------------------------
    op.drop_table('users')
