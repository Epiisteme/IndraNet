from alembic import op
import sqlalchemy as sa

revision = "0001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "enrollments",
        sa.Column("user_id", sa.String(length=128), primary_key=True),
        sa.Column("qbt_token_json", sa.Text(), nullable=False),
        sa.Column("feature_ciphertext", sa.Text(), nullable=False),
        sa.Column("feature_dim", sa.Integer(), nullable=False),
        sa.Column("left_iris_feature_ciphertext", sa.Text(), nullable=True),
        sa.Column("right_iris_feature_ciphertext", sa.Text(), nullable=True),
        sa.Column("fused_iris_feature_ciphertext", sa.Text(), nullable=True),
        sa.Column("left_iris_feature_dim", sa.Integer(), nullable=True),
        sa.Column("right_iris_feature_dim", sa.Integer(), nullable=True),
        sa.Column("fused_iris_feature_dim", sa.Integer(), nullable=True),
        sa.Column("qbt_salt", sa.Text(), nullable=False),
        sa.Column("qbt_commitment", sa.Text(), nullable=False),
        sa.Column("created_at", sa.String(length=64), nullable=False),
        sa.Column("revoked_at", sa.String(length=64), nullable=True),
    )
    op.create_table(
        "audit_log",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("user_id", sa.String(length=128), nullable=True),
        sa.Column("authenticated", sa.Boolean(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("latency_ms", sa.Float(), nullable=True),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("created_at", sa.String(length=64), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("audit_log")
    op.drop_table("enrollments")
