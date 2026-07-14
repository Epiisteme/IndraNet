from alembic import op
import sqlalchemy as sa

revision = "0002_dual_eye_templates"
down_revision = "0001_initial_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("enrollments", sa.Column("left_iris_feature_ciphertext", sa.Text(), nullable=True))
    op.add_column("enrollments", sa.Column("right_iris_feature_ciphertext", sa.Text(), nullable=True))
    op.add_column("enrollments", sa.Column("fused_iris_feature_ciphertext", sa.Text(), nullable=True))
    op.add_column("enrollments", sa.Column("left_iris_feature_dim", sa.Integer(), nullable=True))
    op.add_column("enrollments", sa.Column("right_iris_feature_dim", sa.Integer(), nullable=True))
    op.add_column("enrollments", sa.Column("fused_iris_feature_dim", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("enrollments", "fused_iris_feature_dim")
    op.drop_column("enrollments", "right_iris_feature_dim")
    op.drop_column("enrollments", "left_iris_feature_dim")
    op.drop_column("enrollments", "fused_iris_feature_ciphertext")
    op.drop_column("enrollments", "right_iris_feature_ciphertext")
    op.drop_column("enrollments", "left_iris_feature_ciphertext")
