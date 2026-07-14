from alembic import op
import sqlalchemy as sa

revision = "0002_dual_eye_templates"
down_revision = "0001_initial_schema"
branch_labels = None
depends_on = None


DUAL_EYE_COLUMNS = [
    sa.Column("left_iris_feature_ciphertext", sa.Text(), nullable=True),
    sa.Column("right_iris_feature_ciphertext", sa.Text(), nullable=True),
    sa.Column("fused_iris_feature_ciphertext", sa.Text(), nullable=True),
    sa.Column("left_iris_feature_dim", sa.Integer(), nullable=True),
    sa.Column("right_iris_feature_dim", sa.Integer(), nullable=True),
    sa.Column("fused_iris_feature_dim", sa.Integer(), nullable=True),
]


def _existing_columns() -> set[str]:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    return {column["name"] for column in inspector.get_columns("enrollments")}


def upgrade() -> None:
    existing = _existing_columns()
    for column in DUAL_EYE_COLUMNS:
        if column.name not in existing:
            op.add_column("enrollments", column)


def downgrade() -> None:
    existing = _existing_columns()
    for column in reversed(DUAL_EYE_COLUMNS):
        if column.name in existing:
            op.drop_column("enrollments", column.name)
