from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

from sqlalchemy import (
    Boolean,
    Column,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    insert,
    select,
    update,
)
from sqlalchemy.engine import Engine


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


metadata = MetaData()
enrollments = Table(
    "enrollments",
    metadata,
    Column("user_id", String(128), primary_key=True),
    Column("qbt_token_json", Text, nullable=False),
    Column("feature_ciphertext", Text, nullable=False),
    Column("feature_dim", Integer, nullable=False),
    Column("qbt_salt", Text, nullable=False),
    Column("qbt_commitment", Text, nullable=False),
    Column("created_at", String(64), nullable=False),
    Column("revoked_at", String(64)),
)
audit_log = Table(
    "audit_log",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("event_type", String(64), nullable=False),
    Column("user_id", String(128)),
    Column("authenticated", Boolean),
    Column("confidence", Float),
    Column("latency_ms", Float),
    Column("reason", Text, nullable=False),
    Column("created_at", String(64), nullable=False),
)


@dataclass(frozen=True)
class EnrollmentRecord:
    user_id: str
    qbt_token_json: str
    feature_ciphertext: str
    feature_dim: int
    qbt_salt: str
    qbt_commitment: str
    created_at: str
    revoked_at: str | None = None


@dataclass(frozen=True)
class AuditRecord:
    id: int
    event_type: str
    user_id: str | None
    authenticated: bool | None
    confidence: float | None
    latency_ms: float | None
    reason: str
    created_at: str


class IdentityStore:
    def __init__(self, database_url: str):
        if "://" not in database_url:
            database_url = f"sqlite:///{database_url}"
        self.engine: Engine = create_engine(database_url, pool_pre_ping=True)

    def init(self) -> None:
        metadata.create_all(self.engine)

    def check_connection(self) -> bool:
        try:
            with self.engine.connect() as connection:
                connection.execute(select(1))
            return True
        except Exception:
            return False

    def upsert_enrollment(self, record: EnrollmentRecord) -> None:
        values = record.__dict__
        with self.engine.begin() as connection:
            existing = connection.execute(
                select(enrollments.c.user_id).where(
                    enrollments.c.user_id == record.user_id,
                )
            ).first()
            if existing:
                connection.execute(
                    update(enrollments)
                    .where(enrollments.c.user_id == record.user_id)
                    .values(**values)
                )
            else:
                connection.execute(insert(enrollments).values(**values))

    def revoke_enrollment(self, user_id: str) -> bool:
        with self.engine.begin() as connection:
            result = connection.execute(
                update(enrollments)
                .where(enrollments.c.user_id == user_id, enrollments.c.revoked_at.is_(None))
                .values(revoked_at=utc_now_iso())
            )
            return bool(result.rowcount)

    def get_enrollment(self, user_id: str) -> EnrollmentRecord | None:
        with self.engine.connect() as connection:
            row = (
                connection.execute(
                    select(enrollments).where(
                        enrollments.c.user_id == user_id,
                        enrollments.c.revoked_at.is_(None),
                    )
                )
                .mappings()
                .first()
            )
        return EnrollmentRecord(**row) if row else None

    def list_enrollments(self, include_revoked: bool = False) -> list[EnrollmentRecord]:
        query = select(enrollments)
        if not include_revoked:
            query = query.where(enrollments.c.revoked_at.is_(None))
        with self.engine.connect() as connection:
            rows = (
                connection.execute(query.order_by(enrollments.c.created_at.desc()))
                .mappings()
                .all()
            )
        return [EnrollmentRecord(**row) for row in rows]

    def append_audit(
        self,
        event_type: str,
        *,
        user_id: str | None = None,
        authenticated: bool | None = None,
        confidence: float | None = None,
        latency_ms: float | None = None,
        reason: str = "",
    ) -> None:
        with self.engine.begin() as connection:
            connection.execute(
                insert(audit_log).values(
                    event_type=event_type,
                    user_id=user_id,
                    authenticated=authenticated,
                    confidence=confidence,
                    latency_ms=latency_ms,
                    reason=reason,
                    created_at=utc_now_iso(),
                )
            )

    def list_audit(self, limit: int = 100) -> list[AuditRecord]:
        limit = max(1, min(limit, 1000))
        with self.engine.connect() as connection:
            rows = (
                connection.execute(select(audit_log).order_by(audit_log.c.id.desc()).limit(limit))
                .mappings()
                .all()
            )
        return [AuditRecord(**row) for row in rows]

    def close(self) -> None:
        self.engine.dispose()


def active_user_ids(records: Iterable[EnrollmentRecord]) -> list[str]:
    return [record.user_id for record in records if record.revoked_at is None]
