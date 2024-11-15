import json
from datetime import UTC, datetime, timedelta
from typing import Any, Dict, Optional

from sqlalchemy import Column, DateTime, String, Text, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.pool import StaticPool

Base = declarative_base()


class SessionModel(Base):
    __tablename__ = "sessions"

    session_id = Column(String, primary_key=True)
    metadata = Column(Text, nullable=False)  # JSON string of session metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    last_accessed = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)


class Database:
    """SQLite database for minimal session metadata storage."""

    def __init__(self, db_path: str = "sessions.db"):
        self._engine = create_engine(
            f"sqlite:///{db_path}",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self._engine)

        session_factory = sessionmaker(bind=self._engine)
        self._Session = scoped_session(session_factory)

    def save_session(
        self, session_id: str, metadata: dict[str, Any], ttl: int = 3600
    ) -> None:
        """
        Save session metadata with expiration.

        Args:
            session_id: Unique identifier for the session
            metadata: Session metadata to store
            ttl: Time to live in seconds (default 1 hour)
        """
        session = self._Session()
        try:
            expires_at = datetime.now(tz=UTC) + timedelta(seconds=ttl)

            db_session = (
                session.query(SessionModel).filter_by(session_id=session_id).first()
            )
            if db_session:
                db_session.metadata = json.dumps(metadata)
                db_session.last_accessed = datetime.now(tz=UTC)
                db_session.expires_at = expires_at
            else:
                db_session = SessionModel(
                    session_id=session_id,
                    metadata=json.dumps(metadata),
                    expires_at=expires_at,
                )
                session.add(db_session)

            session.commit()
        finally:
            session.close()

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        """
        Retrieve session metadata if not expired.

        Args:
            session_id: Session identifier

        Returns:
            Session metadata if found and not expired, None otherwise
        """
        session = self._Session()
        try:
            db_session = (
                session.query(SessionModel)
                .filter(
                    SessionModel.session_id == session_id,
                    SessionModel.expires_at > datetime.now(tz=UTC),
                )
                .first()
            )

            if db_session:
                db_session.last_accessed = datetime.now(tz=UTC)
                session.commit()
                return json.loads(db_session.metadata)
            return None
        finally:
            session.close()

    def cleanup_expired(self) -> None:
        """Remove expired sessions."""
        session = self._Session()
        try:
            # Delete expired sessions
            session.query(SessionModel).filter(
                SessionModel.expires_at <= datetime.now(tz=UTC)
            ).delete()

            session.commit()
        finally:
            session.close()

    def close(self) -> None:
        """Close database connection."""
        self._Session.remove()
        self._engine.dispose()
