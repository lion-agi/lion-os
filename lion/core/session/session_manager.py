import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from lion.core.generic import LogManager
from lion.core.session.branch import Branch
from lion.core.session.session import Session
from lion.core.storage.database import Database


class SessionInfo:
    """Simple container for session information."""

    def __init__(self, session_id: str, created_at: datetime, expires_at: datetime):
        self.session_id = session_id
        self.created_at = created_at
        self.last_accessed = created_at
        self.expires_at = expires_at
        self.branches: dict[str, Branch] = {}
        self.default_branch_id: str | None = None

    def to_dict(self) -> dict:
        """Convert session info to dictionary."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "branches": list(self.branches.keys()),
            "default_branch_id": self.default_branch_id,
        }


class SessionManager:
    """Manages agent sessions with minimal persistence."""

    def __init__(self, db_path: str = "sessions.db", default_ttl: int = 3600):
        """
        Initialize SessionManager.

        Args:
            db_path: Path to the SQLite database file
            default_ttl: Default session time-to-live in seconds (1 hour)
        """
        self._db = Database(db_path)
        self._default_ttl = default_ttl
        self._sessions: dict[str, SessionInfo] = {}  # In-memory cache
        self._logger = LogManager(
            persist_dir="./data/logs/sessions",
            file_prefix="session_operations_",
            capacity=1000,
            auto_save_on_exit=True,
        )

    def create_session(self, ttl: int | None = None) -> str:
        """
        Create a new session.

        Args:
            ttl: Optional time-to-live in seconds

        Returns:
            Session ID
        """
        session_id = str(uuid.uuid4())
        now = datetime.now()
        expires_at = now + timedelta(seconds=ttl or self._default_ttl)

        # Create session info
        session_info = SessionInfo(session_id, now, expires_at)

        # Create default branch
        branch = Branch()
        session_info.branches[branch.ln_id] = branch
        session_info.default_branch_id = branch.ln_id

        # Store in memory
        self._sessions[session_id] = session_info

        # Store minimal info in database
        self._db.save_session(
            session_id, session_info.to_dict(), ttl or self._default_ttl
        )

        return session_id

    def get_session(self, session_id: str) -> Session:
        """
        Retrieve a session by ID.

        Args:
            session_id: Session identifier

        Returns:
            Session object
        """
        # Check memory cache first
        if session_id in self._sessions:
            session_info = self._sessions[session_id]
            session_info.last_accessed = datetime.now()

            # Create session object
            session = Session()
            for branch_id, branch in session_info.branches.items():
                session.branches.include(branch)
                if branch_id == session_info.default_branch_id:
                    session.default_branch = branch
            return session

        # Try to load from database
        data = self._db.get_session(session_id)
        if data:
            # Create new session with default branch
            session = Session()
            branch = Branch()
            session.branches.include(branch)
            session.default_branch = branch

            # Create session info
            now = datetime.now()
            session_info = SessionInfo(
                session_id=session_id,
                created_at=datetime.fromisoformat(data["created_at"]),
                expires_at=datetime.fromisoformat(data["expires_at"]),
            )
            session_info.branches[branch.ln_id] = branch
            session_info.default_branch_id = branch.ln_id

            # Cache in memory
            self._sessions[session_id] = session_info

            return session

        raise KeyError(f"Session {session_id} not found")

    def save_branch(self, session_id: str, branch: Branch) -> None:
        """
        Save a branch to a session.

        Args:
            session_id: Session identifier
            branch: Branch to save
        """
        if session_id not in self._sessions:
            raise KeyError(f"Session {session_id} not found")

        session_info = self._sessions[session_id]
        session_info.branches[branch.ln_id] = branch
        session_info.last_accessed = datetime.now()

    def get_branch(self, session_id: str, branch_id: str) -> Branch:
        """
        Get a branch from a session.

        Args:
            session_id: Session identifier
            branch_id: Branch identifier

        Returns:
            Branch object
        """
        if session_id not in self._sessions:
            raise KeyError(f"Session {session_id} not found")

        session_info = self._sessions[session_id]
        if branch_id not in session_info.branches:
            raise KeyError(f"Branch {branch_id} not found in session {session_id}")

        return session_info.branches[branch_id]

    def remove_session(self, session_id: str) -> None:
        """
        Remove a session.

        Args:
            session_id: Session identifier
        """
        # Remove from memory cache
        if session_id in self._sessions:
            del self._sessions[session_id]

        # Remove from database
        self._db.save_session(
            session_id, {"deleted": True}, ttl=0  # Expire immediately
        )

    def get_session_info(self, session_id: str) -> dict:
        """
        Get session information.

        Args:
            session_id: Session identifier

        Returns:
            Dictionary with session information
        """
        if session_id not in self._sessions:
            raise KeyError(f"Session {session_id} not found")

        return self._sessions[session_id].to_dict()

    def cleanup_expired(self) -> None:
        """Remove expired sessions from cache and database."""
        # Clear expired sessions from database
        self._db.cleanup_expired()

        # Clear memory cache of expired sessions
        now = datetime.now()
        expired_sessions = [
            session_id
            for session_id, info in self._sessions.items()
            if info.expires_at <= now
        ]

        for session_id in expired_sessions:
            del self._sessions[session_id]

    def close(self) -> None:
        """Close database connection and dump logs."""
        self._logger.dump()
        self._db.close()
