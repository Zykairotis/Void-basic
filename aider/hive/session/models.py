"""
Database models for the Hive session management system.

This module contains SQLAlchemy models that are shared across the session
management components to avoid circular imports.
"""

from sqlalchemy import Column, String, DateTime, Text, Integer, Boolean, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID

# Base class for all database models
Base = declarative_base()


class SessionModel(Base):
    """Database model for sessions."""
    __tablename__ = 'sessions'

    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False, index=True)
    status = Column(String, nullable=False, index=True)
    priority = Column(Integer, nullable=False, default=2)
    created_at = Column(DateTime, nullable=False, index=True)
    updated_at = Column(DateTime, nullable=False)
    expires_at = Column(DateTime, nullable=True, index=True)

    # Session metadata
    name = Column(String, nullable=False)
    description = Column(Text)
    session_metadata = Column(Text)  # JSON serialized session metadata

    # Configuration
    config = Column(Text)  # JSON serialized configuration

    # Project and type information
    project_path = Column(String, nullable=False, index=True)
    session_type = Column(String, nullable=False, default="default")
    tags = Column(Text)  # JSON array of tags

    def __repr__(self):
        return f"<SessionModel(id='{self.id}', name='{self.name}', status='{self.status}')>"


class SessionEventModel(Base):
    """Database model for session events."""
    __tablename__ = 'session_events'

    id = Column(String, primary_key=True)
    session_id = Column(String, nullable=False, index=True)
    event_type = Column(String, nullable=False, index=True)
    sequence_number = Column(Integer, nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)

    # Event data
    event_data = Column(Text)  # JSON serialized event data
    agent_id = Column(String, nullable=True, index=True)
    correlation_id = Column(String, nullable=True, index=True)

    def __repr__(self):
        return f"<SessionEventModel(id='{self.id}', session_id='{self.session_id}', event_type='{self.event_type}')>"


class SessionSnapshotModel(Base):
    """Database model for session snapshots."""
    __tablename__ = 'session_snapshots'

    id = Column(String, primary_key=True)
    session_id = Column(String, nullable=False, index=True)
    sequence_number = Column(Integer, nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)

    # Snapshot data - using binary for large states
    snapshot_data = Column(LargeBinary)  # Pickled SessionSnapshot
    compression_type = Column(String, default="gzip")

    def __repr__(self):
        return f"<SessionSnapshotModel(id='{self.id}', session_id='{self.session_id}', sequence_number={self.sequence_number})>"
