"""
Advanced Session Management System for Aider Hive Architecture.

This module provides comprehensive session management with persistence, recovery,
and state management capabilities. It implements event sourcing with snapshots,
hierarchical context organization, and robust recovery mechanisms.

Key Features:
- Event sourcing with periodic snapshots for audit trails and recovery
- Persistent session storage with SQLite/PostgreSQL support
- Hierarchical context management and semantic organization
- Concurrent session handling with proper isolation
- Automatic session cleanup and archival
- Recovery mechanisms for system restarts
- Integration with vector databases for context retrieval
"""

import asyncio
import json
import logging
import pickle
import sqlite3
import uuid
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, Tuple
from contextlib import asynccontextmanager

import structlog
from sqlalchemy import create_engine, Column, String, DateTime, Text, Integer, Boolean, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy import event

from ..context.context_store import GlobalContextStore, ContextEntry
from .session_events import SessionEvent, EventType, SessionEventStore
from .context_hierarchy import HierarchicalContextManager
from .session_recovery import SessionRecoveryManager


Base = declarative_base()


class SessionStatus(Enum):
    """Status of a session."""
    CREATED = "created"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    COMPLETED = "completed"
    FAILED = "failed"
    ARCHIVED = "archived"
    EXPIRED = "expired"


class SessionPriority(Enum):
    """Session priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class SessionMetadata:
    """Metadata associated with a session."""
    user_id: str
    project_path: str
    session_type: str = "default"
    priority: SessionPriority = SessionPriority.NORMAL
    tags: List[str] = field(default_factory=list)
    custom_data: Dict[str, Any] = field(default_factory=dict)
    created_by: str = "system"
    expires_at: Optional[datetime] = None


@dataclass
class SessionSnapshot:
    """Complete snapshot of session state at a point in time."""
    session_id: str
    timestamp: datetime
    sequence_number: int
    agent_states: Dict[str, Any]
    context_state: Dict[str, Any]
    conversation_state: Dict[str, Any]
    workflow_state: Dict[str, Any]
    metadata: Dict[str, Any]
    checkpoint_reason: str = "periodic"


@dataclass
class SessionStatistics:
    """Statistics and metrics for a session."""
    total_events: int = 0
    total_messages: int = 0
    agent_activations: Dict[str, int] = field(default_factory=dict)
    context_retrievals: int = 0
    errors_count: int = 0
    last_activity: Optional[datetime] = None
    processing_time_total: float = 0.0
    memory_usage_peak: float = 0.0


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
    last_activity = Column(DateTime, nullable=True, index=True)

    # JSON fields for flexible data
    metadata = Column(Text)  # JSON serialized SessionMetadata
    statistics = Column(Text)  # JSON serialized SessionStatistics
    current_state = Column(Text)  # JSON serialized current state

    # Project and type information
    project_path = Column(String, nullable=False, index=True)
    session_type = Column(String, nullable=False, default="default")
    tags = Column(Text)  # JSON array of tags


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


class SessionSnapshotModel(Base):
    """Database model for session snapshots."""
    __tablename__ = 'session_snapshots'

    id = Column(String, primary_key=True)
    session_id = Column(String, nullable=False, index=True)
    sequence_number = Column(Integer, nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)

    # Snapshot data - using binary for large states
    snapshot_data = Column(LargeBinary)  # Pickled SessionSnapshot
    checkpoint_reason = Column(String, nullable=False)
    size_bytes = Column(Integer, nullable=False)


class SessionManager:
    """
    Advanced session management system with persistence and recovery.

    Provides comprehensive session lifecycle management, event sourcing,
    state persistence, and recovery capabilities for the Hive system.
    """

    def __init__(
        self,
        database_url: str = "sqlite:///./hive_sessions.db",
        context_store: Optional[GlobalContextStore] = None,
        snapshot_interval: int = 100,  # Events between snapshots
        cleanup_interval: int = 3600,  # Seconds between cleanup runs
        max_session_age_hours: int = 168,  # 1 week default
        max_inactive_hours: int = 24,  # 1 day default
        enable_compression: bool = True,
        enable_encryption: bool = False
    ):
        """Initialize the session manager."""
        self.database_url = database_url
        self.context_store = context_store
        self.snapshot_interval = snapshot_interval
        self.cleanup_interval = cleanup_interval
        self.max_session_age_hours = max_session_age_hours
        self.max_inactive_hours = max_inactive_hours
        self.enable_compression = enable_compression
        self.enable_encryption = enable_encryption

        # Core components
        self.logger = structlog.get_logger().bind(component="session_manager")
        self.engine = None
        self.session_factory = None
        self.event_store = None
        self.context_manager = None
        self.recovery_manager = None

        # Active sessions cache
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.session_locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

        # Background tasks
        self.cleanup_task: Optional[asyncio.Task] = None
        self.snapshot_tasks: Dict[str, asyncio.Task] = {}

        # Metrics
        self.metrics = {
            'sessions_created': 0,
            'sessions_restored': 0,
            'snapshots_created': 0,
            'events_stored': 0,
            'cleanup_runs': 0
        }

    async def initialize(self) -> bool:
        """Initialize the session manager and its components."""
        try:
            self.logger.info("Initializing Session Manager")

            # Setup database
            await self._setup_database()

            # Initialize event store
            self.event_store = SessionEventStore(self.session_factory)
            await self.event_store.initialize()

            # Initialize context manager
            self.context_manager = HierarchicalContextManager(
                context_store=self.context_store,
                enable_semantic_search=True,
                enable_caching=True
            )
            await self.context_manager.initialize()

            # Initialize recovery manager
            self.recovery_manager = SessionRecoveryManager(
                session_factory=self.session_factory,
                event_store=self.event_store,
                context_manager=self.context_manager
            )

            # Load active sessions from database
            await self._load_active_sessions()

            # Start background tasks
            await self._start_background_tasks()

            self.logger.info("Session Manager initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize session manager: {e}", exc_info=True)
            return False

    async def create_session(
        self,
        user_id: str,
        project_path: str,
        session_type: str = "default",
        metadata: Optional[SessionMetadata] = None,
        initial_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new session with proper initialization."""
        session_id = str(uuid.uuid4())

        try:
            async with self.session_locks[session_id]:
                self.logger.info(f"Creating session {session_id} for user {user_id}")

                # Create metadata if not provided
                if metadata is None:
                    metadata = SessionMetadata(
                        user_id=user_id,
                        project_path=project_path,
                        session_type=session_type
                    )

                # Initialize session state
                session_state = {
                    'id': session_id,
                    'status': SessionStatus.CREATED,
                    'metadata': metadata,
                    'statistics': SessionStatistics(),
                    'agent_states': {},
                    'context_state': {},
                    'conversation_history': [],
                    'workflow_state': {},
                    'created_at': datetime.utcnow(),
                    'updated_at': datetime.utcnow()
                }

                # Store in database
                await self._store_session(session_state)

                # Cache active session
                self.active_sessions[session_id] = session_state

                # Initialize context hierarchy
                if initial_context:
                    await self.context_manager.initialize_session_context(
                        session_id, initial_context
                    )

                # Create initial event
                await self._create_event(
                    session_id=session_id,
                    event_type=EventType.SESSION_CREATED,
                    event_data={'metadata': asdict(metadata)},
                    agent_id="system"
                )

                # Start snapshot task
                self._start_session_snapshot_task(session_id)

                self.metrics['sessions_created'] += 1

                self.logger.info(f"Session {session_id} created successfully")
                return session_id

        except Exception as e:
            self.logger.error(f"Failed to create session {session_id}: {e}", exc_info=True)
            # Cleanup on failure
            await self._cleanup_failed_session(session_id)
            raise

    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session state, loading from database if not in cache."""
        try:
            # Check cache first
            if session_id in self.active_sessions:
                return self.active_sessions[session_id].copy()

            # Load from database
            async with self._get_db_session() as db_session:
                session_model = db_session.query(SessionModel).filter(
                    SessionModel.id == session_id
                ).first()

                if not session_model:
                    return None

                # Reconstruct session state
                session_state = await self._model_to_session_state(session_model)

                # Cache if active
                if session_state['status'] in [SessionStatus.ACTIVE, SessionStatus.SUSPENDED]:
                    self.active_sessions[session_id] = session_state

                return session_state

        except Exception as e:
            self.logger.error(f"Failed to get session {session_id}: {e}", exc_info=True)
            return None

    async def update_session_state(
        self,
        session_id: str,
        agent_id: str,
        state_update: Dict[str, Any],
        event_type: EventType = EventType.STATE_UPDATED
    ) -> bool:
        """Update session state with proper event logging."""
        try:
            async with self.session_locks[session_id]:
                session_state = self.active_sessions.get(session_id)
                if not session_state:
                    session_state = await self.get_session(session_id)
                    if not session_state:
                        self.logger.error(f"Session {session_id} not found")
                        return False

                # Update agent state
                if 'agent_states' not in session_state:
                    session_state['agent_states'] = {}

                session_state['agent_states'][agent_id] = state_update
                session_state['updated_at'] = datetime.utcnow()
                session_state['statistics'].last_activity = datetime.utcnow()

                # Update statistics
                if agent_id not in session_state['statistics'].agent_activations:
                    session_state['statistics'].agent_activations[agent_id] = 0
                session_state['statistics'].agent_activations[agent_id] += 1

                # Store event
                await self._create_event(
                    session_id=session_id,
                    event_type=event_type,
                    event_data={'state_update': state_update},
                    agent_id=agent_id
                )

                # Update cache
                self.active_sessions[session_id] = session_state

                # Trigger snapshot if needed
                await self._check_snapshot_needed(session_id)

                return True

        except Exception as e:
            self.logger.error(f"Failed to update session {session_id}: {e}", exc_info=True)
            return False

    async def add_conversation_message(
        self,
        session_id: str,
        message: Dict[str, Any],
        agent_id: Optional[str] = None
    ) -> bool:
        """Add a message to the conversation history."""
        try:
            async with self.session_locks[session_id]:
                session_state = self.active_sessions.get(session_id)
                if not session_state:
                    session_state = await self.get_session(session_id)
                    if not session_state:
                        return False

                # Add timestamp and ID if not present
                message.update({
                    'timestamp': datetime.utcnow().isoformat(),
                    'message_id': str(uuid.uuid4()),
                    'agent_id': agent_id
                })

                # Add to conversation history
                if 'conversation_history' not in session_state:
                    session_state['conversation_history'] = []

                session_state['conversation_history'].append(message)
                session_state['updated_at'] = datetime.utcnow()
                session_state['statistics'].total_messages += 1
                session_state['statistics'].last_activity = datetime.utcnow()

                # Store event
                await self._create_event(
                    session_id=session_id,
                    event_type=EventType.MESSAGE_ADDED,
                    event_data={'message': message},
                    agent_id=agent_id
                )

                # Update context if message contains relevant information
                await self._update_context_from_message(session_id, message)

                return True

        except Exception as e:
            self.logger.error(f"Failed to add message to session {session_id}: {e}", exc_info=True)
            return False

    async def suspend_session(self, session_id: str, reason: str = "user_request") -> bool:
        """Suspend a session, preserving state for later resumption."""
        try:
            async with self.session_locks[session_id]:
                session_state = self.active_sessions.get(session_id)
                if not session_state:
                    return False

                # Update status
                session_state['status'] = SessionStatus.SUSPENDED
                session_state['updated_at'] = datetime.utcnow()

                # Create snapshot before suspension
                await self._create_snapshot(session_id, reason="suspension")

                # Store event
                await self._create_event(
                    session_id=session_id,
                    event_type=EventType.SESSION_SUSPENDED,
                    event_data={'reason': reason},
                    agent_id="system"
                )

                # Persist to database
                await self._store_session(session_state)

                self.logger.info(f"Session {session_id} suspended: {reason}")
                return True

        except Exception as e:
            self.logger.error(f"Failed to suspend session {session_id}: {e}", exc_info=True)
            return False

    async def resume_session(self, session_id: str) -> bool:
        """Resume a suspended session."""
        try:
            # Use recovery manager for complex restoration
            success = await self.recovery_manager.restore_session(session_id)

            if success:
                session_state = await self.get_session(session_id)
                if session_state:
                    session_state['status'] = SessionStatus.ACTIVE
                    session_state['updated_at'] = datetime.utcnow()

                    await self._create_event(
                        session_id=session_id,
                        event_type=EventType.SESSION_RESUMED,
                        event_data={},
                        agent_id="system"
                    )

                    # Restart snapshot task
                    self._start_session_snapshot_task(session_id)

                    self.metrics['sessions_restored'] += 1
                    self.logger.info(f"Session {session_id} resumed successfully")

            return success

        except Exception as e:
            self.logger.error(f"Failed to resume session {session_id}: {e}", exc_info=True)
            return False

    async def complete_session(
        self,
        session_id: str,
        completion_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Mark a session as completed and archive its data."""
        try:
            async with self.session_locks[session_id]:
                session_state = self.active_sessions.get(session_id)
                if not session_state:
                    session_state = await self.get_session(session_id)
                    if not session_state:
                        return False

                # Update status
                session_state['status'] = SessionStatus.COMPLETED
                session_state['updated_at'] = datetime.utcnow()

                if completion_data:
                    session_state['completion_data'] = completion_data

                # Create final snapshot
                await self._create_snapshot(session_id, reason="completion")

                # Store completion event
                await self._create_event(
                    session_id=session_id,
                    event_type=EventType.SESSION_COMPLETED,
                    event_data=completion_data or {},
                    agent_id="system"
                )

                # Persist final state
                await self._store_session(session_state)

                # Remove from active sessions
                if session_id in self.active_sessions:
                    del self.active_sessions[session_id]

                # Stop snapshot task
                if session_id in self.snapshot_tasks:
                    self.snapshot_tasks[session_id].cancel()
                    del self.snapshot_tasks[session_id]

                self.logger.info(f"Session {session_id} completed successfully")
                return True

        except Exception as e:
            self.logger.error(f"Failed to complete session {session_id}: {e}", exc_info=True)
            return False

    async def get_session_history(
        self,
        session_id: str,
        include_events: bool = True,
        include_snapshots: bool = False,
        limit: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """Get complete session history with events and snapshots."""
        try:
            session_state = await self.get_session(session_id)
            if not session_state:
                return None

            history = {
                'session': session_state,
                'events': [],
                'snapshots': []
            }

            if include_events:
                events = await self.event_store.get_session_events(
                    session_id, limit=limit
                )
                history['events'] = events

            if include_snapshots:
                snapshots = await self._get_session_snapshots(session_id, limit=limit)
                history['snapshots'] = snapshots

            return history

        except Exception as e:
            self.logger.error(f"Failed to get session history {session_id}: {e}", exc_info=True)
            return None

    async def cleanup_expired_sessions(self) -> Dict[str, int]:
        """Clean up expired and old sessions."""
        try:
            self.logger.info("Starting session cleanup")

            cleanup_stats = {
                'expired': 0,
                'inactive': 0,
                'archived': 0,
                'failed': 0
            }

            current_time = datetime.utcnow()
            max_age = current_time - timedelta(hours=self.max_session_age_hours)
            max_inactive = current_time - timedelta(hours=self.max_inactive_hours)

            async with self._get_db_session() as db_session:
                # Find sessions to clean up
                expired_sessions = db_session.query(SessionModel).filter(
                    SessionModel.expires_at < current_time
                ).all()

                inactive_sessions = db_session.query(SessionModel).filter(
                    SessionModel.last_activity < max_inactive,
                    SessionModel.status.in_(['active', 'suspended'])
                ).all()

                old_sessions = db_session.query(SessionModel).filter(
                    SessionModel.created_at < max_age,
                    SessionModel.status == 'completed'
                ).all()

                # Process expired sessions
                for session in expired_sessions:
                    await self._expire_session(session.id)
                    cleanup_stats['expired'] += 1

                # Process inactive sessions
                for session in inactive_sessions:
                    await self._mark_inactive(session.id)
                    cleanup_stats['inactive'] += 1

                # Archive old completed sessions
                for session in old_sessions:
                    await self._archive_session(session.id)
                    cleanup_stats['archived'] += 1

            self.metrics['cleanup_runs'] += 1
            self.logger.info(f"Session cleanup completed: {cleanup_stats}")

            return cleanup_stats

        except Exception as e:
            self.logger.error(f"Session cleanup failed: {e}", exc_info=True)
            return {'failed': 1}

    async def stop(self) -> None:
        """Stop the session manager and cleanup resources."""
        try:
            self.logger.info("Stopping Session Manager")

            # Cancel background tasks
            if self.cleanup_task:
                self.cleanup_task.cancel()

            for task in self.snapshot_tasks.values():
                task.cancel()

            # Create final snapshots for active sessions
            for session_id in list(self.active_sessions.keys()):
                await self._create_snapshot(session_id, reason="shutdown")

            # Persist active sessions
            for session_id, session_state in self.active_sessions.items():
                await self._store_session(session_state)

            # Close database connections
            if self.engine:
                self.engine.dispose()

            self.logger.info("Session Manager stopped successfully")

        except Exception as e:
            self.logger.error(f"Error stopping session manager: {e}", exc_info=True)

    # Private helper methods

    async def _setup_database(self) -> None:
        """Setup database connection and tables."""
        self.engine = create_engine(
            self.database_url,
            echo=False,
            pool_pre_ping=True
        )

        # Create tables
        Base.metadata.create_all(self.engine)

        # Create session factory
        self.session_factory = sessionmaker(bind=self.engine)

        self.logger.info(f"Database setup completed: {self.database_url}")

    @asynccontextmanager
    async def _get_db_session(self):
        """Get database session with proper cleanup."""
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    async def _store_session(self, session_state: Dict[str, Any]) -> None:
        """Store session state in database."""
        async with self._get_db_session() as db_session:
            session_model = db_session.query(SessionModel).filter(
                SessionModel.id == session_state['id']
            ).first()

            if session_model:
                # Update existing
                session_model.status = session_state['status'].value
                session_model.updated_at = session_state['updated_at']
                session_model.last_activity = session_state['statistics'].last_activity
                session_model.metadata = json.dumps(asdict(session_state['metadata']))
                session_model.statistics = json.dumps(asdict(session_state['statistics']))
                session_model.current_state = json.dumps(session_state, default=str)
            else:
                # Create new
                session_model = SessionModel(
                    id=session_state['id'],
                    user_id=session_state['metadata'].user_id,
                    status=session_state['status'].value,
                    priority=session_state['metadata'].priority.value,
                    created_at=session_state['created_at'],
                    updated_at=session_state['updated_at'],
                    project_path=session_state['metadata'].project_path,
                    session_type=session_state['metadata'].session_type,
                    metadata=json.dumps(asdict(session_state['metadata'])),
                    statistics=json.dumps(asdict(session_state['statistics'])),
                    current_state=json.dumps(session_state, default=str),
                    tags=json.dumps(session_state['metadata'].tags)
                )
                db_session.add(session_model)

    async def _create_event(
        self,
        session_id: str,
        event_type: EventType,
        event_data: Dict[str, Any],
        agent_id: Optional[str] = None,
        correlation_id: Optional[str] = None
    ) -> str:
        """Create and store a session event."""
        event = SessionEvent(
            event_id=str(uuid.uuid4()),
            session_id=session_id,
            event_type=event_type,
            timestamp=datetime.utcnow(),
            event_data=event_data,
            agent_id=agent_id,
            correlation_id=correlation_id
        )

        await self.event_store.store_event(event)
        self.metrics['events_stored'] += 1

        return event.event_id

    async def _create_snapshot(self, session_id: str, reason: str = "periodic") -> bool:
        """Create a snapshot of current session state."""
        try:
            session_state = self.active_sessions.get(session_id)
            if not session_state:
                return False

            # Get next sequence number
            sequence_number = await self._get_next_snapshot_sequence(session_id)

            # Create snapshot
            snapshot = SessionSnapshot(
                session_id=session_id,
                timestamp=datetime.utcnow(),
                sequence_number=sequence_number,
                agent_states=session_state.get('agent_states', {}),
                context_state=await self.context_manager.get_session_context(session_id),
                conversation_state={'history': session_state.get('conversation_history', [])},
                workflow_state=session_state.get('workflow_state', {}),
                metadata=asdict(session_state['metadata']),
                checkpoint_reason=reason
            )

            # Store snapshot
            await self._store_snapshot(snapshot)

            self.metrics['snapshots_created'] += 1
            self.logger.debug(f"Created snapshot for session {session_id}: {reason}")

            return True

        except Exception as e:
            self.logger.error(f"Failed to create snapshot for {session_id}: {e}", exc_info=True)
            return False

    async def _store_snapshot(self, snapshot: SessionSnapshot) -> None:
        """Store snapshot in database."""
        async with self._get_db_session() as db_session:
            # Serialize snapshot data
            snapshot_data = pickle.dumps(snapshot)
            if self.enable_compression:
                import gzip
                snapshot_data = gzip.compress(snapshot_data)

            snapshot_model = SessionSnapshotModel(
                id=str(uuid.uuid4()),
                session_id=snapshot.session_id,
                sequence_number=snapshot.sequence_number,
                timestamp=snapshot.timestamp,
                snapshot_data=snapshot_data,
                checkpoint_reason=snapshot.checkpoint_reason,
                size_bytes=len(snapshot_data)
            )

            db_session.add(snapshot_model)

    def _start_session_snapshot_task(self, session_id: str) -> None:
        """Start periodic snapshot task for a session."""
        async def snapshot_loop():
            while session_id in self.active_sessions:
                try:
                    await asyncio.sleep(60)  # Check every minute
                    await self._check_snapshot_needed(session_id)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"Snapshot task error for {session_id}: {e}")

        task = asyncio.create_task(snapshot_loop())
        self.snapshot_tasks[session_id] = task

    async def _check_snapshot_needed(self, session_id: str) -> None:
        """Check if snapshot is needed based on activity."""
        session_state = self.active_sessions.get(session_id)
        if not session_state:
            return

        # Get event count since last snapshot
        event_count = session_state['statistics'].total_events
        last_snapshot_events = session_state.get('last_snapshot_events', 0)

        if event_count - last_snapshot_events >= self.snapshot_interval:
            await self._create_snapshot(session_id, reason="periodic")
            session_state['last_snapshot_events'] = event_count

    async def _get_next_snapshot_sequence(self, session_id: str) -> int:
        """Get the next sequence number for snapshots."""
        async with self._get_db_session() as db_session:
            last_snapshot = db_session.query(SessionSnapshotModel).filter(
                SessionSnapshotModel.session_id == session_id
            ).order_by(SessionSnapshotModel.sequence_number.desc()).first()

            return (last_snapshot.sequence_number + 1) if last_snapshot else 1

    async def _load_active_sessions(self) -> None:
        """Load active sessions from database on startup."""
        try:
            async with self._get_db_session() as db_session:
                active_sessions = db_session.query(SessionModel).filter(
                    SessionModel.status.in_(['active', 'suspended'])
                ).all()

                for session_model in active_sessions:
                    session_state = await self._model_to_session_state(session_model)
                    self.active_sessions[session_state['id']] = session_state

                    # Start snapshot task if active
                    if session_state['status'] == SessionStatus.ACTIVE:
                        self._start_session_snapshot_task(session_state['id'])

                self.logger.info(f"Loaded {len(active_sessions)} active sessions")

        except Exception as e:
            self.logger.error(f"Failed to load active sessions: {e}", exc_info=True)

    async def _model_to_session_state(self, session_model: SessionModel) -> Dict[str, Any]:
        """Convert database model to session state dictionary."""
        try:
            # Parse JSON fields
            metadata_dict = json.loads(session_model.metadata) if session_model.metadata else {}
            statistics_dict = json.loads(session_model.statistics) if session_model.statistics else {}
            current_state = json.loads(session_model.current_state) if session_model.current_state else {}
            tags = json.loads(session_model.tags) if session_model.tags else []

            # Reconstruct metadata
            metadata = SessionMetadata(
                user_id=metadata_dict.get('user_id', session_model.user_id),
                project_path=metadata_dict.get('project_path', session_model.project_path),
                session_type=metadata_dict.get('session_type', session_model.session_type),
                priority=SessionPriority(metadata_dict.get('priority', session_model.priority)),
                tags=tags,
                custom_data=metadata_dict.get('custom_data', {}),
                created_by=metadata_dict.get('created_by', 'system'),
                expires_at=session_model.expires_at
            )

            # Reconstruct statistics
            statistics = SessionStatistics(
                total_events=statistics_dict.get('total_events', 0),
                total_messages=statistics_dict.get('total_messages', 0),
                agent_activations=statistics_dict.get('agent_activations', {}),
                context_retrievals=statistics_dict.get('context_retrievals', 0),
                errors_count=statistics_dict.get('errors_count', 0),
                last_activity=session_model.last_activity,
                processing_time_total=statistics_dict.get('processing_time_total', 0.0),
                memory_usage_peak=statistics_dict.get('memory_usage_peak', 0.0)
            )

            # Build session state
            session_state = {
                'id': session_model.id,
                'status': SessionStatus(session_model.status),
                'metadata': metadata,
                'statistics': statistics,
                'created_at': session_model.created_at,
                'updated_at': session_model.updated_at,
                'agent_states': current_state.get('agent_states', {}),
                'context_state': current_state.get('context_state', {}),
                'conversation_history': current_state.get('conversation_history', []),
                'workflow_state': current_state.get('workflow_state', {}),
                'last_snapshot_events': current_state.get('last_snapshot_events', 0)
            }

            return session_state

        except Exception as e:
            self.logger.error(f"Failed to convert session model: {e}", exc_info=True)
            raise

    async def _start_background_tasks(self) -> None:
        """Start background maintenance tasks."""
        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(self.cleanup_interval)
                    await self.cleanup_expired_sessions()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"Cleanup task error: {e}")

        self.cleanup_task = asyncio.create_task(cleanup_loop())
        self.logger.info("Background tasks started")

    async def _cleanup_failed_session(self, session_id: str) -> None:
        """Clean up resources for a failed session creation."""
        try:
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]

            if session_id in self.session_locks:
                del self.session_locks[session_id]

            if session_id in self.snapshot_tasks:
                self.snapshot_tasks[session_id].cancel()
                del self.snapshot_tasks[session_id]

            # Remove from database if exists
            async with self._get_db_session() as db_session:
                session_model = db_session.query(SessionModel).filter(
                    SessionModel.id == session_id
                ).first()
                if session_model:
                    db_session.delete(session_model)

        except Exception as e:
            self.logger.error(f"Failed to cleanup failed session {session_id}: {e}")

    async def _update_context_from_message(
        self,
        session_id: str,
        message: Dict[str, Any]
    ) -> None:
        """Update session context based on message content."""
        try:
            # Extract relevant context from message
            if 'content' in message and self.context_manager:
                await self.context_manager.add_message_context(
                    session_id=session_id,
                    message=message,
                    extract_entities=True,
                    update_semantic_index=True
                )
        except Exception as e:
            self.logger.error(f"Failed to update context from message: {e}")

    async def _expire_session(self, session_id: str) -> None:
        """Mark session as expired and clean up."""
        try:
            async with self.session_locks[session_id]:
                session_state = self.active_sessions.get(session_id)
                if session_state:
                    session_state['status'] = SessionStatus.EXPIRED
                    session_state['updated_at'] = datetime.utcnow()
                    await self._store_session(session_state)

                    if session_id in self.active_sessions:
                        del self.active_sessions[session_id]

                await self._create_event(
                    session_id=session_id,
                    event_type=EventType.SESSION_EXPIRED,
                    event_data={'reason': 'expired'},
                    agent_id="system"
                )

        except Exception as e:
            self.logger.error(f"Failed to expire session {session_id}: {e}")

    async def _mark_inactive(self, session_id: str) -> None:
        """Mark session as inactive due to no activity."""
        try:
            await self.suspend_session(session_id, reason="inactivity")
        except Exception as e:
            self.logger.error(f"Failed to mark session inactive {session_id}: {e}")

    async def _archive_session(self, session_id: str) -> None:
        """Archive an old completed session."""
        try:
            async with self._get_db_session() as db_session:
                session_model = db_session.query(SessionModel).filter(
                    SessionModel.id == session_id
                ).first()

                if session_model:
                    session_model.status = SessionStatus.ARCHIVED.value
                    session_model.updated_at = datetime.utcnow()

            await self._create_event(
                session_id=session_id,
                event_type=EventType.SESSION_ARCHIVED,
                event_data={'reason': 'old_completed'},
                agent_id="system"
            )

        except Exception as e:
            self.logger.error(f"Failed to archive session {session_id}: {e}")

    async def _get_session_snapshots(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[SessionSnapshot]:
        """Get snapshots for a session."""
        try:
            async with self._get_db_session() as db_session:
                query = db_session.query(SessionSnapshotModel).filter(
                    SessionSnapshotModel.session_id == session_id
                ).order_by(SessionSnapshotModel.sequence_number.desc())

                if limit:
                    query = query.limit(limit)

                snapshot_models = query.all()

                snapshots = []
                for model in snapshot_models:
                    # Deserialize snapshot data
                    snapshot_data = model.snapshot_data
                    if self.enable_compression:
                        import gzip
                        snapshot_data = gzip.decompress(snapshot_data)

                    snapshot = pickle.loads(snapshot_data)
                    snapshots.append(snapshot)

                return snapshots

        except Exception as e:
            self.logger.error(f"Failed to get snapshots for {session_id}: {e}")
            return []

    # Public utility methods

    async def get_active_sessions(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get list of active sessions, optionally filtered by user."""
        try:
            sessions = []
            for session_id, session_state in self.active_sessions.items():
                if user_id and session_state['metadata'].user_id != user_id:
                    continue

                sessions.append({
                    'id': session_id,
                    'user_id': session_state['metadata'].user_id,
                    'project_path': session_state['metadata'].project_path,
                    'status': session_state['status'].value,
                    'created_at': session_state['created_at'].isoformat(),
                    'last_activity': session_state['statistics'].last_activity.isoformat()
                        if session_state['statistics'].last_activity else None,
                    'total_messages': session_state['statistics'].total_messages,
                    'agent_count': len(session_state.get('agent_states', {}))
                })

            return sessions

        except Exception as e:
            self.logger.error(f"Failed to get active sessions: {e}")
            return []

    async def get_session_statistics(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed statistics for a session."""
        try:
            session_state = await self.get_session(session_id)
            if not session_state:
                return None

            stats = session_state['statistics']
            return {
                'session_id': session_id,
                'total_events': stats.total_events,
                'total_messages': stats.total_messages,
                'agent_activations': stats.agent_activations,
                'context_retrievals': stats.context_retrievals,
                'errors_count': stats.errors_count,
                'processing_time_total': stats.processing_time_total,
                'memory_usage_peak': stats.memory_usage_peak,
                'last_activity': stats.last_activity.isoformat() if stats.last_activity else None,
                'uptime': (datetime.utcnow() - session_state['created_at']).total_seconds(),
                'status': session_state['status'].value
            }

        except Exception as e:
            self.logger.error(f"Failed to get session statistics: {e}")
            return None

    def get_metrics(self) -> Dict[str, Any]:
        """Get session manager metrics."""
        return {
            **self.metrics,
            'active_sessions_count': len(self.active_sessions),
            'background_tasks_count': 1 + len(self.snapshot_tasks),
            'memory_cache_size': len(self.active_sessions)
        }
