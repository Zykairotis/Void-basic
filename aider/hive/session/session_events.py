"""
Session Events System for Event Sourcing in Aider Hive Architecture.

This module implements a comprehensive event sourcing system for session management,
providing audit trails, recovery capabilities, and state reconstruction through
event replay. It supports various event types throughout the session lifecycle
and provides efficient storage and retrieval mechanisms.

Key Features:
- Comprehensive event type definitions for session lifecycle
- Event storage and retrieval with filtering and querying
- Event replay capabilities for state reconstruction
- Efficient serialization and deserialization
- Database integration with proper indexing
- Event correlation and causality tracking
- Batch processing for high-throughput scenarios
"""

import asyncio
import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union, AsyncIterator
from collections import defaultdict

import structlog
from sqlalchemy import and_, or_, desc, asc
from sqlalchemy.orm import Session

from .models import SessionEventModel


class EventType(Enum):
    """Comprehensive event types for session lifecycle and operations."""

    # Session Lifecycle Events
    SESSION_CREATED = "session_created"
    SESSION_STARTED = "session_started"
    SESSION_SUSPENDED = "session_suspended"
    SESSION_RESUMED = "session_resumed"
    SESSION_COMPLETED = "session_completed"
    SESSION_FAILED = "session_failed"
    SESSION_EXPIRED = "session_expired"
    SESSION_ARCHIVED = "session_archived"
    SESSION_DELETED = "session_deleted"

    # Agent Events
    AGENT_ACTIVATED = "agent_activated"
    AGENT_DEACTIVATED = "agent_deactivated"
    AGENT_STATE_CHANGED = "agent_state_changed"
    AGENT_ERROR = "agent_error"
    AGENT_TIMEOUT = "agent_timeout"

    # Message and Communication Events
    MESSAGE_ADDED = "message_added"
    MESSAGE_UPDATED = "message_updated"
    MESSAGE_DELETED = "message_deleted"
    MESSAGE_SENT = "message_sent"
    MESSAGE_RECEIVED = "message_received"
    MESSAGE_FAILED = "message_failed"

    # Context and Data Events
    CONTEXT_UPDATED = "context_updated"
    CONTEXT_RETRIEVED = "context_retrieved"
    CONTEXT_CACHED = "context_cached"
    CONTEXT_INVALIDATED = "context_invalidated"

    # Workflow and Task Events
    WORKFLOW_STARTED = "workflow_started"
    WORKFLOW_COMPLETED = "workflow_completed"
    WORKFLOW_FAILED = "workflow_failed"
    TASK_CREATED = "task_created"
    TASK_ASSIGNED = "task_assigned"
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    TASK_CANCELLED = "task_cancelled"

    # State and Configuration Events
    STATE_UPDATED = "state_updated"
    STATE_SNAPSHOT_CREATED = "state_snapshot_created"
    CONFIG_UPDATED = "config_updated"
    SETTINGS_CHANGED = "settings_changed"

    # Error and Recovery Events
    ERROR_OCCURRED = "error_occurred"
    ERROR_RECOVERED = "error_recovered"
    RETRY_ATTEMPTED = "retry_attempted"
    ROLLBACK_INITIATED = "rollback_initiated"
    ROLLBACK_COMPLETED = "rollback_completed"

    # Performance and Monitoring Events
    PERFORMANCE_MEASURED = "performance_measured"
    METRIC_RECORDED = "metric_recorded"
    HEALTH_CHECK_PERFORMED = "health_check_performed"
    RESOURCE_ALLOCATED = "resource_allocated"
    RESOURCE_DEALLOCATED = "resource_deallocated"

    # User Interaction Events
    USER_ACTION = "user_action"
    USER_INPUT = "user_input"
    USER_FEEDBACK = "user_feedback"
    USER_PREFERENCE_CHANGED = "user_preference_changed"

    # Code Generation and Modification Events
    CODE_GENERATED = "code_generated"
    CODE_MODIFIED = "code_modified"
    CODE_VALIDATED = "code_validated"
    CODE_COMMITTED = "code_committed"
    CODE_REVERTED = "code_reverted"

    # Integration Events
    EXTERNAL_API_CALLED = "external_api_called"
    EXTERNAL_API_RESPONSE = "external_api_response"
    INTEGRATION_STARTED = "integration_started"
    INTEGRATION_COMPLETED = "integration_completed"
    INTEGRATION_FAILED = "integration_failed"


@dataclass
class SessionEvent:
    """
    Represents a single event in the session's event stream.

    This is the core data structure for event sourcing, containing all
    necessary information to reconstruct state and provide audit trails.
    """
    event_id: str
    session_id: str
    event_type: EventType
    timestamp: datetime
    sequence_number: int = 0
    event_data: Dict[str, Any] = field(default_factory=dict)

    # Event metadata
    agent_id: Optional[str] = None
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None
    user_id: Optional[str] = None

    # Event processing metadata
    version: str = "1.0"
    source: str = "hive_system"
    tags: List[str] = field(default_factory=list)

    # Performance and debugging metadata
    processing_time_ms: Optional[float] = None
    memory_usage_mb: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            'event_id': self.event_id,
            'session_id': self.session_id,
            'event_type': self.event_type.value,
            'timestamp': self.timestamp.isoformat(),
            'sequence_number': self.sequence_number,
            'event_data': self.event_data,
            'agent_id': self.agent_id,
            'correlation_id': self.correlation_id,
            'causation_id': self.causation_id,
            'user_id': self.user_id,
            'version': self.version,
            'source': self.source,
            'tags': self.tags,
            'processing_time_ms': self.processing_time_ms,
            'memory_usage_mb': self.memory_usage_mb
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionEvent':
        """Create event from dictionary."""
        return cls(
            event_id=data['event_id'],
            session_id=data['session_id'],
            event_type=EventType(data['event_type']),
            timestamp=datetime.fromisoformat(data['timestamp']),
            sequence_number=data.get('sequence_number', 0),
            event_data=data.get('event_data', {}),
            agent_id=data.get('agent_id'),
            correlation_id=data.get('correlation_id'),
            causation_id=data.get('causation_id'),
            user_id=data.get('user_id'),
            version=data.get('version', '1.0'),
            source=data.get('source', 'hive_system'),
            tags=data.get('tags', []),
            processing_time_ms=data.get('processing_time_ms'),
            memory_usage_mb=data.get('memory_usage_mb')
        )


@dataclass
class EventQuery:
    """Query parameters for filtering and retrieving events."""
    session_id: Optional[str] = None
    event_types: Optional[List[EventType]] = None
    agent_id: Optional[str] = None
    user_id: Optional[str] = None
    correlation_id: Optional[str] = None

    # Time-based filtering
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    # Sequence-based filtering
    min_sequence: Optional[int] = None
    max_sequence: Optional[int] = None

    # Result options
    limit: Optional[int] = None
    offset: int = 0
    order_by: str = "timestamp"  # timestamp, sequence_number
    order_direction: str = "asc"  # asc, desc

    # Additional filters
    tags: Optional[List[str]] = None
    include_data: bool = True


class SessionEventStore:
    """
    Event store implementation for persisting and retrieving session events.

    Provides efficient storage, querying, and replay capabilities for events
    in the session management system. Supports various query patterns and
    batch operations for high-performance scenarios.
    """

    def __init__(self, session_factory, enable_caching: bool = True, cache_size: int = 1000):
        """Initialize the event store."""
        self.session_factory = session_factory
        self.enable_caching = enable_caching
        self.cache_size = cache_size

        self.logger = structlog.get_logger().bind(component="session_event_store")

        # In-memory cache for recently accessed events
        self.event_cache: Dict[str, SessionEvent] = {}
        self.cache_access_order: List[str] = []

        # Sequence number tracking
        self.sequence_counters: Dict[str, int] = defaultdict(int)

        # Metrics
        self.metrics = {
            'events_stored': 0,
            'events_retrieved': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'queries_executed': 0
        }

    async def initialize(self) -> None:
        """Initialize the event store."""
        try:
            # Load sequence counters from database
            await self._load_sequence_counters()
            self.logger.info("Event store initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize event store: {e}", exc_info=True)
            raise

    async def store_event(self, event: SessionEvent) -> bool:
        """
        Store a single event in the event store.

        Args:
            event: The event to store

        Returns:
            bool: True if stored successfully
        """
        try:
            # Assign sequence number if not set
            if event.sequence_number == 0:
                event.sequence_number = await self._get_next_sequence_number(event.session_id)

            # Store in database
            with self.session_factory() as db_session:
                event_model = SessionEventModel(
                    id=event.event_id,
                    session_id=event.session_id,
                    event_type=event.event_type.value,
                    sequence_number=event.sequence_number,
                    timestamp=event.timestamp,
                    event_data=json.dumps(event.event_data),
                    agent_id=event.agent_id,
                    correlation_id=event.correlation_id
                )

                db_session.add(event_model)
                db_session.commit()

            # Update cache
            if self.enable_caching:
                self._cache_event(event)

            self.metrics['events_stored'] += 1

            self.logger.debug(f"Stored event {event.event_id} for session {event.session_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to store event {event.event_id}: {e}", exc_info=True)
            return False

    async def store_events_batch(self, events: List[SessionEvent]) -> int:
        """
        Store multiple events in a single batch operation.

        Args:
            events: List of events to store

        Returns:
            int: Number of events successfully stored
        """
        if not events:
            return 0

        stored_count = 0

        try:
            with self.session_factory() as db_session:
                for event in events:
                    try:
                        # Assign sequence number if not set
                        if event.sequence_number == 0:
                            event.sequence_number = await self._get_next_sequence_number(event.session_id)

                        event_model = SessionEventModel(
                            id=event.event_id,
                            session_id=event.session_id,
                            event_type=event.event_type.value,
                            sequence_number=event.sequence_number,
                            timestamp=event.timestamp,
                            event_data=json.dumps(event.event_data),
                            agent_id=event.agent_id,
                            correlation_id=event.correlation_id
                        )

                        db_session.add(event_model)

                        # Update cache
                        if self.enable_caching:
                            self._cache_event(event)

                        stored_count += 1

                    except Exception as e:
                        self.logger.error(f"Failed to prepare event {event.event_id}: {e}")
                        continue

                db_session.commit()

            self.metrics['events_stored'] += stored_count
            self.logger.info(f"Stored {stored_count}/{len(events)} events in batch")

        except Exception as e:
            self.logger.error(f"Batch store operation failed: {e}", exc_info=True)

        return stored_count

    async def get_event(self, event_id: str) -> Optional[SessionEvent]:
        """Get a single event by ID."""
        try:
            # Check cache first
            if self.enable_caching and event_id in self.event_cache:
                self.metrics['cache_hits'] += 1
                return self.event_cache[event_id]

            # Query database
            with self.session_factory() as db_session:
                event_model = db_session.query(SessionEventModel).filter(
                    SessionEventModel.id == event_id
                ).first()

                if event_model:
                    event = await self._model_to_event(event_model)

                    # Cache the event
                    if self.enable_caching:
                        self._cache_event(event)

                    self.metrics['events_retrieved'] += 1
                    self.metrics['cache_misses'] += 1
                    return event

            return None

        except Exception as e:
            self.logger.error(f"Failed to get event {event_id}: {e}", exc_info=True)
            return None

    async def get_session_events(
        self,
        session_id: str,
        limit: Optional[int] = None,
        offset: int = 0,
        event_types: Optional[List[EventType]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[SessionEvent]:
        """Get events for a specific session with optional filtering."""
        try:
            with self.session_factory() as db_session:
                query = db_session.query(SessionEventModel).filter(
                    SessionEventModel.session_id == session_id
                )

                # Apply filters
                if event_types:
                    event_type_values = [et.value for et in event_types]
                    query = query.filter(SessionEventModel.event_type.in_(event_type_values))

                if start_time:
                    query = query.filter(SessionEventModel.timestamp >= start_time)

                if end_time:
                    query = query.filter(SessionEventModel.timestamp <= end_time)

                # Apply ordering and pagination
                query = query.order_by(asc(SessionEventModel.sequence_number))

                if offset > 0:
                    query = query.offset(offset)

                if limit:
                    query = query.limit(limit)

                event_models = query.all()

                # Convert to events
                events = []
                for model in event_models:
                    event = await self._model_to_event(model)
                    events.append(event)

                    # Cache the event
                    if self.enable_caching:
                        self._cache_event(event)

                self.metrics['events_retrieved'] += len(events)
                self.metrics['queries_executed'] += 1

                return events

        except Exception as e:
            self.logger.error(f"Failed to get session events: {e}", exc_info=True)
            return []

    async def query_events(self, query: EventQuery) -> List[SessionEvent]:
        """Execute a complex event query."""
        try:
            with self.session_factory() as db_session:
                db_query = db_session.query(SessionEventModel)

                # Apply filters
                if query.session_id:
                    db_query = db_query.filter(SessionEventModel.session_id == query.session_id)

                if query.event_types:
                    event_type_values = [et.value for et in query.event_types]
                    db_query = db_query.filter(SessionEventModel.event_type.in_(event_type_values))

                if query.agent_id:
                    db_query = db_query.filter(SessionEventModel.agent_id == query.agent_id)

                if query.correlation_id:
                    db_query = db_query.filter(SessionEventModel.correlation_id == query.correlation_id)

                if query.start_time:
                    db_query = db_query.filter(SessionEventModel.timestamp >= query.start_time)

                if query.end_time:
                    db_query = db_query.filter(SessionEventModel.timestamp <= query.end_time)

                if query.min_sequence is not None:
                    db_query = db_query.filter(SessionEventModel.sequence_number >= query.min_sequence)

                if query.max_sequence is not None:
                    db_query = db_query.filter(SessionEventModel.sequence_number <= query.max_sequence)

                # Apply ordering
                if query.order_by == "sequence_number":
                    order_col = SessionEventModel.sequence_number
                else:
                    order_col = SessionEventModel.timestamp

                if query.order_direction == "desc":
                    db_query = db_query.order_by(desc(order_col))
                else:
                    db_query = db_query.order_by(asc(order_col))

                # Apply pagination
                if query.offset > 0:
                    db_query = db_query.offset(query.offset)

                if query.limit:
                    db_query = db_query.limit(query.limit)

                event_models = db_query.all()

                # Convert to events
                events = []
                for model in event_models:
                    event = await self._model_to_event(model)
                    events.append(event)

                    # Cache the event
                    if self.enable_caching:
                        self._cache_event(event)

                self.metrics['events_retrieved'] += len(events)
                self.metrics['queries_executed'] += 1

                return events

        except Exception as e:
            self.logger.error(f"Failed to execute event query: {e}", exc_info=True)
            return []

    async def replay_events(
        self,
        session_id: str,
        from_sequence: int = 0,
        to_sequence: Optional[int] = None
    ) -> AsyncIterator[SessionEvent]:
        """
        Replay events for a session in sequence order.

        Args:
            session_id: The session to replay events for
            from_sequence: Starting sequence number (inclusive)
            to_sequence: Ending sequence number (inclusive, optional)

        Yields:
            SessionEvent: Events in sequence order
        """
        try:
            with self.session_factory() as db_session:
                query = db_session.query(SessionEventModel).filter(
                    SessionEventModel.session_id == session_id,
                    SessionEventModel.sequence_number >= from_sequence
                )

                if to_sequence is not None:
                    query = query.filter(SessionEventModel.sequence_number <= to_sequence)

                query = query.order_by(asc(SessionEventModel.sequence_number))

                for event_model in query:
                    event = await self._model_to_event(event_model)
                    yield event

        except Exception as e:
            self.logger.error(f"Failed to replay events for session {session_id}: {e}", exc_info=True)

    async def get_event_statistics(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics about stored events."""
        try:
            with self.session_factory() as db_session:
                base_query = db_session.query(SessionEventModel)

                if session_id:
                    base_query = base_query.filter(SessionEventModel.session_id == session_id)

                total_events = base_query.count()

                # Event type distribution
                event_type_counts = {}
                for event_type in EventType:
                    count = base_query.filter(
                        SessionEventModel.event_type == event_type.value
                    ).count()
                    if count > 0:
                        event_type_counts[event_type.value] = count

                # Time-based statistics
                first_event = base_query.order_by(asc(SessionEventModel.timestamp)).first()
                last_event = base_query.order_by(desc(SessionEventModel.timestamp)).first()

                stats = {
                    'total_events': total_events,
                    'event_type_distribution': event_type_counts,
                    'first_event_time': first_event.timestamp.isoformat() if first_event else None,
                    'last_event_time': last_event.timestamp.isoformat() if last_event else None,
                    'store_metrics': self.metrics.copy()
                }

                if session_id:
                    stats['session_id'] = session_id

                return stats

        except Exception as e:
            self.logger.error(f"Failed to get event statistics: {e}", exc_info=True)
            return {'error': str(e)}

    # Private helper methods

    async def _get_next_sequence_number(self, session_id: str) -> int:
        """Get the next sequence number for a session."""
        self.sequence_counters[session_id] += 1
        return self.sequence_counters[session_id]

    async def _load_sequence_counters(self) -> None:
        """Load current sequence numbers from database."""
        try:
            with self.session_factory() as db_session:
                # Get the maximum sequence number for each session
                results = db_session.query(
                    SessionEventModel.session_id,
                    db_session.query(SessionEventModel.sequence_number).filter(
                        SessionEventModel.session_id == SessionEventModel.session_id
                    ).order_by(desc(SessionEventModel.sequence_number)).limit(1).scalar_subquery()
                ).distinct().all()

                for session_id, max_sequence in results:
                    if max_sequence is not None:
                        self.sequence_counters[session_id] = max_sequence

                self.logger.debug(f"Loaded sequence counters for {len(results)} sessions")

        except Exception as e:
            self.logger.error(f"Failed to load sequence counters: {e}", exc_info=True)

    async def _model_to_event(self, event_model: SessionEventModel) -> SessionEvent:
        """Convert database model to SessionEvent."""
        event_data = json.loads(event_model.event_data) if event_model.event_data else {}

        return SessionEvent(
            event_id=event_model.id,
            session_id=event_model.session_id,
            event_type=EventType(event_model.event_type),
            timestamp=event_model.timestamp,
            sequence_number=event_model.sequence_number,
            event_data=event_data,
            agent_id=event_model.agent_id,
            correlation_id=event_model.correlation_id
        )

    def _cache_event(self, event: SessionEvent) -> None:
        """Cache an event with LRU eviction."""
        if not self.enable_caching:
            return

        # Add to cache
        self.event_cache[event.event_id] = event

        # Update access order
        if event.event_id in self.cache_access_order:
            self.cache_access_order.remove(event.event_id)
        self.cache_access_order.append(event.event_id)

        # Evict oldest if cache is full
        while len(self.event_cache) > self.cache_size:
            oldest_id = self.cache_access_order.pop(0)
            del self.event_cache[oldest_id]

    def get_metrics(self) -> Dict[str, Any]:
        """Get event store metrics."""
        cache_hit_rate = 0.0
        total_requests = self.metrics['cache_hits'] + self.metrics['cache_misses']
        if total_requests > 0:
            cache_hit_rate = self.metrics['cache_hits'] / total_requests * 100

        return {
            **self.metrics,
            'cache_hit_rate_percent': cache_hit_rate,
            'cached_events_count': len(self.event_cache),
            'tracked_sessions': len(self.sequence_counters)
        }
