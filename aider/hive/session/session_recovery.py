"""
Session Recovery Manager for Aider Hive Architecture.

This module provides comprehensive fault tolerance and recovery capabilities
for session management. It handles session restoration from snapshots and
event logs, supports various failure scenarios, and ensures data consistency
during recovery operations.

Key Features:
- Session restoration from snapshots and event replay
- Multiple recovery strategies for different failure scenarios
- Transaction rollback and consistency guarantees
- Partial recovery support with error handling
- Concurrent recovery coordination
- Recovery metrics and detailed reporting
- Integration with event sourcing and context management
"""

import asyncio
import json
import pickle
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from contextlib import asynccontextmanager

import structlog
from sqlalchemy.orm import sessionmaker

from .session_events import SessionEvent, EventType, SessionEventStore
from .session_manager import (
    SessionModel, SessionSnapshotModel, SessionStatus, SessionMetadata,
    SessionStatistics, SessionSnapshot
)
from .context_hierarchy import HierarchicalContextManager


class RecoveryStrategy(Enum):
    """Different recovery strategies for various failure scenarios."""
    SNAPSHOT_ONLY = "snapshot_only"              # Restore from latest snapshot only
    EVENT_REPLAY = "event_replay"                # Replay all events from beginning
    SNAPSHOT_PLUS_EVENTS = "snapshot_plus_events"  # Snapshot + replay newer events
    BEST_EFFORT = "best_effort"                  # Try multiple strategies
    PARTIAL_RECOVERY = "partial_recovery"        # Recover what's possible
    ROLLBACK_TO_CHECKPOINT = "rollback_to_checkpoint"  # Rollback to known good state


class RecoveryStatus(Enum):
    """Status of recovery operations."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"


class FailureType(Enum):
    """Types of failures that can be recovered from."""
    SESSION_CORRUPTION = "session_corruption"
    MISSING_SNAPSHOT = "missing_snapshot"
    EVENT_LOG_CORRUPTION = "event_log_corruption"
    CONTEXT_LOSS = "context_loss"
    AGENT_STATE_CORRUPTION = "agent_state_corruption"
    DATABASE_CORRUPTION = "database_corruption"
    PARTIAL_WRITE_FAILURE = "partial_write_failure"
    CONCURRENT_MODIFICATION = "concurrent_modification"
    TIMEOUT_FAILURE = "timeout_failure"
    UNKNOWN_FAILURE = "unknown_failure"


@dataclass
class RecoveryOperation:
    """Represents a recovery operation in progress."""
    operation_id: str
    session_id: str
    strategy: RecoveryStrategy
    status: RecoveryStatus
    failure_type: FailureType

    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    # Recovery details
    target_snapshot_id: Optional[str] = None
    events_to_replay: int = 0
    events_replayed: int = 0

    # Results
    recovered_state: Optional[Dict[str, Any]] = None
    recovery_errors: List[str] = field(default_factory=list)
    partial_results: Dict[str, Any] = field(default_factory=dict)

    # Metrics
    processing_time_ms: float = 0.0
    data_recovered_bytes: int = 0
    consistency_checks_passed: int = 0
    consistency_checks_failed: int = 0


@dataclass
class RecoveryPlan:
    """Recovery plan with multiple strategies and fallbacks."""
    session_id: str
    primary_strategy: RecoveryStrategy
    fallback_strategies: List[RecoveryStrategy] = field(default_factory=list)

    # Constraints
    max_recovery_time_seconds: int = 300
    allow_partial_recovery: bool = True
    require_consistency_check: bool = True

    # Data requirements
    required_components: Set[str] = field(default_factory=lambda: {
        'session_metadata', 'conversation_history', 'agent_states'
    })
    optional_components: Set[str] = field(default_factory=lambda: {
        'context_hierarchy', 'workflow_state', 'performance_metrics'
    })


class SessionRecoveryManager:
    """
    Comprehensive session recovery manager with fault tolerance.

    Provides multiple recovery strategies, handles various failure scenarios,
    and ensures data consistency during recovery operations.
    """

    def __init__(
        self,
        session_factory: sessionmaker,
        event_store: SessionEventStore,
        context_manager: HierarchicalContextManager,
        enable_consistency_checks: bool = True,
        max_concurrent_recoveries: int = 5,
        default_timeout_seconds: int = 300
    ):
        """Initialize the recovery manager."""
        self.session_factory = session_factory
        self.event_store = event_store
        self.context_manager = context_manager
        self.enable_consistency_checks = enable_consistency_checks
        self.max_concurrent_recoveries = max_concurrent_recoveries
        self.default_timeout_seconds = default_timeout_seconds

        self.logger = structlog.get_logger().bind(component="session_recovery")

        # Active recovery operations
        self.active_recoveries: Dict[str, RecoveryOperation] = {}
        self.recovery_locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self.recovery_semaphore = asyncio.Semaphore(max_concurrent_recoveries)

        # Recovery statistics
        self.recovery_stats = {
            'total_recoveries': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0,
            'partial_recoveries': 0,
            'strategy_usage': defaultdict(int),
            'failure_type_counts': defaultdict(int),
            'average_recovery_time_ms': 0.0
        }

        # Recovery cache for frequently accessed data
        self.recovery_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl_seconds = 3600

    async def restore_session(
        self,
        session_id: str,
        strategy: RecoveryStrategy = RecoveryStrategy.BEST_EFFORT,
        recovery_plan: Optional[RecoveryPlan] = None
    ) -> bool:
        """
        Restore a session using the specified strategy.

        Args:
            session_id: The session to restore
            strategy: Recovery strategy to use
            recovery_plan: Custom recovery plan (optional)

        Returns:
            bool: True if recovery was successful
        """
        operation_id = str(uuid.uuid4())

        try:
            async with self.recovery_semaphore:
                async with self.recovery_locks[session_id]:
                    self.logger.info(f"Starting session recovery: {session_id}")

                    # Create recovery plan if not provided
                    if recovery_plan is None:
                        recovery_plan = await self._create_recovery_plan(session_id, strategy)

                    # Detect failure type
                    failure_type = await self._diagnose_failure(session_id)

                    # Create recovery operation
                    operation = RecoveryOperation(
                        operation_id=operation_id,
                        session_id=session_id,
                        strategy=strategy,
                        status=RecoveryStatus.IN_PROGRESS,
                        failure_type=failure_type
                    )

                    self.active_recoveries[operation_id] = operation

                    try:
                        # Execute recovery
                        success = await self._execute_recovery(operation, recovery_plan)

                        # Update operation status
                        operation.status = (RecoveryStatus.COMPLETED if success
                                          else RecoveryStatus.FAILED)
                        operation.completed_at = datetime.utcnow()
                        operation.processing_time_ms = (
                            operation.completed_at - operation.started_at
                        ).total_seconds() * 1000

                        # Update statistics
                        await self._update_recovery_stats(operation, success)

                        if success:
                            self.logger.info(f"Session recovery completed: {session_id}")
                        else:
                            self.logger.error(f"Session recovery failed: {session_id}")

                        return success

                    finally:
                        # Cleanup
                        if operation_id in self.active_recoveries:
                            del self.active_recoveries[operation_id]

        except Exception as e:
            self.logger.error(f"Recovery operation failed: {e}", exc_info=True)
            return False

    async def rollback_session(
        self,
        session_id: str,
        target_snapshot_id: Optional[str] = None,
        target_sequence_number: Optional[int] = None
    ) -> bool:
        """
        Rollback a session to a previous state.

        Args:
            session_id: Session to rollback
            target_snapshot_id: Specific snapshot to rollback to
            target_sequence_number: Event sequence number to rollback to

        Returns:
            bool: True if rollback was successful
        """
        try:
            self.logger.info(f"Rolling back session {session_id}")

            # Find target snapshot
            if target_snapshot_id:
                snapshot = await self._load_snapshot_by_id(target_snapshot_id)
            elif target_sequence_number:
                snapshot = await self._find_snapshot_before_sequence(
                    session_id, target_sequence_number
                )
            else:
                # Use most recent snapshot
                snapshot = await self._load_latest_snapshot(session_id)

            if not snapshot:
                self.logger.error(f"No suitable snapshot found for rollback")
                return False

            # Create recovery operation
            operation = RecoveryOperation(
                operation_id=str(uuid.uuid4()),
                session_id=session_id,
                strategy=RecoveryStrategy.ROLLBACK_TO_CHECKPOINT,
                status=RecoveryStatus.IN_PROGRESS,
                failure_type=FailureType.UNKNOWN_FAILURE,
                target_snapshot_id=snapshot.session_id if hasattr(snapshot, 'session_id') else None
            )

            # Restore from snapshot
            success = await self._restore_from_snapshot(operation, snapshot)

            if success:
                # Replay events after snapshot if needed
                if target_sequence_number and target_sequence_number > snapshot.sequence_number:
                    success = await self._replay_events_to_sequence(
                        operation, session_id,
                        snapshot.sequence_number + 1,
                        target_sequence_number
                    )

            operation.status = RecoveryStatus.COMPLETED if success else RecoveryStatus.FAILED

            self.logger.info(f"Session rollback {'completed' if success else 'failed'}: {session_id}")
            return success

        except Exception as e:
            self.logger.error(f"Session rollback failed: {e}", exc_info=True)
            return False

    async def validate_session_integrity(self, session_id: str) -> Dict[str, Any]:
        """
        Validate the integrity of a session's data.

        Returns:
            Dict containing validation results and any issues found
        """
        try:
            validation_results = {
                'session_id': session_id,
                'is_valid': True,
                'issues': [],
                'warnings': [],
                'components_checked': [],
                'validation_time_ms': 0.0
            }

            start_time = datetime.utcnow()

            # Check session metadata
            session_valid = await self._validate_session_metadata(session_id)
            validation_results['components_checked'].append('session_metadata')
            if not session_valid:
                validation_results['is_valid'] = False
                validation_results['issues'].append('Session metadata is corrupted or missing')

            # Check event log consistency
            events_valid = await self._validate_event_log(session_id)
            validation_results['components_checked'].append('event_log')
            if not events_valid:
                validation_results['is_valid'] = False
                validation_results['issues'].append('Event log has inconsistencies or gaps')

            # Check snapshots
            snapshots_valid = await self._validate_snapshots(session_id)
            validation_results['components_checked'].append('snapshots')
            if not snapshots_valid:
                validation_results['warnings'].append('Some snapshots may be corrupted')

            # Check context integrity
            context_valid = await self._validate_context_integrity(session_id)
            validation_results['components_checked'].append('context')
            if not context_valid:
                validation_results['warnings'].append('Context hierarchy may have issues')

            # Calculate validation time
            end_time = datetime.utcnow()
            validation_results['validation_time_ms'] = (
                end_time - start_time
            ).total_seconds() * 1000

            return validation_results

        except Exception as e:
            self.logger.error(f"Session integrity validation failed: {e}", exc_info=True)
            return {
                'session_id': session_id,
                'is_valid': False,
                'issues': [f'Validation error: {str(e)}'],
                'warnings': [],
                'components_checked': [],
                'validation_time_ms': 0.0
            }

    async def get_recovery_options(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get available recovery options for a session.

        Returns:
            List of recovery options with strategies and success probabilities
        """
        try:
            options = []

            # Check for snapshots
            snapshots = await self._get_session_snapshots(session_id)
            if snapshots:
                latest_snapshot = max(snapshots, key=lambda s: s.sequence_number)
                options.append({
                    'strategy': RecoveryStrategy.SNAPSHOT_ONLY.value,
                    'description': f'Restore from latest snapshot (sequence {latest_snapshot.sequence_number})',
                    'success_probability': 0.95,
                    'data_loss_risk': 'Low',
                    'estimated_time_seconds': 30,
                    'available': True
                })

                options.append({
                    'strategy': RecoveryStrategy.SNAPSHOT_PLUS_EVENTS.value,
                    'description': 'Restore from snapshot and replay newer events',
                    'success_probability': 0.85,
                    'data_loss_risk': 'Minimal',
                    'estimated_time_seconds': 120,
                    'available': True
                })

            # Check for complete event log
            event_count = await self._count_session_events(session_id)
            if event_count > 0:
                options.append({
                    'strategy': RecoveryStrategy.EVENT_REPLAY.value,
                    'description': f'Replay all {event_count} events from beginning',
                    'success_probability': 0.70,
                    'data_loss_risk': None,
                    'estimated_time_seconds': event_count * 2,
                    'available': True
                })

            # Always offer best effort
            options.append({
                'strategy': RecoveryStrategy.BEST_EFFORT.value,
                'description': 'Try multiple strategies automatically',
                'success_probability': 0.90,
                'data_loss_risk': 'Variable',
                'estimated_time_seconds': 300,
                'available': True
            })

            # Partial recovery
            options.append({
                'strategy': RecoveryStrategy.PARTIAL_RECOVERY.value,
                'description': 'Recover available components, accept data loss',
                'success_probability': 0.98,
                'data_loss_risk': 'High',
                'estimated_time_seconds': 60,
                'available': True
            })

            return options

        except Exception as e:
            self.logger.error(f"Failed to get recovery options: {e}")
            return []

    # Private helper methods

    async def _create_recovery_plan(
        self,
        session_id: str,
        strategy: RecoveryStrategy
    ) -> RecoveryPlan:
        """Create a recovery plan based on available data and strategy."""
        try:
            plan = RecoveryPlan(
                session_id=session_id,
                primary_strategy=strategy
            )

            # Add fallback strategies based on primary strategy
            if strategy == RecoveryStrategy.BEST_EFFORT:
                plan.fallback_strategies = [
                    RecoveryStrategy.SNAPSHOT_PLUS_EVENTS,
                    RecoveryStrategy.SNAPSHOT_ONLY,
                    RecoveryStrategy.PARTIAL_RECOVERY
                ]
            elif strategy == RecoveryStrategy.SNAPSHOT_PLUS_EVENTS:
                plan.fallback_strategies = [
                    RecoveryStrategy.SNAPSHOT_ONLY,
                    RecoveryStrategy.PARTIAL_RECOVERY
                ]
            elif strategy == RecoveryStrategy.EVENT_REPLAY:
                plan.fallback_strategies = [
                    RecoveryStrategy.SNAPSHOT_PLUS_EVENTS,
                    RecoveryStrategy.PARTIAL_RECOVERY
                ]

            return plan

        except Exception as e:
            self.logger.error(f"Failed to create recovery plan: {e}")
            # Return minimal plan
            return RecoveryPlan(session_id=session_id, primary_strategy=strategy)

    async def _diagnose_failure(self, session_id: str) -> FailureType:
        """Diagnose the type of failure that occurred."""
        try:
            # Check if session exists in database
            session_exists = await self._check_session_exists(session_id)
            if not session_exists:
                return FailureType.SESSION_CORRUPTION

            # Check for missing snapshots
            snapshots = await self._get_session_snapshots(session_id)
            if not snapshots:
                return FailureType.MISSING_SNAPSHOT

            # Check event log integrity
            events_valid = await self._validate_event_log(session_id)
            if not events_valid:
                return FailureType.EVENT_LOG_CORRUPTION

            # Check context
            context_valid = await self._validate_context_integrity(session_id)
            if not context_valid:
                return FailureType.CONTEXT_LOSS

            # Default to unknown failure
            return FailureType.UNKNOWN_FAILURE

        except Exception as e:
            self.logger.error(f"Failure diagnosis failed: {e}")
            return FailureType.UNKNOWN_FAILURE

    async def _execute_recovery(
        self,
        operation: RecoveryOperation,
        recovery_plan: RecoveryPlan
    ) -> bool:
        """Execute the recovery operation according to the plan."""
        try:
            strategies_to_try = [recovery_plan.primary_strategy] + recovery_plan.fallback_strategies

            for strategy in strategies_to_try:
                operation.strategy = strategy
                self.logger.info(f"Attempting recovery with strategy: {strategy.value}")

                success = False

                if strategy == RecoveryStrategy.SNAPSHOT_ONLY:
                    success = await self._recover_from_snapshot_only(operation)
                elif strategy == RecoveryStrategy.EVENT_REPLAY:
                    success = await self._recover_from_event_replay(operation)
                elif strategy == RecoveryStrategy.SNAPSHOT_PLUS_EVENTS:
                    success = await self._recover_from_snapshot_plus_events(operation)
                elif strategy == RecoveryStrategy.PARTIAL_RECOVERY:
                    success = await self._recover_partial(operation)
                elif strategy == RecoveryStrategy.ROLLBACK_TO_CHECKPOINT:
                    success = await self._recover_rollback_to_checkpoint(operation)

                if success:
                    # Verify recovery if consistency checks are enabled
                    if self.enable_consistency_checks:
                        validation_result = await self.validate_session_integrity(operation.session_id)
                        if validation_result['is_valid']:
                            return True
                        else:
                            self.logger.warning(f"Recovery passed but consistency check failed")
                            continue
                    else:
                        return True

                self.logger.warning(f"Recovery strategy {strategy.value} failed, trying next")

            # All strategies failed
            operation.status = RecoveryStatus.FAILED
            return False

        except Exception as e:
            self.logger.error(f"Recovery execution failed: {e}", exc_info=True)
            operation.recovery_errors.append(str(e))
            return False

    async def _recover_from_snapshot_only(self, operation: RecoveryOperation) -> bool:
        """Recover session from the latest snapshot only."""
        try:
            snapshot = await self._load_latest_snapshot(operation.session_id)
            if not snapshot:
                operation.recovery_errors.append("No snapshots available")
                return False

            operation.target_snapshot_id = getattr(snapshot, 'id', 'unknown')
            return await self._restore_from_snapshot(operation, snapshot)

        except Exception as e:
            operation.recovery_errors.append(f"Snapshot recovery failed: {str(e)}")
            return False

    async def _recover_from_event_replay(self, operation: RecoveryOperation) -> bool:
        """Recover session by replaying all events from the beginning."""
        try:
            # Get all events for the session
            events = await self.event_store.get_session_events(
                operation.session_id,
                limit=None  # Get all events
            )

            if not events:
                operation.recovery_errors.append("No events available for replay")
                return False

            operation.events_to_replay = len(events)

            # Create initial session state
            initial_state = await self._create_initial_session_state(operation.session_id)
            if not initial_state:
                operation.recovery_errors.append("Could not create initial session state")
                return False

            # Replay events
            return await self._replay_events(operation, initial_state, events)

        except Exception as e:
            operation.recovery_errors.append(f"Event replay failed: {str(e)}")
            return False

    async def _recover_from_snapshot_plus_events(self, operation: RecoveryOperation) -> bool:
        """Recover from snapshot and replay events that occurred after it."""
        try:
            # Load latest snapshot
            snapshot = await self._load_latest_snapshot(operation.session_id)
            if not snapshot:
                operation.recovery_errors.append("No snapshot available")
                return False

            operation.target_snapshot_id = getattr(snapshot, 'id', 'unknown')

            # Restore from snapshot
            success = await self._restore_from_snapshot(operation, snapshot)
            if not success:
                return False

            # Get events after snapshot
            events_after_snapshot = await self.event_store.get_session_events(
                operation.session_id,
                limit=None
            )

            # Filter events that occurred after the snapshot
            replay_events = [
                event for event in events_after_snapshot
                if event.sequence_number > snapshot.sequence_number
            ]

            if replay_events:
                operation.events_to_replay = len(replay_events)

                # Get current state and replay newer events
                current_state = operation.recovered_state
                return await self._replay_events(operation, current_state, replay_events)

            return True

        except Exception as e:
            operation.recovery_errors.append(f"Snapshot + events recovery failed: {str(e)}")
            return False

    async def _recover_partial(self, operation: RecoveryOperation) -> bool:
        """Attempt partial recovery, salvaging what data is available."""
        try:
            recovered_components = {}

            # Try to recover session metadata
            try:
                metadata = await self._recover_session_metadata(operation.session_id)
                if metadata:
                    recovered_components['metadata'] = metadata
            except Exception as e:
                operation.recovery_errors.append(f"Metadata recovery failed: {str(e)}")

            # Try to recover conversation history from events
            try:
                conversation = await self._recover_conversation_from_events(operation.session_id)
                if conversation:
                    recovered_components['conversation'] = conversation
            except Exception as e:
                operation.recovery_errors.append(f"Conversation recovery failed: {str(e)}")

            # Try to recover context
            try:
                context = await self._recover_context_data(operation.session_id)
                if context:
                    recovered_components['context'] = context
            except Exception as e:
                operation.recovery_errors.append(f"Context recovery failed: {str(e)}")

            # Build partial session state
            if recovered_components:
                operation.partial_results = recovered_components
                operation.recovered_state = self._build_partial_session_state(recovered_components)
                operation.status = RecoveryStatus.PARTIAL
                return True

            operation.recovery_errors.append("No components could be recovered")
            return False

        except Exception as e:
            operation.recovery_errors.append(f"Partial recovery failed: {str(e)}")
            return False

    async def _recover_rollback_to_checkpoint(self, operation: RecoveryOperation) -> bool:
        """Rollback to a specific checkpoint (snapshot)."""
        try:
            if operation.target_snapshot_id:
                snapshot = await self._load_snapshot_by_id(operation.target_snapshot_id)
            else:
                snapshot = await self._load_latest_snapshot(operation.session_id)

            if not snapshot:
                operation.recovery_errors.append("No checkpoint available for rollback")
                return False

            return await self._restore_from_snapshot(operation, snapshot)

        except Exception as e:
            operation.recovery_errors.append(f"Checkpoint rollback failed: {str(e)}")
            return False

    async def _restore_from_snapshot(
        self,
        operation: RecoveryOperation,
        snapshot: SessionSnapshot
    ) -> bool:
        """Restore session state from a snapshot."""
        try:
            # Reconstruct session state from snapshot
            session_state = {
                'id': snapshot.session_id,
                'status': SessionStatus.SUSPENDED,  # Will be activated after recovery
                'agent_states': snapshot.agent_states,
                'context_state': snapshot.context_state,
                'conversation_history': snapshot.conversation_state.get('history', []),
                'workflow_state': snapshot.workflow_state,
                'metadata': SessionMetadata(**snapshot.metadata),
                'statistics': SessionStatistics(),  # Reset statistics
                'created_at': datetime.utcnow(),
                'updated_at': datetime.utcnow(),
                'recovered_from_snapshot': True,
                'recovery_snapshot_id': getattr(snapshot, 'id', 'unknown'),
                'recovery_timestamp': datetime.utcnow()
            }

            # Store recovered state
            success = await self._store_recovered_session(session_state)
            if success:
                operation.recovered_state = session_state
                operation.data_recovered_bytes += len(json.dumps(session_state, default=str))

            return success

        except Exception as e:
            self.logger.error(f"Snapshot restoration failed: {e}", exc_info=True)
            return False

    async def _replay_events(
        self,
        operation: RecoveryOperation,
        initial_state: Dict[str, Any],
        events: List[SessionEvent]
    ) -> bool:
        """Replay a list of events to reconstruct session state."""
        try:
            current_state = initial_state.copy()

            for event in events:
                try:
                    # Apply event to current state
                    success = await self._apply_event_to_state(current_state, event)
                    if success:
                        operation.events_replayed += 1
                    else:
                        operation.recovery_errors.append(
                            f"Failed to apply event {event.event_id} (type: {event.event_type.value})"
                        )
                        # Continue with other events for best effort

                except Exception as e:
                    operation.recovery_errors.append(
                        f"Error applying event {event.event_id}: {str(e)}"
                    )
                    continue

            # Store final state
            if operation.events_replayed > 0:
                operation.recovered_state = current_state
                return await self._store_recovered_session(current_state)

            return False

        except Exception as e:
            self.logger.error(f"Event replay failed: {e}", exc_info=True)
            return False

    async def _apply_event_to_state(
        self,
        state: Dict[str, Any],
        event: SessionEvent
    ) -> bool:
        """Apply a single event to the session state."""
        try:
            event_data = event.event_data

            if event.event_type == EventType.MESSAGE_ADDED:
                # Add message to conversation history
                if 'conversation_history' not in state:
                    state['conversation_history'] = []

                message = event_data.get('message', {})
                state['conversation_history'].append(message)

            elif event.event_type == EventType.STATE_UPDATED:
                # Update agent state
                if 'agent_states' not in state:
                    state['agent_states'] = {}

                state_update = event_data.get('state_update', {})
                if event.agent_id:
                    state['agent_states'][event.agent_id] = state_update

            elif event.event_type == EventType.CONTEXT_UPDATED:
                # Update context
                if 'context_state' not in state:
                    state['context_state'] = {}

                context_update = event_data.get('context_update', {})
                state['context_state'].update(context_update)

            elif event.event_type in [EventType.SESSION_SUSPENDED, EventType.SESSION_RESUMED]:
                # Update session status
                if event.event_type == EventType.SESSION_SUSPENDED:
                    state['status'] = SessionStatus.SUSPENDED
                else:
                    state['status'] = SessionStatus.ACTIVE

            # Update timestamp
            state['updated_at'] = event.timestamp

            return True

        except Exception as e:
            self.logger.error(f"Failed to apply event: {e}")
            return False

    async def _store_recovered_session(self, session_state: Dict[str, Any]) -> bool:
        """Store recovered session state to database."""
        try:
            # This would integrate with the actual session storage mechanism
            # For now, we'll create a simplified version

            async with self._get_db_session() as db_session:
                # Check if session already exists
                session_model = db_session.query(SessionModel).filter(
                    SessionModel.id == session_state['id']
                ).first()

                if session_model:
                    # Update existing session
                    session_model.status = session_state['status'].value
                    session_model.updated_at = session_state['updated_at']
                    session_model.current_state = json.dumps(session_state, default=str)
                    session_model.metadata = json.dumps(asdict(session_state['metadata']))
                    session_model.statistics = json.dumps(asdict(session_state['statistics']))
                else:
                    # Create new session record
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

                db_session.commit()
                return True

        except Exception as e:
            self.logger.error(f"Failed to store recovered session: {e}", exc_info=True)
            return False

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

    async def _load_latest_snapshot(self, session_id: str) -> Optional[SessionSnapshot]:
        """Load the most recent snapshot for a session."""
        try:
            async with self._get_db_session() as db_session:
                snapshot_model = db_session.query(SessionSnapshotModel).filter(
                    SessionSnapshotModel.session_id == session_id
                ).order_by(SessionSnapshotModel.sequence_number.desc()).first()

                if snapshot_model:
                    return await self._deserialize_snapshot(snapshot_model)

            return None

        except Exception as e:
            self.logger.error(f"Failed to load latest snapshot: {e}")
            return None

    async def _load_snapshot_by_id(self, snapshot_id: str) -> Optional[SessionSnapshot]:
        """Load a specific snapshot by ID."""
        try:
            async with self._get_db_session() as db_session:
                snapshot_model = db_session.query(SessionSnapshotModel).filter(
                    SessionSnapshotModel.id == snapshot_id
                ).first()

                if snapshot_model:
                    return await self._deserialize_snapshot(snapshot_model)

            return None

        except Exception as e:
            self.logger.error(f"Failed to load snapshot by ID: {e}")
            return None

    async def _get_session_snapshots(self, session_id: str) -> List[SessionSnapshot]:
        """Get all snapshots for a session."""
        try:
            snapshots = []
            async with self._get_db_session() as db_session:
                snapshot_models = db_session.query(SessionSnapshotModel).filter(
                    SessionSnapshotModel.session_id == session_id
                ).order_by(SessionSnapshotModel.sequence_number).all()

                for model in snapshot_models:
                    snapshot = await self._deserialize_snapshot(model)
                    if snapshot:
                        snapshots.append(snapshot)

            return snapshots

        except Exception as e:
            self.logger.error(f"Failed to get session snapshots: {e}")
            return []

    async def _deserialize_snapshot(self, snapshot_model: SessionSnapshotModel) -> Optional[SessionSnapshot]:
        """Deserialize a snapshot from database model."""
        try:
            snapshot_data = snapshot_model.snapshot_data

            # Decompress if needed
            try:
                import gzip
                snapshot_data = gzip.decompress(snapshot_data)
            except:
                pass  # Data might not be compressed

            # Deserialize
            snapshot = pickle.loads(snapshot_data)
            return snapshot

        except Exception as e:
            self.logger.error(f"Failed to deserialize snapshot: {e}")
            return None

    async def _validate_session_metadata(self, session_id: str) -> bool:
        """Validate session metadata integrity."""
        try:
            async with self._get_db_session() as db_session:
                session_model = db_session.query(SessionModel).filter(
                    SessionModel.id == session_id
                ).first()

                if not session_model:
                    return False

                # Check required fields
                if not all([
                    session_model.user_id,
                    session_model.status,
                    session_model.created_at,
                    session_model.project_path
                ]):
                    return False

                # Validate JSON fields
                try:
                    if session_model.metadata:
                        json.loads(session_model.metadata)
                    if session_model.current_state:
                        json.loads(session_model.current_state)
                except json.JSONDecodeError:
                    return False

                return True

        except Exception as e:
            self.logger.error(f"Session metadata validation failed: {e}")
            return False

    async def _validate_event_log(self, session_id: str) -> bool:
        """Validate event log consistency."""
        try:
            events = await self.event_store.get_session_events(session_id)
            if not events:
                return True  # No events is valid

            # Check sequence number consistency
            expected_sequence = 1
            for event in sorted(events, key=lambda e: e.sequence_number):
                if event.sequence_number != expected_sequence:
                    self.logger.warning(f"Event sequence gap: expected {expected_sequence}, got {event.sequence_number}")
                    return False
                expected_sequence += 1

            return True

        except Exception as e:
            self.logger.error(f"Event log validation failed: {e}")
            return False

    async def _validate_snapshots(self, session_id: str) -> bool:
        """Validate snapshot integrity."""
        try:
            snapshots = await self._get_session_snapshots(session_id)
            if not snapshots:
                return True  # No snapshots is valid

            # Check if snapshots can be deserialized
            for snapshot in snapshots:
                if not isinstance(snapshot, SessionSnapshot):
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Snapshot validation failed: {e}")
            return False

    async def _validate_context_integrity(self, session_id: str) -> bool:
        """Validate context hierarchy integrity."""
        try:
            if not self.context_manager:
                return True

            context_state = await self.context_manager.get_session_context(session_id)
            return len(context_state) > 0

        except Exception as e:
            self.logger.error(f"Context integrity validation failed: {e}")
            return False

    async def _check_session_exists(self, session_id: str) -> bool:
        """Check if session exists in database."""
        try:
            async with self._get_db_session() as db_session:
                count = db_session.query(SessionModel).filter(
                    SessionModel.id == session_id
                ).count()
                return count > 0

        except Exception as e:
            self.logger.error(f"Session existence check failed: {e}")
            return False

    async def _count_session_events(self, session_id: str) -> int:
        """Count total events for a session."""
        try:
            events = await self.event_store.get_session_events(session_id)
            return len(events)

        except Exception as e:
            self.logger.error(f"Event counting failed: {e}")
            return 0

    async def _create_initial_session_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Create minimal initial session state for event replay."""
        try:
            # Try to get basic metadata from database
            async with self._get_db_session() as db_session:
                session_model = db_session.query(SessionModel).filter(
                    SessionModel.id == session_id
                ).first()

                if session_model:
                    metadata_dict = json.loads(session_model.metadata) if session_model.metadata else {}
                    metadata = SessionMetadata(
                        user_id=session_model.user_id,
                        project_path=session_model.project_path,
                        session_type=session_model.session_type
                    )
                else:
                    # Create minimal metadata
                    metadata = SessionMetadata(
                        user_id="unknown",
                        project_path="/unknown",
                        session_type="recovery"
                    )

                initial_state = {
                    'id': session_id,
                    'status': SessionStatus.SUSPENDED,
                    'metadata': metadata,
                    'statistics': SessionStatistics(),
                    'agent_states': {},
                    'context_state': {},
                    'conversation_history': [],
                    'workflow_state': {},
                    'created_at': datetime.utcnow(),
                    'updated_at': datetime.utcnow()
                }

                return initial_state

        except Exception as e:
            self.logger.error(f"Failed to create initial session state: {e}")
            return None

    async def _recover_session_metadata(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Attempt to recover session metadata."""
        try:
            async with self._get_db_session() as db_session:
                session_model = db_session.query(SessionModel).filter(
                    SessionModel.id == session_id
                ).first()

                if session_model:
                    return {
                        'id': session_model.id,
                        'user_id': session_model.user_id,
                        'status': session_model.status,
                        'created_at': session_model.created_at,
                        'project_path': session_model.project_path,
                        'session_type': session_model.session_type
                    }

            return None

        except Exception as e:
            self.logger.error(f"Metadata recovery failed: {e}")
            return None

    async def _recover_conversation_from_events(self, session_id: str) -> Optional[List[Dict[str, Any]]]:
        """Recover conversation history from message events."""
        try:
            events = await self.event_store.get_session_events(
                session_id,
                event_types=[EventType.MESSAGE_ADDED]
            )

            conversation = []
            for event in sorted(events, key=lambda e: e.sequence_number):
                message = event.event_data.get('message', {})
                if message:
                    conversation.append(message)

            return conversation if conversation else None

        except Exception as e:
            self.logger.error(f"Conversation recovery failed: {e}")
            return None

    async def _recover_context_data(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Attempt to recover context data."""
        try:
            if not self.context_manager:
                return None

            context_state = await self.context_manager.get_session_context(session_id)
            return context_state if context_state else None

        except Exception as e:
            self.logger.error(f"Context recovery failed: {e}")
            return None

    def _build_partial_session_state(self, recovered_components: Dict[str, Any]) -> Dict[str, Any]:
        """Build partial session state from recovered components."""
        try:
            # Start with minimal state
            session_state = {
                'id': recovered_components.get('metadata', {}).get('id', 'unknown'),
                'status': SessionStatus.SUSPENDED,
                'agent_states': {},
                'context_state': {},
                'conversation_history': [],
                'workflow_state': {},
                'created_at': datetime.utcnow(),
                'updated_at': datetime.utcnow(),
                'recovery_type': 'partial',
                'recovered_components': list(recovered_components.keys())
            }

            # Add recovered components
            if 'metadata' in recovered_components:
                metadata_data = recovered_components['metadata']
                session_state['metadata'] = SessionMetadata(
                    user_id=metadata_data.get('user_id', 'unknown'),
                    project_path=metadata_data.get('project_path', '/unknown'),
                    session_type=metadata_data.get('session_type', 'recovery')
                )

            if 'conversation' in recovered_components:
                session_state['conversation_history'] = recovered_components['conversation']

            if 'context' in recovered_components:
                session_state['context_state'] = recovered_components['context']

            session_state['statistics'] = SessionStatistics()

            return session_state

        except Exception as e:
            self.logger.error(f"Failed to build partial session state: {e}")
            return {}

    async def _find_snapshot_before_sequence(
        self,
        session_id: str,
        sequence_number: int
    ) -> Optional[SessionSnapshot]:
        """Find the most recent snapshot before a given sequence number."""
        try:
            async with self._get_db_session() as db_session:
                snapshot_model = db_session.query(SessionSnapshotModel).filter(
                    SessionSnapshotModel.session_id == session_id,
                    SessionSnapshotModel.sequence_number <= sequence_number
                ).order_by(SessionSnapshotModel.sequence_number.desc()).first()

                if snapshot_model:
                    return await self._deserialize_snapshot(snapshot_model)

            return None

        except Exception as e:
            self.logger.error(f"Failed to find snapshot before sequence: {e}")
            return None

    async def _replay_events_to_sequence(
        self,
        operation: RecoveryOperation,
        session_id: str,
        start_sequence: int,
        end_sequence: int
    ) -> bool:
        """Replay events within a sequence range."""
        try:
            events = await self.event_store.get_session_events(session_id)

            # Filter events in range
            replay_events = [
                event for event in events
                if start_sequence <= event.sequence_number <= end_sequence
            ]

            if not replay_events:
                return True

            operation.events_to_replay = len(replay_events)

            # Apply events to current state
            current_state = operation.recovered_state
            if not current_state:
                return False

            for event in sorted(replay_events, key=lambda e: e.sequence_number):
                success = await self._apply_event_to_state(current_state, event)
                if success:
                    operation.events_replayed += 1

            return operation.events_replayed > 0

        except Exception as e:
            self.logger.error(f"Event replay to sequence failed: {e}")
            return False

    async def _update_recovery_stats(self, operation: RecoveryOperation, success: bool) -> None:
        """Update recovery statistics."""
        try:
            self.recovery_stats['total_recoveries'] += 1

            if success:
                self.recovery_stats['successful_recoveries'] += 1
            elif operation.status == RecoveryStatus.PARTIAL:
                self.recovery_stats['partial_recoveries'] += 1
            else:
                self.recovery_stats['failed_recoveries'] += 1

            self.recovery_stats['strategy_usage'][operation.strategy.value] += 1
            self.recovery_stats['failure_type_counts'][operation.failure_type.value] += 1

            # Update average recovery time
            if operation.processing_time_ms > 0:
                total_time = (self.recovery_stats['average_recovery_time_ms'] *
                            (self.recovery_stats['total_recoveries'] - 1) +
                            operation.processing_time_ms)
                self.recovery_stats['average_recovery_time_ms'] = (
                    total_time / self.recovery_stats['total_recoveries']
                )

        except Exception as e:
            self.logger.error(f"Failed to update recovery stats: {e}")

    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get comprehensive recovery statistics."""
        return {
            **self.recovery_stats,
            'active_recoveries': len(self.active_recoveries),
            'cache_entries': len(self.recovery_cache),
            'success_rate': (
                self.recovery_stats['successful_recoveries'] /
                max(1, self.recovery_stats['total_recoveries']) * 100
            )
        }

    def get_active_recoveries(self) -> List[Dict[str, Any]]:
        """Get information about currently active recovery operations."""
        active_ops = []
        for operation in self.active_recoveries.values():
            active_ops.append({
                'operation_id': operation.operation_id,
                'session_id': operation.session_id,
                'strategy': operation.strategy.value,
                'status': operation.status.value,
                'started_at': operation.started_at.isoformat(),
                'processing_time_ms': (datetime.utcnow() - operation.started_at).total_seconds() * 1000,
                'events_replayed': operation.events_replayed,
                'events_to_replay': operation.events_to_replay,
                'progress_percent': (
                    operation.events_replayed / max(1, operation.events_to_replay) * 100
                    if operation.events_to_replay > 0 else 0
                )
            })
        return active_ops
