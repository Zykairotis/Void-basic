"""
InterAgentContextManager: Seamless context sharing and synchronization system for the Aider Hive Architecture.

This module provides advanced context management capabilities for multi-agent collaboration including:
- Real-time context sharing between agents
- Subscription-based context updates and notifications
- Context synchronization and conflict resolution
- Access control and permission management
- Context dependency tracking and propagation
- Intelligent context prefetching and caching
- Cross-agent context coherence maintenance
- Performance-optimized context distribution
"""

import asyncio
import logging
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Callable, Tuple, Union
from weakref import WeakSet, WeakKeyDictionary

import structlog

from .context_store import (
    GlobalContextStore,
    ContextEntry,
    ContextType,
    ContextScope,
    ContextQuery,
    ContextSearchResult,
)


class ContextShareMode(Enum):
    """Context sharing modes between agents."""
    READ_ONLY = "read_only"
    READ_WRITE = "read_write"
    EXCLUSIVE_WRITE = "exclusive_write"
    COLLABORATIVE = "collaborative"
    BROADCAST = "broadcast"


class ContextUpdateType(Enum):
    """Types of context updates."""
    CREATED = "created"
    UPDATED = "updated"
    DELETED = "deleted"
    ACCESSED = "accessed"
    LOCKED = "locked"
    UNLOCKED = "unlocked"


class ContextSyncStrategy(Enum):
    """Context synchronization strategies."""
    IMMEDIATE = "immediate"
    BATCHED = "batched"
    PERIODIC = "periodic"
    ON_DEMAND = "on_demand"
    EVENTUAL = "eventual"


class ConflictResolution(Enum):
    """Conflict resolution strategies."""
    LAST_WRITER_WINS = "last_writer_wins"
    FIRST_WRITER_WINS = "first_writer_wins"
    MERGE = "merge"
    MANUAL = "manual"
    VERSION_BRANCH = "version_branch"


@dataclass
class ContextSubscription:
    """Represents a context subscription by an agent."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = ""
    context_patterns: List[str] = field(default_factory=list)
    context_types: List[ContextType] = field(default_factory=list)
    scopes: List[ContextScope] = field(default_factory=list)

    # Subscription behavior
    share_mode: ContextShareMode = ContextShareMode.READ_ONLY
    update_types: Set[ContextUpdateType] = field(default_factory=lambda: {ContextUpdateType.UPDATED})
    sync_strategy: ContextSyncStrategy = ContextSyncStrategy.IMMEDIATE

    # Callback for notifications
    callback: Optional[Callable] = None

    # Filtering and conditions
    filter_function: Optional[Callable[[ContextEntry], bool]] = None
    min_relevance_score: float = 0.0

    # Subscription metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_notification: Optional[datetime] = None
    notification_count: int = 0
    is_active: bool = True


@dataclass
class ContextUpdate:
    """Represents a context update notification."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    context_entry_id: str = ""
    update_type: ContextUpdateType = ContextUpdateType.UPDATED
    agent_id: str = ""  # Agent that made the update

    # Update details
    previous_version: Optional[int] = None
    new_version: Optional[int] = None
    changes: Dict[str, Any] = field(default_factory=dict)

    # Timing
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Propagation tracking
    notified_agents: Set[str] = field(default_factory=set)
    pending_agents: Set[str] = field(default_factory=set)


@dataclass
class ContextLock:
    """Represents a context lock for exclusive access."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    context_entry_id: str = ""
    agent_id: str = ""
    lock_type: str = "write"  # read, write, exclusive

    # Lock timing
    acquired_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    timeout_seconds: float = 300.0  # 5 minutes default

    # Lock metadata
    reason: str = ""
    auto_release: bool = True


@dataclass
class ContextConflict:
    """Represents a context update conflict."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    context_entry_id: str = ""

    # Conflicting updates
    update1: ContextUpdate = None
    update2: ContextUpdate = None

    # Conflict details
    conflict_type: str = "concurrent_modification"
    resolution_strategy: ConflictResolution = ConflictResolution.LAST_WRITER_WINS

    # Resolution tracking
    detected_at: datetime = field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None
    resolution_result: Optional[str] = None
    resolved_by_agent: Optional[str] = None


@dataclass
class ContextSharingMetrics:
    """Metrics for context sharing performance and usage."""
    total_subscriptions: int = 0
    active_subscriptions: int = 0
    total_updates: int = 0
    total_notifications: int = 0

    # Performance metrics
    average_notification_time: float = 0.0
    average_sync_time: float = 0.0
    cache_hit_rate: float = 0.0

    # Conflict metrics
    total_conflicts: int = 0
    resolved_conflicts: int = 0
    pending_conflicts: int = 0

    # Lock metrics
    active_locks: int = 0
    lock_contention_rate: float = 0.0
    average_lock_hold_time: float = 0.0


class ContextAccessController:
    """Manages access control and permissions for context sharing."""

    def __init__(self):
        self.agent_permissions: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
        self.context_permissions: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))

    def grant_permission(self, agent_id: str, context_id: str, permission: str) -> None:
        """Grant a specific permission to an agent for a context."""
        self.agent_permissions[agent_id][context_id].add(permission)
        self.context_permissions[context_id][agent_id].add(permission)

    def revoke_permission(self, agent_id: str, context_id: str, permission: str) -> None:
        """Revoke a specific permission from an agent for a context."""
        self.agent_permissions[agent_id][context_id].discard(permission)
        self.context_permissions[context_id][agent_id].discard(permission)

    def has_permission(self, agent_id: str, context_id: str, permission: str) -> bool:
        """Check if an agent has a specific permission for a context."""
        return permission in self.agent_permissions[agent_id][context_id]

    def get_agent_permissions(self, agent_id: str, context_id: str) -> Set[str]:
        """Get all permissions an agent has for a context."""
        return self.agent_permissions[agent_id][context_id].copy()

    def get_context_agents(self, context_id: str) -> Dict[str, Set[str]]:
        """Get all agents and their permissions for a context."""
        return dict(self.context_permissions[context_id])


class InterAgentContextManager:
    """
    Advanced context sharing and synchronization system for multi-agent collaboration.

    Features:
    - Real-time context sharing with subscription-based updates
    - Conflict detection and resolution
    - Access control and permission management
    - Context locking for exclusive access
    - Intelligent context prefetching and caching
    - Performance-optimized context distribution
    - Cross-agent context coherence maintenance
    """

    def __init__(
        self,
        context_store: GlobalContextStore,
        default_sync_strategy: ContextSyncStrategy = ContextSyncStrategy.IMMEDIATE,
        default_conflict_resolution: ConflictResolution = ConflictResolution.LAST_WRITER_WINS,
        max_lock_duration: float = 3600.0,  # 1 hour
        notification_batch_size: int = 50,
        sync_interval: float = 5.0,  # 5 seconds
    ):
        """
        Initialize the inter-agent context manager.

        Args:
            context_store: Global context store instance
            default_sync_strategy: Default synchronization strategy
            default_conflict_resolution: Default conflict resolution strategy
            max_lock_duration: Maximum lock duration in seconds
            notification_batch_size: Batch size for notifications
            sync_interval: Interval for periodic synchronization
        """
        self.context_store = context_store
        self.default_sync_strategy = default_sync_strategy
        self.default_conflict_resolution = default_conflict_resolution
        self.max_lock_duration = max_lock_duration
        self.notification_batch_size = notification_batch_size
        self.sync_interval = sync_interval

        # Logging
        self.logger = structlog.get_logger().bind(component="context_manager")

        # Subscription management
        self.subscriptions: Dict[str, ContextSubscription] = {}
        self.agent_subscriptions: Dict[str, Set[str]] = defaultdict(set)
        self.context_subscribers: Dict[str, Set[str]] = defaultdict(set)

        # Update tracking
        self.pending_updates: deque = deque()
        self.update_history: deque = deque(maxlen=1000)
        self.notification_queue: Dict[ContextSyncStrategy, deque] = {
            strategy: deque() for strategy in ContextSyncStrategy
        }

        # Lock management
        self.active_locks: Dict[str, ContextLock] = {}
        self.lock_queue: Dict[str, deque] = defaultdict(deque)

        # Conflict management
        self.active_conflicts: Dict[str, ContextConflict] = {}
        self.conflict_history: deque = deque(maxlen=100)

        # Access control
        self.access_controller = ContextAccessController()

        # Metrics and monitoring
        self.metrics = ContextSharingMetrics()

        # Caching for performance
        self.context_cache: Dict[str, Tuple[ContextEntry, datetime]] = {}
        self.cache_ttl = 300.0  # 5 minutes

        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()
        self._shutdown_event = asyncio.Event()

        # Event handlers
        self.on_context_shared: List[Callable] = []
        self.on_conflict_detected: List[Callable] = []
        self.on_conflict_resolved: List[Callable] = []

    async def start(self) -> None:
        """Start the context manager and background tasks."""
        try:
            self.logger.info("Starting inter-agent context manager")

            # Register with context store for updates
            self.context_store.on_entry_added.append(self._handle_context_added)
            self.context_store.on_entry_updated.append(self._handle_context_updated)
            self.context_store.on_entry_removed.append(self._handle_context_removed)

            # Start background tasks
            self._start_background_tasks()

            self.logger.info("Inter-agent context manager started successfully")

        except Exception as e:
            self.logger.error("Failed to start context manager", error=str(e))
            raise

    async def stop(self) -> None:
        """Stop the context manager gracefully."""
        self.logger.info("Stopping inter-agent context manager")

        # Signal shutdown
        self._shutdown_event.set()

        # Cancel background tasks
        await self._cancel_background_tasks()

        # Release all locks
        await self._release_all_locks()

        self.logger.info("Inter-agent context manager stopped")

    async def share_context(
        self,
        agent_id: str,
        context_key: str,
        content: Any,
        context_type: ContextType,
        scope: ContextScope = ContextScope.PROJECT,
        recipients: Optional[List[str]] = None,
        share_mode: ContextShareMode = ContextShareMode.READ_WRITE,
        **kwargs
    ) -> str:
        """
        Share context with other agents.

        Args:
            agent_id: ID of the sharing agent
            context_key: Key for the context
            content: Context content to share
            context_type: Type of context
            scope: Context scope
            recipients: Specific agents to share with (None for all)
            share_mode: Sharing mode
            **kwargs: Additional context store parameters

        Returns:
            Context entry ID
        """
        try:
            # Store the context
            entry_id = await self.context_store.store_context(
                key=context_key,
                content=content,
                context_type=context_type,
                scope=scope,
                **kwargs
            )

            # Set up permissions
            if recipients:
                for recipient_id in recipients:
                    self._setup_sharing_permissions(recipient_id, entry_id, share_mode)
            else:
                # Grant permissions to all subscribed agents
                for subscription in self.subscriptions.values():
                    if self._matches_subscription(entry_id, subscription):
                        self._setup_sharing_permissions(subscription.agent_id, entry_id, share_mode)

            # Create update notification
            update = ContextUpdate(
                context_entry_id=entry_id,
                update_type=ContextUpdateType.CREATED,
                agent_id=agent_id,
            )

            # Queue for notification
            await self._queue_update_notification(update)

            # Trigger callbacks
            for callback in self.on_context_shared:
                try:
                    await callback(agent_id, entry_id, share_mode)
                except Exception as e:
                    self.logger.warning("Error in context shared callback", error=str(e))

            self.logger.info(
                "Context shared",
                agent_id=agent_id,
                context_key=context_key,
                entry_id=entry_id,
                recipients=len(recipients) if recipients else "all"
            )

            return entry_id

        except Exception as e:
            self.logger.error("Error sharing context", agent_id=agent_id, context_key=context_key, error=str(e))
            raise

    async def subscribe_to_context(
        self,
        agent_id: str,
        patterns: Optional[List[str]] = None,
        context_types: Optional[List[ContextType]] = None,
        scopes: Optional[List[ContextScope]] = None,
        callback: Optional[Callable] = None,
        share_mode: ContextShareMode = ContextShareMode.READ_ONLY,
        sync_strategy: Optional[ContextSyncStrategy] = None,
        filter_function: Optional[Callable[[ContextEntry], bool]] = None,
    ) -> str:
        """
        Subscribe an agent to context updates.

        Args:
            agent_id: ID of the subscribing agent
            patterns: Context key patterns to match
            context_types: Context types to subscribe to
            scopes: Context scopes to subscribe to
            callback: Callback function for notifications
            share_mode: Desired sharing mode
            sync_strategy: Synchronization strategy
            filter_function: Custom filter function

        Returns:
            Subscription ID
        """
        subscription = ContextSubscription(
            agent_id=agent_id,
            context_patterns=patterns or ["*"],
            context_types=context_types or [],
            scopes=scopes or [],
            share_mode=share_mode,
            sync_strategy=sync_strategy or self.default_sync_strategy,
            callback=callback,
            filter_function=filter_function,
        )

        # Store subscription
        self.subscriptions[subscription.id] = subscription
        self.agent_subscriptions[agent_id].add(subscription.id)

        # Update metrics
        self.metrics.total_subscriptions += 1
        self.metrics.active_subscriptions = len([s for s in self.subscriptions.values() if s.is_active])

        self.logger.info(
            "Context subscription created",
            agent_id=agent_id,
            subscription_id=subscription.id,
            patterns=patterns,
            context_types=[t.value for t in (context_types or [])],
        )

        return subscription.id

    async def unsubscribe_from_context(self, subscription_id: str) -> bool:
        """
        Unsubscribe from context updates.

        Args:
            subscription_id: Subscription ID to remove

        Returns:
            True if unsubscribed successfully
        """
        if subscription_id not in self.subscriptions:
            return False

        subscription = self.subscriptions[subscription_id]
        agent_id = subscription.agent_id

        # Remove subscription
        del self.subscriptions[subscription_id]
        self.agent_subscriptions[agent_id].discard(subscription_id)

        # Update metrics
        self.metrics.active_subscriptions = len([s for s in self.subscriptions.values() if s.is_active])

        self.logger.info("Context subscription removed", subscription_id=subscription_id, agent_id=agent_id)
        return True

    async def update_shared_context(
        self,
        agent_id: str,
        context_key: Optional[str] = None,
        entry_id: Optional[str] = None,
        content: Optional[Any] = None,
        **kwargs
    ) -> bool:
        """
        Update shared context with conflict detection.

        Args:
            agent_id: ID of the updating agent
            context_key: Context key to update
            entry_id: Context entry ID to update
            content: New content
            **kwargs: Additional update parameters

        Returns:
            True if updated successfully
        """
        try:
            # Get the context entry
            if entry_id:
                entry = await self.context_store.retrieve_context(entry_id=entry_id)
            elif context_key:
                entry = await self.context_store.retrieve_context(key=context_key)
            else:
                raise ValueError("Either context_key or entry_id must be provided")

            if not entry:
                self.logger.warning("Context entry not found", context_key=context_key, entry_id=entry_id)
                return False

            # Check permissions
            if not self.access_controller.has_permission(agent_id, entry.id, "write"):
                self.logger.warning("Agent lacks write permission", agent_id=agent_id, entry_id=entry.id)
                return False

            # Check for locks
            if await self._is_locked_by_other(entry.id, agent_id):
                self.logger.warning("Context is locked by another agent", entry_id=entry.id)
                return False

            # Store previous version for conflict detection
            previous_version = entry.version

            # Update the context
            updated_entry_id = await self.context_store.store_context(
                key=entry.key,
                content=content,
                context_type=entry.context_type,
                scope=entry.scope,
                **kwargs
            )

            # Create update notification
            update = ContextUpdate(
                context_entry_id=updated_entry_id,
                update_type=ContextUpdateType.UPDATED,
                agent_id=agent_id,
                previous_version=previous_version,
                new_version=entry.version + 1,
                changes={"content": content} if content is not None else {},
            )

            # Check for conflicts
            conflict = await self._detect_conflicts(update)
            if conflict:
                await self._handle_conflict(conflict)
                return False

            # Queue for notification
            await self._queue_update_notification(update)

            # Invalidate cache
            self.context_cache.pop(entry.id, None)

            self.logger.info(
                "Shared context updated",
                agent_id=agent_id,
                entry_id=updated_entry_id,
                previous_version=previous_version,
            )

            return True

        except Exception as e:
            self.logger.error("Error updating shared context", agent_id=agent_id, error=str(e))
            return False

    async def get_shared_context(
        self,
        agent_id: str,
        context_key: Optional[str] = None,
        entry_id: Optional[str] = None,
        use_cache: bool = True,
    ) -> Optional[ContextEntry]:
        """
        Get shared context with permission checking and caching.

        Args:
            agent_id: ID of the requesting agent
            context_key: Context key to retrieve
            entry_id: Context entry ID to retrieve
            use_cache: Whether to use cached version

        Returns:
            Context entry or None if not accessible
        """
        try:
            # Try cache first
            cache_key = entry_id or context_key
            if use_cache and cache_key in self.context_cache:
                cached_entry, cached_time = self.context_cache[cache_key]
                if time.time() - cached_time.timestamp() < self.cache_ttl:
                    # Check permissions
                    if self.access_controller.has_permission(agent_id, cached_entry.id, "read"):
                        return cached_entry

            # Retrieve from context store
            entry = await self.context_store.retrieve_context(key=context_key, entry_id=entry_id)

            if not entry:
                return None

            # Check permissions
            if not self.access_controller.has_permission(agent_id, entry.id, "read"):
                self.logger.warning("Agent lacks read permission", agent_id=agent_id, entry_id=entry.id)
                return None

            # Cache the entry
            if cache_key:
                self.context_cache[cache_key] = (entry, datetime.utcnow())

            # Create access notification
            update = ContextUpdate(
                context_entry_id=entry.id,
                update_type=ContextUpdateType.ACCESSED,
                agent_id=agent_id,
            )

            # Queue for notification (low priority)
            await self._queue_update_notification(update)

            return entry

        except Exception as e:
            self.logger.error("Error getting shared context", agent_id=agent_id, error=str(e))
            return None

    async def acquire_context_lock(
        self,
        agent_id: str,
        context_entry_id: str,
        lock_type: str = "write",
        timeout_seconds: float = 300.0,
        reason: str = "",
    ) -> Optional[str]:
        """
        Acquire a lock on a context entry.

        Args:
            agent_id: ID of the agent requesting the lock
            context_entry_id: Context entry to lock
            lock_type: Type of lock (read, write, exclusive)
            timeout_seconds: Lock timeout
            reason: Reason for the lock

        Returns:
            Lock ID if successful, None otherwise
        """
        try:
            # Check if already locked
            if context_entry_id in self.active_locks:
                existing_lock = self.active_locks[context_entry_id]
                if existing_lock.agent_id != agent_id:
                    # Add to wait queue
                    self.lock_queue[context_entry_id].append((agent_id, lock_type, timeout_seconds, reason))
                    self.logger.info("Lock request queued", agent_id=agent_id, context_entry_id=context_entry_id)
                    return None

            # Create lock
            lock = ContextLock(
                context_entry_id=context_entry_id,
                agent_id=agent_id,
                lock_type=lock_type,
                timeout_seconds=min(timeout_seconds, self.max_lock_duration),
                reason=reason,
                expires_at=datetime.utcnow() + timedelta(seconds=timeout_seconds),
            )

            # Store lock
            self.active_locks[context_entry_id] = lock

            # Create lock notification
            update = ContextUpdate(
                context_entry_id=context_entry_id,
                update_type=ContextUpdateType.LOCKED,
                agent_id=agent_id,
            )

            await self._queue_update_notification(update)

            # Update metrics
            self.metrics.active_locks = len(self.active_locks)

            self.logger.info(
                "Context lock acquired",
                agent_id=agent_id,
                context_entry_id=context_entry_id,
                lock_id=lock.id,
                lock_type=lock_type,
            )

            return lock.id

        except Exception as e:
            self.logger.error("Error acquiring context lock", agent_id=agent_id, error=str(e))
            return None

    async def release_context_lock(self, agent_id: str, lock_id: str) -> bool:
        """
        Release a context lock.

        Args:
            agent_id: ID of the agent releasing the lock
            lock_id: Lock ID to release

        Returns:
            True if released successfully
        """
        try:
            # Find the lock
            lock = None
            context_entry_id = None

            for entry_id, active_lock in self.active_locks.items():
                if active_lock.id == lock_id:
                    if active_lock.agent_id != agent_id:
                        self.logger.warning("Agent cannot release lock owned by another", agent_id=agent_id, lock_id=lock_id)
                        return False

                    lock = active_lock
                    context_entry_id = entry_id
                    break

            if not lock:
                return False

            # Remove lock
            del self.active_locks[context_entry_id]

            # Process queue
            await self._process_lock_queue(context_entry_id)

            # Create unlock notification
            update = ContextUpdate(
                context_entry_id=context_entry_id,
                update_type=ContextUpdateType.UNLOCKED,
                agent_id=agent_id,
            )

            await self._queue_update_notification(update)

            # Update metrics
            self.metrics.active_locks = len(self.active_locks)
            hold_time = (datetime.utcnow() - lock.acquired_at).total_seconds()
            self.metrics.average_lock_hold_time = (self.metrics.average_lock_hold_time + hold_time) / 2

            self.logger.info("Context lock released", agent_id=agent_id, lock_id=lock_id, hold_time=hold_time)
            return True

        except Exception as e:
            self.logger.error("Error releasing context lock", agent_id=agent_id, lock_id=lock_id, error=str(e))
            return False

    def get_sharing_metrics(self) -> Dict[str, Any]:
        """Get comprehensive context sharing metrics."""
        return {
            'subscriptions': {
                'total': self.metrics.total_subscriptions,
                'active': self.metrics.active_subscriptions,
                'by_agent': {
                    agent_id: len(subs)
                    for agent_id, subs in self.agent_subscriptions.items()
                },
            },
            'updates': {
                'total': self.metrics.total_updates,
                'notifications': self.metrics.total_notifications,
                'pending': len(self.pending_updates),
            },
            'performance': {
                'average_notification_time_ms': self.metrics.average_notification_time * 1000,
                'average_sync_time_ms': self.metrics.average_sync_time * 1000,
                'cache_hit_rate': self.metrics.cache_hit_rate,
            },
            'conflicts': {
                'total': self.metrics.total_conflicts,
                'resolved': self.metrics.resolved_conflicts,
                'pending': self.metrics.pending_conflicts,
                'active': len(self.active_conflicts),
            },
            'locks': {
                'active': self.metrics.active_locks,
                'contention_rate': self.metrics.lock_contention_rate,
                'average_hold_time_seconds': self.metrics.average_lock_hold_time,
                'queued_requests': sum(len(queue) for queue in self.lock_queue.values()),
            },
            'cache': {
                'entries': len(self.context_cache),
                'hit_rate': self.metrics.cache_hit_rate,
                'size_mb': sum(len(str(entry)) for entry, _ in self.context_cache.values()) / (1024 * 1024),
            }
        }

    # Private helper methods

    def _setup_sharing_permissions(self, agent_id: str, context_id: str, share_mode: ContextShareMode) -> None:
        """Set up sharing permissions based on share mode."""
        if share_mode == ContextShareMode.READ_ONLY:
            self.access_controller.grant_permission(agent_id, context_id, "read")
        elif share_mode == ContextShareMode.READ_WRITE:
            self.access_controller.grant_permission(agent_id, context_id, "read")
            self.access_controller.grant_permission(agent_id, context_id, "write")
        elif share_mode == ContextShareMode.EXCLUSIVE_WRITE:
            self.access_controller.grant_permission(agent_id, context_id, "read")
            self.access_controller.grant_permission(agent_id, context_id, "write")
            self.access_controller.grant_permission(agent_id, context_id, "exclusive")
        elif share_mode in [ContextShareMode.COLLABORATIVE, ContextShareMode.BROADCAST]:
            self.access_controller.grant_permission(agent_id, context_id, "read")
            self.access_controller.grant_permission(agent_id, context_id, "write")
            self.access_controller.grant_permission(agent_id, context_id, "collaborate")

    def _matches_subscription(self, context_entry_id: str, subscription: ContextSubscription) -> bool:
        """Check if a context entry matches a subscription."""
        # Get the context entry
        entry = self.context_store.memory_store.get(context_entry_id)
        if not entry:
            return False

        # Check context types
        if subscription.context_types and entry.context_type not in subscription.context_types:
            return False

        # Check scopes
        if subscription.scopes and entry.scope not in subscription.scopes:
            return False

        # Check patterns
        if subscription.context_patterns:
            pattern_match = False
            for pattern in subscription.context_patterns:
                if pattern == "*" or pattern in entry.key:
                    pattern_match = True
                    break
            if not pattern_match:
                return False

        # Apply custom filter
        if subscription.filter_function:
            try:
                if not subscription.filter_function(entry):
                    return False
            except Exception as e:
                self.logger.warning("Error in subscription filter", error=str(e))
                return False

        return True

    async def _handle_context_added(self, entry: ContextEntry) -> None:
        """Handle context entry addition."""
        update = ContextUpdate(
            context_entry_id=entry.id,
            update_type=ContextUpdateType.CREATED,
            agent_id="system",
        )
        await self._queue_update_notification(update)

    async def _handle_context_updated(self, entry: ContextEntry) -> None:
        """Handle context entry update."""
        update = ContextUpdate(
            context_entry_id=entry.id,
            update_type=ContextUpdateType.UPDATED,
            agent_id="system",
            new_version=entry.version,
        )
        await self._queue_update_notification(update)

    async def _handle_context_removed(self, entry: ContextEntry) -> None:
        """Handle context entry removal."""
        update = ContextUpdate(
            context_entry_id=entry.id,
            update_type=ContextUpdateType.DELETED,
            agent_id="system",
        )
        await self._queue_update_notification(update)

        # Clean up related data
        self.context_cache.pop(entry.id, None)
        if entry.id in self.active_locks:
            del self.active_locks[entry.id]

    async def _queue_update_notification(self, update: ContextUpdate) -> None:
        """Queue update notification based on sync strategy."""
        # Find matching subscriptions
        matching_subscriptions = []
        for subscription in self.subscriptions.values():
            if not subscription.is_active:
                continue

            if update.update_type not in subscription.update_types:
                continue

            if self._matches_subscription(update.context_entry_id, subscription):
                matching_subscriptions.append(subscription)

        # Group by sync strategy
        for subscription in matching_subscriptions:
            update.pending_agents.add(subscription.agent_id)
            self.notification_queue[subscription.sync_strategy].append((update, subscription))

        # Update metrics
        self.metrics.total_updates += 1
        self.update_history.append(update)

    async def _is_locked_by_other(self, context_entry_id: str, agent_id: str) -> bool:
        """Check if context is locked by another agent."""
        if context_entry_id not in self.active_locks:
            return False

        lock = self.active_locks[context_entry_id]
        return lock.agent_id != agent_id

    async def _detect_conflicts(self, update: ContextUpdate) -> Optional[ContextConflict]:
        """Detect potential conflicts in context updates."""
        # Check for concurrent modifications
        context_entry_id = update.context_entry_id

        # Look for recent updates to the same context
        recent_threshold = datetime.utcnow() - timedelta(seconds=10)
        for historical_update in reversed(self.update_history):
            if (historical_update.context_entry_id == context_entry_id and
                historical_update.timestamp > recent_threshold and
                historical_update.agent_id != update.agent_id and
                historical_update.update_type == ContextUpdateType.UPDATED):

                # Potential conflict detected
                conflict = ContextConflict(
                    context_entry_id=context_entry_id,
                    update1=historical_update,
                    update2=update,
                    resolution_strategy=self.default_conflict_resolution,
                )

                return conflict

        return None

    async def _handle_conflict(self, conflict: ContextConflict) -> None:
        """Handle detected context conflict."""
        self.active_conflicts[conflict.id] = conflict
        self.metrics.total_conflicts += 1
        self.metrics.pending_conflicts += 1

        # Trigger callbacks
        for callback in self.on_conflict_detected:
            try:
                await callback(conflict)
            except Exception as e:
                self.logger.warning("Error in conflict detection callback", error=str(e))

        # Apply resolution strategy
        if conflict.resolution_strategy == ConflictResolution.LAST_WRITER_WINS:
            # The newer update wins (already applied)
            conflict.resolved_at = datetime.utcnow()
            conflict.resolution_result = "last_writer_wins"
            self.metrics.resolved_conflicts += 1
            self.metrics.pending_conflicts -= 1

        self.logger.warning(
            "Context conflict detected and resolved",
            conflict_id=conflict.id,
            context_entry_id=conflict.context_entry_id,
            resolution=conflict.resolution_strategy.value,
        )

    async def _process_lock_queue(self, context_entry_id: str) -> None:
        """Process pending lock requests for a context entry."""
        if context_entry_id not in self.lock_queue:
            return

        queue = self.lock_queue[context_entry_id]
        if not queue:
            return

        # Get next request
        agent_id, lock_type, timeout_seconds, reason = queue.popleft()

        # Try to acquire lock
        lock_id = await self.acquire_context_lock(
            agent_id=agent_id,
            context_entry_id=context_entry_id,
            lock_type=lock_type,
            timeout_seconds=timeout_seconds,
            reason=reason,
        )

        if lock_id:
            self.logger.info("Queued lock request processed", agent_id=agent_id, lock_id=lock_id)

    def _start_background_tasks(self) -> None:
        """Start background processing tasks."""
        # Notification processing task
        notification_task = asyncio.create_task(self._notification_processor())
        self._background_tasks.add(notification_task)
        notification_task.add_done_callback(self._background_tasks.discard)

        # Lock cleanup task
        lock_cleanup_task = asyncio.create_task(self._lock_cleanup_loop())
        self._background_tasks.add(lock_cleanup_task)
        lock_cleanup_task.add_done_callback(self._background_tasks.discard)

        # Cache cleanup task
        cache_cleanup_task = asyncio.create_task(self._cache_cleanup_loop())
        self._background_tasks.add(cache_cleanup_task)
        cache_cleanup_task.add_done_callback(self._background_tasks.discard)

        # Metrics collection task
        metrics_task = asyncio.create_task(self._metrics_collection_loop())
        self._background_tasks.add(metrics_task)
        metrics_task.add_done_callback(self._background_tasks.discard)

    async def _notification_processor(self) -> None:
        """Process queued notifications based on sync strategies."""
        while not self._shutdown_event.is_set():
            try:
                # Process immediate notifications
                await self._process_notifications(ContextSyncStrategy.IMMEDIATE)

                # Process batched notifications
                await self._process_notifications(ContextSyncStrategy.BATCHED)

                # Process periodic notifications
                await self._process_notifications(ContextSyncStrategy.PERIODIC)

                await asyncio.sleep(1.0)  # Process every second

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Error in notification processor", error=str(e))

    async def _process_notifications(self, strategy: ContextSyncStrategy) -> None:
        """Process notifications for a specific sync strategy."""
        queue = self.notification_queue[strategy]

        if strategy == ContextSyncStrategy.IMMEDIATE:
            # Process all immediately
            while queue:
                update, subscription = queue.popleft()
                await self._send_notification(update, subscription)

        elif strategy == ContextSyncStrategy.BATCHED:
            # Process in batches
            batch = []
            while queue and len(batch) < self.notification_batch_size:
                batch.append(queue.popleft())

            if batch:
                await self._send_batch_notifications(batch)

        elif strategy == ContextSyncStrategy.PERIODIC:
            # Process all periodically
            if queue:
                batch = []
                while queue:
                    batch.append(queue.popleft())
                await self._send_batch_notifications(batch)

    async def _send_notification(self, update: ContextUpdate, subscription: ContextSubscription) -> None:
        """Send individual notification."""
        start_time = time.time()

        try:
            if subscription.callback:
                await subscription.callback(update)

            # Update subscription metrics
            subscription.last_notification = datetime.utcnow()
            subscription.notification_count += 1

            # Update global metrics
            self.metrics.total_notifications += 1
            notification_time = time.time() - start_time
            self.metrics.average_notification_time = (
                self.metrics.average_notification_time + notification_time
            ) / 2

            # Mark as notified
            update.notified_agents.add(subscription.agent_id)
            update.pending_agents.discard(subscription.agent_id)

        except Exception as e:
            self.logger.error(
                "Error sending notification",
                subscription_id=subscription.id,
                agent_id=subscription.agent_id,
                error=str(e)
            )

    async def _send_batch_notifications(self, batch: List[Tuple[ContextUpdate, ContextSubscription]]) -> None:
        """Send batch of notifications."""
        # Group by agent
        agent_batches = defaultdict(list)
        for update, subscription in batch:
            agent_batches[subscription.agent_id].append((update, subscription))

        # Send to each agent
        for agent_id, agent_notifications in agent_batches.items():
            try:
                # Use the first subscription's callback for the batch
                if agent_notifications and agent_notifications[0][1].callback:
                    updates = [update for update, _ in agent_notifications]
                    await agent_notifications[0][1].callback(updates)

                # Update metrics
                for update, subscription in agent_notifications:
                    subscription.last_notification = datetime.utcnow()
                    subscription.notification_count += 1
                    update.notified_agents.add(agent_id)
                    update.pending_agents.discard(agent_id)

                self.metrics.total_notifications += len(agent_notifications)

            except Exception as e:
                self.logger.error("Error sending batch notifications", agent_id=agent_id, error=str(e))

    async def _lock_cleanup_loop(self) -> None:
        """Clean up expired locks."""
        while not self._shutdown_event.is_set():
            try:
                current_time = datetime.utcnow()
                expired_locks = []

                for context_id, lock in list(self.active_locks.items()):
                    if lock.expires_at and current_time > lock.expires_at:
                        expired_locks.append((context_id, lock))

                # Release expired locks
                for context_id, lock in expired_locks:
                    self.logger.info("Releasing expired lock", lock_id=lock.id, agent_id=lock.agent_id)
                    await self.release_context_lock(lock.agent_id, lock.id)

                await asyncio.sleep(30)  # Check every 30 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Error in lock cleanup loop", error=str(e))

    async def _cache_cleanup_loop(self) -> None:
        """Clean up expired cache entries."""
        while not self._shutdown_event.is_set():
            try:
                current_time = time.time()
                expired_keys = []

                for key, (entry, cached_time) in list(self.context_cache.items()):
                    if current_time - cached_time.timestamp() > self.cache_ttl:
                        expired_keys.append(key)

                # Remove expired entries
                for key in expired_keys:
                    del self.context_cache[key]

                if expired_keys:
                    self.logger.debug("Cleaned up expired cache entries", count=len(expired_keys))

                await asyncio.sleep(60)  # Clean every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Error in cache cleanup loop", error=str(e))

    async def _metrics_collection_loop(self) -> None:
        """Collect and update metrics."""
        while not self._shutdown_event.is_set():
            try:
                # Update cache hit rate
                cache_hits = len([1 for key, (entry, cached_time) in self.context_cache.items()
                                if time.time() - cached_time.timestamp() < self.cache_ttl])
                cache_total = len(self.context_cache)
                self.metrics.cache_hit_rate = cache_hits / cache_total if cache_total > 0 else 0.0

                # Update lock contention rate
                total_lock_requests = sum(len(queue) for queue in self.lock_queue.values())
                active_locks = len(self.active_locks)
                self.metrics.lock_contention_rate = (
                    total_lock_requests / (total_lock_requests + active_locks)
                    if (total_lock_requests + active_locks) > 0 else 0.0
                )

                await asyncio.sleep(30)  # Update every 30 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Error in metrics collection loop", error=str(e))

    async def _cancel_background_tasks(self) -> None:
        """Cancel all background tasks."""
        for task in self._background_tasks:
            if not task.done():
                task.cancel()

        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)

        self._background_tasks.clear()

    async def _release_all_locks(self) -> None:
        """Release all active locks during shutdown."""
        for context_id, lock in list(self.active_locks.items()):
            try:
                await self.release_context_lock(lock.agent_id, lock.id)
            except Exception as e:
                self.logger.warning("Error releasing lock during shutdown", lock_id=lock.id, error=str(e))
