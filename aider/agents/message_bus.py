"""
MessageBus: High-performance inter-agent communication system for the Aider Hive Architecture.

This module implements a robust publisher-subscriber message bus with support for:
- Topic-based routing and subscriptions
- Priority-based message handling
- Message persistence and reliability
- Dead letter queues for failed messages
- Broadcasting and multicasting
- Message filtering and routing rules
- Performance monitoring and metrics
"""

import asyncio
import logging
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union
from weakref import WeakKeyDictionary, WeakSet

import structlog

from .base_agent import AgentMessage, MessagePriority


class MessageBusState(Enum):
    """Message bus operational states."""
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    SHUTTING_DOWN = "shutting_down"
    SHUTDOWN = "shutdown"


class DeliveryStatus(Enum):
    """Message delivery status tracking."""
    PENDING = "pending"
    DELIVERED = "delivered"
    FAILED = "failed"
    EXPIRED = "expired"
    DEAD_LETTER = "dead_letter"


@dataclass
class Subscription:
    """Represents a subscription to messages."""
    subscriber_id: str
    handler: Callable[[AgentMessage], None]
    topics: Set[str] = field(default_factory=set)
    message_types: Set[str] = field(default_factory=set)
    priority_filter: Optional[MessagePriority] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    is_active: bool = True


@dataclass
class MessageDeliveryInfo:
    """Tracks message delivery information."""
    message_id: str
    status: DeliveryStatus = DeliveryStatus.PENDING
    attempts: int = 0
    max_attempts: int = 3
    last_attempt: Optional[datetime] = None
    delivered_to: Set[str] = field(default_factory=set)
    failed_to: Set[str] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class MessageBusMetrics:
    """Performance and operational metrics for the message bus."""
    messages_published: int = 0
    messages_delivered: int = 0
    messages_failed: int = 0
    messages_expired: int = 0
    active_subscriptions: int = 0
    average_delivery_time: float = 0.0
    peak_queue_size: int = 0
    current_queue_size: int = 0
    uptime: float = 0.0


class MessageBus:
    """
    High-performance message bus for inter-agent communication.

    Features:
    - Asynchronous pub/sub messaging
    - Topic-based routing
    - Priority queuing
    - Message persistence and retry logic
    - Dead letter queue handling
    - Broadcasting capabilities
    - Performance monitoring
    """

    def __init__(
        self,
        max_queue_size: int = 10000,
        max_retry_attempts: int = 3,
        message_ttl: int = 300,  # 5 minutes
        enable_persistence: bool = False,
        enable_dead_letter_queue: bool = True,
    ):
        """
        Initialize the message bus.

        Args:
            max_queue_size: Maximum number of messages in queue
            max_retry_attempts: Maximum delivery retry attempts
            message_ttl: Message time-to-live in seconds
            enable_persistence: Enable message persistence (for reliability)
            enable_dead_letter_queue: Enable dead letter queue for failed messages
        """
        self.max_queue_size = max_queue_size
        self.max_retry_attempts = max_retry_attempts
        self.message_ttl = message_ttl
        self.enable_persistence = enable_persistence
        self.enable_dead_letter_queue = enable_dead_letter_queue

        # State management
        self.state = MessageBusState.INITIALIZING
        self.started_at: Optional[datetime] = None

        # Logging
        self.logger = structlog.get_logger().bind(component="message_bus")

        # Subscriptions management
        self.subscriptions: Dict[str, Subscription] = {}
        self.topic_subscribers: Dict[str, Set[str]] = defaultdict(set)
        self.type_subscribers: Dict[str, Set[str]] = defaultdict(set)
        self.priority_subscribers: Dict[MessagePriority, Set[str]] = defaultdict(set)

        # Message queues (priority-based)
        self.message_queues: Dict[MessagePriority, asyncio.Queue] = {
            priority: asyncio.Queue() for priority in MessagePriority
        }

        # Message tracking and persistence
        self.pending_messages: Dict[str, MessageDeliveryInfo] = {}
        self.dead_letter_queue: deque = deque(maxlen=1000)
        self.message_history: deque = deque(maxlen=5000)  # Recent message history

        # Performance metrics
        self.metrics = MessageBusMetrics()

        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()
        self._shutdown_event = asyncio.Event()

        # Rate limiting and flow control
        self.rate_limiter = asyncio.Semaphore(100)  # Max 100 concurrent operations
        self.delivery_semaphore = asyncio.Semaphore(50)  # Max 50 concurrent deliveries

    async def start(self) -> None:
        """Start the message bus and background processing."""
        try:
            self.logger.info("Starting message bus")
            self.state = MessageBusState.INITIALIZING

            # Start background tasks
            self._start_background_tasks()

            self.state = MessageBusState.RUNNING
            self.started_at = datetime.utcnow()

            self.logger.info("Message bus started successfully")

        except Exception as e:
            self.state = MessageBusState.SHUTDOWN
            self.logger.error("Failed to start message bus", error=str(e))
            raise

    async def stop(self) -> None:
        """Stop the message bus gracefully."""
        self.logger.info("Stopping message bus")
        self.state = MessageBusState.SHUTTING_DOWN

        # Signal shutdown to background tasks
        self._shutdown_event.set()

        # Process remaining messages (with timeout)
        await self._drain_queues(timeout=30.0)

        # Cancel background tasks
        await self._cancel_background_tasks()

        # Clear data structures
        self.subscriptions.clear()
        self.topic_subscribers.clear()
        self.type_subscribers.clear()
        self.priority_subscribers.clear()

        self.state = MessageBusState.SHUTDOWN
        self.logger.info("Message bus stopped")

    async def subscribe(
        self,
        subscriber_id: str,
        handler: Callable[[AgentMessage], None],
        topics: Optional[List[str]] = None,
        message_types: Optional[List[str]] = None,
        priority_filter: Optional[MessagePriority] = None,
    ) -> str:
        """
        Subscribe to messages.

        Args:
            subscriber_id: Unique identifier for the subscriber
            handler: Async function to handle received messages
            topics: List of topics to subscribe to (None for all)
            message_types: List of message types to subscribe to (None for all)
            priority_filter: Only receive messages of this priority or higher

        Returns:
            Subscription ID
        """
        if self.state != MessageBusState.RUNNING:
            raise RuntimeError("Message bus is not running")

        subscription = Subscription(
            subscriber_id=subscriber_id,
            handler=handler,
            topics=set(topics or []),
            message_types=set(message_types or []),
            priority_filter=priority_filter,
        )

        subscription_id = f"{subscriber_id}_{uuid.uuid4().hex[:8]}"
        self.subscriptions[subscription_id] = subscription

        # Update topic and type indexes
        if topics:
            for topic in topics:
                self.topic_subscribers[topic].add(subscription_id)
        if message_types:
            for msg_type in message_types:
                self.type_subscribers[msg_type].add(subscription_id)
        if priority_filter:
            self.priority_subscribers[priority_filter].add(subscription_id)

        self.metrics.active_subscriptions = len(self.subscriptions)

        self.logger.info(
            "New subscription created",
            subscription_id=subscription_id,
            subscriber_id=subscriber_id,
            topics=topics,
            message_types=message_types,
        )

        return subscription_id

    async def unsubscribe(self, subscriber_id: str) -> int:
        """
        Unsubscribe from all messages for a subscriber.

        Args:
            subscriber_id: Subscriber to unsubscribe

        Returns:
            Number of subscriptions removed
        """
        removed_count = 0

        # Find and remove all subscriptions for this subscriber
        to_remove = [
            sub_id for sub_id, sub in self.subscriptions.items()
            if sub.subscriber_id == subscriber_id
        ]

        for subscription_id in to_remove:
            subscription = self.subscriptions[subscription_id]

            # Remove from topic indexes
            for topic in subscription.topics:
                self.topic_subscribers[topic].discard(subscription_id)

            # Remove from message type indexes
            for msg_type in subscription.message_types:
                self.type_subscribers[msg_type].discard(subscription_id)

            # Remove from priority indexes
            if subscription.priority_filter:
                self.priority_subscribers[subscription.priority_filter].discard(subscription_id)

            # Remove subscription
            del self.subscriptions[subscription_id]
            removed_count += 1

        self.metrics.active_subscriptions = len(self.subscriptions)

        self.logger.info(
            "Subscriptions removed",
            subscriber_id=subscriber_id,
            count=removed_count
        )

        return removed_count

    async def publish(
        self,
        message: AgentMessage,
        topic: Optional[str] = None,
        broadcast: bool = False,
    ) -> str:
        """
        Publish a message to the bus.

        Args:
            message: Message to publish
            topic: Optional topic for routing
            broadcast: If True, send to all subscribers

        Returns:
            Message ID
        """
        if self.state != MessageBusState.RUNNING:
            raise RuntimeError("Message bus is not running")

        async with self.rate_limiter:
            # Validate message
            if not message.id:
                message.id = str(uuid.uuid4())

            # Set topic if provided
            if topic:
                message.payload['_topic'] = topic

            # Check queue capacity
            current_size = sum(queue.qsize() for queue in self.message_queues.values())
            if current_size >= self.max_queue_size:
                self.logger.warning("Message queue at capacity, dropping message")
                return message.id

            # Create delivery tracking
            delivery_info = MessageDeliveryInfo(
                message_id=message.id,
                max_attempts=self.max_retry_attempts,
            )
            self.pending_messages[message.id] = delivery_info

            # Add to appropriate priority queue
            await self.message_queues[message.priority].put({
                'message': message,
                'topic': topic,
                'broadcast': broadcast,
                'queued_at': time.time(),
            })

            # Update metrics
            self.metrics.messages_published += 1
            current_queue_size = sum(queue.qsize() for queue in self.message_queues.values())
            self.metrics.current_queue_size = current_queue_size
            if current_queue_size > self.metrics.peak_queue_size:
                self.metrics.peak_queue_size = current_queue_size

            # Add to history
            self.message_history.append({
                'message_id': message.id,
                'sender_id': message.sender_id,
                'message_type': message.message_type,
                'priority': message.priority,
                'timestamp': message.timestamp,
                'topic': topic,
            })

            self.logger.debug(
                "Message published",
                message_id=message.id,
                sender_id=message.sender_id,
                message_type=message.message_type,
                topic=topic,
                broadcast=broadcast,
            )

            return message.id

    async def publish_to_topic(self, topic: str, message: AgentMessage) -> str:
        """Publish a message to a specific topic."""
        return await self.publish(message, topic=topic)

    async def broadcast(self, message: AgentMessage) -> str:
        """Broadcast a message to all subscribers."""
        return await self.publish(message, broadcast=True)

    def get_subscribers(self, topic: Optional[str] = None) -> List[str]:
        """Get list of subscribers for a topic or all subscribers."""
        if topic:
            subscription_ids = self.topic_subscribers.get(topic, set())
            return [
                self.subscriptions[sub_id].subscriber_id
                for sub_id in subscription_ids
                if sub_id in self.subscriptions
            ]
        else:
            return list(set(sub.subscriber_id for sub in self.subscriptions.values()))

    def get_metrics(self) -> Dict[str, Any]:
        """Get current message bus metrics."""
        uptime = 0.0
        if self.started_at:
            uptime = (datetime.utcnow() - self.started_at).total_seconds()

        return {
            'state': self.state.value,
            'uptime': uptime,
            'metrics': {
                'messages_published': self.metrics.messages_published,
                'messages_delivered': self.metrics.messages_delivered,
                'messages_failed': self.metrics.messages_failed,
                'messages_expired': self.metrics.messages_expired,
                'active_subscriptions': self.metrics.active_subscriptions,
                'average_delivery_time': self.metrics.average_delivery_time,
                'current_queue_size': self.metrics.current_queue_size,
                'peak_queue_size': self.metrics.peak_queue_size,
            },
            'queue_sizes': {
                priority.name: queue.qsize()
                for priority, queue in self.message_queues.items()
            },
            'pending_messages': len(self.pending_messages),
            'dead_letter_queue_size': len(self.dead_letter_queue),
        }

    def _start_background_tasks(self) -> None:
        """Start background processing tasks."""
        # Message delivery processors (one per priority level)
        for priority in MessagePriority:
            task = asyncio.create_task(self._message_processor(priority))
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

        # Cleanup and maintenance tasks
        cleanup_task = asyncio.create_task(self._cleanup_expired_messages())
        self._background_tasks.add(cleanup_task)
        cleanup_task.add_done_callback(self._background_tasks.discard)

        retry_task = asyncio.create_task(self._retry_failed_messages())
        self._background_tasks.add(retry_task)
        retry_task.add_done_callback(self._background_tasks.discard)

    async def _message_processor(self, priority: MessagePriority) -> None:
        """Process messages from a specific priority queue."""
        queue = self.message_queues[priority]

        while not self._shutdown_event.is_set():
            try:
                # Get message with timeout
                message_data = await asyncio.wait_for(queue.get(), timeout=1.0)

                # Process the message
                await self._deliver_message(message_data)

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(
                    "Error in message processor",
                    priority=priority.name,
                    error=str(e)
                )

    async def _deliver_message(self, message_data: Dict[str, Any]) -> None:
        """Deliver a message to appropriate subscribers."""
        message = message_data['message']
        topic = message_data.get('topic')
        broadcast = message_data.get('broadcast', False)
        queued_at = message_data.get('queued_at', time.time())

        start_time = time.time()

        try:
            async with self.delivery_semaphore:
                # Find matching subscribers
                subscribers = self._find_matching_subscribers(message, topic, broadcast)

                if not subscribers:
                    self.logger.debug(
                        "No subscribers found for message",
                        message_id=message.id,
                        message_type=message.message_type,
                        topic=topic,
                    )
                    return

                # Deliver to each subscriber
                delivery_tasks = []
                for subscription_id in subscribers:
                    if subscription_id in self.subscriptions:
                        task = asyncio.create_task(
                            self._deliver_to_subscriber(subscription_id, message)
                        )
                        delivery_tasks.append(task)

                # Wait for all deliveries to complete
                if delivery_tasks:
                    results = await asyncio.gather(*delivery_tasks, return_exceptions=True)

                    # Process results
                    successful_deliveries = sum(1 for result in results if result is True)
                    failed_deliveries = len(results) - successful_deliveries

                    # Update delivery tracking
                    delivery_info = self.pending_messages.get(message.id)
                    if delivery_info:
                        if failed_deliveries == 0:
                            delivery_info.status = DeliveryStatus.DELIVERED
                        else:
                            delivery_info.status = DeliveryStatus.FAILED
                            delivery_info.attempts += 1

                    # Update metrics
                    delivery_time = time.time() - start_time
                    self.metrics.average_delivery_time = (
                        self.metrics.average_delivery_time + delivery_time
                    ) / 2

                    if successful_deliveries > 0:
                        self.metrics.messages_delivered += successful_deliveries
                    if failed_deliveries > 0:
                        self.metrics.messages_failed += failed_deliveries

        except Exception as e:
            self.logger.error(
                "Error delivering message",
                message_id=message.id,
                error=str(e)
            )

            # Mark as failed
            delivery_info = self.pending_messages.get(message.id)
            if delivery_info:
                delivery_info.status = DeliveryStatus.FAILED
                delivery_info.attempts += 1

    def _find_matching_subscribers(
        self,
        message: AgentMessage,
        topic: Optional[str],
        broadcast: bool,
    ) -> Set[str]:
        """Find subscribers that should receive this message."""
        if broadcast:
            return set(self.subscriptions.keys())

        matching_subscribers = set()

        # Direct recipient
        if message.recipient_id:
            for sub_id, sub in self.subscriptions.items():
                if sub.subscriber_id == message.recipient_id:
                    matching_subscribers.add(sub_id)

        # Topic-based routing
        if topic and topic in self.topic_subscribers:
            matching_subscribers.update(self.topic_subscribers[topic])

        # Message type routing
        if message.message_type in self.type_subscribers:
            matching_subscribers.update(self.type_subscribers[message.message_type])

        # Priority filtering
        filtered_subscribers = set()
        for sub_id in matching_subscribers:
            subscription = self.subscriptions[sub_id]
            if (subscription.priority_filter is None or
                message.priority.value <= subscription.priority_filter.value):
                filtered_subscribers.add(sub_id)

        return filtered_subscribers

    async def _deliver_to_subscriber(
        self,
        subscription_id: str,
        message: AgentMessage,
    ) -> bool:
        """Deliver message to a specific subscriber."""
        try:
            subscription = self.subscriptions[subscription_id]

            if not subscription.is_active:
                return False

            # Call the handler
            await subscription.handler(message)

            # Update delivery tracking
            delivery_info = self.pending_messages.get(message.id)
            if delivery_info:
                delivery_info.delivered_to.add(subscription.subscriber_id)

            return True

        except Exception as e:
            self.logger.error(
                "Failed to deliver message to subscriber",
                subscription_id=subscription_id,
                message_id=message.id,
                error=str(e)
            )

            # Update delivery tracking
            delivery_info = self.pending_messages.get(message.id)
            if delivery_info:
                delivery_info.failed_to.add(subscription_id)

            return False

    async def _cleanup_expired_messages(self) -> None:
        """Clean up expired messages and delivery tracking."""
        while not self._shutdown_event.is_set():
            try:
                current_time = datetime.utcnow()
                expired_messages = []

                for message_id, delivery_info in self.pending_messages.items():
                    message_age = (current_time - delivery_info.created_at).total_seconds()

                    if message_age > self.message_ttl:
                        expired_messages.append(message_id)

                # Remove expired messages
                for message_id in expired_messages:
                    delivery_info = self.pending_messages.pop(message_id)
                    delivery_info.status = DeliveryStatus.EXPIRED

                    if self.enable_dead_letter_queue:
                        self.dead_letter_queue.append({
                            'message_id': message_id,
                            'status': DeliveryStatus.EXPIRED,
                            'expired_at': current_time,
                            'delivery_info': delivery_info,
                        })

                    self.metrics.messages_expired += 1

                # Sleep before next cleanup
                await asyncio.sleep(60)  # Clean up every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Error in cleanup task", error=str(e))

    async def _retry_failed_messages(self) -> None:
        """Retry delivery of failed messages."""
        while not self._shutdown_event.is_set():
            try:
                retry_messages = []

                for message_id, delivery_info in self.pending_messages.items():
                    if (delivery_info.status == DeliveryStatus.FAILED and
                        delivery_info.attempts < delivery_info.max_attempts):

                        # Check if enough time has passed since last attempt
                        if (delivery_info.last_attempt is None or
                            (datetime.utcnow() - delivery_info.last_attempt).total_seconds() > 30):
                            retry_messages.append(message_id)

                # TODO: Implement retry logic (would need to store original message data)
                # This is a placeholder for the retry mechanism

                await asyncio.sleep(30)  # Check for retries every 30 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Error in retry task", error=str(e))

    async def _drain_queues(self, timeout: float = 30.0) -> None:
        """Drain remaining messages from queues during shutdown."""
        start_time = time.time()

        while time.time() - start_time < timeout:
            total_messages = sum(queue.qsize() for queue in self.message_queues.values())
            if total_messages == 0:
                break

            await asyncio.sleep(0.1)

        self.logger.info("Queue draining completed")

    async def _cancel_background_tasks(self) -> None:
        """Cancel all background tasks."""
        for task in self._background_tasks:
            if not task.done():
                task.cancel()

        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)

        self._background_tasks.clear()

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check and return status.

        Returns:
            Dictionary containing health status information
        """
        try:
            current_time = datetime.utcnow()

            # Check message bus operational status
            is_running = self.state == MessageBusState.RUNNING

            # Check queue health
            total_queued_messages = sum(queue.qsize() for queue in self.message_queues.values())
            max_queue_size = getattr(self, 'max_queue_size', 1000)
            queue_utilization = (total_queued_messages / max_queue_size) * 100 if max_queue_size > 0 else 0

            # Check subscription health
            total_subscriptions = sum(len(subs) for subs in self.subscriptions.values())
            active_subscribers = len(self.subscriber_queues)

            # Check performance metrics
            messages_sent = self.metrics.get('messages_sent', 0)
            messages_delivered = self.metrics.get('messages_delivered', 0)
            messages_failed = self.metrics.get('messages_failed', 0)

            delivery_rate = 0.0
            if messages_sent > 0:
                delivery_rate = (messages_delivered / messages_sent) * 100

            # Check background task health
            active_tasks = len([task for task in self._background_tasks if not task.done()])

            # Determine overall health status
            is_healthy = (
                is_running and
                queue_utilization < 90.0 and  # Not overloaded
                delivery_rate >= 95.0 and     # High delivery success rate
                active_tasks > 0               # Background tasks running
            )

            health_status = {
                "status": "healthy" if is_healthy else "degraded",
                "timestamp": current_time.isoformat(),
                "message_bus_specific": {
                    "state": self.state.value,
                    "is_running": is_running,
                    "queue_health": {
                        "total_queued_messages": total_queued_messages,
                        "max_queue_size": max_queue_size,
                        "queue_utilization_percent": queue_utilization,
                        "active_queues": len(self.message_queues)
                    },
                    "subscription_health": {
                        "total_subscriptions": total_subscriptions,
                        "active_subscribers": active_subscribers,
                        "subscription_topics": len(self.subscriptions)
                    },
                    "performance_metrics": {
                        "messages_sent": messages_sent,
                        "messages_delivered": messages_delivered,
                        "messages_failed": messages_failed,
                        "delivery_rate_percent": delivery_rate,
                        "average_delivery_time": self.metrics.get('avg_delivery_time', 0.0)
                    },
                    "background_tasks": {
                        "active_tasks": active_tasks,
                        "total_tasks": len(self._background_tasks)
                    }
                }
            }

            # Add any critical issues
            issues = []
            if not is_running:
                issues.append(f"Message bus not running (state: {self.state.value})")
            if queue_utilization >= 90.0:
                issues.append(f"High queue utilization: {queue_utilization:.1f}%")
            if delivery_rate < 95.0 and messages_sent > 0:
                issues.append(f"Low delivery rate: {delivery_rate:.1f}%")
            if active_tasks == 0:
                issues.append("No background tasks running")

            if issues:
                health_status["issues"] = issues

            return health_status

        except Exception as e:
            self.logger.error(f"Health check failed: {e}", exc_info=True)
            return {
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
                "error_type": type(e).__name__
            }
