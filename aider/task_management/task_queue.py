"""
HiveTaskManager: Intelligent task scheduling and routing system for the Aider Hive Architecture.

This module provides advanced task management capabilities including:
- Priority-based task scheduling
- Task dependency management
- Intelligent routing and load balancing
- Retry logic and error handling
- Performance monitoring and metrics
- Dynamic priority adjustment
- Task batching and optimization
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
from weakref import WeakSet

import structlog


class TaskState(Enum):
    """Task execution states."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"
    TIMEOUT = "timeout"


class TaskPriority(Enum):
    """Task priority levels for scheduling."""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


class TaskType(Enum):
    """Types of tasks that can be executed."""
    CODE_GENERATION = "code_generation"
    CODE_ANALYSIS = "code_analysis"
    CONTEXT_BUILDING = "context_building"
    GIT_OPERATION = "git_operation"
    FILE_OPERATION = "file_operation"
    SEARCH_OPERATION = "search_operation"
    VALIDATION = "validation"
    OPTIMIZATION = "optimization"
    MONITORING = "monitoring"
    MAINTENANCE = "maintenance"


@dataclass
class TaskDependency:
    """Represents a task dependency relationship."""
    task_id: str
    dependency_type: str = "completion"  # completion, data, resource
    timeout: Optional[float] = None


@dataclass
class TaskResource:
    """Represents a resource requirement for task execution."""
    resource_type: str
    amount: float = 1.0
    exclusive: bool = False


@dataclass
class TaskResult:
    """Result of task execution."""
    task_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Task:
    """Represents a task in the hive system."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: TaskType = TaskType.CODE_GENERATION
    priority: TaskPriority = TaskPriority.NORMAL
    state: TaskState = TaskState.PENDING

    # Task execution details
    handler: Optional[Callable] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)

    # Scheduling and dependencies
    dependencies: List[TaskDependency] = field(default_factory=list)
    required_capabilities: List[str] = field(default_factory=list)
    required_resources: List[TaskResource] = field(default_factory=list)
    target_agent_type: Optional[str] = None

    # Execution constraints
    timeout: Optional[float] = None
    max_retries: int = 3
    retry_delay: float = 1.0
    retry_backoff: float = 2.0

    # Timing
    created_at: datetime = field(default_factory=datetime.utcnow)
    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Execution tracking
    attempts: int = 0
    assigned_agent: Optional[str] = None
    result: Optional[TaskResult] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)


@dataclass
class TaskMetrics:
    """Task execution metrics and statistics."""
    total_tasks_submitted: int = 0
    total_tasks_completed: int = 0
    total_tasks_failed: int = 0
    total_tasks_cancelled: int = 0
    total_tasks_timeout: int = 0

    average_execution_time: float = 0.0
    average_queue_time: float = 0.0
    peak_queue_size: int = 0
    current_queue_size: int = 0

    tasks_by_priority: Dict[TaskPriority, int] = field(default_factory=lambda: {p: 0 for p in TaskPriority})
    tasks_by_type: Dict[TaskType, int] = field(default_factory=lambda: {t: 0 for t in TaskType})

    throughput_per_minute: float = 0.0
    error_rate: float = 0.0


class TaskScheduler:
    """Advanced task scheduler with multiple scheduling strategies."""

    def __init__(self, strategy: str = "priority_fifo"):
        self.strategy = strategy
        self.priority_queues: Dict[TaskPriority, deque] = {
            priority: deque() for priority in TaskPriority
        }

    def enqueue(self, task: Task) -> None:
        """Add a task to the appropriate queue."""
        self.priority_queues[task.priority].append(task)
        task.state = TaskState.QUEUED
        task.scheduled_at = datetime.utcnow()

    def dequeue(self) -> Optional[Task]:
        """Get the next task to execute based on scheduling strategy."""
        if self.strategy == "priority_fifo":
            return self._dequeue_priority_fifo()
        elif self.strategy == "priority_srtf":
            return self._dequeue_priority_srtf()
        elif self.strategy == "weighted_round_robin":
            return self._dequeue_weighted_round_robin()
        else:
            return self._dequeue_priority_fifo()

    def _dequeue_priority_fifo(self) -> Optional[Task]:
        """Dequeue highest priority task (FIFO within priority)."""
        for priority in TaskPriority:
            if self.priority_queues[priority]:
                return self.priority_queues[priority].popleft()
        return None

    def _dequeue_priority_srtf(self) -> Optional[Task]:
        """Dequeue based on priority and estimated execution time (Shortest Remaining Time First)."""
        best_task = None
        best_queue = None
        best_priority = None

        for priority in TaskPriority:
            queue = self.priority_queues[priority]
            if queue:
                # For now, just take the first task (could be enhanced with time estimation)
                task = queue[0]
                if best_task is None or priority.value < best_priority.value:
                    best_task = task
                    best_queue = queue
                    best_priority = priority

        if best_task and best_queue:
            best_queue.remove(best_task)
            return best_task

        return None

    def _dequeue_weighted_round_robin(self) -> Optional[Task]:
        """Dequeue using weighted round-robin based on priority."""
        # Priority weights (higher priority = more chances to be selected)
        weights = {
            TaskPriority.CRITICAL: 8,
            TaskPriority.HIGH: 4,
            TaskPriority.NORMAL: 2,
            TaskPriority.LOW: 1,
            TaskPriority.BACKGROUND: 1,
        }

        # Simple weighted selection
        import random
        total_weight = sum(weights[p] * len(self.priority_queues[p]) for p in TaskPriority)

        if total_weight == 0:
            return None

        target = random.uniform(0, total_weight)
        current = 0

        for priority in TaskPriority:
            queue = self.priority_queues[priority]
            if queue:
                current += weights[priority] * len(queue)
                if current >= target:
                    return queue.popleft()

        return None

    def get_queue_sizes(self) -> Dict[TaskPriority, int]:
        """Get current queue sizes by priority."""
        return {priority: len(queue) for priority, queue in self.priority_queues.items()}

    def get_total_queued(self) -> int:
        """Get total number of queued tasks."""
        return sum(len(queue) for queue in self.priority_queues.values())


class DependencyManager:
    """Manages task dependencies and execution ordering."""

    def __init__(self):
        self.dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        self.reverse_dependencies: Dict[str, Set[str]] = defaultdict(set)
        self.completed_tasks: Set[str] = set()
        self.blocked_tasks: Dict[str, Set[str]] = defaultdict(set)

    def add_dependency(self, task_id: str, depends_on: str) -> None:
        """Add a dependency relationship."""
        self.dependency_graph[task_id].add(depends_on)
        self.reverse_dependencies[depends_on].add(task_id)

        if depends_on not in self.completed_tasks:
            self.blocked_tasks[depends_on].add(task_id)

    def mark_completed(self, task_id: str) -> Set[str]:
        """Mark a task as completed and return unblocked tasks."""
        self.completed_tasks.add(task_id)
        unblocked = set()

        # Check tasks that were waiting for this one
        for waiting_task in self.blocked_tasks[task_id]:
            # Remove this dependency
            self.dependency_graph[waiting_task].discard(task_id)

            # If no more dependencies, unblock the task
            if not self.dependency_graph[waiting_task]:
                unblocked.add(waiting_task)

        # Clean up
        del self.blocked_tasks[task_id]

        return unblocked

    def is_ready(self, task_id: str) -> bool:
        """Check if a task is ready to execute (all dependencies satisfied)."""
        dependencies = self.dependency_graph.get(task_id, set())
        return all(dep in self.completed_tasks for dep in dependencies)

    def get_blocked_tasks(self) -> Dict[str, Set[str]]:
        """Get tasks blocked by dependencies."""
        return dict(self.blocked_tasks)


class ResourceManager:
    """Manages resource allocation and constraints for task execution."""

    def __init__(self):
        self.available_resources: Dict[str, float] = defaultdict(float)
        self.allocated_resources: Dict[str, float] = defaultdict(float)
        self.resource_reservations: Dict[str, Dict[str, float]] = defaultdict(dict)

    def set_resource_limit(self, resource_type: str, limit: float) -> None:
        """Set the limit for a resource type."""
        self.available_resources[resource_type] = limit

    def can_allocate(self, task_id: str, resources: List[TaskResource]) -> bool:
        """Check if resources can be allocated for a task."""
        for resource in resources:
            available = self.available_resources[resource.resource_type]
            allocated = self.allocated_resources[resource.resource_type]

            if available - allocated < resource.amount:
                return False

        return True

    def allocate_resources(self, task_id: str, resources: List[TaskResource]) -> bool:
        """Allocate resources for a task."""
        if not self.can_allocate(task_id, resources):
            return False

        for resource in resources:
            self.allocated_resources[resource.resource_type] += resource.amount
            self.resource_reservations[task_id][resource.resource_type] = resource.amount

        return True

    def release_resources(self, task_id: str) -> None:
        """Release resources allocated to a task."""
        if task_id in self.resource_reservations:
            for resource_type, amount in self.resource_reservations[task_id].items():
                self.allocated_resources[resource_type] -= amount

            del self.resource_reservations[task_id]

    def get_resource_usage(self) -> Dict[str, Dict[str, float]]:
        """Get current resource usage statistics."""
        return {
            resource_type: {
                'available': self.available_resources[resource_type],
                'allocated': self.allocated_resources[resource_type],
                'utilization': (
                    self.allocated_resources[resource_type] /
                    max(1, self.available_resources[resource_type])
                ),
            }
            for resource_type in self.available_resources
        }


class HiveTaskManager:
    """
    Advanced task management system for the Aider Hive Architecture.

    Features:
    - Priority-based task scheduling
    - Task dependency management
    - Resource allocation and constraints
    - Intelligent routing and load balancing
    - Retry logic and error handling
    - Performance monitoring and metrics
    - Dynamic priority adjustment
    """

    def __init__(
        self,
        max_concurrent_tasks: int = 100,
        max_queue_size: int = 10000,
        default_timeout: float = 300.0,
        scheduling_strategy: str = "priority_fifo",
        enable_metrics: bool = True,
    ):
        """
        Initialize the task manager.

        Args:
            max_concurrent_tasks: Maximum number of concurrent task executions
            max_queue_size: Maximum number of tasks in queue
            default_timeout: Default task timeout in seconds
            scheduling_strategy: Task scheduling strategy
            enable_metrics: Enable performance metrics collection
        """
        self.max_concurrent_tasks = max_concurrent_tasks
        self.max_queue_size = max_queue_size
        self.default_timeout = default_timeout
        self.enable_metrics = enable_metrics

        # Core components
        self.scheduler = TaskScheduler(scheduling_strategy)
        self.dependency_manager = DependencyManager()
        self.resource_manager = ResourceManager()

        # Task storage and tracking
        self.tasks: Dict[str, Task] = {}
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.completed_tasks: deque = deque(maxlen=1000)
        self.failed_tasks: deque = deque(maxlen=100)

        # State management
        self.is_running = False
        self.started_at: Optional[datetime] = None

        # Logging
        self.logger = structlog.get_logger().bind(component="task_manager")

        # Metrics
        self.metrics = TaskMetrics()
        self.metrics_history: deque = deque(maxlen=100)

        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()
        self._shutdown_event = asyncio.Event()

        # Concurrency control
        self.execution_semaphore = asyncio.Semaphore(max_concurrent_tasks)

        # Task handlers registry
        self.task_handlers: Dict[TaskType, Callable] = {}

        # Agent pool reference (set externally)
        self.agent_pool = None

    async def initialize(self) -> None:
        """Initialize the task manager components."""
        try:
            self.logger.info("Initializing task manager")

            # Initialize resource limits
            self._initialize_resources()

            # Initialize task handlers
            self._initialize_task_handlers()

            # Validate configuration
            self._validate_config()

            self.logger.info("Task manager initialized successfully")

        except Exception as e:
            self.logger.error("Failed to initialize task manager", error=str(e))
            raise

    async def health_check(self) -> Dict[str, Any]:
        """Perform a comprehensive health check of the task manager."""
        try:
            health_status = {
                "status": "healthy",
                "is_running": self.is_running,
                "uptime": None,
                "queue_size": self.scheduler.get_total_queued(),
                "running_tasks": len(self.running_tasks),
                "completed_tasks": len([t for t in self.tasks.values() if t.state == TaskState.COMPLETED]),
                "failed_tasks": len([t for t in self.tasks.values() if t.state == TaskState.FAILED]),
                "resource_usage": {},
                "background_tasks_count": len(self._background_tasks),
                "memory_usage": "N/A",  # Could integrate psutil for actual memory usage
                "errors": []
            }

            # Calculate uptime if running
            if self.is_running and self.started_at:
                uptime = datetime.utcnow() - self.started_at
                health_status["uptime"] = str(uptime)

            # Check resource manager health
            if hasattr(self.resource_manager, 'get_resource_usage'):
                health_status["resource_usage"] = self.resource_manager.get_resource_usage()

            # Check for any warning conditions
            if self.scheduler.get_total_queued() > self.max_queue_size * 0.8:
                health_status["errors"].append("Queue size approaching limit")

            if len(self.running_tasks) > self.max_concurrent_tasks * 0.9:
                health_status["errors"].append("High concurrent task load")

            # Check if background tasks are running
            if self.is_running and len(self._background_tasks) == 0:
                health_status["errors"].append("Background tasks not running")
                health_status["status"] = "degraded"

            # Overall health determination
            if health_status["errors"]:
                health_status["status"] = "degraded" if health_status["status"] == "healthy" else "unhealthy"

            return health_status

        except Exception as e:
            self.logger.error("Health check failed", error=str(e))
            return {
                "status": "unhealthy",
                "error": str(e),
                "is_running": self.is_running
            }

    def _initialize_task_handlers(self) -> None:
        """Initialize default task handlers."""
        # This will be extended when integrating with specific agents
        self.logger.debug("Task handlers initialized")

    def _validate_config(self) -> None:
        """Validate task manager configuration."""
        if self.max_concurrent_tasks <= 0:
            raise ValueError("max_concurrent_tasks must be positive")
        if self.max_queue_size <= 0:
            raise ValueError("max_queue_size must be positive")
        if self.default_timeout <= 0:
            raise ValueError("default_timeout must be positive")

    async def start(self) -> None:
        """Start the task manager and background processing."""
        try:
            self.logger.info("Starting task manager")

            # Initialize resource limits
            self._initialize_resources()

            # Start background tasks
            self._start_background_tasks()

            self.is_running = True
            self.started_at = datetime.utcnow()

            self.logger.info("Task manager started successfully")

        except Exception as e:
            self.logger.error("Failed to start task manager", error=str(e))
            raise

    async def stop(self) -> None:
        """Stop the task manager gracefully."""
        self.logger.info("Stopping task manager")

        # Signal shutdown
        self._shutdown_event.set()
        self.is_running = False

        # Cancel running tasks
        await self._cancel_running_tasks()

        # Cancel background tasks
        await self._cancel_background_tasks()

        self.logger.info("Task manager stopped")

    async def submit_task(
        self,
        task_type: TaskType,
        payload: Dict[str, Any],
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout: Optional[float] = None,
        dependencies: Optional[List[str]] = None,
        required_capabilities: Optional[List[str]] = None,
        target_agent_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Submit a new task for execution.

        Args:
            task_type: Type of task to execute
            payload: Task payload data
            priority: Task priority level
            timeout: Task timeout in seconds
            dependencies: List of task IDs this task depends on
            required_capabilities: Required agent capabilities
            target_agent_type: Preferred agent type for execution
            metadata: Additional task metadata

        Returns:
            Task ID for tracking
        """
        if not self.is_running:
            raise RuntimeError("Task manager is not running")

        # Check queue capacity
        if self.scheduler.get_total_queued() >= self.max_queue_size:
            raise RuntimeError("Task queue is at capacity")

        # Create task
        task = Task(
            type=task_type,
            priority=priority,
            payload=payload,
            timeout=timeout or self.default_timeout,
            required_capabilities=required_capabilities or [],
            target_agent_type=target_agent_type,
            metadata=metadata or {},
        )

        # Add dependencies
        if dependencies:
            for dep_id in dependencies:
                if dep_id in self.tasks:
                    task.dependencies.append(TaskDependency(task_id=dep_id))
                    self.dependency_manager.add_dependency(task.id, dep_id)

        # Store task
        self.tasks[task.id] = task

        # Queue task if ready, otherwise it will be queued when dependencies complete
        if self.dependency_manager.is_ready(task.id):
            self.scheduler.enqueue(task)

        # Update metrics
        if self.enable_metrics:
            self.metrics.total_tasks_submitted += 1
            self.metrics.tasks_by_priority[priority] += 1
            self.metrics.tasks_by_type[task_type] += 1

        self.logger.info(
            "Task submitted",
            task_id=task.id,
            task_type=task_type.value,
            priority=priority.value,
            dependencies=len(dependencies) if dependencies else 0,
        )

        return task.id

    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a task."""
        if task_id not in self.tasks:
            return None

        task = self.tasks[task_id]

        return {
            'id': task.id,
            'type': task.type.value,
            'state': task.state.value,
            'priority': task.priority.value,
            'created_at': task.created_at.isoformat(),
            'started_at': task.started_at.isoformat() if task.started_at else None,
            'completed_at': task.completed_at.isoformat() if task.completed_at else None,
            'attempts': task.attempts,
            'assigned_agent': task.assigned_agent,
            'result': task.result.__dict__ if task.result else None,
            'metadata': task.metadata,
        }

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a task."""
        if task_id not in self.tasks:
            return False

        task = self.tasks[task_id]

        if task.state in [TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED]:
            return False

        # Cancel running task
        if task_id in self.running_tasks:
            self.running_tasks[task_id].cancel()
            del self.running_tasks[task_id]

        # Remove from queues
        for queue in self.scheduler.priority_queues.values():
            if task in queue:
                queue.remove(task)
                break

        # Update state
        task.state = TaskState.CANCELLED
        task.completed_at = datetime.utcnow()

        # Release resources
        self.resource_manager.release_resources(task_id)

        # Update metrics
        if self.enable_metrics:
            self.metrics.total_tasks_cancelled += 1

        self.logger.info("Task cancelled", task_id=task_id)
        return True

    def register_task_handler(self, task_type: TaskType, handler: Callable) -> None:
        """Register a handler for a specific task type."""
        self.task_handlers[task_type] = handler
        self.logger.info("Task handler registered", task_type=task_type.value)

    def calculate_priority(self, context: Dict[str, Any]) -> TaskPriority:
        """
        Calculate task priority based on context.

        This is a sophisticated priority calculation that considers:
        - User urgency indicators
        - System load
        - Task dependencies
        - Resource availability
        """
        # Base priority
        priority = TaskPriority.NORMAL

        # Check urgency indicators
        if context.get('urgent', False):
            priority = TaskPriority.CRITICAL
        elif context.get('high_priority', False):
            priority = TaskPriority.HIGH
        elif context.get('background', False):
            priority = TaskPriority.BACKGROUND

        # Adjust based on system load
        current_load = len(self.running_tasks) / self.max_concurrent_tasks
        if current_load > 0.8 and priority == TaskPriority.NORMAL:
            priority = TaskPriority.LOW

        # Consider task type importance
        task_type = context.get('task_type')
        if task_type == TaskType.CODE_GENERATION:
            # Code generation is usually high priority
            if priority == TaskPriority.NORMAL:
                priority = TaskPriority.HIGH
        elif task_type == TaskType.MAINTENANCE:
            # Maintenance is usually lower priority
            if priority == TaskPriority.NORMAL:
                priority = TaskPriority.LOW

        return priority

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive task manager metrics."""
        uptime = 0.0
        if self.started_at:
            uptime = (datetime.utcnow() - self.started_at).total_seconds()

        # Calculate throughput
        if uptime > 0:
            self.metrics.throughput_per_minute = (
                self.metrics.total_tasks_completed / (uptime / 60)
            )

        # Calculate error rate
        total_processed = (
            self.metrics.total_tasks_completed +
            self.metrics.total_tasks_failed
        )
        if total_processed > 0:
            self.metrics.error_rate = self.metrics.total_tasks_failed / total_processed

        # Update queue metrics
        queue_sizes = self.scheduler.get_queue_sizes()
        self.metrics.current_queue_size = sum(queue_sizes.values())
        if self.metrics.current_queue_size > self.metrics.peak_queue_size:
            self.metrics.peak_queue_size = self.metrics.current_queue_size

        return {
            'uptime': uptime,
            'is_running': self.is_running,
            'queue_sizes': {p.name: size for p, size in queue_sizes.items()},
            'running_tasks': len(self.running_tasks),
            'metrics': {
                'total_submitted': self.metrics.total_tasks_submitted,
                'total_completed': self.metrics.total_tasks_completed,
                'total_failed': self.metrics.total_tasks_failed,
                'total_cancelled': self.metrics.total_tasks_cancelled,
                'total_timeout': self.metrics.total_tasks_timeout,
                'average_execution_time': self.metrics.average_execution_time,
                'average_queue_time': self.metrics.average_queue_time,
                'current_queue_size': self.metrics.current_queue_size,
                'peak_queue_size': self.metrics.peak_queue_size,
                'throughput_per_minute': self.metrics.throughput_per_minute,
                'error_rate': self.metrics.error_rate,
            },
            'resource_usage': self.resource_manager.get_resource_usage(),
        }

    def _initialize_resources(self) -> None:
        """Initialize default resource limits."""
        self.resource_manager.set_resource_limit('cpu', 100.0)
        self.resource_manager.set_resource_limit('memory', 1000.0)
        self.resource_manager.set_resource_limit('io', 50.0)
        self.resource_manager.set_resource_limit('network', 20.0)

    def _start_background_tasks(self) -> None:
        """Start background processing tasks."""
        # Task execution loop
        execution_task = asyncio.create_task(self._task_execution_loop())
        self._background_tasks.add(execution_task)
        execution_task.add_done_callback(self._background_tasks.discard)

        # Metrics collection
        if self.enable_metrics:
            metrics_task = asyncio.create_task(self._metrics_collection_loop())
            self._background_tasks.add(metrics_task)
            metrics_task.add_done_callback(self._background_tasks.discard)

        # Cleanup task
        cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._background_tasks.add(cleanup_task)
        cleanup_task.add_done_callback(self._background_tasks.discard)

    async def _task_execution_loop(self) -> None:
        """Main task execution loop."""
        while not self._shutdown_event.is_set():
            try:
                # Get next task
                task = self.scheduler.dequeue()

                if task is None:
                    await asyncio.sleep(0.1)
                    continue

                # Check resource availability
                if not self.resource_manager.can_allocate(task.id, task.required_resources):
                    # Put task back and wait
                    self.scheduler.enqueue(task)
                    await asyncio.sleep(1.0)
                    continue

                # Execute task
                execution_task = asyncio.create_task(self._execute_task(task))
                self.running_tasks[task.id] = execution_task
                execution_task.add_done_callback(
                    lambda t, task_id=task.id: self.running_tasks.pop(task_id, None)
                )

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Error in task execution loop", error=str(e))

    async def _execute_task(self, task: Task) -> None:
        """Execute a single task."""
        async with self.execution_semaphore:
            try:
                # Allocate resources
                self.resource_manager.allocate_resources(task.id, task.required_resources)

                # Update task state
                task.state = TaskState.RUNNING
                task.started_at = datetime.utcnow()
                task.attempts += 1

                self.logger.info("Starting task execution", task_id=task.id, task_type=task.type.value)

                # Get appropriate agent or handler
                result = None
                if self.agent_pool and task.target_agent_type:
                    agent = await self.agent_pool.get_agent(
                        task.target_agent_type,
                        required_capabilities=task.required_capabilities,
                        timeout=task.timeout,
                    )
                    if agent:
                        task.assigned_agent = agent.agent_id
                        # Execute via agent (simplified - would need proper message handling)
                        result = await self._execute_via_agent(agent, task)

                # Fallback to direct handler
                if result is None and task.type in self.task_handlers:
                    handler = self.task_handlers[task.type]
                    result = await asyncio.wait_for(
                        handler(task.payload, task.context),
                        timeout=task.timeout
                    )

                # Create task result
                execution_time = (datetime.utcnow() - task.started_at).total_seconds()
                task.result = TaskResult(
                    task_id=task.id,
                    success=True,
                    result=result,
                    execution_time=execution_time,
                )

                # Update task state
                task.state = TaskState.COMPLETED
                task.completed_at = datetime.utcnow()

                # Mark dependencies as satisfied
                unblocked_tasks = self.dependency_manager.mark_completed(task.id)
                for unblocked_id in unblocked_tasks:
                    if unblocked_id in self.tasks:
                        unblocked_task = self.tasks[unblocked_id]
                        self.scheduler.enqueue(unblocked_task)

                # Update metrics
                if self.enable_metrics:
                    self.metrics.total_tasks_completed += 1
                    self.metrics.average_execution_time = (
                        self.metrics.average_execution_time + execution_time
                    ) / 2

                self.completed_tasks.append(task)

                self.logger.info(
                    "Task completed successfully",
                    task_id=task.id,
                    execution_time=execution_time,
                )

            except asyncio.TimeoutError:
                await self._handle_task_timeout(task)
            except Exception as e:
                await self._handle_task_failure(task, str(e))
            finally:
                # Release resources
                self.resource_manager.release_resources(task.id)

    async def _execute_via_agent(self, agent, task: Task) -> Any:
        """Execute task via an agent (placeholder for agent integration)."""
        # This would be implemented based on the agent's interface
        # For now, return a placeholder result
        return {"status": "executed_via_agent", "agent_id": agent.agent_id}

    async def _handle_task_timeout(self, task: Task) -> None:
        """Handle task timeout."""
        task.state = TaskState.TIMEOUT
        task.completed_at = datetime.utcnow()

        task.result = TaskResult(
            task_id=task.id,
            success=False,
            error="Task execution timeout",
        )

        if self.enable_metrics:
            self.metrics.total_tasks_timeout += 1

        self.failed_tasks.append(task)

        self.logger.warning("Task timed out", task_id=task.id, timeout=task.timeout)

    async def _handle_task_failure(self, task: Task, error: str) -> None:
        """Handle task failure and potentially retry."""
        task.result = TaskResult(
            task_id=task.id,
            success=False,
            error=error,
        )

        # Check if we should retry
        if task.attempts < task.max_retries:
            # Schedule retry
            task.state = TaskState.RETRYING
            retry_delay = task.retry_delay * (task.retry_backoff ** (task.attempts - 1))

            self.logger.info(
                "Scheduling task retry",
                task_id=task.id,
                attempt=task.attempts,
                retry_delay=retry_delay,
            )

            # Schedule retry after delay
            asyncio.create_task(self._schedule_retry(task, retry_delay))
        else:
            # Mark as failed
            task.state = TaskState.FAILED
            task.completed_at = datetime.utcnow()

            if self.enable_metrics:
                self.metrics.total_tasks_failed += 1

            self.failed_tasks.append(task)

            self.logger.error(
                "Task failed permanently",
                task_id=task.id,
                error=error,
                attempts=task.attempts,
            )

    async def _schedule_retry(self, task: Task, delay: float) -> None:
        """Schedule a task retry after a delay."""
        await asyncio.sleep(delay)
        if not self._shutdown_event.is_set():
            self.scheduler.enqueue(task)

    async def _metrics_collection_loop(self) -> None:
        """Background metrics collection and analysis."""
        while not self._shutdown_event.is_set():
            try:
                # Collect current metrics snapshot
                current_metrics = self.get_metrics()
                self.metrics_history.append({
                    'timestamp': datetime.utcnow(),
                    'metrics': current_metrics,
                })

                # Calculate queue wait times
                total_queue_time = 0
                queued_count = 0

                for task in self.tasks.values():
                    if task.state == TaskState.QUEUED and task.scheduled_at:
                        queue_time = (datetime.utcnow() - task.scheduled_at).total_seconds()
                        total_queue_time += queue_time
                        queued_count += 1

                if queued_count > 0:
                    self.metrics.average_queue_time = total_queue_time / queued_count

                await asyncio.sleep(60)  # Collect metrics every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Error in metrics collection loop", error=str(e))

    async def _cleanup_loop(self) -> None:
        """Background cleanup of completed tasks and maintenance."""
        while not self._shutdown_event.is_set():
            try:
                current_time = datetime.utcnow()

                # Clean up old completed tasks
                expired_tasks = []
                for task_id, task in self.tasks.items():
                    if (task.state in [TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED] and
                        task.completed_at and
                        (current_time - task.completed_at).total_seconds() > 3600):  # 1 hour
                        expired_tasks.append(task_id)

                for task_id in expired_tasks:
                    del self.tasks[task_id]

                # Log cleanup statistics
                if expired_tasks:
                    self.logger.info("Cleaned up expired tasks", count=len(expired_tasks))

                await asyncio.sleep(300)  # Cleanup every 5 minutes

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Error in cleanup loop", error=str(e))

    async def _cancel_running_tasks(self) -> None:
        """Cancel all running tasks."""
        if not self.running_tasks:
            return

        self.logger.info("Cancelling running tasks", count=len(self.running_tasks))

        # Cancel all running tasks
        for task_id, asyncio_task in list(self.running_tasks.items()):
            if not asyncio_task.done():
                asyncio_task.cancel()

        # Wait for cancellation to complete
        if self.running_tasks:
            await asyncio.gather(*self.running_tasks.values(), return_exceptions=True)

        self.running_tasks.clear()

    async def _cancel_background_tasks(self) -> None:
        """Cancel all background tasks."""
        for task in self._background_tasks:
            if not task.done():
                task.cancel()

        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)

        self._background_tasks.clear()
