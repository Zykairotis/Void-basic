"""
Aider Hive Task Management Module

This module provides advanced task scheduling, routing, and management capabilities
for the Aider CLI Multi-Agent Hive Architecture.

Core Components:
- HiveTaskManager: Intelligent task scheduling and execution management
- TaskRouter: Smart task classification and routing to appropriate agents
- TaskScheduler: Priority-based task queue management
- DependencyManager: Task dependency resolution and ordering
- ResourceManager: Resource allocation and constraint management

Key Features:
- Priority-based task scheduling with multiple strategies
- Intelligent task classification using pattern matching and ML
- Dynamic agent routing based on capabilities and performance
- Task dependency management and execution ordering
- Resource allocation and constraint enforcement
- Comprehensive performance monitoring and metrics
- Retry logic and error handling with exponential backoff
- Load balancing and queue optimization

Usage Example:
    ```python
    from aider.task_management import HiveTaskManager, TaskRouter, TaskType, TaskPriority

    # Initialize task management system
    task_manager = HiveTaskManager(
        max_concurrent_tasks=50,
        scheduling_strategy="priority_fifo"
    )

    # Initialize task router
    task_router = TaskRouter(
        default_strategy=RoutingStrategy.HYBRID,
        enable_learning=True
    )

    # Start the systems
    await task_manager.start()

    # Submit a task
    task_id = await task_manager.submit_task(
        task_type=TaskType.CODE_GENERATION,
        payload={"code": "def hello_world(): return 'Hello, World!'"},
        priority=TaskPriority.HIGH,
        required_capabilities=["code_generation", "syntax_validation"]
    )

    # Route the task
    task = task_manager.tasks[task_id]
    routing_decision = task_router.route_task(task)
    ```

Performance Considerations:
- Supports thousands of concurrent tasks with efficient queue management
- Optimized for low-latency task routing and scheduling
- Memory-efficient task history and metrics collection
- Scalable architecture with configurable resource limits

Security Features:
- Resource constraint enforcement to prevent resource exhaustion
- Task timeout protection against infinite loops
- Secure task payload handling and validation
- Audit trail for all task operations
"""

__version__ = "1.0.0"
__author__ = "Aider Development Team"

# Core task management classes
from .task_queue import (
    # Main task manager
    HiveTaskManager,

    # Task definitions and enums
    Task,
    TaskState,
    TaskPriority,
    TaskType,
    TaskResult,
    TaskDependency,
    TaskResource,
    TaskMetrics,

    # Queue management components
    TaskScheduler,
    DependencyManager,
    ResourceManager,
)

# Task routing and classification
from .task_router import (
    # Main router
    TaskRouter,
    TaskClassifier,

    # Routing enums and strategies
    RoutingStrategy,
    ClassificationConfidence,

    # Routing data structures
    RoutingRule,
    ClassificationResult,
    RoutingDecision,
    AgentPerformanceMetrics,
)

# Re-export all public classes and enums
__all__ = [
    # Core task management
    "HiveTaskManager",
    "Task",
    "TaskState",
    "TaskPriority",
    "TaskType",
    "TaskResult",
    "TaskDependency",
    "TaskResource",
    "TaskMetrics",

    # Queue management
    "TaskScheduler",
    "DependencyManager",
    "ResourceManager",

    # Task routing
    "TaskRouter",
    "TaskClassifier",
    "RoutingStrategy",
    "ClassificationConfidence",
    "RoutingRule",
    "ClassificationResult",
    "RoutingDecision",
    "AgentPerformanceMetrics",
]

# Module metadata
FRAMEWORK_NAME = "Aider Hive Task Management Framework"
FRAMEWORK_VERSION = __version__

def get_framework_info():
    """Get information about the task management framework."""
    return {
        "name": FRAMEWORK_NAME,
        "version": FRAMEWORK_VERSION,
        "components": {
            "task_manager": "Advanced task scheduling and execution management",
            "task_router": "Intelligent task classification and routing system",
            "task_scheduler": "Priority-based task queue with multiple strategies",
            "dependency_manager": "Task dependency resolution and ordering",
            "resource_manager": "Resource allocation and constraint management",
            "task_classifier": "ML-enhanced task analysis and classification",
        },
        "features": [
            "Priority-based task scheduling",
            "Intelligent agent routing and load balancing",
            "Task dependency management",
            "Resource allocation and constraints",
            "Performance monitoring and metrics",
            "Retry logic with exponential backoff",
            "Machine learning-enhanced decisions",
            "Comprehensive audit trail",
        ],
        "supported_strategies": {
            "scheduling": ["priority_fifo", "priority_srtf", "weighted_round_robin"],
            "routing": ["capability_match", "load_balanced", "performance_optimized", "rule_based", "hybrid"],
        }
    }

def create_default_task_system(
    max_concurrent_tasks: int = 100,
    max_queue_size: int = 10000,
    routing_strategy: RoutingStrategy = RoutingStrategy.HYBRID,
    enable_metrics: bool = True
):
    """
    Create a default task management system with recommended settings.

    Args:
        max_concurrent_tasks: Maximum concurrent task executions
        max_queue_size: Maximum tasks in queue
        routing_strategy: Default routing strategy
        enable_metrics: Enable performance metrics collection

    Returns:
        Tuple of (HiveTaskManager, TaskRouter) ready for use
    """
    # Create task manager
    task_manager = HiveTaskManager(
        max_concurrent_tasks=max_concurrent_tasks,
        max_queue_size=max_queue_size,
        default_timeout=300.0,
        scheduling_strategy="priority_fifo",
        enable_metrics=enable_metrics,
    )

    # Create task router
    task_router = TaskRouter(
        default_strategy=routing_strategy,
        enable_learning=True,
        performance_weight=0.3,
        load_weight=0.4,
        capability_weight=0.3,
    )

    return task_manager, task_router

# Convenience functions for common task types
def create_code_generation_task(
    code_specification: str,
    priority: TaskPriority = TaskPriority.NORMAL,
    timeout: float = 120.0,
    metadata: dict = None
) -> dict:
    """Create a code generation task payload."""
    return {
        "task_type": TaskType.CODE_GENERATION,
        "payload": {
            "specification": code_specification,
            "type": "generation"
        },
        "priority": priority,
        "timeout": timeout,
        "required_capabilities": ["code_generation", "syntax_validation"],
        "metadata": metadata or {}
    }

def create_code_analysis_task(
    code_content: str,
    analysis_type: str = "general",
    priority: TaskPriority = TaskPriority.NORMAL,
    timeout: float = 60.0,
    metadata: dict = None
) -> dict:
    """Create a code analysis task payload."""
    return {
        "task_type": TaskType.CODE_ANALYSIS,
        "payload": {
            "code": code_content,
            "analysis_type": analysis_type
        },
        "priority": priority,
        "timeout": timeout,
        "required_capabilities": ["code_analysis", "pattern_recognition"],
        "metadata": metadata or {}
    }

def create_git_operation_task(
    operation: str,
    parameters: dict,
    priority: TaskPriority = TaskPriority.NORMAL,
    timeout: float = 30.0,
    metadata: dict = None
) -> dict:
    """Create a git operation task payload."""
    return {
        "task_type": TaskType.GIT_OPERATION,
        "payload": {
            "operation": operation,
            "parameters": parameters
        },
        "priority": priority,
        "timeout": timeout,
        "required_capabilities": ["git_operations", "version_control"],
        "metadata": metadata or {}
    }

def create_context_building_task(
    scope: str,
    parameters: dict = None,
    priority: TaskPriority = TaskPriority.NORMAL,
    timeout: float = 90.0,
    metadata: dict = None
) -> dict:
    """Create a context building task payload."""
    return {
        "task_type": TaskType.CONTEXT_BUILDING,
        "payload": {
            "scope": scope,
            "parameters": parameters or {}
        },
        "priority": priority,
        "timeout": timeout,
        "required_capabilities": ["context_analysis", "semantic_understanding"],
        "metadata": metadata or {}
    }

# Task priority helpers
def get_priority_from_context(context: dict) -> TaskPriority:
    """Determine task priority based on context clues."""
    if context.get("urgent", False) or context.get("critical", False):
        return TaskPriority.CRITICAL
    elif context.get("high_priority", False) or context.get("important", False):
        return TaskPriority.HIGH
    elif context.get("background", False) or context.get("low_priority", False):
        return TaskPriority.LOW
    elif context.get("batch", False) or context.get("maintenance", False):
        return TaskPriority.BACKGROUND
    else:
        return TaskPriority.NORMAL

# Resource estimation helpers
def estimate_task_resources(task_type: TaskType, payload_size: int) -> list:
    """Estimate resource requirements for a task."""
    base_resources = {
        TaskType.CODE_GENERATION: [
            TaskResource("cpu", 2.0),
            TaskResource("memory", 100.0)
        ],
        TaskType.CODE_ANALYSIS: [
            TaskResource("cpu", 1.5),
            TaskResource("memory", 50.0)
        ],
        TaskType.CONTEXT_BUILDING: [
            TaskResource("cpu", 1.0),
            TaskResource("memory", 200.0),
            TaskResource("io", 5.0)
        ],
        TaskType.GIT_OPERATION: [
            TaskResource("cpu", 0.5),
            TaskResource("io", 10.0)
        ],
        TaskType.FILE_OPERATION: [
            TaskResource("io", 5.0),
            TaskResource("memory", 20.0)
        ],
        TaskType.SEARCH_OPERATION: [
            TaskResource("cpu", 1.0),
            TaskResource("io", 3.0)
        ],
    }

    resources = base_resources.get(task_type, [TaskResource("cpu", 1.0)])

    # Scale resources based on payload size
    scale_factor = max(1.0, payload_size / 1000.0)
    for resource in resources:
        resource.amount *= scale_factor

    return resources
