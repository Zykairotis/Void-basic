"""
BaseAgent: Abstract base class for all agents in the Aider Hive Architecture.

This module provides the foundational framework for implementing specialized agents
with modern asyncio patterns, message bus communication, and lifecycle management.
"""

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, Set, Callable, List, Tuple, TYPE_CHECKING
from weakref import WeakSet

import structlog

if TYPE_CHECKING:
    from .message_bus import MessageBus


class AgentState(Enum):
    """Agent lifecycle states."""
    INITIALIZING = "initializing"
    READY = "ready"
    BUSY = "busy"
    IDLE = "idle"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"
    SHUTDOWN = "shutdown"


class MessagePriority(Enum):
    """Message priority levels for task routing."""
    URGENT = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


@dataclass
class AgentMessage:
    """Standard message format for inter-agent communication."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str = ""
    recipient_id: str = ""
    message_type: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: datetime = field(default_factory=datetime.utcnow)
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    expires_at: Optional[datetime] = None


@dataclass(frozen=True)
class AgentCapability:
    """Defines what an agent can do."""
    name: str
    description: str
    input_types: Tuple[str, ...]
    output_types: Tuple[str, ...]
    cost_estimate: float = 1.0  # Relative cost for resource management

    # Capability Constants
    DEPLOYMENT_AUTOMATION = None  # Will be initialized after class definition
    CI_CD_PIPELINE = None
    INFRASTRUCTURE_MANAGEMENT = None
    MONITORING = None
    ROLLBACK_MANAGEMENT = None
    AI_INTEGRATION = None
    TEST_GENERATION = None
    TEST_EXECUTION = None
    QUALITY_ANALYSIS = None
    PERFORMANCE_TESTING = None
    SECURITY_TESTING = None
    WORKFLOW_ORCHESTRATION = None
    MULTI_AGENT_COORDINATION = None
    QUALITY_ASSURANCE = None
    ERROR_RECOVERY = None
    CODE_GENERATION = None
    CODE_ANALYSIS = None
    CONTEXT_UNDERSTANDING = None
    PROJECT_ANALYSIS = None
    GIT_OPERATIONS = None
    VERSION_CONTROL = None


# Initialize AgentCapability constants after class definition
AgentCapability.DEPLOYMENT_AUTOMATION = AgentCapability(
    name="deployment_automation",
    description="Automate deployment processes and pipelines",
    input_types=("deployment_config", "environment_spec"),
    output_types=("deployment_result", "status_report")
)

AgentCapability.CI_CD_PIPELINE = AgentCapability(
    name="ci_cd_pipeline",
    description="Manage continuous integration and deployment pipelines",
    input_types=("pipeline_config", "code_changes"),
    output_types=("pipeline_result", "build_artifacts")
)

AgentCapability.INFRASTRUCTURE_MANAGEMENT = AgentCapability(
    name="infrastructure_management",
    description="Manage cloud infrastructure and resources",
    input_types=("infrastructure_spec", "scaling_requirements"),
    output_types=("infrastructure_status", "resource_allocation")
)

AgentCapability.MONITORING = AgentCapability(
    name="monitoring",
    description="Monitor system health and performance metrics",
    input_types=("monitoring_config", "alert_rules"),
    output_types=("health_report", "performance_metrics")
)

AgentCapability.ROLLBACK_MANAGEMENT = AgentCapability(
    name="rollback_management",
    description="Manage deployment rollbacks and recovery",
    input_types=("rollback_config", "failure_detection"),
    output_types=("rollback_result", "recovery_status")
)

AgentCapability.AI_INTEGRATION = AgentCapability(
    name="ai_integration",
    description="Integrate AI capabilities into workflows",
    input_types=("ai_config", "model_requirements"),
    output_types=("ai_response", "model_output")
)

AgentCapability.TEST_GENERATION = AgentCapability(
    name="test_generation",
    description="Generate automated tests for code and systems",
    input_types=("code_analysis", "test_requirements"),
    output_types=("test_suite", "test_coverage")
)

AgentCapability.TEST_EXECUTION = AgentCapability(
    name="test_execution",
    description="Execute automated test suites",
    input_types=("test_suite", "execution_config"),
    output_types=("test_results", "coverage_report")
)

AgentCapability.QUALITY_ANALYSIS = AgentCapability(
    name="quality_analysis",
    description="Analyze code and system quality metrics",
    input_types=("code_base", "quality_standards"),
    output_types=("quality_report", "improvement_suggestions")
)

AgentCapability.PERFORMANCE_TESTING = AgentCapability(
    name="performance_testing",
    description="Execute performance and load testing",
    input_types=("performance_config", "load_scenarios"),
    output_types=("performance_report", "bottleneck_analysis")
)

AgentCapability.SECURITY_TESTING = AgentCapability(
    name="security_testing",
    description="Execute security vulnerability testing",
    input_types=("security_config", "vulnerability_scan"),
    output_types=("security_report", "vulnerability_list")
)

AgentCapability.WORKFLOW_ORCHESTRATION = AgentCapability(
    name="workflow_orchestration",
    description="Orchestrate complex multi-agent workflows",
    input_types=("workflow_definition", "agent_coordination"),
    output_types=("workflow_result", "execution_status")
)

AgentCapability.MULTI_AGENT_COORDINATION = AgentCapability(
    name="multi_agent_coordination",
    description="Coordinate multiple agents in parallel workflows",
    input_types=("coordination_plan", "agent_states"),
    output_types=("coordination_result", "agent_status")
)

AgentCapability.QUALITY_ASSURANCE = AgentCapability(
    name="quality_assurance",
    description="Ensure quality across development workflows",
    input_types=("quality_standards", "workflow_output"),
    output_types=("quality_validation", "compliance_report")
)

AgentCapability.ERROR_RECOVERY = AgentCapability(
    name="error_recovery",
    description="Handle errors and implement recovery strategies",
    input_types=("error_context", "recovery_options"),
    output_types=("recovery_result", "error_resolution")
)

AgentCapability.CODE_GENERATION = AgentCapability(
    name="code_generation",
    description="Generate code using AI assistance",
    input_types=("code_requirements", "context_analysis"),
    output_types=("generated_code", "implementation_notes")
)

AgentCapability.CODE_ANALYSIS = AgentCapability(
    name="code_analysis",
    description="Analyze code structure and patterns",
    input_types=("source_code", "analysis_criteria"),
    output_types=("code_insights", "refactor_suggestions")
)

AgentCapability.CONTEXT_UNDERSTANDING = AgentCapability(
    name="context_understanding",
    description="Understand project context and requirements",
    input_types=("project_files", "documentation"),
    output_types=("context_model", "requirement_analysis")
)

AgentCapability.PROJECT_ANALYSIS = AgentCapability(
    name="project_analysis",
    description="Analyze project structure and dependencies",
    input_types=("project_structure", "dependency_graph"),
    output_types=("project_insights", "architecture_analysis")
)

AgentCapability.GIT_OPERATIONS = AgentCapability(
    name="git_operations",
    description="Execute git version control operations",
    input_types=("git_commands", "repository_state"),
    output_types=("git_result", "repository_status")
)

AgentCapability.VERSION_CONTROL = AgentCapability(
    name="version_control",
    description="Manage version control workflows",
    input_types=("version_strategy", "change_management"),
    output_types=("version_result", "change_tracking")
)


@dataclass
class AgentMetrics:
    """Agent performance and health metrics."""
    tasks_completed: int = 0
    tasks_failed: int = 0
    average_response_time: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    last_activity: Optional[datetime] = None
    error_count: int = 0
    uptime: float = 0.0


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the Aider Hive Architecture.

    Provides common functionality for:
    - Asyncio-based lifecycle management
    - Message bus communication
    - Health monitoring and metrics
    - Error handling and recovery
    - Context management
    - Resource management
    """

    def __init__(
        self,
        agent_id: str,
        agent_type: str,
        config: Optional[Dict[str, Any]] = None,
        message_bus: Optional['MessageBus'] = None,
    ):
        """
        Initialize the base agent.

        Args:
            agent_id: Unique identifier for this agent instance
            agent_type: Type/class of the agent (e.g., 'code', 'context', 'git')
            config: Agent-specific configuration
            message_bus: Reference to the message bus for communication
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.config = config or {}
        self.message_bus = message_bus

        # State management
        self.state = AgentState.INITIALIZING
        self.created_at = datetime.utcnow()
        self.started_at: Optional[datetime] = None

        # Logging
        self.logger = structlog.get_logger().bind(
            agent_id=agent_id,
            agent_type=agent_type
        )

        # Metrics and monitoring
        self.metrics = AgentMetrics()
        self.health_check_interval = self.config.get('health_check_interval', 30.0)

        # Communication
        self.message_handlers: Dict[str, Callable] = {}
        self.message_queue = asyncio.Queue()
        self.subscription_topics: Set[str] = set()

        # Task management
        self.active_tasks: WeakSet = WeakSet()
        self.max_concurrent_tasks = self.config.get('max_concurrent_tasks', 10)
        self.task_semaphore = asyncio.Semaphore(self.max_concurrent_tasks)

        # Shutdown management
        self._shutdown_event = asyncio.Event()
        self._background_tasks: Set[asyncio.Task] = set()

        # Register default message handlers
        self._register_default_handlers()

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the agent. Must be implemented by subclasses.

        This method should:
        - Set up any required resources
        - Connect to external services
        - Validate configuration
        - Register capabilities
        """
        pass

    @abstractmethod
    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """
        Process an incoming message. Must be implemented by subclasses.

        Args:
            message: The message to process

        Returns:
            Optional response message
        """
        pass

    @abstractmethod
    def get_capabilities(self) -> List[AgentCapability]:
        """
        Return list of capabilities this agent provides.

        Returns:
            List of agent capabilities
        """
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check and return status.

        Returns:
            Dictionary containing health status information
        """
        pass

    async def start(self) -> None:
        """Start the agent and begin processing messages."""
        try:
            self.logger.info("Starting agent")
            self.state = AgentState.INITIALIZING

            # Initialize the agent
            await self.initialize()

            # Start background tasks
            self._start_background_tasks()

            # Connect to message bus if available
            if self.message_bus:
                await self.message_bus.subscribe(self.agent_id, self._handle_message)

            self.state = AgentState.READY
            self.started_at = datetime.utcnow()

            self.logger.info("Agent started successfully")

        except Exception as e:
            self.state = AgentState.ERROR
            self.logger.error("Failed to start agent", error=str(e))
            raise

    async def stop(self) -> None:
        """Stop the agent gracefully."""
        self.logger.info("Stopping agent")
        self.state = AgentState.SHUTTING_DOWN

        # Signal shutdown to background tasks
        self._shutdown_event.set()

        # Wait for active tasks to complete (with timeout)
        try:
            await asyncio.wait_for(
                self._wait_for_tasks_completion(),
                timeout=self.config.get('shutdown_timeout', 30.0)
            )
        except asyncio.TimeoutError:
            self.logger.warning("Shutdown timeout reached, cancelling remaining tasks")
            await self._cancel_remaining_tasks()

        # Unsubscribe from message bus
        if self.message_bus:
            await self.message_bus.unsubscribe(self.agent_id)

        # Cancel background tasks
        await self._cancel_background_tasks()

        self.state = AgentState.SHUTDOWN
        self.logger.info("Agent stopped")

    async def send_message(
        self,
        recipient_id: str,
        message_type: str,
        payload: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
        correlation_id: Optional[str] = None,
    ) -> str:
        """
        Send a message to another agent.

        Args:
            recipient_id: ID of the recipient agent
            message_type: Type of message being sent
            payload: Message payload data
            priority: Message priority level
            correlation_id: Optional correlation ID for request/response tracking

        Returns:
            Message ID for tracking
        """
        message = AgentMessage(
            sender_id=self.agent_id,
            recipient_id=recipient_id,
            message_type=message_type,
            payload=payload,
            priority=priority,
            correlation_id=correlation_id,
        )

        if self.message_bus:
            await self.message_bus.publish(message)
        else:
            self.logger.warning("No message bus available, cannot send message")

        return message.id

    async def request_response(
        self,
        recipient_id: str,
        message_type: str,
        payload: Dict[str, Any],
        timeout: float = 30.0,
    ) -> Optional[AgentMessage]:
        """
        Send a message and wait for a response.

        Args:
            recipient_id: ID of the recipient agent
            message_type: Type of message being sent
            payload: Message payload data
            timeout: Maximum time to wait for response

        Returns:
            Response message or None if timeout
        """
        correlation_id = str(uuid.uuid4())

        # Set up response handler
        response_future = asyncio.Future()
        self.message_handlers[f"response_{correlation_id}"] = lambda msg: response_future.set_result(msg)

        try:
            # Send the request
            await self.send_message(
                recipient_id=recipient_id,
                message_type=message_type,
                payload=payload,
                correlation_id=correlation_id,
            )

            # Wait for response
            response = await asyncio.wait_for(response_future, timeout=timeout)
            return response

        except asyncio.TimeoutError:
            self.logger.warning(
                "Request timeout",
                recipient_id=recipient_id,
                message_type=message_type,
                timeout=timeout
            )
            return None

        finally:
            # Clean up handler
            self.message_handlers.pop(f"response_{correlation_id}", None)

    def register_message_handler(self, message_type: str, handler: Callable) -> None:
        """
        Register a handler for a specific message type.

        Args:
            message_type: Type of message to handle
            handler: Async function to handle the message
        """
        self.message_handlers[message_type] = handler

    async def update_metrics(self, **kwargs) -> None:
        """Update agent metrics."""
        for key, value in kwargs.items():
            if hasattr(self.metrics, key):
                setattr(self.metrics, key, value)

        self.metrics.last_activity = datetime.utcnow()

    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        uptime = 0.0
        if self.started_at:
            uptime = (datetime.utcnow() - self.started_at).total_seconds()

        return {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'state': self.state.value,
            'uptime': uptime,
            'metrics': {
                'tasks_completed': self.metrics.tasks_completed,
                'tasks_failed': self.metrics.tasks_failed,
                'average_response_time': self.metrics.average_response_time,
                'error_count': self.metrics.error_count,
            },
            'active_tasks': len(self.active_tasks),
            'capabilities': [cap.name for cap in self.get_capabilities()],
        }

    async def _handle_message(self, message: AgentMessage) -> None:
        """Internal message handler."""
        try:
            self.state = AgentState.BUSY
            start_time = asyncio.get_event_loop().time()

            # Check if this is a response to a previous request
            if message.correlation_id:
                handler_key = f"response_{message.correlation_id}"
                if handler_key in self.message_handlers:
                    handler = self.message_handlers[handler_key]
                    await handler(message)
                    return

            # Check for specific message type handlers
            if message.message_type in self.message_handlers:
                handler = self.message_handlers[message.message_type]
                await handler(message)
            else:
                # Use the abstract process_message method
                response = await self.process_message(message)

                # Send response if one was generated and reply_to is specified
                if response and message.reply_to:
                    response.recipient_id = message.reply_to
                    response.correlation_id = message.correlation_id
                    if self.message_bus:
                        await self.message_bus.publish(response)

            # Update metrics
            processing_time = asyncio.get_event_loop().time() - start_time
            await self.update_metrics(
                tasks_completed=self.metrics.tasks_completed + 1,
                average_response_time=(
                    self.metrics.average_response_time + processing_time
                ) / 2,
            )

        except Exception as e:
            self.logger.error("Error processing message", error=str(e), message_id=message.id)
            await self.update_metrics(
                tasks_failed=self.metrics.tasks_failed + 1,
                error_count=self.metrics.error_count + 1,
            )
        finally:
            self.state = AgentState.IDLE

    def _register_default_handlers(self) -> None:
        """Register default message handlers."""
        self.register_message_handler('ping', self._handle_ping)
        self.register_message_handler('health_check', self._handle_health_check)
        self.register_message_handler('get_status', self._handle_get_status)
        self.register_message_handler('get_capabilities', self._handle_get_capabilities)

    async def _handle_ping(self, message: AgentMessage) -> None:
        """Handle ping messages."""
        if message.reply_to:
            response = AgentMessage(
                sender_id=self.agent_id,
                recipient_id=message.reply_to,
                message_type='pong',
                payload={'timestamp': datetime.utcnow().isoformat()},
                correlation_id=message.correlation_id,
            )
            if self.message_bus:
                await self.message_bus.publish(response)

    async def _handle_health_check(self, message: AgentMessage) -> None:
        """Handle health check messages."""
        health_status = await self.health_check()
        if message.reply_to:
            response = AgentMessage(
                sender_id=self.agent_id,
                recipient_id=message.reply_to,
                message_type='health_check_response',
                payload=health_status,
                correlation_id=message.correlation_id,
            )
            if self.message_bus:
                await self.message_bus.publish(response)

    async def _handle_get_status(self, message: AgentMessage) -> None:
        """Handle status request messages."""
        status = self.get_status()
        if message.reply_to:
            response = AgentMessage(
                sender_id=self.agent_id,
                recipient_id=message.reply_to,
                message_type='status_response',
                payload=status,
                correlation_id=message.correlation_id,
            )
            if self.message_bus:
                await self.message_bus.publish(response)

    async def _handle_get_capabilities(self, message: AgentMessage) -> None:
        """Handle capabilities request messages."""
        capabilities = [
            {
                'name': cap.name,
                'description': cap.description,
                'input_types': cap.input_types,
                'output_types': cap.output_types,
                'cost_estimate': cap.cost_estimate,
            }
            for cap in self.get_capabilities()
        ]

        if message.reply_to:
            response = AgentMessage(
                sender_id=self.agent_id,
                recipient_id=message.reply_to,
                message_type='capabilities_response',
                payload={'capabilities': capabilities},
                correlation_id=message.correlation_id,
            )
            if self.message_bus:
                await self.message_bus.publish(response)

    def _start_background_tasks(self) -> None:
        """Start background monitoring tasks."""
        # Health check task
        health_task = asyncio.create_task(self._health_check_loop())
        self._background_tasks.add(health_task)
        health_task.add_done_callback(self._background_tasks.discard)

    async def _health_check_loop(self) -> None:
        """Background health check loop."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.health_check_interval)
                await self.health_check()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Health check failed", error=str(e))

    async def _wait_for_tasks_completion(self) -> None:
        """Wait for all active tasks to complete."""
        while self.active_tasks:
            await asyncio.sleep(0.1)

    async def _cancel_remaining_tasks(self) -> None:
        """Cancel any remaining active tasks."""
        for task in list(self.active_tasks):
            if not task.done():
                task.cancel()

    async def _cancel_background_tasks(self) -> None:
        """Cancel all background tasks."""
        for task in self._background_tasks:
            if not task.done():
                task.cancel()

        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)

        self._background_tasks.clear()
