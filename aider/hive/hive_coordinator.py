"""
HiveCoordinator: Central coordination system for the Aider Multi-Agent Hive Architecture.

This module serves as the main integration layer that:
- Manages the lifecycle of all agents in the hive
- Coordinates communication and message routing
- Handles system-wide configuration and health monitoring
- Provides external APIs for CLI and other interfaces
- Manages resource allocation and scaling
- Ensures fault tolerance and recovery
"""

import asyncio
import json
import logging
import os
import signal
import sys
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import structlog

from ..agents.base_agent import BaseAgent, AgentMessage, MessagePriority
from ..agents.message_bus import MessageBus
from ..agents.agent_pool import AgentPool, AgentSelectionStrategy
from ..agents.orchestrator_agent import OrchestratorAgent
from ..agents.code_agent import CodeAgent
from ..agents.context_agent import ContextAgent
from ..agents.git_agent import GitAgent
from ..task_management.task_queue import HiveTaskManager
from ..context.context_store import GlobalContextStore, StorageBackend


class HiveState(Enum):
    """States of the hive system."""
    INITIALIZING = "initializing"
    STARTING = "starting"
    RUNNING = "running"
    DEGRADED = "degraded"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class AgentType(Enum):
    """Types of agents in the hive."""
    ORCHESTRATOR = "orchestrator"
    CODE = "code"
    CONTEXT = "context"
    GIT = "git"


@dataclass
class AgentConfig:
    """Configuration for an individual agent."""
    agent_type: str
    enabled: bool = True
    instances: int = 1
    auto_scaling: bool = False
    min_instances: int = 1
    max_instances: int = 5
    target_cpu: float = 80.0
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HiveConfig:
    """Complete hive system configuration."""
    system: Dict[str, Any] = field(default_factory=dict)
    agents: Dict[str, AgentConfig] = field(default_factory=dict)
    messaging: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    performance: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HiveMetrics:
    """System-wide metrics for the hive."""
    total_requests_processed: int = 0
    active_agents: int = 0
    total_agents: int = 0
    message_throughput: float = 0.0
    average_response_time: float = 0.0
    error_rate: float = 0.0
    uptime_seconds: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    agent_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class HealthStatus:
    """Health status of the hive system."""
    is_healthy: bool = True
    status: HiveState = HiveState.STOPPED
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    last_check: datetime = field(default_factory=datetime.utcnow)
    component_health: Dict[str, bool] = field(default_factory=dict)


class HiveCoordinator:
    """
    Central coordinator for the Aider Multi-Agent Hive Architecture.

    Responsibilities:
    - Agent lifecycle management
    - Inter-agent communication coordination
    - System health monitoring and metrics
    - Configuration management
    - Resource allocation and scaling
    - External API provision
    - Fault tolerance and recovery
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        project_root: Optional[str] = None,
        debug_mode: bool = False
    ):
        """Initialize the hive coordinator."""
        # Core system setup
        self.project_root = project_root or os.getcwd()
        self.debug_mode = debug_mode
        self.coordinator_id = str(uuid.uuid4())

        # State management
        self.state = HiveState.INITIALIZING
        self.started_at: Optional[datetime] = None
        self.shutdown_event = asyncio.Event()

        # Logging setup
        self.logger = structlog.get_logger().bind(
            coordinator_id=self.coordinator_id,
            project_root=self.project_root
        )

        # Configuration
        self.config = self._load_configuration(config_path)

        # Core components
        self.message_bus: Optional[MessageBus] = None
        self.agent_pool: Optional[AgentPool] = None
        self.task_manager: Optional[HiveTaskManager] = None
        self.context_store: Optional[GlobalContextStore] = None

        # Agent management
        self.agents: Dict[str, BaseAgent] = {}
        self.agent_configs: Dict[str, AgentConfig] = {}
        self.agent_instances: Dict[str, List[BaseAgent]] = defaultdict(list)

        # Health and monitoring
        self.health_status = HealthStatus()
        self.metrics = HiveMetrics()
        self.health_check_interval = self.config.performance.get('health_check_interval', 30.0)
        self.metrics_collection_interval = self.config.performance.get('metrics_interval', 60.0)

        # Background tasks
        self.background_tasks: Set[asyncio.Task] = set()

        # Request tracking
        self.active_requests: Dict[str, Dict[str, Any]] = {}
        self.request_history: List[Dict[str, Any]] = []
        self.max_history_size = 1000

        # Performance tracking
        self.response_times: List[float] = []
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.last_metrics_reset = datetime.utcnow()

    async def initialize(self) -> bool:
        """Initialize the hive system."""
        try:
            self.logger.info("Initializing Aider Hive System", coordinator_id=self.coordinator_id)
            self.state = HiveState.INITIALIZING

            # Initialize core components
            await self._initialize_message_bus()
            await self._initialize_context_store()
            await self._initialize_task_manager()
            await self._initialize_agent_pool()

            # Load agent configurations
            self._load_agent_configurations()

            # Initialize agents
            await self._initialize_agents()

            # Setup monitoring
            await self._setup_monitoring()

            # Register signal handlers
            self._setup_signal_handlers()

            self.logger.info("Hive system initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize hive system: {e}", exc_info=True)
            self.state = HiveState.ERROR
            return False

    async def start(self) -> bool:
        """Start the hive system."""
        try:
            self.logger.info("Starting Aider Hive System")
            self.state = HiveState.STARTING
            self.started_at = datetime.utcnow()

            # Start core components
            if self.message_bus:
                await self.message_bus.start()

            if self.task_manager:
                await self.task_manager.start()

            if self.context_store:
                await self.context_store.start()

            # Start agents
            await self._start_agents()

            # Start background tasks
            await self._start_background_tasks()

            # Perform initial health check
            await self._perform_health_check()

            self.state = HiveState.RUNNING
            self.logger.info("Hive system started successfully")

            return True

        except Exception as e:
            self.logger.error(f"Failed to start hive system: {e}", exc_info=True)
            self.state = HiveState.ERROR
            return False

    async def stop(self, timeout: float = 30.0) -> None:
        """Stop the hive system gracefully."""
        try:
            self.logger.info("Stopping Aider Hive System")
            self.state = HiveState.STOPPING

            # Signal shutdown
            self.shutdown_event.set()

            # Stop background tasks
            await self._stop_background_tasks(timeout)

            # Stop agents
            await self._stop_agents(timeout)

            # Stop core components
            if self.task_manager:
                await self.task_manager.stop()

            if self.context_store:
                await self.context_store.stop()

            if self.message_bus:
                await self.message_bus.stop()

            self.state = HiveState.STOPPED
            self.logger.info("Hive system stopped successfully")

        except Exception as e:
            self.logger.error(f"Error during hive system shutdown: {e}", exc_info=True)
            self.state = HiveState.ERROR

    async def process_request(
        self,
        request: str,
        context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process a user request through the hive system."""
        if not request_id:
            request_id = str(uuid.uuid4())

        start_time = time.time()

        self.logger.info(
            "Processing user request",
            request_id=request_id,
            user_id=user_id,
            request_preview=request[:100] + "..." if len(request) > 100 else request
        )

        try:
            # Validate system state
            if self.state != HiveState.RUNNING:
                return {
                    'request_id': request_id,
                    'success': False,
                    'error': f'Hive system not running (state: {self.state.value})',
                    'processing_time': time.time() - start_time
                }

            # Track active request
            self.active_requests[request_id] = {
                'request': request,
                'context': context,
                'user_id': user_id,
                'started_at': datetime.utcnow(),
                'status': 'processing'
            }

            # Get orchestrator agent
            orchestrator = self._get_orchestrator_agent()
            if not orchestrator:
                return {
                    'request_id': request_id,
                    'success': False,
                    'error': 'No orchestrator agent available',
                    'processing_time': time.time() - start_time
                }

            # Process through orchestrator
            result = await orchestrator.process_user_request(
                request=request,
                context=context,
                user_id=user_id
            )

            # Update metrics
            processing_time = time.time() - start_time
            self._update_request_metrics(request_id, processing_time, result.get('status') == 'completed')

            # Clean up tracking
            if request_id in self.active_requests:
                del self.active_requests[request_id]

            # Add processing metadata
            result.update({
                'processing_time': processing_time,
                'coordinator_id': self.coordinator_id,
                'hive_state': self.state.value
            })

            self.logger.info(
                "Request processed",
                request_id=request_id,
                success=result.get('status') == 'completed',
                processing_time=processing_time
            )

            return result

        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(
                "Request processing failed",
                request_id=request_id,
                error=str(e),
                processing_time=processing_time,
                exc_info=True
            )

            # Update error metrics
            self._update_request_metrics(request_id, processing_time, False)

            # Clean up tracking
            if request_id in self.active_requests:
                del self.active_requests[request_id]

            return {
                'request_id': request_id,
                'success': False,
                'error': str(e),
                'processing_time': processing_time
            }

    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        try:
            # Perform health check
            await self._perform_health_check()

            # Collect current metrics
            await self._collect_metrics()

            # Get agent statuses
            agent_statuses = {}
            for agent_id, agent in self.agents.items():
                try:
                    agent_statuses[agent_id] = agent.get_status()
                except Exception as e:
                    agent_statuses[agent_id] = {'error': str(e)}

            return {
                'hive_state': self.state.value,
                'health_status': {
                    'is_healthy': self.health_status.is_healthy,
                    'issues': self.health_status.issues,
                    'warnings': self.health_status.warnings,
                    'last_check': self.health_status.last_check.isoformat(),
                    'component_health': self.health_status.component_health
                },
                'metrics': {
                    'total_requests_processed': self.metrics.total_requests_processed,
                    'active_agents': self.metrics.active_agents,
                    'total_agents': self.metrics.total_agents,
                    'message_throughput': self.metrics.message_throughput,
                    'average_response_time': self.metrics.average_response_time,
                    'error_rate': self.metrics.error_rate,
                    'uptime_seconds': self.metrics.uptime_seconds,
                    'memory_usage_mb': self.metrics.memory_usage_mb,
                    'cpu_usage_percent': self.metrics.cpu_usage_percent
                },
                'agents': agent_statuses,
                'active_requests': len(self.active_requests),
                'configuration': {
                    'project_root': self.project_root,
                    'debug_mode': self.debug_mode,
                    'agent_count': len(self.agents)
                },
                'coordinator_id': self.coordinator_id,
                'started_at': self.started_at.isoformat() if self.started_at else None
            }

        except Exception as e:
            self.logger.error(f"Failed to get system status: {e}", exc_info=True)
            return {
                'error': str(e),
                'hive_state': self.state.value,
                'coordinator_id': self.coordinator_id
            }

    async def scale_agent(self, agent_type: str, target_instances: int) -> Dict[str, Any]:
        """Scale an agent type to the target number of instances."""
        try:
            if agent_type not in self.agent_configs:
                return {'success': False, 'error': f'Unknown agent type: {agent_type}'}

            config = self.agent_configs[agent_type]
            current_instances = len(self.agent_instances[agent_type])

            if target_instances < config.min_instances:
                return {'success': False, 'error': f'Target instances below minimum: {config.min_instances}'}

            if target_instances > config.max_instances:
                return {'success': False, 'error': f'Target instances above maximum: {config.max_instances}'}

            if target_instances == current_instances:
                return {'success': True, 'message': 'Already at target instances', 'instances': current_instances}

            if target_instances > current_instances:
                # Scale up
                for i in range(target_instances - current_instances):
                    agent = await self._create_agent_instance(agent_type, config)
                    if agent:
                        self.agent_instances[agent_type].append(agent)
                        await agent.start()
            else:
                # Scale down
                instances_to_remove = current_instances - target_instances
                for i in range(instances_to_remove):
                    if self.agent_instances[agent_type]:
                        agent = self.agent_instances[agent_type].pop()
                        await agent.stop()
                        if agent.agent_id in self.agents:
                            del self.agents[agent.agent_id]

            new_count = len(self.agent_instances[agent_type])
            self.logger.info(f"Scaled {agent_type} from {current_instances} to {new_count} instances")

            return {
                'success': True,
                'agent_type': agent_type,
                'previous_instances': current_instances,
                'current_instances': new_count,
                'target_instances': target_instances
            }

        except Exception as e:
            self.logger.error(f"Failed to scale agent {agent_type}: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}

    def get_metrics(self) -> HiveMetrics:
        """Get current system metrics."""
        return self.metrics

    def get_health_status(self) -> HealthStatus:
        """Get current health status."""
        return self.health_status

    # Private helper methods

    def _load_configuration(self, config_path: Optional[str]) -> HiveConfig:
        """Load hive configuration from file or use defaults."""
        try:
            if config_path and os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                self.logger.info(f"Loaded configuration from {config_path}")
            else:
                config_data = self._get_default_configuration()
                self.logger.info("Using default configuration")

            return HiveConfig(
                system=config_data.get('system', {}),
                agents={
                    name: AgentConfig(
                        agent_type=name,
                        enabled=agent_config.get('enabled', True),
                        instances=agent_config.get('instances', 1),
                        auto_scaling=agent_config.get('auto_scaling', {}).get('enabled', False),
                        min_instances=agent_config.get('auto_scaling', {}).get('min_instances', 1),
                        max_instances=agent_config.get('auto_scaling', {}).get('max_instances', 5),
                        target_cpu=agent_config.get('auto_scaling', {}).get('target_cpu', 80.0),
                        config=agent_config.get('configuration', {})
                    )
                    for name, agent_config in config_data.get('agents', {}).items()
                },
                messaging=config_data.get('messaging', {}),
                context=config_data.get('context', {}),
                performance=config_data.get('performance', {})
            )

        except Exception as e:
            self.logger.warning(f"Failed to load configuration: {e}, using defaults")
            return HiveConfig(**self._get_default_configuration())

    def _get_default_configuration(self) -> Dict[str, Any]:
        """Get default hive configuration."""
        return {
            'system': {
                'version': '1.0.0',
                'environment': 'development',
                'debug_mode': self.debug_mode,
                'log_level': 'INFO'
            },
            'agents': {
                'orchestrator': {
                    'enabled': True,
                    'instances': 1,
                    'configuration': {
                        'max_concurrent_workflows': 5,
                        'default_timeout': 300.0
                    }
                },
                'code': {
                    'enabled': True,
                    'instances': 1,
                    'auto_scaling': {
                        'enabled': False,
                        'min_instances': 1,
                        'max_instances': 3,
                        'target_cpu': 80.0
                    },
                    'configuration': {
                        'max_file_size': 1048576,
                        'generation_timeout': 120.0
                    }
                },
                'context': {
                    'enabled': True,
                    'instances': 1,
                    'configuration': {
                        'max_context_entries': 10000,
                        'context_ttl': 3600,
                        'enable_semantic_search': True
                    }
                },
                'git': {
                    'enabled': True,
                    'instances': 1,
                    'configuration': {
                        'auto_stage_changes': True,
                        'commit_message_style': 'conventional'
                    }
                }
            },
            'messaging': {
                'backend': 'memory',
                'max_queue_size': 1000,
                'message_ttl': 300,
                'compression': False,
                'encryption': False
            },
            'context': {
                'storage_backend': 'memory',
                'max_memory_entries': 10000,
                'vector_dimension': 384,
                'enable_compression': False,
                'cache_ttl': 300
            },
            'performance': {
                'monitoring_enabled': True,
                'metrics_interval': 60,
                'profiling_enabled': False,
                'health_check_interval': 30,
                'performance_targets': {
                    'agent_startup_time': 5.0,
                    'message_latency': 0.1,
                    'context_retrieval': 0.5,
                    'task_throughput': 10.0
                }
            }
        }

    async def _initialize_message_bus(self) -> None:
        """Initialize the message bus."""
        try:
            messaging_config = self.config.messaging
            self.message_bus = MessageBus(
                max_queue_size=messaging_config.get('max_queue_size', 10000),
                max_retry_attempts=messaging_config.get('max_retry_attempts', 3),
                message_ttl=messaging_config.get('message_ttl', 300),
                enable_persistence=messaging_config.get('enable_persistence', False),
                enable_dead_letter_queue=messaging_config.get('enable_dead_letter_queue', True)
            )
            await self.message_bus.start()
            self.logger.debug("Message bus initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize message bus: {e}")
            raise

    async def _initialize_context_store(self) -> None:
        """Initialize the global context store."""
        try:
            context_config = self.config.context
            self.context_store = GlobalContextStore(
                storage_backend=StorageBackend.HYBRID,
                max_memory_entries=context_config.get('max_memory_entries', 10000),
                max_cache_size=context_config.get('max_cache_size', 1000),
                vector_dimension=context_config.get('vector_dimension', 384),
                enable_compression=context_config.get('enable_compression', False),
                enable_encryption=context_config.get('enable_encryption', False)
            )
            await self.context_store.initialize()
            self.logger.debug("Context store initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize context store: {e}")
            raise

    async def _initialize_task_manager(self) -> None:
        """Initialize the task manager."""
        try:
            system_config = self.config.system
            self.task_manager = HiveTaskManager(
                max_concurrent_tasks=system_config.get('max_concurrent_tasks', 100),
                max_queue_size=system_config.get('max_queue_size', 10000),
                default_timeout=system_config.get('default_timeout', 300.0),
                scheduling_strategy=system_config.get('scheduling_strategy', 'priority_fifo'),
                enable_metrics=system_config.get('enable_metrics', True)
            )
            await self.task_manager.initialize()
            self.logger.debug("Task manager initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize task manager: {e}")
            raise

    async def _initialize_agent_pool(self) -> None:
        """Initialize the agent pool."""
        try:
            system_config = self.config.system
            self.agent_pool = AgentPool(
                message_bus=self.message_bus,
                selection_strategy=AgentSelectionStrategy.LEAST_LOADED,
                auto_scaling_enabled=system_config.get('auto_scaling_enabled', True),
                max_total_agents=system_config.get('max_total_agents', 100),
                health_check_interval=system_config.get('health_check_interval', 30.0),
                scaling_check_interval=system_config.get('scaling_check_interval', 60.0)
            )
            await self.agent_pool.start()
            self.logger.debug("Agent pool initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize agent pool: {e}")
            raise

    def _load_agent_configurations(self) -> None:
        """Load agent configurations."""
        self.agent_configs = {}
        for agent_type, config in self.config.agents.items():
            if config.enabled:
                self.agent_configs[agent_type] = config
                self.logger.debug(f"Loaded configuration for {agent_type} agent")

    async def _initialize_agents(self) -> None:
        """Initialize all configured agents."""
        for agent_type, config in self.agent_configs.items():
            try:
                for i in range(config.instances):
                    agent = await self._create_agent_instance(agent_type, config)
                    if agent:
                        self.agents[agent.agent_id] = agent
                        self.agent_instances[agent_type].append(agent)
                        await agent.initialize()

                self.logger.info(f"Initialized {config.instances} instances of {agent_type} agent")

            except Exception as e:
                self.logger.error(f"Failed to initialize {agent_type} agent: {e}")
                # Continue with other agents

    async def _create_agent_instance(self, agent_type: str, config: AgentConfig) -> Optional[BaseAgent]:
        """Create a single agent instance."""
        try:
            agent_id = f"{agent_type}_{uuid.uuid4().hex[:8]}"

            agent_config = {
                **config.config,
                'project_root': self.project_root,
                'debug_mode': self.debug_mode
            }

            if agent_type == 'orchestrator':
                agent = OrchestratorAgent(
                    agent_id=agent_id,
                    config=agent_config,
                    message_bus=self.message_bus
                )
            elif agent_type == 'code':
                agent = CodeAgent(
                    agent_id=agent_id,
                    config=agent_config,
                    message_bus=self.message_bus
                )
            elif agent_type == 'context':
                agent = ContextAgent(
                    agent_id=agent_id,
                    config=agent_config,
                    message_bus=self.message_bus
                )
            elif agent_type == 'git':
                agent = GitAgent(
                    agent_id=agent_id,
                    config=agent_config,
                    message_bus=self.message_bus
                )
            else:
                self.logger.error(f"Unknown agent type: {agent_type}")
                return None

            return agent

        except Exception as e:
            self.logger.error(f"Failed to create {agent_type} agent instance: {e}")
            return None

    async def _start_agents(self) -> None:
        """Start all initialized agents."""
        for agent_id, agent in self.agents.items():
            try:
                await agent.start()
                self.logger.debug(f"Started agent {agent_id}")
            except Exception as e:
                self.logger.error(f"Failed to start agent {agent_id}: {e}")

    async def _stop_agents(self, timeout: float) -> None:
        """Stop all agents gracefully."""
        if not self.agents:
            return

        # Create stop tasks for all agents
        stop_tasks = []
        for agent_id, agent in self.agents.items():
            task = asyncio.create_task(agent.stop())
            stop_tasks.append(task)

        try:
            # Wait for all agents to stop
            await asyncio.wait_for(
                asyncio.gather(*stop_tasks, return_exceptions=True),
                timeout=timeout
            )
            self.logger.info("All agents stopped successfully")
        except asyncio.TimeoutError:
            self.logger.warning(f"Agent shutdown timed out after {timeout}s")
        except Exception as e:
            self.logger.error(f"Error stopping agents: {e}")

    async def _setup_monitoring(self) -> None:
        """Setup system monitoring."""
        self.logger.debug("Setting up system monitoring")
        # Monitoring setup is handled by background tasks

    async def _start_background_tasks(self) -> None:
        """Start background monitoring and maintenance tasks."""
        # Health check task
        health_task = asyncio.create_task(self._health_check_loop())
        self.background_tasks.add(health_task)

        # Metrics collection task
        metrics_task = asyncio.create_task(self._metrics_collection_loop())
        self.background_tasks.add(metrics_task)

        # Auto-scaling task
        scaling_task = asyncio.create_task(self._auto_scaling_loop())
        self.background_tasks.add(scaling_task)

        self.logger.debug(f"Started {len(self.background_tasks)} background tasks")

    async def _stop_background_tasks(self, timeout: float) -> None:
        """Stop all background tasks."""
        if not self.background_tasks:
            return

        # Cancel all tasks
        for task in self.background_tasks:
            task.cancel()

        try:
            # Wait for cancellation
            await asyncio.wait_for(
                asyncio.gather(*self.background_tasks, return_exceptions=True),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            self.logger.warning(f"Background task shutdown timed out after {timeout}s")

        self.background_tasks.clear()

    async def _health_check_loop(self) -> None:
        """Background health check loop."""
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(self.health_check_interval)
                if not self.shutdown_event.is_set():
                    await self._perform_health_check()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in health check loop: {e}")

    async def _metrics_collection_loop(self) -> None:
        """Background metrics collection loop."""
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(self.metrics_collection_interval)
                if not self.shutdown_event.is_set():
                    await self._collect_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in metrics collection loop: {e}")

    async def _auto_scaling_loop(self) -> None:
        """Background auto-scaling loop."""
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(60.0)  # Check every minute
                if not self.shutdown_event.is_set():
                    await self._check_auto_scaling()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in auto-scaling loop: {e}")

    async def _perform_health_check(self) -> None:
        """Perform comprehensive health check."""
        try:
            issues = []
            warnings = []
            component_health = {}

            # Check message bus health
            if self.message_bus:
                try:
                    bus_health = await self.message_bus.health_check()
                    component_health['message_bus'] = bus_health
                    if not bus_health:
                        issues.append("Message bus is unhealthy")
                except Exception as e:
                    issues.append(f"Message bus health check failed: {e}")
                    component_health['message_bus'] = False

            # Check agent health
            healthy_agents = 0
            total_agents = len(self.agents)

            for agent_id, agent in self.agents.items():
                try:
                    agent_health = await agent.health_check()
                    component_health[agent_id] = agent_health
                    if agent_health:
                        healthy_agents += 1
                    else:
                        warnings.append(f"Agent {agent_id} is unhealthy")
                except Exception as e:
                    warnings.append(f"Agent {agent_id} health check failed: {e}")
                    component_health[agent_id] = False

            # Check task manager health
            if self.task_manager:
                try:
                    task_health = await self.task_manager.health_check()
                    component_health['task_manager'] = task_health
                    if not task_health:
                        warnings.append("Task manager is unhealthy")
                except Exception as e:
                    warnings.append(f"Task manager health check failed: {e}")
                    component_health['task_manager'] = False

            # Check context store health
            if self.context_store:
                try:
                    context_health = await self.context_store.health_check()
                    component_health['context_store'] = context_health
                    if not context_health:
                        warnings.append("Context store is unhealthy")
                except Exception as e:
                    warnings.append(f"Context store health check failed: {e}")
                    component_health['context_store'] = False

            # Determine overall health
            critical_issues = len(issues) > 0
            agent_health_ratio = healthy_agents / total_agents if total_agents > 0 else 1.0

            is_healthy = (not critical_issues and
                         agent_health_ratio >= 0.5 and
                         self.state == HiveState.RUNNING)

            # Update health status
            self.health_status = HealthStatus(
                is_healthy=is_healthy,
                status=self.state,
                issues=issues,
                warnings=warnings,
                last_check=datetime.utcnow(),
                component_health=component_health
            )

            # Update system state if needed
            if critical_issues and self.state == HiveState.RUNNING:
                self.state = HiveState.DEGRADED
            elif not critical_issues and self.state == HiveState.DEGRADED:
                self.state = HiveState.RUNNING

        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            self.health_status.is_healthy = False
            self.health_status.issues.append(f"Health check error: {e}")

    async def _collect_metrics(self) -> None:
        """Collect system-wide metrics."""
        try:
            current_time = datetime.utcnow()

            # Calculate uptime
            if self.started_at:
                uptime = (current_time - self.started_at).total_seconds()
            else:
                uptime = 0.0

            # Count active agents
            active_agents = sum(1 for agent in self.agents.values()
                              if agent.state.value in ['running', 'idle'])

            # Calculate message throughput
            message_throughput = 0.0
            if self.message_bus:
                try:
                    bus_metrics = self.message_bus.get_metrics()
                    message_throughput = bus_metrics.get('messages_per_second', 0.0)
                except:
                    pass

            # Calculate average response time
            if self.response_times:
                avg_response_time = sum(self.response_times) / len(self.response_times)
                # Keep only recent response times (last 100)
                if len(self.response_times) > 100:
                    self.response_times = self.response_times[-100:]
            else:
                avg_response_time = 0.0

            # Calculate error rate
            total_requests = self.metrics.total_requests_processed
            total_errors = sum(self.error_counts.values())
            error_rate = (total_errors / total_requests * 100) if total_requests > 0 else 0.0

            # Get system resource usage (simplified)
            memory_usage_mb = 0.0
            cpu_usage_percent = 0.0
            try:
                import psutil
                process = psutil.Process()
                memory_usage_mb = process.memory_info().rss / 1024 / 1024
                cpu_usage_percent = process.cpu_percent()
            except ImportError:
                # psutil not available, use placeholder values
                pass

            # Collect agent-specific metrics
            agent_metrics = {}
            for agent_id, agent in self.agents.items():
                try:
                    agent_metrics[agent_id] = agent.get_metrics()
                except Exception as e:
                    agent_metrics[agent_id] = {'error': str(e)}

            # Update metrics
            self.metrics = HiveMetrics(
                total_requests_processed=self.metrics.total_requests_processed,
                active_agents=active_agents,
                total_agents=len(self.agents),
                message_throughput=message_throughput,
                average_response_time=avg_response_time,
                error_rate=error_rate,
                uptime_seconds=uptime,
                memory_usage_mb=memory_usage_mb,
                cpu_usage_percent=cpu_usage_percent,
                agent_metrics=agent_metrics
            )

        except Exception as e:
            self.logger.error(f"Metrics collection failed: {e}")

    async def _check_auto_scaling(self) -> None:
        """Check and apply auto-scaling rules."""
        try:
            for agent_type, config in self.agent_configs.items():
                if not config.auto_scaling:
                    continue

                current_instances = len(self.agent_instances[agent_type])

                # Get agent metrics to determine load
                total_cpu = 0.0
                agent_count = 0

                for agent in self.agent_instances[agent_type]:
                    try:
                        metrics = agent.get_metrics()
                        if 'cpu_usage' in metrics:
                            total_cpu += metrics['cpu_usage']
                            agent_count += 1
                    except:
                        continue

                if agent_count == 0:
                    continue

                avg_cpu = total_cpu / agent_count

                # Scaling decisions
                target_instances = current_instances

                if avg_cpu > config.target_cpu and current_instances < config.max_instances:
                    # Scale up
                    target_instances = min(current_instances + 1, config.max_instances)
                    self.logger.info(f"Auto-scaling up {agent_type}: {current_instances} -> {target_instances}")

                elif avg_cpu < (config.target_cpu * 0.5) and current_instances > config.min_instances:
                    # Scale down
                    target_instances = max(current_instances - 1, config.min_instances)
                    self.logger.info(f"Auto-scaling down {agent_type}: {current_instances} -> {target_instances}")

                if target_instances != current_instances:
                    await self.scale_agent(agent_type, target_instances)

        except Exception as e:
            self.logger.error(f"Auto-scaling check failed: {e}")

    def _get_orchestrator_agent(self) -> Optional[OrchestratorAgent]:
        """Get an available orchestrator agent."""
        orchestrators = self.agent_instances.get('orchestrator', [])

        for agent in orchestrators:
            if hasattr(agent, 'state') and agent.state.value in ['running', 'idle']:
                return agent

        # Fallback: return any orchestrator
        return orchestrators[0] if orchestrators else None

    def _update_request_metrics(self, request_id: str, processing_time: float, success: bool) -> None:
        """Update metrics for a processed request."""
        try:
            self.metrics.total_requests_processed += 1
            self.response_times.append(processing_time)

            if not success:
                self.error_counts[request_id] = self.error_counts.get(request_id, 0) + 1

            # Store in request history
            self.request_history.append({
                'request_id': request_id,
                'processing_time': processing_time,
                'success': success,
                'timestamp': datetime.utcnow().isoformat()
            })

            # Limit history size
            if len(self.request_history) > self.max_history_size:
                self.request_history = self.request_history[-self.max_history_size:]

        except Exception as e:
            self.logger.error(f"Failed to update request metrics: {e}")

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        try:
            def signal_handler(signum, frame):
                self.logger.info(f"Received signal {signum}, initiating shutdown")
                asyncio.create_task(self.stop())

            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)

            if hasattr(signal, 'SIGHUP'):
                signal.signal(signal.SIGHUP, signal_handler)

        except Exception as e:
            self.logger.warning(f"Failed to setup signal handlers: {e}")

    # Context manager support
    async def __aenter__(self):
        """Async context manager entry."""
        if await self.initialize():
            if await self.start():
                return self
        raise Exception("Failed to initialize and start hive coordinator")

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()

    def __str__(self) -> str:
        """String representation of the hive coordinator."""
        return (f"HiveCoordinator(id={self.coordinator_id[:8]}, "
                f"state={self.state.value}, "
                f"agents={len(self.agents)}, "
                f"project_root={self.project_root})")

    def __repr__(self) -> str:
        """Detailed representation of the hive coordinator."""
        return (f"HiveCoordinator("
                f"coordinator_id='{self.coordinator_id}', "
                f"state={self.state}, "
                f"project_root='{self.project_root}', "
                f"debug_mode={self.debug_mode}, "
                f"agents={list(self.agents.keys())}, "
                f"started_at={self.started_at})")


# Convenience functions for external use

async def create_hive_coordinator(
    config_path: Optional[str] = None,
    project_root: Optional[str] = None,
    debug_mode: bool = False
) -> HiveCoordinator:
    """Create and initialize a hive coordinator."""
    coordinator = HiveCoordinator(
        config_path=config_path,
        project_root=project_root,
        debug_mode=debug_mode
    )

    if not await coordinator.initialize():
        raise Exception("Failed to initialize hive coordinator")

    return coordinator

async def run_hive_system(
    config_path: Optional[str] = None,
    project_root: Optional[str] = None,
    debug_mode: bool = False
) -> None:
    """Run the hive system until shutdown."""
    async with create_hive_coordinator(config_path, project_root, debug_mode) as coordinator:
        await coordinator.start()

        try:
            # Wait for shutdown signal
            await coordinator.shutdown_event.wait()
        except KeyboardInterrupt:
            coordinator.logger.info("Received keyboard interrupt")
        finally:
            coordinator.logger.info("Hive system shutting down")
