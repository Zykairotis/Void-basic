"""
AgentPool: Advanced agent lifecycle management and resource allocation system.

This module provides intelligent agent pool management with features including:
- Dynamic agent scaling based on load
- Health monitoring and auto-recovery
- Load balancing and intelligent routing
- Resource optimization and allocation
- Performance metrics and analytics
- Fault tolerance and graceful degradation
"""

import asyncio
import logging
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Type, Callable, Union
from weakref import WeakSet

import structlog

# Import autonomous workflow agents
from .workflow_orchestrator import WorkflowOrchestrator
from .quality_agent import QualityAgent
from .deployment_agent import DeploymentAgent

from .base_agent import BaseAgent, AgentState, AgentCapability, AgentMetrics
from .message_bus import MessageBus


class PoolState(Enum):
    """Agent pool operational states."""
    INITIALIZING = "initializing"
    RUNNING = "running"
    SCALING = "scaling"
    DRAINING = "draining"
    SHUTDOWN = "shutdown"


class AgentSelectionStrategy(Enum):
    """Strategies for selecting agents from the pool."""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    FASTEST_RESPONSE = "fastest_response"
    RANDOM = "random"
    CAPABILITY_MATCH = "capability_match"


@dataclass
class AgentRegistration:
    """Agent type registration information."""
    agent_type: str
    agent_class: Type[BaseAgent]
    default_config: Dict[str, Any] = field(default_factory=dict)
    min_instances: int = 1
    max_instances: int = 10
    target_cpu_utilization: float = 0.7
    scaling_cooldown: int = 300  # seconds
    health_check_interval: int = 30  # seconds
    capabilities: List[AgentCapability] = field(default_factory=list)


@dataclass
class AgentInstance:
    """Represents an agent instance in the pool."""
    instance_id: str
    agent_type: str
    agent: BaseAgent
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_health_check: Optional[datetime] = None
    health_status: Dict[str, Any] = field(default_factory=dict)
    is_healthy: bool = True
    current_load: float = 0.0
    total_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0


@dataclass
class PoolMetrics:
    """Pool-wide performance and operational metrics."""
    total_agents: int = 0
    healthy_agents: int = 0
    unhealthy_agents: int = 0
    average_load: float = 0.0
    peak_load: float = 0.0
    total_requests_processed: int = 0
    total_requests_failed: int = 0
    average_response_time: float = 0.0
    scaling_events: int = 0
    last_scaling_event: Optional[datetime] = None


class LoadBalancer:
    """Intelligent load balancer for agent selection."""

    def __init__(self, strategy: AgentSelectionStrategy = AgentSelectionStrategy.LEAST_LOADED):
        self.strategy = strategy
        self.round_robin_counters: Dict[str, int] = defaultdict(int)

    def select_agent(
        self,
        agents: List[AgentInstance],
        request_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[AgentInstance]:
        """
        Select the best agent for handling a request.

        Args:
            agents: List of available healthy agents
            request_context: Optional context for capability matching

        Returns:
            Selected agent instance or None if no suitable agent
        """
        if not agents:
            return None

        # Filter by capability if specified
        if request_context and 'required_capabilities' in request_context:
            required_caps = request_context['required_capabilities']
            capable_agents = []
            for agent in agents:
                agent_caps = {cap.name for cap in agent.agent.get_capabilities()}
                if all(cap in agent_caps for cap in required_caps):
                    capable_agents.append(agent)
            agents = capable_agents

        if not agents:
            return None

        if self.strategy == AgentSelectionStrategy.ROUND_ROBIN:
            return self._select_round_robin(agents)
        elif self.strategy == AgentSelectionStrategy.LEAST_LOADED:
            return self._select_least_loaded(agents)
        elif self.strategy == AgentSelectionStrategy.FASTEST_RESPONSE:
            return self._select_fastest_response(agents)
        elif self.strategy == AgentSelectionStrategy.RANDOM:
            return self._select_random(agents)
        else:
            return agents[0]  # Fallback

    def _select_round_robin(self, agents: List[AgentInstance]) -> AgentInstance:
        """Round-robin selection."""
        agent_type = agents[0].agent_type
        counter = self.round_robin_counters[agent_type]
        selected = agents[counter % len(agents)]
        self.round_robin_counters[agent_type] = (counter + 1) % len(agents)
        return selected

    def _select_least_loaded(self, agents: List[AgentInstance]) -> AgentInstance:
        """Select agent with lowest current load."""
        return min(agents, key=lambda a: a.current_load)

    def _select_fastest_response(self, agents: List[AgentInstance]) -> AgentInstance:
        """Select agent with fastest average response time."""
        return min(agents, key=lambda a: a.average_response_time or float('inf'))

    def _select_random(self, agents: List[AgentInstance]) -> AgentInstance:
        """Random selection."""
        import random
        return random.choice(agents)


class AgentPool:
    """
    Advanced agent pool management system.

    Features:
    - Dynamic scaling based on load and performance
    - Intelligent agent selection and load balancing
    - Health monitoring and auto-recovery
    - Resource optimization and allocation
    - Performance analytics and metrics
    - Fault tolerance and graceful degradation
    """

    def __init__(
        self,
        message_bus: Optional[MessageBus] = None,
        selection_strategy: AgentSelectionStrategy = AgentSelectionStrategy.LEAST_LOADED,
        auto_scaling_enabled: bool = True,
        max_total_agents: int = 100,
        health_check_interval: float = 30.0,
        scaling_check_interval: float = 60.0,
    ):
        """
        Initialize the agent pool.

        Args:
            message_bus: Message bus for agent communication
            selection_strategy: Strategy for selecting agents
            auto_scaling_enabled: Enable automatic scaling
            max_total_agents: Maximum total agents across all types
            health_check_interval: Health check frequency in seconds
            scaling_check_interval: Scaling evaluation frequency in seconds
        """
        self.message_bus = message_bus
        self.auto_scaling_enabled = auto_scaling_enabled
        self.max_total_agents = max_total_agents
        self.health_check_interval = health_check_interval
        self.scaling_check_interval = scaling_check_interval

        # State management
        self.state = PoolState.INITIALIZING
        self.started_at: Optional[datetime] = None

        # Logging
        self.logger = structlog.get_logger().bind(component="agent_pool")

        # Agent management
        self.registered_types: Dict[str, AgentRegistration] = {}
        self.agent_instances: Dict[str, AgentInstance] = {}
        self.agents_by_type: Dict[str, List[str]] = defaultdict(list)
        self.healthy_agents: Dict[str, List[str]] = defaultdict(list)

        # Load balancing
        self.load_balancer = LoadBalancer(selection_strategy)

        # Metrics and monitoring
        self.metrics = PoolMetrics()
        self.request_history: deque = deque(maxlen=1000)
        self.scaling_history: deque = deque(maxlen=100)

        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()
        self._shutdown_event = asyncio.Event()

        # Request tracking
        self.active_requests: Dict[str, Dict[str, Any]] = {}
        self.request_semaphore = asyncio.Semaphore(1000)  # Max concurrent requests

    async def start(self) -> None:
        """Start the agent pool and background monitoring."""
        try:
            self.logger.info("Starting agent pool")
            self.state = PoolState.INITIALIZING

            # Start background tasks
            self._start_background_tasks()

            # Register autonomous workflow agents
            await self._register_autonomous_workflow_agents()

            # Initialize minimum instances for registered types
            for agent_type, registration in self.registered_types.items():
                await self._ensure_minimum_instances(agent_type)

            self.state = PoolState.RUNNING
            self.started_at = datetime.utcnow()

            self.logger.info("Agent pool started successfully")

        except Exception as e:
            self.state = PoolState.SHUTDOWN
            self.logger.error("Failed to start agent pool", error=str(e))
            raise

    async def stop(self) -> None:
        """Stop the agent pool gracefully."""
        self.logger.info("Stopping agent pool")
        self.state = PoolState.DRAINING

        # Signal shutdown
        self._shutdown_event.set()

        # Stop all agents
        stop_tasks = []
        for instance in self.agent_instances.values():
            task = asyncio.create_task(instance.agent.stop())
            stop_tasks.append(task)

        if stop_tasks:
            await asyncio.gather(*stop_tasks, return_exceptions=True)

        # Cancel background tasks
        await self._cancel_background_tasks()

        # Clear data structures
        self.agent_instances.clear()
        self.agents_by_type.clear()
        self.healthy_agents.clear()

        self.state = PoolState.SHUTDOWN
        self.logger.info("Agent pool stopped")

    def register_agent_type(
        self,
        agent_type: str,
        agent_class: Type[BaseAgent],
        config: Optional[Dict[str, Any]] = None,
        min_instances: int = 1,
        max_instances: int = 10,
        target_cpu_utilization: float = 0.7,
    ) -> None:
        """
        Register a new agent type with the pool.

        Args:
            agent_type: Unique identifier for the agent type
            agent_class: Agent class to instantiate
            config: Default configuration for agents of this type
            min_instances: Minimum number of instances to maintain
            max_instances: Maximum number of instances allowed
            target_cpu_utilization: Target CPU utilization for scaling
        """
        registration = AgentRegistration(
            agent_type=agent_type,
            agent_class=agent_class,
            default_config=config or {},
            min_instances=min_instances,
            max_instances=max_instances,
            target_cpu_utilization=target_cpu_utilization,
        )

        self.registered_types[agent_type] = registration

        self.logger.info(
            "Agent type registered",
            agent_type=agent_type,
            min_instances=min_instances,
            max_instances=max_instances,
        )

    async def get_agent(
        self,
        agent_type: str,
        required_capabilities: Optional[List[str]] = None,
        timeout: float = 30.0,
    ) -> Optional[BaseAgent]:
        """
        Get an available agent of the specified type.

        Args:
            agent_type: Type of agent needed
            required_capabilities: List of required capabilities
            timeout: Maximum time to wait for an agent

        Returns:
            Available agent instance or None
        """
        if self.state != PoolState.RUNNING:
            raise RuntimeError("Agent pool is not running")

        async with self.request_semaphore:
            start_time = time.time()

            try:
                # Get healthy agents of the requested type
                healthy_agent_ids = self.healthy_agents.get(agent_type, [])
                healthy_agents = [
                    self.agent_instances[agent_id]
                    for agent_id in healthy_agent_ids
                    if agent_id in self.agent_instances
                ]

                if not healthy_agents:
                    # Try to scale up if auto-scaling is enabled
                    if self.auto_scaling_enabled:
                        await self._scale_up_if_needed(agent_type)

                        # Wait briefly for new agents to become available
                        await asyncio.sleep(1.0)

                        healthy_agent_ids = self.healthy_agents.get(agent_type, [])
                        healthy_agents = [
                            self.agent_instances[agent_id]
                            for agent_id in healthy_agent_ids
                            if agent_id in self.agent_instances
                        ]

                    if not healthy_agents:
                        self.logger.warning("No healthy agents available", agent_type=agent_type)
                        return None

                # Select the best agent
                request_context = {}
                if required_capabilities:
                    request_context['required_capabilities'] = required_capabilities

                selected_instance = self.load_balancer.select_agent(healthy_agents, request_context)

                if selected_instance:
                    # Track the request
                    request_id = str(uuid.uuid4())
                    self.active_requests[request_id] = {
                        'agent_id': selected_instance.instance_id,
                        'agent_type': agent_type,
                        'start_time': start_time,
                        'required_capabilities': required_capabilities,
                    }

                    # Update agent load
                    selected_instance.current_load += 1
                    selected_instance.total_requests += 1

                    # Schedule request completion tracking
                    asyncio.create_task(self._track_request_completion(request_id, selected_instance))

                    return selected_instance.agent

                return None

            except Exception as e:
                self.logger.error("Error getting agent", agent_type=agent_type, error=str(e))
                return None

    async def return_agent(self, agent: BaseAgent) -> None:
        """
        Return an agent to the pool (decrease its load).

        Args:
            agent: Agent instance to return
        """
        # Find the agent instance
        for instance in self.agent_instances.values():
            if instance.agent == agent:
                if instance.current_load > 0:
                    instance.current_load -= 1
                break

    async def get_pool_status(self) -> Dict[str, Any]:
        """Get comprehensive pool status and metrics."""
        # Update metrics
        await self._update_pool_metrics()

        uptime = 0.0
        if self.started_at:
            uptime = (datetime.utcnow() - self.started_at).total_seconds()

        status = {
            'state': self.state.value,
            'uptime': uptime,
            'metrics': {
                'total_agents': self.metrics.total_agents,
                'healthy_agents': self.metrics.healthy_agents,
                'unhealthy_agents': self.metrics.unhealthy_agents,
                'average_load': self.metrics.average_load,
                'peak_load': self.metrics.peak_load,
                'total_requests_processed': self.metrics.total_requests_processed,
                'total_requests_failed': self.metrics.total_requests_failed,
                'average_response_time': self.metrics.average_response_time,
                'scaling_events': self.metrics.scaling_events,
            },
            'agent_types': {},
            'active_requests': len(self.active_requests),
        }

        # Add per-type statistics
        for agent_type in self.registered_types.keys():
            type_agents = [
                self.agent_instances[agent_id]
                for agent_id in self.agents_by_type[agent_type]
                if agent_id in self.agent_instances
            ]

            healthy_count = len(self.healthy_agents.get(agent_type, []))
            total_count = len(type_agents)

            status['agent_types'][agent_type] = {
                'total_instances': total_count,
                'healthy_instances': healthy_count,
                'unhealthy_instances': total_count - healthy_count,
                'average_load': sum(a.current_load for a in type_agents) / max(1, total_count),
                'total_requests': sum(a.total_requests for a in type_agents),
                'failed_requests': sum(a.failed_requests for a in type_agents),
            }

        return status

    async def scale_agent_type(self, agent_type: str, target_instances: int) -> bool:
        """
        Manually scale an agent type to a target number of instances.

        Args:
            agent_type: Agent type to scale
            target_instances: Target number of instances

        Returns:
            True if scaling was successful
        """
        if agent_type not in self.registered_types:
            self.logger.error("Unknown agent type", agent_type=agent_type)
            return False

        registration = self.registered_types[agent_type]
        current_count = len(self.agents_by_type[agent_type])

        # Validate target
        if target_instances < registration.min_instances:
            target_instances = registration.min_instances
        elif target_instances > registration.max_instances:
            target_instances = registration.max_instances

        try:
            if target_instances > current_count:
                # Scale up
                for _ in range(target_instances - current_count):
                    await self._create_agent_instance(agent_type)
            elif target_instances < current_count:
                # Scale down
                await self._scale_down_agent_type(agent_type, current_count - target_instances)

            self.logger.info(
                "Manual scaling completed",
                agent_type=agent_type,
                from_count=current_count,
                to_count=target_instances,
            )
            return True

        except Exception as e:
            self.logger.error("Failed to scale agent type", agent_type=agent_type, error=str(e))
            return False

    async def _create_agent_instance(self, agent_type: str) -> Optional[str]:
        """Create a new agent instance."""
        try:
            registration = self.registered_types[agent_type]

            # Check total agent limit
            if len(self.agent_instances) >= self.max_total_agents:
                self.logger.warning("Cannot create agent: total limit reached")
                return None

            # Create agent instance
            instance_id = f"{agent_type}_{uuid.uuid4().hex[:8]}"
            agent = registration.agent_class(
                agent_id=instance_id,
                agent_type=agent_type,
                config=registration.default_config.copy(),
                message_bus=self.message_bus,
            )

            # Create instance tracking
            instance = AgentInstance(
                instance_id=instance_id,
                agent_type=agent_type,
                agent=agent,
            )

            # Start the agent
            await agent.start()

            # Add to tracking structures
            self.agent_instances[instance_id] = instance
            self.agents_by_type[agent_type].append(instance_id)
            self.healthy_agents[agent_type].append(instance_id)

            self.logger.info("Agent instance created", instance_id=instance_id, agent_type=agent_type)
            return instance_id

        except Exception as e:
            self.logger.error("Failed to create agent instance", agent_type=agent_type, error=str(e))
            return None

    async def _remove_agent_instance(self, instance_id: str) -> bool:
        """Remove an agent instance."""
        try:
            if instance_id not in self.agent_instances:
                return False

            instance = self.agent_instances[instance_id]
            agent_type = instance.agent_type

            # Stop the agent
            await instance.agent.stop()

            # Remove from tracking structures
            del self.agent_instances[instance_id]
            self.agents_by_type[agent_type].remove(instance_id)
            if instance_id in self.healthy_agents[agent_type]:
                self.healthy_agents[agent_type].remove(instance_id)

            self.logger.info("Agent instance removed", instance_id=instance_id, agent_type=agent_type)
            return True

        except Exception as e:
            self.logger.error("Failed to remove agent instance", instance_id=instance_id, error=str(e))
            return False

    async def _ensure_minimum_instances(self, agent_type: str) -> None:
        """Ensure minimum number of instances for an agent type."""
        registration = self.registered_types[agent_type]
        current_count = len(self.agents_by_type[agent_type])

        for _ in range(registration.min_instances - current_count):
            await self._create_agent_instance(agent_type)

    async def _scale_up_if_needed(self, agent_type: str) -> bool:
        """Scale up an agent type if needed."""
        registration = self.registered_types[agent_type]
        current_count = len(self.agents_by_type[agent_type])

        if current_count < registration.max_instances:
            await self._create_agent_instance(agent_type)
            return True

        return False

    async def _scale_down_agent_type(self, agent_type: str, remove_count: int) -> None:
        """Scale down an agent type by removing instances."""
        agent_ids = self.agents_by_type[agent_type].copy()

        # Sort by load (remove least loaded first)
        instances = [self.agent_instances[aid] for aid in agent_ids if aid in self.agent_instances]
        instances.sort(key=lambda i: i.current_load)

        removed = 0
        for instance in instances:
            if removed >= remove_count:
                break

            if instance.current_load == 0:  # Only remove idle agents
                await self._remove_agent_instance(instance.instance_id)
                removed += 1

    def _start_background_tasks(self) -> None:
        """Start background monitoring and management tasks."""
        # Health monitoring
        health_task = asyncio.create_task(self._health_monitor_loop())
        self._background_tasks.add(health_task)
        health_task.add_done_callback(self._background_tasks.discard)

        # Auto-scaling
        if self.auto_scaling_enabled:
            scaling_task = asyncio.create_task(self._auto_scaling_loop())
            self._background_tasks.add(scaling_task)
            scaling_task.add_done_callback(self._background_tasks.discard)

        # Metrics collection
        metrics_task = asyncio.create_task(self._metrics_collection_loop())
        self._background_tasks.add(metrics_task)
        metrics_task.add_done_callback(self._background_tasks.discard)

    async def _health_monitor_loop(self) -> None:
        """Monitor agent health continuously."""
        while not self._shutdown_event.is_set():
            try:
                for instance_id, instance in list(self.agent_instances.items()):
                    try:
                        # Perform health check
                        health_status = await instance.agent.health_check()
                        instance.health_status = health_status
                        instance.last_health_check = datetime.utcnow()

                        # Determine health
                        is_healthy = health_status.get('healthy', True)

                        # Update healthy agents list
                        agent_type = instance.agent_type
                        if is_healthy and not instance.is_healthy:
                            # Agent recovered
                            instance.is_healthy = True
                            if instance_id not in self.healthy_agents[agent_type]:
                                self.healthy_agents[agent_type].append(instance_id)
                            self.logger.info("Agent recovered", instance_id=instance_id)

                        elif not is_healthy and instance.is_healthy:
                            # Agent became unhealthy
                            instance.is_healthy = False
                            if instance_id in self.healthy_agents[agent_type]:
                                self.healthy_agents[agent_type].remove(instance_id)
                            self.logger.warning("Agent became unhealthy", instance_id=instance_id)

                    except Exception as e:
                        # Mark as unhealthy on health check failure
                        self.logger.error("Health check failed", instance_id=instance_id, error=str(e))
                        instance.is_healthy = False
                        if instance_id in self.healthy_agents[instance.agent_type]:
                            self.healthy_agents[instance.agent_type].remove(instance_id)

                await asyncio.sleep(self.health_check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Error in health monitor loop", error=str(e))

    async def _auto_scaling_loop(self) -> None:
        """Auto-scaling evaluation loop."""
        while not self._shutdown_event.is_set():
            try:
                for agent_type, registration in self.registered_types.items():
                    await self._evaluate_scaling(agent_type, registration)

                await asyncio.sleep(self.scaling_check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Error in auto-scaling loop", error=str(e))

    async def _evaluate_scaling(self, agent_type: str, registration: AgentRegistration) -> None:
        """Evaluate if scaling is needed for an agent type."""
        try:
            current_instances = [
                self.agent_instances[aid] for aid in self.agents_by_type[agent_type]
                if aid in self.agent_instances
            ]

            if not current_instances:
                return

            # Calculate average load
            total_load = sum(instance.current_load for instance in current_instances)
            average_load = total_load / len(current_instances)

            # Scale up if average load is high
            if (average_load > registration.target_cpu_utilization and
                len(current_instances) < registration.max_instances):

                await self._create_agent_instance(agent_type)
                self.metrics.scaling_events += 1
                self.metrics.last_scaling_event = datetime.utcnow()

                self.scaling_history.append({
                    'timestamp': datetime.utcnow(),
                    'agent_type': agent_type,
                    'action': 'scale_up',
                    'from_count': len(current_instances),
                    'to_count': len(current_instances) + 1,
                    'average_load': average_load,
                })

            # Scale down if average load is low (only idle agents)
            elif (average_load < registration.target_cpu_utilization * 0.3 and
                  len(current_instances) > registration.min_instances):

                # Find idle agents
                idle_instances = [i for i in current_instances if i.current_load == 0]
                if idle_instances:
                    await self._remove_agent_instance(idle_instances[0].instance_id)
                    self.metrics.scaling_events += 1
                    self.metrics.last_scaling_event = datetime.utcnow()

                    self.scaling_history.append({
                        'timestamp': datetime.utcnow(),
                        'agent_type': agent_type,
                        'action': 'scale_down',
                        'from_count': len(current_instances),
                        'to_count': len(current_instances) - 1,
                        'average_load': average_load,
                    })

        except Exception as e:
            self.logger.error("Error evaluating scaling", agent_type=agent_type, error=str(e))

    async def _metrics_collection_loop(self) -> None:
        """Collect and update pool metrics."""
        while not self._shutdown_event.is_set():
            try:
                await self._update_pool_metrics()
                await asyncio.sleep(30)  # Update metrics every 30 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Error in metrics collection loop", error=str(e))

    async def _update_pool_metrics(self) -> None:
        """Update pool-wide metrics."""
        try:
            total_agents = len(self.agent_instances)
            healthy_count = sum(len(agents) for agents in self.healthy_agents.values())

            total_load = sum(instance.current_load for instance in self.agent_instances.values())
            average_load = total_load / max(1, total_agents)

            self.metrics.total_agents = total_agents
            self.metrics.healthy_agents = healthy_count
            self.metrics.unhealthy_agents = total_agents - healthy_count
            self.metrics.average_load = average_load

            if average_load > self.metrics.peak_load:
                self.metrics.peak_load = average_load

            # Update response time metrics
            total_requests = sum(instance.total_requests for instance in self.agent_instances.values())
            if total_requests > 0:
                total_response_time = sum(
                    instance.average_response_time * instance.total_requests
                    for instance in self.agent_instances.values()
                    if instance.average_response_time > 0
                )
                self.metrics.average_response_time = total_response_time / total_requests

        except Exception as e:
            self.logger.error("Error updating metrics", error=str(e))

    async def _track_request_completion(self, request_id: str, instance: AgentInstance) -> None:
        """Track when a request completes and update metrics."""
        # This would typically be called when the agent finishes processing
        # For now, we'll simulate with a timeout
        await asyncio.sleep(0.1)  # Brief delay to simulate processing

        if request_id in self.active_requests:
            request_info = self.active_requests.pop(request_id)

            # Update instance metrics
            processing_time = time.time() - request_info['start_time']
            if instance.average_response_time == 0:
                instance.average_response_time = processing_time
            else:
                instance.average_response_time = (instance.average_response_time + processing_time) / 2

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check and return status.

        Returns:
            Dictionary containing health status information
        """
        try:
            current_time = datetime.utcnow()

            # Check agent pool operational status
            is_running = self.state == AgentPoolState.RUNNING

            # Check agent instances health
            total_instances = len(self.agent_instances)
            healthy_instances = 0
            unhealthy_instances = 0
            busy_instances = 0

            for agent_type, instances in self.agent_instances.items():
                for instance in instances:
                    if instance.state == AgentInstanceState.RUNNING:
                        if instance.current_load < 0.8:  # Consider healthy if load < 80%
                            healthy_instances += 1
                        else:
                            busy_instances += 1
                    else:
                        unhealthy_instances += 1

            # Check scaling health
            auto_scaling_enabled = self.config.get('auto_scaling', {}).get('enabled', False)
            scaling_health = True

            if auto_scaling_enabled:
                # Check if any agent type is at capacity
                for agent_type, instances in self.agent_instances.items():
                    type_config = self.config.get('agents', {}).get(agent_type, {})
                    max_instances = type_config.get('auto_scaling', {}).get('max_instances', 5)
                    if len(instances) >= max_instances and all(inst.current_load > 0.9 for inst in instances):
                        scaling_health = False
                        break

            # Check request processing
            active_requests = len(self.active_requests)
            max_requests = self.config.get('max_concurrent_requests', 100)
            request_utilization = (active_requests / max_requests) * 100 if max_requests > 0 else 0

            # Calculate performance metrics
            total_requests = self.pool_metrics.get('total_requests', 0)
            completed_requests = self.pool_metrics.get('completed_requests', 0)
            failed_requests = self.pool_metrics.get('failed_requests', 0)

            success_rate = 0.0
            if total_requests > 0:
                success_rate = (completed_requests / total_requests) * 100

            # Check background task health
            active_tasks = len([task for task in self._background_tasks if not task.done()])

            # Determine overall health status
            is_healthy = (
                is_running and
                healthy_instances > 0 and
                unhealthy_instances == 0 and
                scaling_health and
                request_utilization < 90.0 and
                success_rate >= 90.0 and
                active_tasks > 0
            )

            health_status = {
                "status": "healthy" if is_healthy else "degraded",
                "timestamp": current_time.isoformat(),
                "agent_pool_specific": {
                    "state": self.state.value,
                    "is_running": is_running,
                    "instance_health": {
                        "total_instances": total_instances,
                        "healthy_instances": healthy_instances,
                        "unhealthy_instances": unhealthy_instances,
                        "busy_instances": busy_instances,
                        "agent_types": list(self.agent_instances.keys())
                    },
                    "scaling_health": {
                        "auto_scaling_enabled": auto_scaling_enabled,
                        "scaling_health": scaling_health
                    },
                    "request_processing": {
                        "active_requests": active_requests,
                        "max_requests": max_requests,
                        "request_utilization_percent": request_utilization
                    },
                    "performance_metrics": {
                        "total_requests": total_requests,
                        "completed_requests": completed_requests,
                        "failed_requests": failed_requests,
                        "success_rate_percent": success_rate,
                        "average_response_time": self.pool_metrics.get('avg_response_time', 0.0)
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
                issues.append(f"Agent pool not running (state: {self.state.value})")
            if total_instances == 0:
                issues.append("No agent instances available")
            if unhealthy_instances > 0:
                issues.append(f"{unhealthy_instances} unhealthy agent instances")
            if not scaling_health:
                issues.append("Auto-scaling at capacity limits")
            if request_utilization >= 90.0:
                issues.append(f"High request utilization: {request_utilization:.1f}%")
            if success_rate < 90.0 and total_requests > 0:
                issues.append(f"Low success rate: {success_rate:.1f}%")
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

    async def _register_autonomous_workflow_agents(self):
        """Register autonomous workflow agents with the pool."""

        # Register WorkflowOrchestrator
        self.register_agent_type(
            agent_type="workflow_orchestrator",
            agent_class=WorkflowOrchestrator,
            config={
                'max_concurrent_workflows': 5,
                'human_review_enabled': True,
                'auto_deployment_enabled': False
            },
            min_instances=1,
            max_instances=3,
            target_cpu_utilization=0.6
        )

        # Register QualityAgent
        self.register_agent_type(
            agent_type="quality",
            agent_class=QualityAgent,
            config={
                'ai_test_generation': True,
                'self_healing_tests': True,
                'parallel_execution': True,
                'max_concurrent_tests': 10
            },
            min_instances=1,
            max_instances=5,
            target_cpu_utilization=0.7
        )

        # Register DeploymentAgent
        self.register_agent_type(
            agent_type="deployment",
            agent_class=DeploymentAgent,
            config={
                'docker_enabled': True,
                'kubernetes_enabled': False,
                'auto_rollback_enabled': True,
                'security_scanning_enabled': True
            },
            min_instances=1,
            max_instances=3,
            target_cpu_utilization=0.8
        )

        self.logger.info("Autonomous workflow agents registered successfully")

    async def _cancel_background_tasks(self):
        """Cancel all background tasks."""
        for task in self._background_tasks:
            if not task.done():
                task.cancel()

        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)

        self._background_tasks.clear()
