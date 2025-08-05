"""
Aider Hive Agents Module

This module provides the core agent framework for the Aider CLI Multi-Agent Hive Architecture.
It includes specialized agents, message bus communication, and intelligent agent pool management.

Core Components:
- BaseAgent: Abstract base class for all agents
- MessageBus: High-performance inter-agent communication system
- AgentPool: Advanced agent lifecycle management and resource allocation

Agent Specializations:
- OrchestratorAgent: Master coordinator for all agent activities
- CodeAgent: Specialized for code generation and analysis
- ContextAgent: Manages project context and semantic understanding
- GitAgent: Handles Git operations and intelligent version control

Usage Example:
    ```python
    from aider.agents import MessageBus, AgentPool, BaseAgent

    # Initialize message bus
    message_bus = MessageBus()
    await message_bus.start()

    # Create agent pool
    agent_pool = AgentPool(message_bus=message_bus)
    await agent_pool.start()

    # Register agent types
    agent_pool.register_agent_type(
        agent_type="code",
        agent_class=CodeAgent,
        min_instances=2,
        max_instances=10
    )

    # Get an agent
    code_agent = await agent_pool.get_agent("code")
    ```
"""

__version__ = "1.0.0"
__author__ = "Aider Development Team"

# Core agent framework
from .base_agent import (
    BaseAgent,
    AgentState,
    MessagePriority,
    AgentMessage,
    AgentCapability,
    AgentMetrics,
)

# Message bus communication
from .message_bus import (
    MessageBus,
    MessageBusState,
    DeliveryStatus,
    Subscription,
    MessageDeliveryInfo,
    MessageBusMetrics,
)

# Agent pool management
from .agent_pool import (
    AgentPool,
    PoolState,
    AgentSelectionStrategy,
    AgentRegistration,
    AgentInstance,
    PoolMetrics,
    LoadBalancer,
)

# Specialized agents
from .orchestrator_agent import OrchestratorAgent, RequestType, RequestComplexity, RequestAnalysis
from .code_agent import CodeAgent, CodeLanguage, CodeOperation, CodeQuality, CodeGenerationRequest
from .context_agent import ContextAgent, ContextType, ContextScope, ContextEntry, ContextQuery
from .git_agent import GitAgent, GitOperation, ChangeType, ConflictResolutionStrategy, GitStatus

# Re-export all public classes and enums
__all__ = [
    # Core agent framework
    "BaseAgent",
    "AgentState",
    "MessagePriority",
    "AgentMessage",
    "AgentCapability",
    "AgentMetrics",

    # Message bus
    "MessageBus",
    "MessageBusState",
    "DeliveryStatus",
    "Subscription",
    "MessageDeliveryInfo",
    "MessageBusMetrics",

    # Agent pool
    "AgentPool",
    "PoolState",
    "AgentSelectionStrategy",
    "AgentRegistration",
    "AgentInstance",
    "PoolMetrics",
    "LoadBalancer",

    # Specialized agents
    "OrchestratorAgent",
    "RequestType",
    "RequestComplexity",
    "RequestAnalysis",
    "CodeAgent",
    "CodeLanguage",
    "CodeOperation",
    "CodeQuality",
    "CodeGenerationRequest",
    "ContextAgent",
    "ContextType",
    "ContextScope",
    "ContextEntry",
    "ContextQuery",
    "GitAgent",
    "GitOperation",
    "ChangeType",
    "ConflictResolutionStrategy",
    "GitStatus",
]

# Module metadata
FRAMEWORK_NAME = "Aider Hive Agent Framework"
FRAMEWORK_VERSION = __version__

def get_framework_info():
    """Get information about the agent framework."""
    return {
        "name": FRAMEWORK_NAME,
        "version": FRAMEWORK_VERSION,
        "components": {
            "base_agent": "Abstract agent foundation with lifecycle management",
            "message_bus": "High-performance pub/sub communication system",
            "agent_pool": "Intelligent agent scaling and resource management",
            "orchestrator_agent": "Central coordination and workflow management",
            "code_agent": "Code generation, modification, and review",
            "context_agent": "Project context management and semantic search",
            "git_agent": "Intelligent git operations and version control",
        },
        "features": [
            "Asynchronous agent communication",
            "Dynamic agent scaling",
            "Health monitoring and auto-recovery",
            "Load balancing and intelligent routing",
            "Performance metrics and analytics",
            "Fault tolerance and graceful degradation",
        ]
    }
