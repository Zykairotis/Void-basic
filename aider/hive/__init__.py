"""
Hive module for the Aider Multi-Agent Hive Architecture.

This module provides the core coordination and management functionality
for the multi-agent system, including:
- HiveCoordinator for system orchestration
- Agent lifecycle management
- Inter-agent communication coordination
- System health monitoring and metrics
- Configuration management
"""

from .hive_coordinator import (
    HiveCoordinator,
    HiveState,
    HiveConfig,
    HiveMetrics,
    HealthStatus,
    AgentConfig,
    create_hive_coordinator,
    run_hive_system
)

__all__ = [
    'HiveCoordinator',
    'HiveState',
    'HiveConfig',
    'HiveMetrics',
    'HealthStatus',
    'AgentConfig',
    'create_hive_coordinator',
    'run_hive_system'
]

__version__ = '1.0.0'
