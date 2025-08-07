"""
Session Management Module for Aider Hive Architecture.

This module provides comprehensive session management capabilities including:
- Advanced session lifecycle management with persistence
- Event sourcing with audit trails and recovery
- Hierarchical context management with semantic search
- Fault-tolerant recovery mechanisms
- Real-time session monitoring and metrics

Key Components:
- SessionManager: Core session management and coordination
- SessionEventStore: Event sourcing and audit trail management
- HierarchicalContextManager: Advanced context organization and retrieval
- SessionRecoveryManager: Fault tolerance and recovery operations
"""

from .session_manager import (
    SessionManager,
    SessionStatus,
    SessionPriority,
    SessionMetadata,
    SessionSnapshot,
    SessionStatistics,
    HiveMetrics,
    HealthStatus,
    AgentConfig,
    SessionModel,
    SessionSnapshotModel,
    create_hive_coordinator,
    run_hive_system
)

from .session_events import (
    SessionEvent,
    EventType,
    EventQuery,
    SessionEventStore,
    SessionEventModel
)

from .context_hierarchy import (
    HierarchicalContextManager,
    ContextChunk,
    ContextTier,
    ContextType,
    ContextHierarchy,
    ContextRetrievalResult,
    RetrievalStrategy,
    SemanticChunker,
    VectorSearchEngine
)

from .session_recovery import (
    SessionRecoveryManager,
    RecoveryStrategy,
    RecoveryStatus,
    RecoveryOperation,
    RecoveryPlan,
    FailureType
)

__all__ = [
    # Session Management
    'SessionManager',
    'SessionStatus',
    'SessionPriority',
    'SessionMetadata',
    'SessionSnapshot',
    'SessionStatistics',
    'HiveMetrics',
    'HealthStatus',
    'AgentConfig',
    'SessionModel',
    'SessionSnapshotModel',
    'create_hive_coordinator',
    'run_hive_system',

    # Event Management
    'SessionEvent',
    'EventType',
    'EventQuery',
    'SessionEventStore',
    'SessionEventModel',

    # Context Management
    'HierarchicalContextManager',
    'ContextChunk',
    'ContextTier',
    'ContextType',
    'ContextHierarchy',
    'ContextRetrievalResult',
    'RetrievalStrategy',
    'SemanticChunker',
    'VectorSearchEngine',

    # Recovery Management
    'SessionRecoveryManager',
    'RecoveryStrategy',
    'RecoveryStatus',
    'RecoveryOperation',
    'RecoveryPlan',
    'FailureType'
]

__version__ = '1.0.0'
