"""
Aider Hive Context Management Module

This module provides advanced context storage, sharing, and synchronization capabilities
for the Aider CLI Multi-Agent Hive Architecture.

Core Components:
- GlobalContextStore: Advanced context storage with vector search capabilities
- InterAgentContextManager: Real-time context sharing and synchronization between agents
- VectorIndex: Semantic similarity search for context entries
- ContextCache: High-performance LRU cache for frequently accessed context
- ContextAccessController: Permission management for context access

Key Features:
- Vector-based semantic storage and search using embeddings
- Real-time context sharing with subscription-based notifications
- Intelligent caching and prefetching mechanisms
- Context versioning and conflict resolution
- Access control and permission management
- Cross-agent context coherence maintenance
- Performance-optimized context distribution
- Memory-efficient storage with compression support

Usage Example:
    ```python
    from aider.context import (
        GlobalContextStore, InterAgentContextManager,
        ContextType, ContextScope, ContextQuery
    )

    # Initialize context store
    context_store = GlobalContextStore(
        storage_backend=StorageBackend.HYBRID,
        max_memory_entries=10000,
        enable_compression=True
    )

    # Initialize context manager
    context_manager = InterAgentContextManager(
        context_store=context_store,
        default_sync_strategy=ContextSyncStrategy.IMMEDIATE
    )

    # Start the systems
    await context_store.start()
    await context_manager.start()

    # Store context
    entry_id = await context_store.store_context(
        key="project_structure",
        content={"files": ["main.py", "utils.py"], "dependencies": ["requests"]},
        context_type=ContextType.PROJECT_METADATA,
        scope=ContextScope.PROJECT,
        keywords=["python", "project", "structure"]
    )

    # Share context between agents
    await context_manager.share_context(
        agent_id="agent_1",
        context_key="project_structure",
        content=project_data,
        context_type=ContextType.PROJECT_METADATA,
        recipients=["agent_2", "agent_3"]
    )

    # Subscribe to context updates
    subscription_id = await context_manager.subscribe_to_context(
        agent_id="agent_2",
        context_types=[ContextType.CODE_STRUCTURE, ContextType.FILE_CONTENT],
        callback=lambda update: print(f"Context updated: {update.context_entry_id}")
    )
    ```

Performance Considerations:
- Supports millions of context entries with efficient vector indexing
- Optimized for low-latency context retrieval and sharing
- Memory-efficient storage with configurable compression
- Scalable architecture with distributed caching support

Security Features:
- Fine-grained access control with permission management
- Context encryption support for sensitive information
- Secure inter-agent communication with authentication
- Audit trail for all context operations
"""

__version__ = "1.0.0"
__author__ = "Aider Development Team"

# Core context storage
from .context_store import (
    # Main context store
    GlobalContextStore,

    # Context data structures
    ContextEntry,
    ContextQuery,
    ContextSearchResult,
    ContextMetrics,

    # Context enums
    ContextType,
    ContextScope,
    StorageBackend,

    # Supporting components
    VectorIndex,
    ContextCache,
)

# Context sharing and management
from .context_manager import (
    # Main context manager
    InterAgentContextManager,

    # Context sharing structures
    ContextSubscription,
    ContextUpdate,
    ContextLock,
    ContextConflict,
    ContextSharingMetrics,

    # Context sharing enums
    ContextShareMode,
    ContextUpdateType,
    ContextSyncStrategy,
    ConflictResolution,

    # Access control
    ContextAccessController,
)

# Re-export all public classes and enums
__all__ = [
    # Core context storage
    "GlobalContextStore",
    "ContextEntry",
    "ContextQuery",
    "ContextSearchResult",
    "ContextMetrics",
    "ContextType",
    "ContextScope",
    "StorageBackend",
    "VectorIndex",
    "ContextCache",

    # Context sharing and management
    "InterAgentContextManager",
    "ContextSubscription",
    "ContextUpdate",
    "ContextLock",
    "ContextConflict",
    "ContextSharingMetrics",
    "ContextShareMode",
    "ContextUpdateType",
    "ContextSyncStrategy",
    "ConflictResolution",
    "ContextAccessController",
]

# Module metadata
FRAMEWORK_NAME = "Aider Hive Context Management Framework"
FRAMEWORK_VERSION = __version__

def get_framework_info():
    """Get information about the context management framework."""
    return {
        "name": FRAMEWORK_NAME,
        "version": FRAMEWORK_VERSION,
        "components": {
            "context_store": "Advanced context storage with vector search capabilities",
            "context_manager": "Real-time context sharing and synchronization system",
            "vector_index": "Semantic similarity search using vector embeddings",
            "context_cache": "High-performance LRU cache for context entries",
            "access_controller": "Permission management for secure context access",
        },
        "features": [
            "Vector-based semantic storage and search",
            "Real-time context sharing between agents",
            "Subscription-based context notifications",
            "Context versioning and conflict resolution",
            "Access control and permission management",
            "Intelligent caching and prefetching",
            "Cross-agent context coherence",
            "Performance-optimized distribution",
            "Memory-efficient storage with compression",
            "Comprehensive audit trails",
        ],
        "supported_backends": ["memory", "disk", "redis", "vector_db", "hybrid"],
        "supported_sync_strategies": ["immediate", "batched", "periodic", "on_demand", "eventual"],
    }

def create_default_context_system(
    max_memory_entries: int = 10000,
    max_cache_size: int = 1000,
    vector_dimension: int = 512,
    enable_compression: bool = True,
    enable_encryption: bool = False,
    sync_strategy: ContextSyncStrategy = ContextSyncStrategy.IMMEDIATE,
    conflict_resolution: ConflictResolution = ConflictResolution.LAST_WRITER_WINS,
):
    """
    Create a default context management system with recommended settings.

    Args:
        max_memory_entries: Maximum entries in memory storage
        max_cache_size: Maximum entries in cache
        vector_dimension: Dimension for vector embeddings
        enable_compression: Enable content compression
        enable_encryption: Enable content encryption
        sync_strategy: Default synchronization strategy
        conflict_resolution: Default conflict resolution strategy

    Returns:
        Tuple of (GlobalContextStore, InterAgentContextManager) ready for use
    """
    # Create context store
    context_store = GlobalContextStore(
        storage_backend=StorageBackend.HYBRID,
        max_memory_entries=max_memory_entries,
        max_cache_size=max_cache_size,
        vector_dimension=vector_dimension,
        enable_compression=enable_compression,
        enable_encryption=enable_encryption,
    )

    # Create context manager
    context_manager = InterAgentContextManager(
        context_store=context_store,
        default_sync_strategy=sync_strategy,
        default_conflict_resolution=conflict_resolution,
    )

    return context_store, context_manager

# Convenience functions for common context operations
def create_file_context(
    file_path: str,
    content: str,
    language: str = "python",
    scope: ContextScope = ContextScope.PROJECT,
    metadata: dict = None
) -> dict:
    """Create a file content context payload."""
    return {
        "key": f"file:{file_path}",
        "content": {
            "path": file_path,
            "content": content,
            "language": language,
            "size": len(content),
        },
        "context_type": ContextType.FILE_CONTENT,
        "scope": scope,
        "keywords": [language, "file", "source"],
        "tags": {"file_type", language},
        "metadata": metadata or {}
    }

def create_code_structure_context(
    structure_data: dict,
    scope: ContextScope = ContextScope.PROJECT,
    metadata: dict = None
) -> dict:
    """Create a code structure context payload."""
    return {
        "key": "code_structure",
        "content": structure_data,
        "context_type": ContextType.CODE_STRUCTURE,
        "scope": scope,
        "keywords": ["structure", "architecture", "code"],
        "tags": {"structure", "code"},
        "metadata": metadata or {}
    }

def create_conversation_context(
    conversation_id: str,
    messages: list,
    scope: ContextScope = ContextScope.SESSION,
    metadata: dict = None
) -> dict:
    """Create a conversation history context payload."""
    return {
        "key": f"conversation:{conversation_id}",
        "content": {
            "conversation_id": conversation_id,
            "messages": messages,
            "message_count": len(messages),
        },
        "context_type": ContextType.CONVERSATION_HISTORY,
        "scope": scope,
        "keywords": ["conversation", "history", "chat"],
        "tags": {"conversation", "history"},
        "metadata": metadata or {}
    }

def create_git_context(
    repository_path: str,
    branch: str,
    commit_hash: str,
    changes: list = None,
    scope: ContextScope = ContextScope.PROJECT,
    metadata: dict = None
) -> dict:
    """Create a git history context payload."""
    return {
        "key": f"git:{repository_path}:{branch}",
        "content": {
            "repository_path": repository_path,
            "branch": branch,
            "commit_hash": commit_hash,
            "changes": changes or [],
        },
        "context_type": ContextType.GIT_HISTORY,
        "scope": scope,
        "keywords": ["git", "version", "control", branch],
        "tags": {"git", "vcs"},
        "metadata": metadata or {}
    }

# Context query builders
def build_semantic_query(
    query_text: str,
    embedding: list = None,
    similarity_threshold: float = 0.7,
    max_results: int = 10,
    context_types: list = None,
    scopes: list = None
) -> ContextQuery:
    """Build a semantic search query."""
    return ContextQuery(
        query_text=query_text,
        embedding=embedding,
        similarity_threshold=similarity_threshold,
        max_results=max_results,
        context_types=context_types,
        scopes=scopes,
        sort_by="relevance"
    )

def build_keyword_query(
    keywords: list,
    context_types: list = None,
    scopes: list = None,
    max_results: int = 10
) -> ContextQuery:
    """Build a keyword-based search query."""
    return ContextQuery(
        keywords=keywords,
        context_types=context_types,
        scopes=scopes,
        max_results=max_results,
        sort_by="relevance"
    )

def build_recent_context_query(
    hours_back: int = 24,
    context_types: list = None,
    scopes: list = None,
    max_results: int = 50
) -> ContextQuery:
    """Build a query for recently updated context."""
    from datetime import datetime, timedelta

    return ContextQuery(
        updated_after=datetime.utcnow() - timedelta(hours=hours_back),
        context_types=context_types,
        scopes=scopes,
        max_results=max_results,
        sort_by="updated_at",
        sort_order="desc"
    )

# Context sharing helpers
def create_broadcast_subscription(
    agent_id: str,
    callback: callable,
    context_types: list = None,
    sync_strategy: ContextSyncStrategy = ContextSyncStrategy.IMMEDIATE
) -> dict:
    """Create a broadcast subscription configuration."""
    return {
        "agent_id": agent_id,
        "patterns": ["*"],  # Listen to all
        "context_types": context_types,
        "callback": callback,
        "share_mode": ContextShareMode.BROADCAST,
        "sync_strategy": sync_strategy,
    }

def create_collaborative_subscription(
    agent_id: str,
    callback: callable,
    patterns: list = None,
    context_types: list = None,
    filter_function: callable = None
) -> dict:
    """Create a collaborative subscription configuration."""
    return {
        "agent_id": agent_id,
        "patterns": patterns or ["*"],
        "context_types": context_types,
        "callback": callback,
        "share_mode": ContextShareMode.COLLABORATIVE,
        "sync_strategy": ContextSyncStrategy.IMMEDIATE,
        "filter_function": filter_function,
    }

# Performance optimization helpers
def estimate_context_size(content: any) -> int:
    """Estimate the storage size of context content."""
    import json
    import pickle

    try:
        # Try JSON serialization first (more standard)
        json_size = len(json.dumps(content))
        return json_size
    except (TypeError, ValueError):
        try:
            # Fallback to pickle
            pickle_size = len(pickle.dumps(content))
            return pickle_size
        except Exception:
            # Final fallback to string representation
            return len(str(content))

def optimize_context_for_storage(content: any, compress: bool = True) -> tuple:
    """Optimize content for efficient storage."""
    import json
    import zlib

    # Serialize content
    if isinstance(content, (dict, list)):
        serialized = json.dumps(content, separators=(',', ':'))
    else:
        serialized = str(content)

    # Compress if requested
    if compress and len(serialized) > 1000:  # Only compress larger content
        compressed = zlib.compress(serialized.encode())
        return compressed, True

    return serialized.encode(), False

def calculate_context_relevance(entry: ContextEntry, query_terms: list) -> float:
    """Calculate relevance score for a context entry."""
    score = 0.0
    content_str = str(entry.content).lower()

    # Term frequency scoring
    for term in query_terms:
        term_lower = term.lower()
        if term_lower in content_str:
            # Count occurrences
            count = content_str.count(term_lower)
            score += count * 0.1

    # Keyword matches
    keyword_matches = len(set(query_terms).intersection(entry.keywords))
    if keyword_matches > 0:
        score += keyword_matches * 0.2

    # Recency bonus
    from datetime import datetime, timedelta
    if entry.updated_at > datetime.utcnow() - timedelta(days=7):
        score += 0.1

    # Access frequency bonus
    if entry.access_count > 0:
        score += min(0.1, entry.access_count * 0.01)

    return min(score, 1.0)
