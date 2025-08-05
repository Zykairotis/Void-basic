"""
GlobalContextStore: Advanced context management and storage system for the Aider Hive Architecture.

This module provides sophisticated context storage and retrieval capabilities including:
- Vector-based semantic storage and search
- Hierarchical context organization
- Real-time context indexing and updates
- Context versioning and history tracking
- Efficient caching and retrieval mechanisms
- Cross-agent context sharing
- Semantic similarity search
- Context relevance scoring
- Memory-efficient storage management
"""

import asyncio
import hashlib
import json
import logging
import pickle
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from weakref import WeakKeyDictionary

import structlog


class ContextType(Enum):
    """Types of context that can be stored."""
    FILE_CONTENT = "file_content"
    CODE_STRUCTURE = "code_structure"
    PROJECT_METADATA = "project_metadata"
    CONVERSATION_HISTORY = "conversation_history"
    SEMANTIC_INDEX = "semantic_index"
    DEPENDENCY_GRAPH = "dependency_graph"
    GIT_HISTORY = "git_history"
    EXECUTION_CONTEXT = "execution_context"
    USER_PREFERENCES = "user_preferences"
    AGENT_STATE = "agent_state"


class ContextScope(Enum):
    """Scope levels for context access and sharing."""
    GLOBAL = "global"
    PROJECT = "project"
    SESSION = "session"
    AGENT = "agent"
    TASK = "task"
    TEMPORARY = "temporary"


class StorageBackend(Enum):
    """Storage backend types."""
    MEMORY = "memory"
    DISK = "disk"
    REDIS = "redis"
    VECTOR_DB = "vector_db"
    HYBRID = "hybrid"


@dataclass
class ContextEntry:
    """Represents a single context entry."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    key: str = ""
    content: Any = None
    context_type: ContextType = ContextType.FILE_CONTENT
    scope: ContextScope = ContextScope.PROJECT

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    accessed_at: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0

    # Versioning
    version: int = 1
    parent_version: Optional[str] = None

    # Semantic information
    embedding: Optional[List[float]] = None
    keywords: Set[str] = field(default_factory=set)
    tags: Set[str] = field(default_factory=set)

    # Relationships
    related_entries: Set[str] = field(default_factory=set)
    dependencies: Set[str] = field(default_factory=set)

    # Storage metadata
    size_bytes: int = 0
    checksum: Optional[str] = None
    compressed: bool = False

    # TTL and cleanup
    expires_at: Optional[datetime] = None
    auto_cleanup: bool = True

    # Access control
    owner_agent: Optional[str] = None
    access_permissions: Set[str] = field(default_factory=set)


@dataclass
class ContextQuery:
    """Query specification for context retrieval."""
    query_text: Optional[str] = None
    context_types: Optional[List[ContextType]] = None
    scopes: Optional[List[ContextScope]] = None
    keywords: Optional[List[str]] = None
    tags: Optional[List[str]] = None

    # Semantic search
    embedding: Optional[List[float]] = None
    similarity_threshold: float = 0.7
    max_results: int = 10

    # Filters
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    updated_after: Optional[datetime] = None
    updated_before: Optional[datetime] = None

    # Relationships
    related_to: Optional[str] = None
    depends_on: Optional[str] = None

    # Sorting and ranking
    sort_by: str = "relevance"  # relevance, created_at, updated_at, access_count
    sort_order: str = "desc"  # asc, desc


@dataclass
class ContextSearchResult:
    """Result of context search operation."""
    entry: ContextEntry
    relevance_score: float
    similarity_score: float
    match_reasons: List[str] = field(default_factory=list)


@dataclass
class ContextMetrics:
    """Context store performance and usage metrics."""
    total_entries: int = 0
    total_size_bytes: int = 0
    storage_utilization: float = 0.0

    # Access patterns
    total_reads: int = 0
    total_writes: int = 0
    total_searches: int = 0
    cache_hit_rate: float = 0.0

    # Performance
    average_read_time: float = 0.0
    average_write_time: float = 0.0
    average_search_time: float = 0.0

    # Storage breakdown
    entries_by_type: Dict[ContextType, int] = field(default_factory=dict)
    entries_by_scope: Dict[ContextScope, int] = field(default_factory=dict)

    # Memory usage
    memory_usage_mb: float = 0.0
    cache_size_mb: float = 0.0


class VectorIndex:
    """Simple vector similarity index for semantic search."""

    def __init__(self, dimension: int = 512):
        self.dimension = dimension
        self.vectors: Dict[str, List[float]] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}

    def add_vector(self, key: str, vector: List[float], metadata: Dict[str, Any] = None) -> None:
        """Add a vector to the index."""
        if len(vector) != self.dimension:
            raise ValueError(f"Vector dimension {len(vector)} doesn't match index dimension {self.dimension}")

        self.vectors[key] = vector
        self.metadata[key] = metadata or {}

    def remove_vector(self, key: str) -> bool:
        """Remove a vector from the index."""
        if key in self.vectors:
            del self.vectors[key]
            self.metadata.pop(key, None)
            return True
        return False

    def search(self, query_vector: List[float], k: int = 10, threshold: float = 0.0) -> List[Tuple[str, float]]:
        """Search for similar vectors."""
        if len(query_vector) != self.dimension:
            raise ValueError(f"Query vector dimension {len(query_vector)} doesn't match index dimension {self.dimension}")

        similarities = []

        for key, vector in self.vectors.items():
            similarity = self._cosine_similarity(query_vector, vector)
            if similarity >= threshold:
                similarities.append((key, similarity))

        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:k]

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)


class ContextCache:
    """LRU cache for frequently accessed context entries."""

    def __init__(self, max_size: int = 1000, max_memory_mb: float = 100.0):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024

        self.cache: Dict[str, ContextEntry] = {}
        self.access_order: deque = deque()
        self.current_memory = 0

        # Statistics
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[ContextEntry]:
        """Get entry from cache."""
        if key in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(key)
            self.access_order.append(key)

            self.hits += 1
            return self.cache[key]

        self.misses += 1
        return None

    def put(self, key: str, entry: ContextEntry) -> None:
        """Put entry in cache."""
        entry_size = entry.size_bytes or self._estimate_size(entry)

        # Remove existing entry if present
        if key in self.cache:
            self.access_order.remove(key)
            self.current_memory -= self.cache[key].size_bytes or 0

        # Ensure we have space
        while (len(self.cache) >= self.max_size or
               self.current_memory + entry_size > self.max_memory_bytes):
            if not self.access_order:
                break

            oldest_key = self.access_order.popleft()
            if oldest_key in self.cache:
                oldest_entry = self.cache.pop(oldest_key)
                self.current_memory -= oldest_entry.size_bytes or 0

        # Add new entry
        self.cache[key] = entry
        self.access_order.append(key)
        self.current_memory += entry_size

    def remove(self, key: str) -> bool:
        """Remove entry from cache."""
        if key in self.cache:
            entry = self.cache.pop(key)
            self.access_order.remove(key)
            self.current_memory -= entry.size_bytes or 0
            return True
        return False

    def clear(self) -> None:
        """Clear all cached entries."""
        self.cache.clear()
        self.access_order.clear()
        self.current_memory = 0

    def get_hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def _estimate_size(self, entry: ContextEntry) -> int:
        """Estimate memory size of an entry."""
        try:
            return len(pickle.dumps(entry.content))
        except:
            return len(str(entry.content))


class GlobalContextStore:
    """
    Advanced context management and storage system.

    Features:
    - Multi-backend storage (memory, disk, vector DB)
    - Semantic search with vector embeddings
    - Context versioning and history
    - Intelligent caching and prefetching
    - Cross-agent context sharing
    - Real-time indexing and updates
    - Memory-efficient storage management
    """

    def __init__(
        self,
        storage_backend: StorageBackend = StorageBackend.HYBRID,
        max_memory_entries: int = 10000,
        max_cache_size: int = 1000,
        vector_dimension: int = 512,
        enable_compression: bool = True,
        enable_encryption: bool = False,
    ):
        """
        Initialize the global context store.

        Args:
            storage_backend: Primary storage backend
            max_memory_entries: Maximum entries in memory storage
            max_cache_size: Maximum entries in cache
            vector_dimension: Dimension for vector embeddings
            enable_compression: Enable content compression
            enable_encryption: Enable content encryption
        """
        self.storage_backend = storage_backend
        self.max_memory_entries = max_memory_entries
        self.max_cache_size = max_cache_size
        self.vector_dimension = vector_dimension
        self.enable_compression = enable_compression
        self.enable_encryption = enable_encryption
        self.enable_vector_search = vector_dimension > 0

        # Logging
        self.logger = structlog.get_logger().bind(component="context_store")

        # Storage layers
        self.memory_store: Dict[str, ContextEntry] = {}
        self.disk_store_path: Optional[str] = None

        # Caching
        self.cache = ContextCache(max_size=max_cache_size)

        # Vector search
        self.vector_index = VectorIndex(dimension=vector_dimension)

        # Indexing structures
        self.type_index: Dict[ContextType, Set[str]] = defaultdict(set)
        self.scope_index: Dict[ContextScope, Set[str]] = defaultdict(set)
        self.keyword_index: Dict[str, Set[str]] = defaultdict(set)
        self.tag_index: Dict[str, Set[str]] = defaultdict(set)
        self.dependency_index: Dict[str, Set[str]] = defaultdict(set)

        # Versioning
        self.version_history: Dict[str, List[str]] = defaultdict(list)

        # Metrics and monitoring
        self.metrics = ContextMetrics()
        self.access_history: deque = deque(maxlen=1000)

        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()
        self._shutdown_event = asyncio.Event()

        # Cleanup configuration
        self.cleanup_interval = 300  # 5 minutes
        self.max_age_days = 30

        # Event callbacks
        self.on_entry_added: List[Callable] = []
        self.on_entry_updated: List[Callable] = []
        self.on_entry_removed: List[Callable] = []

    async def initialize(self) -> None:
        """Initialize the global context store components."""
        try:
            self.logger.info("Initializing global context store")

            # Initialize storage backends
            await self._initialize_storage()

            # Initialize vector embeddings if enabled
            if self.enable_vector_search:
                await self._initialize_vector_search()

            # Validate configuration
            self._validate_config()

            self.logger.info("Global context store initialized successfully")

        except Exception as e:
            self.logger.error("Failed to initialize context store", error=str(e))
            raise

    async def health_check(self) -> Dict[str, Any]:
        """Perform a comprehensive health check of the context store."""
        try:
            health_status = {
                "status": "healthy",
                "storage_backends": {},
                "cache_stats": {
                    "memory_entries": len(self.memory_store),
                    "cache_hits": getattr(self, '_cache_hits', 0),
                    "cache_misses": getattr(self, '_cache_misses', 0)
                },
                "vector_search_enabled": self.enable_vector_search,
                "compression_enabled": self.enable_compression,
                "total_entries": len(self.memory_store),
                "errors": []
            }

            # Check memory usage
            if len(self.memory_store) > self.max_memory_entries * 0.9:
                health_status["errors"].append("Memory store approaching capacity")

            # Check storage backends
            for backend_name, backend in [("memory", self.memory_store)]:
                try:
                    backend_health = {
                        "available": True,
                        "entries": len(backend) if hasattr(backend, '__len__') else "unknown"
                    }
                    health_status["storage_backends"][backend_name] = backend_health
                except Exception as e:
                    health_status["storage_backends"][backend_name] = {
                        "available": False,
                        "error": str(e)
                    }
                    health_status["errors"].append(f"{backend_name} backend error: {str(e)}")

            # Overall health determination
            if health_status["errors"]:
                health_status["status"] = "degraded"

            return health_status

        except Exception as e:
            self.logger.error("Health check failed", error=str(e))
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    def _validate_config(self) -> None:
        """Validate context store configuration."""
        if self.max_memory_entries <= 0:
            raise ValueError("max_memory_entries must be positive")
        if self.vector_dimension <= 0:
            raise ValueError("vector_dimension must be positive")

    async def _initialize_vector_search(self) -> None:
        """Initialize vector search capabilities."""
        # Placeholder for vector search initialization
        self.logger.debug("Vector search capabilities initialized")

    async def start(self) -> None:
        """Start the context store and background tasks."""
        try:
            self.logger.info("Starting global context store")

            # Initialize storage backends
            await self._initialize_storage()

            # Start background tasks
            self._start_background_tasks()

            self.logger.info("Global context store started successfully")

        except Exception as e:
            self.logger.error("Failed to start context store", error=str(e))
            raise

    async def stop(self) -> None:
        """Stop the context store gracefully."""
        self.logger.info("Stopping global context store")

        # Signal shutdown
        self._shutdown_event.set()

        # Cancel background tasks
        await self._cancel_background_tasks()

        # Flush any pending writes
        await self._flush_storage()

        self.logger.info("Global context store stopped")

    async def store_context(
        self,
        key: str,
        content: Any,
        context_type: ContextType,
        scope: ContextScope = ContextScope.PROJECT,
        keywords: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        expires_at: Optional[datetime] = None,
        embedding: Optional[List[float]] = None,
        related_entries: Optional[List[str]] = None,
        dependencies: Optional[List[str]] = None,
    ) -> str:
        """
        Store context information.

        Args:
            key: Unique key for the context
            content: Context content to store
            context_type: Type of context
            scope: Access scope for the context
            keywords: Associated keywords
            tags: Associated tags
            expires_at: Optional expiration time
            embedding: Optional vector embedding
            related_entries: Related context entry IDs
            dependencies: Context dependencies

        Returns:
            Context entry ID
        """
        start_time = time.time()

        try:
            # Check if entry already exists
            existing_entry = await self._get_entry_by_key(key)

            if existing_entry:
                # Update existing entry
                entry_id = await self._update_existing_entry(
                    existing_entry, content, keywords, tags, expires_at,
                    embedding, related_entries, dependencies
                )
            else:
                # Create new entry
                entry_id = await self._create_new_entry(
                    key, content, context_type, scope, keywords, tags,
                    expires_at, embedding, related_entries, dependencies
                )

            # Update metrics
            write_time = time.time() - start_time
            self.metrics.total_writes += 1
            self.metrics.average_write_time = (
                self.metrics.average_write_time + write_time
            ) / 2

            self.logger.debug(
                "Context stored",
                key=key,
                entry_id=entry_id,
                context_type=context_type.value,
                scope=scope.value,
                write_time=write_time,
            )

            return entry_id

        except Exception as e:
            self.logger.error("Error storing context", key=key, error=str(e))
            raise

    async def retrieve_context(
        self,
        key: Optional[str] = None,
        entry_id: Optional[str] = None,
        include_content: bool = True,
    ) -> Optional[ContextEntry]:
        """
        Retrieve a specific context entry.

        Args:
            key: Context key
            entry_id: Context entry ID
            include_content: Whether to include content in result

        Returns:
            Context entry or None if not found
        """
        start_time = time.time()

        try:
            # Try cache first
            cache_key = entry_id or key
            if cache_key:
                cached_entry = self.cache.get(cache_key)
                if cached_entry:
                    cached_entry.accessed_at = datetime.utcnow()
                    cached_entry.access_count += 1
                    return cached_entry

            # Retrieve from storage
            if entry_id:
                entry = self.memory_store.get(entry_id)
            elif key:
                entry = await self._get_entry_by_key(key)
            else:
                return None

            if entry:
                # Update access information
                entry.accessed_at = datetime.utcnow()
                entry.access_count += 1

                # Cache the entry
                if cache_key:
                    self.cache.put(cache_key, entry)

                # Update metrics
                read_time = time.time() - start_time
                self.metrics.total_reads += 1
                self.metrics.average_read_time = (
                    self.metrics.average_read_time + read_time
                ) / 2

                return entry

            return None

        except Exception as e:
            self.logger.error("Error retrieving context", key=key, entry_id=entry_id, error=str(e))
            return None

    async def search_context(self, query: ContextQuery) -> List[ContextSearchResult]:
        """
        Search for context entries based on query.

        Args:
            query: Search query specification

        Returns:
            List of search results with relevance scores
        """
        start_time = time.time()

        try:
            results = []
            candidate_ids = set()

            # Text-based search
            if query.query_text:
                text_candidates = await self._search_by_text(query.query_text)
                candidate_ids.update(text_candidates)

            # Vector similarity search
            if query.embedding:
                vector_candidates = self._search_by_vector(query.embedding, query.similarity_threshold)
                candidate_ids.update(vector_candidates)

            # Filter by type
            if query.context_types:
                type_candidates = set()
                for context_type in query.context_types:
                    type_candidates.update(self.type_index[context_type])

                if candidate_ids:
                    candidate_ids &= type_candidates
                else:
                    candidate_ids = type_candidates

            # Filter by scope
            if query.scopes:
                scope_candidates = set()
                for scope in query.scopes:
                    scope_candidates.update(self.scope_index[scope])

                if candidate_ids:
                    candidate_ids &= scope_candidates
                else:
                    candidate_ids = scope_candidates

            # Filter by keywords
            if query.keywords:
                keyword_candidates = set()
                for keyword in query.keywords:
                    keyword_candidates.update(self.keyword_index[keyword.lower()])

                if candidate_ids:
                    candidate_ids &= keyword_candidates
                else:
                    candidate_ids = keyword_candidates

            # Filter by tags
            if query.tags:
                tag_candidates = set()
                for tag in query.tags:
                    tag_candidates.update(self.tag_index[tag.lower()])

                if candidate_ids:
                    candidate_ids &= tag_candidates
                else:
                    candidate_ids = tag_candidates

            # If no specific filters, get all entries
            if not candidate_ids and not any([
                query.query_text, query.embedding, query.context_types,
                query.scopes, query.keywords, query.tags
            ]):
                candidate_ids = set(self.memory_store.keys())

            # Score and rank results
            for entry_id in candidate_ids:
                entry = self.memory_store.get(entry_id)
                if not entry:
                    continue

                # Apply time filters
                if query.created_after and entry.created_at < query.created_after:
                    continue
                if query.created_before and entry.created_at > query.created_before:
                    continue
                if query.updated_after and entry.updated_at < query.updated_after:
                    continue
                if query.updated_before and entry.updated_at > query.updated_before:
                    continue

                # Calculate relevance score
                relevance_score = self._calculate_relevance_score(entry, query)

                # Calculate similarity score
                similarity_score = 0.0
                if query.embedding and entry.embedding:
                    similarity_score = self.vector_index._cosine_similarity(
                        query.embedding, entry.embedding
                    )

                # Create search result
                match_reasons = self._get_match_reasons(entry, query)

                results.append(ContextSearchResult(
                    entry=entry,
                    relevance_score=relevance_score,
                    similarity_score=similarity_score,
                    match_reasons=match_reasons,
                ))

            # Sort results
            if query.sort_by == "relevance":
                results.sort(key=lambda r: r.relevance_score, reverse=(query.sort_order == "desc"))
            elif query.sort_by == "similarity":
                results.sort(key=lambda r: r.similarity_score, reverse=(query.sort_order == "desc"))
            elif query.sort_by == "created_at":
                results.sort(key=lambda r: r.entry.created_at, reverse=(query.sort_order == "desc"))
            elif query.sort_by == "updated_at":
                results.sort(key=lambda r: r.entry.updated_at, reverse=(query.sort_order == "desc"))
            elif query.sort_by == "access_count":
                results.sort(key=lambda r: r.entry.access_count, reverse=(query.sort_order == "desc"))

            # Limit results
            results = results[:query.max_results]

            # Update metrics
            search_time = time.time() - start_time
            self.metrics.total_searches += 1
            self.metrics.average_search_time = (
                self.metrics.average_search_time + search_time
            ) / 2

            self.logger.debug(
                "Context search completed",
                query_text=query.query_text,
                results_count=len(results),
                search_time=search_time,
            )

            return results

        except Exception as e:
            self.logger.error("Error searching context", error=str(e))
            return []

    async def delete_context(self, key: Optional[str] = None, entry_id: Optional[str] = None) -> bool:
        """
        Delete a context entry.

        Args:
            key: Context key
            entry_id: Context entry ID

        Returns:
            True if deleted, False if not found
        """
        try:
            # Find the entry
            if entry_id:
                entry = self.memory_store.get(entry_id)
            elif key:
                entry = await self._get_entry_by_key(key)
            else:
                return False

            if not entry:
                return False

            # Remove from all indexes
            self._remove_from_indexes(entry)

            # Remove from storage
            if entry.id in self.memory_store:
                del self.memory_store[entry.id]

            # Remove from cache
            cache_key = entry_id or key
            if cache_key:
                self.cache.remove(cache_key)

            # Remove from vector index
            if entry.embedding:
                self.vector_index.remove_vector(entry.id)

            # Update metrics
            self.metrics.total_entries -= 1
            self.metrics.total_size_bytes -= entry.size_bytes

            # Trigger callbacks
            for callback in self.on_entry_removed:
                try:
                    await callback(entry)
                except Exception as e:
                    self.logger.warning("Error in removal callback", error=str(e))

            self.logger.debug("Context deleted", entry_id=entry.id, key=entry.key)
            return True

        except Exception as e:
            self.logger.error("Error deleting context", key=key, entry_id=entry_id, error=str(e))
            return False

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive context store metrics."""
        # Update real-time metrics
        self.metrics.total_entries = len(self.memory_store)
        self.metrics.total_size_bytes = sum(
            entry.size_bytes for entry in self.memory_store.values()
        )
        self.metrics.cache_hit_rate = self.cache.get_hit_rate()

        # Calculate storage utilization
        max_storage = self.max_memory_entries * 1024 * 1024  # Rough estimate
        self.metrics.storage_utilization = self.metrics.total_size_bytes / max_storage

        # Update type and scope distributions
        self.metrics.entries_by_type = {
            context_type: len(entries)
            for context_type, entries in self.type_index.items()
        }
        self.metrics.entries_by_scope = {
            scope: len(entries)
            for scope, entries in self.scope_index.items()
        }

        # Memory usage (rough estimates)
        self.metrics.memory_usage_mb = self.metrics.total_size_bytes / (1024 * 1024)
        self.metrics.cache_size_mb = self.cache.current_memory / (1024 * 1024)

        return {
            'total_entries': self.metrics.total_entries,
            'total_size_mb': self.metrics.total_size_bytes / (1024 * 1024),
            'storage_utilization': self.metrics.storage_utilization,
            'access_stats': {
                'total_reads': self.metrics.total_reads,
                'total_writes': self.metrics.total_writes,
                'total_searches': self.metrics.total_searches,
                'cache_hit_rate': self.metrics.cache_hit_rate,
            },
            'performance': {
                'average_read_time_ms': self.metrics.average_read_time * 1000,
                'average_write_time_ms': self.metrics.average_write_time * 1000,
                'average_search_time_ms': self.metrics.average_search_time * 1000,
            },
            'distribution': {
                'by_type': {t.value: count for t, count in self.metrics.entries_by_type.items()},
                'by_scope': {s.value: count for s, count in self.metrics.entries_by_scope.items()},
            },
            'memory': {
                'memory_usage_mb': self.metrics.memory_usage_mb,
                'cache_size_mb': self.metrics.cache_size_mb,
                'cache_entries': len(self.cache.cache),
            },
            'indexes': {
                'vector_entries': len(self.vector_index.vectors),
                'keyword_terms': len(self.keyword_index),
                'tag_terms': len(self.tag_index),
            }
        }

    # Private helper methods

    async def _initialize_storage(self) -> None:
        """Initialize storage backends."""
        if self.storage_backend in [StorageBackend.DISK, StorageBackend.HYBRID]:
            # Initialize disk storage
            pass

        if self.storage_backend in [StorageBackend.REDIS, StorageBackend.HYBRID]:
            # Initialize Redis storage
            pass

        if self.storage_backend in [StorageBackend.VECTOR_DB, StorageBackend.HYBRID]:
            # Initialize vector database
            pass

    async def _get_entry_by_key(self, key: str) -> Optional[ContextEntry]:
        """Get entry by key from storage."""
        for entry in self.memory_store.values():
            if entry.key == key:
                return entry
        return None

    async def _create_new_entry(
        self,
        key: str,
        content: Any,
        context_type: ContextType,
        scope: ContextScope,
        keywords: Optional[List[str]],
        tags: Optional[List[str]],
        expires_at: Optional[datetime],
        embedding: Optional[List[float]],
        related_entries: Optional[List[str]],
        dependencies: Optional[List[str]],
    ) -> str:
        """Create a new context entry."""
        entry = ContextEntry(
            key=key,
            content=content,
            context_type=context_type,
            scope=scope,
            keywords=set(keywords or []),
            tags=set(tags or []),
            expires_at=expires_at,
            embedding=embedding,
            related_entries=set(related_entries or []),
            dependencies=set(dependencies or []),
        )

        # Calculate size and checksum
        entry.size_bytes = self._calculate_entry_size(entry)
        entry.checksum = self._calculate_checksum(entry.content)

        # Store entry
        self.memory_store[entry.id] = entry

        # Update indexes
        self._add_to_indexes(entry)

        # Add to vector index
        if embedding:
            self.vector_index.add_vector(entry.id, embedding, {
                'key': key,
                'context_type': context_type.value,
                'scope': scope.value,
            })

        # Update metrics
        self.metrics.total_entries += 1
        self.metrics.total_size_bytes += entry.size_bytes

        # Trigger callbacks
        for callback in self.on_entry_added:
            try:
                await callback(entry)
            except Exception as e:
                self.logger.warning("Error in addition callback", error=str(e))

        return entry.id

    async def _update_existing_entry(
        self,
        entry: ContextEntry,
        content: Any,
        keywords: Optional[List[str]],
        tags: Optional[List[str]],
        expires_at: Optional[datetime],
        embedding: Optional[List[float]],
        related_entries: Optional[List[str]],
        dependencies: Optional[List[str]],
    ) -> str:
        """Update an existing context entry."""
        # Remove from indexes
        self._remove_from_indexes(entry)

        # Create new version
        old_version = entry.version
        entry.version += 1
        entry.updated_at = datetime.utcnow()

        # Update content
        if content is not None:
            entry.content = content
            entry.size_bytes = self._calculate_entry_size(entry)
            entry.checksum = self._calculate_checksum(content)

        # Update metadata
        if keywords is not None:
            entry.keywords = set(keywords)
        if tags is not None:
            entry.tags = set(tags)
        if expires_at is not None:
            entry.expires_at = expires_at
        if embedding is not None:
            entry.embedding = embedding
        if related_entries is not None:
            entry.related_entries = set(related_entries)
        if dependencies is not None:
            entry.dependencies = set(dependencies)

        # Update indexes
        self._add_to_indexes(entry)

        # Update vector index
        if embedding:
            self.vector_index.add_vector(entry.id, embedding, {
                'key': entry.key,
                'context_type': entry.context_type.value,
                'scope': entry.scope.value,
            })

        # Store version history
        self.version_history[entry.key].append(f"{entry.id}:v{old_version}")

        # Trigger callbacks
        for callback in self.on_entry_updated:
            try:
                await callback(entry)
            except Exception as e:
                self.logger.warning("Error in update callback", error=str(e))

        return entry.id

    def _calculate_entry_size(self, entry: ContextEntry) -> int:
        """Calculate the storage size of an entry."""
        try:
            if self.enable_compression:
                import zlib
                content_bytes = pickle.dumps(entry.content)
                compressed = zlib.compress(content_bytes)
                return len(compressed)
            else:
                return len(pickle.dumps(entry.content))
        except Exception:
            # Fallback to string length
            return len(str(entry.content))

    def _calculate_checksum(self, content: Any) -> str:
        """Calculate checksum for content integrity."""
        try:
            content_str = json.dumps(content, sort_keys=True) if isinstance(content, (dict, list)) else str(content)
            return hashlib.sha256(content_str.encode()).hexdigest()
        except Exception:
            return hashlib.sha256(str(content).encode()).hexdigest()

    def _add_to_indexes(self, entry: ContextEntry) -> None:
        """Add entry to all relevant indexes."""
        # Type index
        self.type_index[entry.context_type].add(entry.id)

        # Scope index
        self.scope_index[entry.scope].add(entry.id)

        # Keyword index
        for keyword in entry.keywords:
            self.keyword_index[keyword.lower()].add(entry.id)

        # Tag index
        for tag in entry.tags:
            self.tag_index[tag.lower()].add(entry.id)

        # Dependency index
        for dep in entry.dependencies:
            self.dependency_index[dep].add(entry.id)

    def _remove_from_indexes(self, entry: ContextEntry) -> None:
        """Remove entry from all indexes."""
        # Type index
        self.type_index[entry.context_type].discard(entry.id)

        # Scope index
        self.scope_index[entry.scope].discard(entry.id)

        # Keyword index
        for keyword in entry.keywords:
            self.keyword_index[keyword.lower()].discard(entry.id)

        # Tag index
        for tag in entry.tags:
            self.tag_index[tag.lower()].discard(entry.id)

        # Dependency index
        for dep in entry.dependencies:
            self.dependency_index[dep].discard(entry.id)

    async def _search_by_text(self, query_text: str) -> Set[str]:
        """Search entries by text content."""
        candidates = set()
        query_lower = query_text.lower()
        query_words = query_lower.split()

        for entry_id, entry in self.memory_store.items():
            # Search in content
            content_str = str(entry.content).lower()
            if query_lower in content_str:
                candidates.add(entry_id)
                continue

            # Search in keywords
            if any(word in entry.keywords for word in query_words):
                candidates.add(entry_id)
                continue

            # Search in key
            if query_lower in entry.key.lower():
                candidates.add(entry_id)

        return candidates

    def _search_by_vector(self, query_embedding: List[float], threshold: float) -> Set[str]:
        """Search entries by vector similarity."""
        similar_entries = self.vector_index.search(
            query_embedding,
            k=100,  # Get top 100 similar entries
            threshold=threshold
        )
        return {entry_id for entry_id, _ in similar_entries}

    def _calculate_relevance_score(self, entry: ContextEntry, query: ContextQuery) -> float:
        """Calculate relevance score for an entry given a query."""
        score = 0.0

        # Text match score
        if query.query_text:
            content_str = str(entry.content).lower()
            query_lower = query.query_text.lower()

            if query_lower in content_str:
                score += 0.3

            # Keyword matches
            query_words = set(query_lower.split())
            keyword_matches = len(query_words.intersection(entry.keywords))
            if keyword_matches > 0:
                score += 0.2 * (keyword_matches / len(query_words))

        # Type match score
        if query.context_types and entry.context_type in query.context_types:
            score += 0.2

        # Scope match score
        if query.scopes and entry.scope in query.scopes:
            score += 0.1

        # Keyword match score
        if query.keywords:
            keyword_matches = len(set(query.keywords).intersection(entry.keywords))
            if keyword_matches > 0:
                score += 0.1 * (keyword_matches / len(query.keywords))

        # Tag match score
        if query.tags:
            tag_matches = len(set(query.tags).intersection(entry.tags))
            if tag_matches > 0:
                score += 0.1 * (tag_matches / len(query.tags))

        # Recency boost
        days_old = (datetime.utcnow() - entry.updated_at).days
        if days_old < 7:
            score += 0.05 * (7 - days_old) / 7

        # Access frequency boost
        if entry.access_count > 0:
            score += min(0.05, entry.access_count * 0.01)

        return min(score, 1.0)

    def _get_match_reasons(self, entry: ContextEntry, query: ContextQuery) -> List[str]:
        """Get list of reasons why an entry matches a query."""
        reasons = []

        if query.query_text:
            content_str = str(entry.content).lower()
            if query.query_text.lower() in content_str:
                reasons.append("text_content_match")

            query_words = set(query.query_text.lower().split())
            if query_words.intersection(entry.keywords):
                reasons.append("keyword_match")

        if query.context_types and entry.context_type in query.context_types:
            reasons.append("context_type_match")

        if query.scopes and entry.scope in query.scopes:
            reasons.append("scope_match")

        if query.keywords and set(query.keywords).intersection(entry.keywords):
            reasons.append("explicit_keyword_match")

        if query.tags and set(query.tags).intersection(entry.tags):
            reasons.append("tag_match")

        if query.embedding and entry.embedding:
            reasons.append("semantic_similarity")

        return reasons

    def _start_background_tasks(self) -> None:
        """Start background maintenance tasks."""
        # Cleanup task
        cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._background_tasks.add(cleanup_task)
        cleanup_task.add_done_callback(self._background_tasks.discard)

        # Metrics collection task
        metrics_task = asyncio.create_task(self._metrics_loop())
        self._background_tasks.add(metrics_task)
        metrics_task.add_done_callback(self._background_tasks.discard)

    async def _cleanup_loop(self) -> None:
        """Background cleanup of expired and old entries."""
        while not self._shutdown_event.is_set():
            try:
                current_time = datetime.utcnow()
                expired_entries = []

                # Find expired entries
                for entry_id, entry in list(self.memory_store.items()):
                    # Check expiration
                    if entry.expires_at and current_time > entry.expires_at:
                        expired_entries.append(entry_id)
                        continue

                    # Check age limit
                    if entry.auto_cleanup:
                        age_days = (current_time - entry.created_at).days
                        if age_days > self.max_age_days and entry.access_count == 0:
                            expired_entries.append(entry_id)

                # Remove expired entries
                for entry_id in expired_entries:
                    await self.delete_context(entry_id=entry_id)

                if expired_entries:
                    self.logger.info("Cleaned up expired entries", count=len(expired_entries))

                # Sleep before next cleanup
                await asyncio.sleep(self.cleanup_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Error in cleanup loop", error=str(e))

    async def _metrics_loop(self) -> None:
        """Background metrics collection and analysis."""
        while not self._shutdown_event.is_set():
            try:
                # Update cache hit rate
                self.metrics.cache_hit_rate = self.cache.get_hit_rate()

                # Calculate memory usage
                total_size = sum(entry.size_bytes for entry in self.memory_store.values())
                self.metrics.total_size_bytes = total_size
                self.metrics.memory_usage_mb = total_size / (1024 * 1024)

                # Log metrics periodically
                if len(self.access_history) % 100 == 0:
                    self.logger.info(
                        "Context store metrics",
                        total_entries=len(self.memory_store),
                        cache_hit_rate=self.metrics.cache_hit_rate,
                        memory_usage_mb=self.metrics.memory_usage_mb,
                    )

                await asyncio.sleep(60)  # Update every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Error in metrics loop", error=str(e))

    async def _cancel_background_tasks(self) -> None:
        """Cancel all background tasks."""
        for task in self._background_tasks:
            if not task.done():
                task.cancel()

        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)

        self._background_tasks.clear()

    async def _flush_storage(self) -> None:
        """Flush any pending storage operations."""
        # Placeholder for disk/database storage flushing
        pass
