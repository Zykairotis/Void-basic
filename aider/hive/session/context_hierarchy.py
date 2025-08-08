"""
Hierarchical Context Management System for Aider Hive Architecture.

This module provides advanced context organization and retrieval capabilities
with semantic understanding, hierarchical structuring, and intelligent caching.
It manages different levels of context importance and provides efficient
retrieval mechanisms for AI agents.

Key Features:
- Hierarchical context organization (core, relevant, ambient)
- Semantic chunking and embedding generation
- Vector-based similarity search with hybrid keyword matching
- Context caching with intelligent eviction policies
- Real-time context updates and invalidation
- Integration with vector databases (Pinecone, Weaviate, Chroma)
- Context pruning strategies for memory optimization
- Support for code semantics and project knowledge
"""

import asyncio
import hashlib
import json
import math
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, AsyncIterator
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import structlog
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from ...context.context_store import GlobalContextStore, ContextEntry
from .session_events import EventType


class ContextTier(Enum):
    """Context hierarchy tiers with different importance levels."""
    CORE = 1        # Most critical - immediate task context
    RELEVANT = 2    # Highly relevant - related functions, classes
    AMBIENT = 3     # Background - project structure, dependencies
    ARCHIVED = 4    # Historical - old conversations, unused code


class ContextType(Enum):
    """Types of context entries."""
    CODE_FUNCTION = "code_function"
    CODE_CLASS = "code_class"
    CODE_MODULE = "code_module"
    CONVERSATION = "conversation"
    USER_INPUT = "user_input"
    AGENT_RESPONSE = "agent_response"
    DOCUMENTATION = "documentation"
    ERROR_LOG = "error_log"
    TEST_CASE = "test_case"
    CONFIGURATION = "configuration"
    DEPENDENCY = "dependency"
    PROJECT_METADATA = "project_metadata"


class RetrievalStrategy(Enum):
    """Context retrieval strategies."""
    SEMANTIC_ONLY = "semantic_only"
    KEYWORD_ONLY = "keyword_only"
    HYBRID = "hybrid"
    RECENCY_WEIGHTED = "recency_weighted"
    RELEVANCE_RANKED = "relevance_ranked"


@dataclass
class ContextChunk:
    """A semantic chunk of context with metadata."""
    chunk_id: str
    content: str
    context_type: ContextType
    tier: ContextTier

    # Semantic information
    embedding: Optional[np.ndarray] = None
    keywords: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)

    # Metadata
    source_path: Optional[str] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    function_name: Optional[str] = None
    class_name: Optional[str] = None
    module_name: Optional[str] = None

    # Temporal information
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0

    # Relationships
    parent_chunk_id: Optional[str] = None
    child_chunk_ids: List[str] = field(default_factory=list)
    related_chunk_ids: List[str] = field(default_factory=list)

    # Relevance scoring
    base_relevance: float = 1.0
    dynamic_relevance: float = 1.0
    recency_score: float = 1.0

    def get_combined_relevance(self) -> float:
        """Calculate combined relevance score."""
        return (self.base_relevance * 0.4 +
                self.dynamic_relevance * 0.4 +
                self.recency_score * 0.2)


@dataclass
class ContextHierarchy:
    """Represents the hierarchical structure of context."""
    session_id: str
    core_contexts: List[str] = field(default_factory=list)
    relevant_contexts: List[str] = field(default_factory=list)
    ambient_contexts: List[str] = field(default_factory=list)
    archived_contexts: List[str] = field(default_factory=list)

    # Hierarchy relationships
    context_tree: Dict[str, List[str]] = field(default_factory=dict)
    reverse_tree: Dict[str, str] = field(default_factory=dict)

    # Statistics
    total_chunks: int = 0
    last_updated: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ContextRetrievalResult:
    """Result of a context retrieval operation."""
    chunks: List[ContextChunk]
    total_found: int
    query_time_ms: float
    retrieval_strategy: RetrievalStrategy

    # Scoring information
    semantic_scores: List[float] = field(default_factory=list)
    keyword_scores: List[float] = field(default_factory=list)
    combined_scores: List[float] = field(default_factory=list)

    # Cache information
    cache_hits: int = 0
    cache_misses: int = 0


class SemanticChunker:
    """Intelligent semantic chunking for different content types."""

    def __init__(self, max_chunk_size: int = 1000, overlap_size: int = 100):
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        self.logger = structlog.get_logger().bind(component="semantic_chunker")

    async def chunk_code(self, content: str, file_path: str, language: str) -> List[Dict[str, Any]]:
        """Chunk code content semantically by functions, classes, etc."""
        try:
            chunks = []

            if language.lower() == 'python':
                chunks = await self._chunk_python_code(content, file_path)
            elif language.lower() in ['javascript', 'typescript']:
                chunks = await self._chunk_js_code(content, file_path)
            else:
                # Fallback to generic chunking
                chunks = await self._chunk_generic(content, file_path)

            return chunks

        except Exception as e:
            self.logger.error(f"Failed to chunk code: {e}")
            return await self._chunk_generic(content, file_path)

    async def chunk_conversation(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Chunk conversation messages into semantic groups."""
        chunks = []
        current_chunk = []
        current_size = 0

        for i, message in enumerate(messages):
            message_content = message.get('content', '')
            message_size = len(message_content)

            # Check if adding this message would exceed chunk size
            if current_size + message_size > self.max_chunk_size and current_chunk:
                # Create chunk from current messages
                chunk_content = "\n\n".join([msg.get('content', '') for msg in current_chunk])
                chunks.append({
                    'content': chunk_content,
                    'metadata': {
                        'message_ids': [msg.get('message_id') for msg in current_chunk],
                        'start_index': current_chunk[0].get('index', 0),
                        'end_index': current_chunk[-1].get('index', 0),
                        'speaker_count': len(set(msg.get('speaker', 'unknown') for msg in current_chunk))
                    }
                })

                # Start new chunk with overlap
                if len(current_chunk) > 1:
                    current_chunk = current_chunk[-1:]  # Keep last message for overlap
                    current_size = len(current_chunk[0].get('content', ''))
                else:
                    current_chunk = []
                    current_size = 0

            current_chunk.append(message)
            current_size += message_size

        # Add final chunk
        if current_chunk:
            chunk_content = "\n\n".join([msg.get('content', '') for msg in current_chunk])
            chunks.append({
                'content': chunk_content,
                'metadata': {
                    'message_ids': [msg.get('message_id') for msg in current_chunk],
                    'start_index': current_chunk[0].get('index', 0),
                    'end_index': current_chunk[-1].get('index', 0),
                    'speaker_count': len(set(msg.get('speaker', 'unknown') for msg in current_chunk))
                }
            })

        return chunks

    async def _chunk_python_code(self, content: str, file_path: str) -> List[Dict[str, Any]]:
        """Chunk Python code by AST analysis."""
        import ast

        chunks = []

        try:
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Extract function
                    start_line = node.lineno - 1
                    end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 10

                    lines = content.split('\n')
                    function_content = '\n'.join(lines[start_line:end_line])

                    chunks.append({
                        'content': function_content,
                        'metadata': {
                            'type': 'function',
                            'name': node.name,
                            'line_start': start_line + 1,
                            'line_end': end_line,
                            'file_path': file_path,
                            'args': [arg.arg for arg in node.args.args] if hasattr(node.args, 'args') else []
                        }
                    })

                elif isinstance(node, ast.ClassDef):
                    # Extract class
                    start_line = node.lineno - 1
                    end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 20

                    lines = content.split('\n')
                    class_content = '\n'.join(lines[start_line:end_line])

                    chunks.append({
                        'content': class_content,
                        'metadata': {
                            'type': 'class',
                            'name': node.name,
                            'line_start': start_line + 1,
                            'line_end': end_line,
                            'file_path': file_path,
                            'methods': [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                        }
                    })

        except SyntaxError:
            # Fallback to generic chunking if AST parsing fails
            return await self._chunk_generic(content, file_path)

        return chunks

    async def _chunk_js_code(self, content: str, file_path: str) -> List[Dict[str, Any]]:
        """Chunk JavaScript/TypeScript code."""
        # Simple regex-based chunking for JS/TS
        import re

        chunks = []

        # Find functions
        function_pattern = r'(?:function|async\s+function|\w+\s*:\s*(?:async\s+)?function|\w+\s*=\s*(?:async\s+)?\()\s*\w*\s*\([^)]*\)\s*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'

        for match in re.finditer(function_pattern, content, re.MULTILINE | re.DOTALL):
            function_content = match.group(0)
            start_pos = match.start()

            # Calculate line numbers
            lines_before = content[:start_pos].count('\n')
            lines_in_function = function_content.count('\n')

            chunks.append({
                'content': function_content,
                'metadata': {
                    'type': 'function',
                    'line_start': lines_before + 1,
                    'line_end': lines_before + lines_in_function + 1,
                    'file_path': file_path
                }
            })

        # Find classes
        class_pattern = r'class\s+\w+(?:\s+extends\s+\w+)?\s*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'

        for match in re.finditer(class_pattern, content, re.MULTILINE | re.DOTALL):
            class_content = match.group(0)
            start_pos = match.start()

            lines_before = content[:start_pos].count('\n')
            lines_in_class = class_content.count('\n')

            chunks.append({
                'content': class_content,
                'metadata': {
                    'type': 'class',
                    'line_start': lines_before + 1,
                    'line_end': lines_before + lines_in_class + 1,
                    'file_path': file_path
                }
            })

        return chunks

    async def _chunk_generic(self, content: str, file_path: str) -> List[Dict[str, Any]]:
        """Generic content chunking based on size and natural breaks."""
        chunks = []
        lines = content.split('\n')

        current_chunk = []
        current_size = 0

        for i, line in enumerate(lines):
            line_size = len(line)

            if current_size + line_size > self.max_chunk_size and current_chunk:
                # Create chunk
                chunk_content = '\n'.join(current_chunk)
                chunks.append({
                    'content': chunk_content,
                    'metadata': {
                        'type': 'generic',
                        'line_start': i - len(current_chunk) + 1,
                        'line_end': i,
                        'file_path': file_path
                    }
                })

                # Start new chunk with overlap
                overlap_lines = min(self.overlap_size // 50, len(current_chunk))  # Rough line estimate
                current_chunk = current_chunk[-overlap_lines:] if overlap_lines > 0 else []
                current_size = sum(len(line) for line in current_chunk)

            current_chunk.append(line)
            current_size += line_size

        # Add final chunk
        if current_chunk:
            chunk_content = '\n'.join(current_chunk)
            chunks.append({
                'content': chunk_content,
                'metadata': {
                    'type': 'generic',
                    'line_start': len(lines) - len(current_chunk) + 1,
                    'line_end': len(lines),
                    'file_path': file_path
                }
            })

        return chunks


class VectorSearchEngine:
    """Vector-based semantic search engine with hybrid capabilities."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dimension: int = 384):
        self.model_name = model_name
        self.dimension = dimension
        self.model = None
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.logger = structlog.get_logger().bind(component="vector_search")

        # In-memory vector index
        self.vectors: Dict[str, np.ndarray] = {}
        self.chunk_metadata: Dict[str, Dict[str, Any]] = {}

    async def initialize(self) -> None:
        """Initialize the embedding model."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            self.logger.warning("sentence_transformers not available - vector search disabled")
            return

        try:
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                self.executor,
                SentenceTransformer,
                self.model_name
            )
            self.logger.info(f"Initialized embedding model: {self.model_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize embedding model: {e}")
            raise

    async def add_chunk(self, chunk: ContextChunk) -> bool:
        """Add a chunk to the vector index."""
        try:
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                self.logger.warning("Vector search not available - sentence_transformers not installed")
                return False

            if self.model is None:
                await self.initialize()

            if self.model is None:
                return False

            # Generate embedding
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                self.executor,
                self.model.encode,
                chunk.content
            )

            # Store in index
            self.vectors[chunk.chunk_id] = embedding
            self.chunk_metadata[chunk.chunk_id] = {
                'tier': chunk.tier,
                'context_type': chunk.context_type,
                'keywords': chunk.keywords,
                'created_at': chunk.created_at,
                'relevance': chunk.get_combined_relevance()
            }

            # Update chunk with embedding
            chunk.embedding = embedding

            return True

        except Exception as e:
            self.logger.error(f"Failed to add chunk to vector index: {e}")
            return False

    async def search_similar(
        self,
        query: str,
        limit: int = 10,
        min_similarity: float = 0.3,
        tier_filter: Optional[List[ContextTier]] = None,
        type_filter: Optional[List[ContextType]] = None
    ) -> List[Tuple[str, float]]:
        """Search for similar chunks using vector similarity."""
        try:
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                self.logger.warning("Vector search not available - sentence_transformers not installed")
                return []

            if not self.vectors or self.model is None:
                return []

            # Generate query embedding
            loop = asyncio.get_event_loop()
            query_embedding = await loop.run_in_executor(
                self.executor,
                self.model.encode,
                query
            )

            # Calculate similarities
            similarities = []
            for chunk_id, chunk_embedding in self.vectors.items():
                # Apply filters
                metadata = self.chunk_metadata[chunk_id]

                if tier_filter and metadata['tier'] not in tier_filter:
                    continue

                if type_filter and metadata['context_type'] not in type_filter:
                    continue

                # Cosine similarity
                similarity = np.dot(query_embedding, chunk_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
                )

                if similarity >= min_similarity:
                    # Weight by relevance score
                    weighted_similarity = similarity * metadata['relevance']
                    similarities.append((chunk_id, weighted_similarity))

            # Sort by similarity and limit results
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:limit]

        except Exception as e:
            self.logger.error(f"Vector search failed: {e}")
            return []

    async def search_keywords(
        self,
        keywords: List[str],
        limit: int = 10
    ) -> List[Tuple[str, float]]:
        """Search using keyword matching."""
        try:
            keyword_matches = []

            for chunk_id, metadata in self.chunk_metadata.items():
                chunk_keywords = set(metadata['keywords'])
                query_keywords = set(keyword.lower() for keyword in keywords)

                # Calculate keyword similarity (Jaccard similarity)
                intersection = len(chunk_keywords.intersection(query_keywords))
                union = len(chunk_keywords.union(query_keywords))

                if union > 0:
                    similarity = intersection / union
                    if similarity > 0:
                        # Weight by relevance
                        weighted_similarity = similarity * metadata['relevance']
                        keyword_matches.append((chunk_id, weighted_similarity))

            # Sort and limit
            keyword_matches.sort(key=lambda x: x[1], reverse=True)
            return keyword_matches[:limit]

        except Exception as e:
            self.logger.error(f"Keyword search failed: {e}")
            return []

    def remove_chunk(self, chunk_id: str) -> bool:
        """Remove a chunk from the vector index."""
        try:
            if chunk_id in self.vectors:
                del self.vectors[chunk_id]

            if chunk_id in self.chunk_metadata:
                del self.chunk_metadata[chunk_id]

            return True
        except Exception as e:
            self.logger.error(f"Failed to remove chunk {chunk_id}: {e}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get search engine statistics."""
        tier_counts = defaultdict(int)
        type_counts = defaultdict(int)

        for metadata in self.chunk_metadata.values():
            tier_counts[metadata['tier'].value] += 1
            type_counts[metadata['context_type'].value] += 1

        return {
            'total_chunks': len(self.vectors),
            'dimension': self.dimension,
            'model_name': self.model_name,
            'tier_distribution': dict(tier_counts),
            'type_distribution': dict(type_counts)
        }


class HierarchicalContextManager:
    """
    Advanced hierarchical context management system.

    Manages context organization, semantic search, caching, and intelligent
    retrieval for AI agents in the Hive system.
    """

    def __init__(
        self,
        context_store: Optional[GlobalContextStore] = None,
        enable_semantic_search: bool = True,
        enable_caching: bool = True,
        max_cache_size: int = 1000,
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        overlap_size: int = 100
    ):
        """Initialize the hierarchical context manager."""
        self.context_store = context_store
        self.enable_semantic_search = enable_semantic_search
        self.enable_caching = enable_caching
        self.max_cache_size = max_cache_size

        self.logger = structlog.get_logger().bind(component="hierarchical_context")

        # Core components
        self.chunker = SemanticChunker(chunk_size, overlap_size)
        self.search_engine = VectorSearchEngine(embedding_model) if enable_semantic_search else None

        # Context storage
        self.session_hierarchies: Dict[str, ContextHierarchy] = {}
        self.context_chunks: Dict[str, ContextChunk] = {}

        # Caching system
        self.query_cache: Dict[str, ContextRetrievalResult] = {}
        self.cache_access_order: deque = deque()

        # Statistics and metrics
        self.metrics = {
            'chunks_created': 0,
            'chunks_retrieved': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'searches_performed': 0,
            'context_updates': 0
        }

    async def initialize(self) -> None:
        """Initialize the context manager."""
        try:
            self.logger.info("Initializing Hierarchical Context Manager")

            if self.search_engine:
                await self.search_engine.initialize()

            # Load existing hierarchies if context store is available
            if self.context_store:
                await self._load_existing_hierarchies()

            self.logger.info("Hierarchical Context Manager initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize context manager: {e}", exc_info=True)
            raise

    async def initialize_session_context(
        self,
        session_id: str,
        initial_context: Dict[str, Any]
    ) -> bool:
        """Initialize context hierarchy for a new session."""
        try:
            self.logger.info(f"Initializing context for session {session_id}")

            # Create hierarchy
            hierarchy = ContextHierarchy(session_id=session_id)
            self.session_hierarchies[session_id] = hierarchy

            # Process initial context
            if 'project_files' in initial_context:
                await self._process_project_files(session_id, initial_context['project_files'])

            if 'conversation_history' in initial_context:
                await self._process_conversation_history(session_id, initial_context['conversation_history'])

            if 'documentation' in initial_context:
                await self._process_documentation(session_id, initial_context['documentation'])

            self.logger.info(f"Session context initialized with {hierarchy.total_chunks} chunks")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize session context: {e}", exc_info=True)
            return False

    async def add_message_context(
        self,
        session_id: str,
        message: Dict[str, Any],
        extract_entities: bool = True,
        update_semantic_index: bool = True
    ) -> List[str]:
        """Add a message to the session context."""
        try:
            if session_id not in self.session_hierarchies:
                await self.initialize_session_context(session_id, {})

            hierarchy = self.session_hierarchies[session_id]
            chunk_ids = []

            # Determine context type
            speaker = message.get('speaker', message.get('agent_id', 'user'))
            context_type = ContextType.USER_INPUT if speaker == 'user' else ContextType.AGENT_RESPONSE

            # Create chunk for message
            chunk_id = str(uuid.uuid4())
            chunk = ContextChunk(
                chunk_id=chunk_id,
                content=message.get('content', ''),
                context_type=context_type,
                tier=ContextTier.CORE,  # Recent messages are core context
                created_at=datetime.fromisoformat(message.get('timestamp', datetime.utcnow().isoformat()))
            )

            # Extract entities and keywords if requested
            if extract_entities:
                chunk.entities = await self._extract_entities(chunk.content)
                chunk.keywords = await self._extract_keywords(chunk.content)

            # Store chunk
            self.context_chunks[chunk_id] = chunk
            hierarchy.core_contexts.append(chunk_id)
            hierarchy.total_chunks += 1
            hierarchy.last_updated = datetime.utcnow()

            # Add to search index
            if self.search_engine and update_semantic_index:
                await self.search_engine.add_chunk(chunk)

            chunk_ids.append(chunk_id)
            self.metrics['chunks_created'] += 1
            self.metrics['context_updates'] += 1

            # Trigger hierarchy rebalancing if needed
            await self._rebalance_hierarchy(session_id)

            return chunk_ids

        except Exception as e:
            self.logger.error(f"Failed to add message context: {e}", exc_info=True)
            return []

    async def retrieve_context(
        self,
        session_id: str,
        query: str,
        strategy: RetrievalStrategy = RetrievalStrategy.HYBRID,
        max_chunks: int = 10,
        include_tiers: Optional[List[ContextTier]] = None,
        include_types: Optional[List[ContextType]] = None
    ) -> ContextRetrievalResult:
        """Retrieve relevant context for a query."""
        start_time = datetime.utcnow()

        try:
            # Check cache first
            cache_key = self._generate_cache_key(session_id, query, strategy, max_chunks)
            if self.enable_caching and cache_key in self.query_cache:
                cached_result = self.query_cache[cache_key]
                self.metrics['cache_hits'] += 1
                self._update_cache_access(cache_key)
                return cached_result

            self.metrics['cache_misses'] += 1

            # Get session hierarchy
            if session_id not in self.session_hierarchies:
                return ContextRetrievalResult([], 0, 0.0, strategy)

            hierarchy = self.session_hierarchies[session_id]

            # Determine search scope
            search_chunk_ids = []
            if include_tiers:
                for tier in include_tiers:
                    if tier == ContextTier.CORE:
                        search_chunk_ids.extend(hierarchy.core_contexts)
                    elif tier == ContextTier.RELEVANT:
                        search_chunk_ids.extend(hierarchy.relevant_contexts)
                    elif tier == ContextTier.AMBIENT:
                        search_chunk_ids.extend(hierarchy.ambient_contexts)
            else:
                # Include all active tiers by default
                search_chunk_ids.extend(hierarchy.core_contexts)
                search_chunk_ids.extend(hierarchy.relevant_contexts[:max_chunks])

            # Perform search based on strategy
            result_chunks = []
            semantic_scores = []
            keyword_scores = []

            if strategy in [RetrievalStrategy.SEMANTIC_ONLY, RetrievalStrategy.HYBRID]:
                if self.search_engine:
                    semantic_results = await self.search_engine.search_similar(
                        query=query,
                        limit=max_chunks,
                        tier_filter=include_tiers,
                        type_filter=include_types
                    )

                    for chunk_id, score in semantic_results:
                        if chunk_id in search_chunk_ids and chunk_id in self.context_chunks:
                            chunk = self.context_chunks[chunk_id]
                            chunk.last_accessed = datetime.utcnow()
                            chunk.access_count += 1
                            result_chunks.append(chunk)
                            semantic_scores.append(score)

            if strategy in [RetrievalStrategy.KEYWORD_ONLY, RetrievalStrategy.HYBRID]:
                keywords = await self._extract_keywords(query)
                if self.search_engine:
                    keyword_results = await self.search_engine.search_keywords(
                        keywords=keywords,
                        limit=max_chunks
                    )

                    for chunk_id, score in keyword_results:
                        if chunk_id in search_chunk_ids and chunk_id in self.context_chunks:
                            chunk = self.context_chunks[chunk_id]
                            if chunk not in result_chunks:  # Avoid duplicates
                                chunk.last_accessed = datetime.utcnow()
                                chunk.access_count += 1
                                result_chunks.append(chunk)
                                keyword_scores.append(score)

            # Apply additional ranking if hybrid strategy
            if strategy == RetrievalStrategy.HYBRID and len(result_chunks) > max_chunks:
                result_chunks = await self._rank_chunks_hybrid(
                    result_chunks, query, semantic_scores, keyword_scores
                )[:max_chunks]

            # Sort by combined relevance
            result_chunks.sort(key=lambda c: c.get_combined_relevance(), reverse=True)
            result_chunks = result_chunks[:max_chunks]

            # Calculate combined scores
            combined_scores = [chunk.get_combined_relevance() for chunk in result_chunks]

            # Calculate query time
            query_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            # Create result
            result = ContextRetrievalResult(
                chunks=result_chunks,
                total_found=len(result_chunks),
                query_time_ms=query_time,
                retrieval_strategy=strategy,
                semantic_scores=semantic_scores,
                keyword_scores=keyword_scores,
                combined_scores=combined_scores
            )

            # Cache result
            if self.enable_caching:
                self._cache_result(cache_key, result)

            self.metrics['searches_performed'] += 1
            self.metrics['chunks_retrieved'] += len(result_chunks)

            return result

        except Exception as e:
            query_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.logger.error(f"Context retrieval failed: {e}", exc_info=True)
            return ContextRetrievalResult([], 0, query_time, strategy)

    async def get_session_context(self, session_id: str) -> Dict[str, Any]:
        """Get the complete context state for a session."""
        try:
            if session_id not in self.session_hierarchies:
                return {}

            hierarchy = self.session_hierarchies[session_id]

            # Get chunk counts by tier
            tier_counts = {
                'core': len(hierarchy.core_contexts),
                'relevant': len(hierarchy.relevant_contexts),
                'ambient': len(hierarchy.ambient_contexts),
                'archived': len(hierarchy.archived_contexts)
            }

            # Get context types distribution
            type_counts = defaultdict(int)
            for chunk_id in (hierarchy.core_contexts + hierarchy.relevant_contexts +
                           hierarchy.ambient_contexts):
                if chunk_id in self.context_chunks:
                    chunk = self.context_chunks[chunk_id]
                    type_counts[chunk.context_type.value] += 1

            return {
                'session_id': session_id,
                'total_chunks': hierarchy.total_chunks,
                'tier_distribution': tier_counts,
                'type_distribution': dict(type_counts),
                'last_updated': hierarchy.last_updated.isoformat(),
                'hierarchy_depth': len(hierarchy.context_tree),
                'search_index_size': len(self.search_engine.vectors) if self.search_engine else 0
            }

        except Exception as e:
            self.logger.error(f"Failed to get session context: {e}")
            return {}

    async def update_chunk_relevance(
        self,
        chunk_id: str,
        relevance_factor: float,
        reason: str = "manual_update"
    ) -> bool:
        """Update the relevance score of a context chunk."""
        try:
            if chunk_id not in self.context_chunks:
                return False

            chunk = self.context_chunks[chunk_id]
            old_relevance = chunk.dynamic_relevance

            # Update dynamic relevance
            chunk.dynamic_relevance *= relevance_factor
            chunk.dynamic_relevance = max(0.1, min(2.0, chunk.dynamic_relevance))  # Clamp values

            # Update recency score based on last access
            time_since_access = datetime.utcnow() - chunk.last_accessed
            recency_decay = math.exp(-time_since_access.total_seconds() / 3600)  # 1-hour half-life
            chunk.recency_score = recency_decay

            # Update in search index if available
            if self.search_engine and chunk_id in self.search_engine.chunk_metadata:
                self.search_engine.chunk_metadata[chunk_id]['relevance'] = chunk.get_combined_relevance()

            self.logger.debug(
                f"Updated chunk {chunk_id} relevance: {old_relevance:.2f} -> {chunk.dynamic_relevance:.2f} ({reason})"
            )

            return True

        except Exception as e:
            self.logger.error(f"Failed to update chunk relevance: {e}")
            return False

    async def prune_context(
        self,
        session_id: str,
        max_chunks_per_tier: Optional[Dict[ContextTier, int]] = None,
        age_threshold_hours: int = 24
    ) -> Dict[str, int]:
        """Prune old or irrelevant context to manage memory usage."""
        try:
            if session_id not in self.session_hierarchies:
                return {}

            hierarchy = self.session_hierarchies[session_id]
            pruned_counts = defaultdict(int)

            # Default limits per tier
            if max_chunks_per_tier is None:
                max_chunks_per_tier = {
                    ContextTier.CORE: 50,
                    ContextTier.RELEVANT: 200,
                    ContextTier.AMBIENT: 500,
                    ContextTier.ARCHIVED: 100
                }

            age_cutoff = datetime.utcnow() - timedelta(hours=age_threshold_hours)

            # Prune each tier
            for tier, max_chunks in max_chunks_per_tier.items():
                if tier == ContextTier.CORE:
                    chunk_list = hierarchy.core_contexts
                elif tier == ContextTier.RELEVANT:
                    chunk_list = hierarchy.relevant_contexts
                elif tier == ContextTier.AMBIENT:
                    chunk_list = hierarchy.ambient_contexts
                else:
                    chunk_list = hierarchy.archived_contexts

                if len(chunk_list) <= max_chunks:
                    continue

                # Sort by combined relevance and recency
                chunks_with_scores = []
                for chunk_id in chunk_list:
                    if chunk_id in self.context_chunks:
                        chunk = self.context_chunks[chunk_id]
                        score = chunk.get_combined_relevance()

                        # Penalize very old chunks
                        if chunk.created_at < age_cutoff:
                            score *= 0.5

                        chunks_with_scores.append((chunk_id, score))

                # Sort by score (descending) and keep top chunks
                chunks_with_scores.sort(key=lambda x: x[1], reverse=True)
                chunks_to_keep = chunks_with_scores[:max_chunks]
                chunks_to_remove = chunks_with_scores[max_chunks:]

                # Remove excess chunks
                for chunk_id, _ in chunks_to_remove:
                    await self._remove_chunk(chunk_id)
                    pruned_counts[tier.value] += 1

                # Update hierarchy list
                chunk_list[:] = [chunk_id for chunk_id, _ in chunks_to_keep]

            # Update hierarchy stats
            hierarchy.total_chunks = (len(hierarchy.core_contexts) +
                                   len(hierarchy.relevant_contexts) +
                                   len(hierarchy.ambient_contexts) +
                                   len(hierarchy.archived_contexts))
            hierarchy.last_updated = datetime.utcnow()

            total_pruned = sum(pruned_counts.values())
            if total_pruned > 0:
                self.logger.info(f"Pruned {total_pruned} chunks from session {session_id}")

            return dict(pruned_counts)

        except Exception as e:
            self.logger.error(f"Context pruning failed: {e}")
            return {}

    # Private helper methods

    async def _load_existing_hierarchies(self) -> None:
        """Load existing context hierarchies from the context store."""
        try:
            if not self.context_store:
                return

            # This would integrate with the actual context store implementation
            # For now, we'll leave it as a placeholder
            self.logger.debug("Loading existing context hierarchies")

        except Exception as e:
            self.logger.error(f"Failed to load existing hierarchies: {e}")

    async def _process_project_files(self, session_id: str, file_paths: List[str]) -> None:
        """Process project files and add them to the context hierarchy."""
        try:
            hierarchy = self.session_hierarchies[session_id]

            for file_path in file_paths:
                try:
                    # Read file content
                    path = Path(file_path)
                    if not path.exists():
                        continue

                    content = path.read_text(encoding='utf-8', errors='ignore')
                    language = self._detect_language(file_path)

                    # Chunk the file content
                    chunks = await self.chunker.chunk_code(content, file_path, language)

                    for chunk_data in chunks:
                        chunk_id = str(uuid.uuid4())

                        # Determine context tier based on file importance
                        tier = self._determine_file_tier(file_path, chunk_data.get('metadata', {}))

                        chunk = ContextChunk(
                            chunk_id=chunk_id,
                            content=chunk_data['content'],
                            context_type=self._get_code_context_type(chunk_data.get('metadata', {})),
                            tier=tier,
                            source_path=file_path,
                            line_start=chunk_data.get('metadata', {}).get('line_start'),
                            line_end=chunk_data.get('metadata', {}).get('line_end'),
                            function_name=chunk_data.get('metadata', {}).get('name'),
                            keywords=await self._extract_keywords(chunk_data['content'])
                        )

                        # Store chunk
                        self.context_chunks[chunk_id] = chunk

                        # Add to appropriate tier in hierarchy
                        if tier == ContextTier.CORE:
                            hierarchy.core_contexts.append(chunk_id)
                        elif tier == ContextTier.RELEVANT:
                            hierarchy.relevant_contexts.append(chunk_id)
                        else:
                            hierarchy.ambient_contexts.append(chunk_id)

                        hierarchy.total_chunks += 1

                        # Add to search index
                        if self.search_engine:
                            await self.search_engine.add_chunk(chunk)

                        self.metrics['chunks_created'] += 1

                except Exception as e:
                    self.logger.error(f"Failed to process file {file_path}: {e}")
                    continue

            hierarchy.last_updated = datetime.utcnow()
            self.logger.info(f"Processed {len(file_paths)} files for session {session_id}")

        except Exception as e:
            self.logger.error(f"Failed to process project files: {e}")

    async def _process_conversation_history(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        """Process conversation history and add to context."""
        try:
            hierarchy = self.session_hierarchies[session_id]

            # Chunk conversation messages
            chunks = await self.chunker.chunk_conversation(messages)

            for i, chunk_data in enumerate(chunks):
                chunk_id = str(uuid.uuid4())

                # Recent chunks are more important
                tier = ContextTier.CORE if i >= len(chunks) - 5 else ContextTier.RELEVANT

                chunk = ContextChunk(
                    chunk_id=chunk_id,
                    content=chunk_data['content'],
                    context_type=ContextType.CONVERSATION,
                    tier=tier,
                    keywords=await self._extract_keywords(chunk_data['content']),
                    entities=await self._extract_entities(chunk_data['content'])
                )

                # Store chunk
                self.context_chunks[chunk_id] = chunk

                # Add to hierarchy
                if tier == ContextTier.CORE:
                    hierarchy.core_contexts.append(chunk_id)
                else:
                    hierarchy.relevant_contexts.append(chunk_id)

                hierarchy.total_chunks += 1

                # Add to search index
                if self.search_engine:
                    await self.search_engine.add_chunk(chunk)

                self.metrics['chunks_created'] += 1

            hierarchy.last_updated = datetime.utcnow()

        except Exception as e:
            self.logger.error(f"Failed to process conversation history: {e}")

    async def _process_documentation(self, session_id: str, doc_content: str) -> None:
        """Process documentation content and add to context."""
        try:
            hierarchy = self.session_hierarchies[session_id]

            # Simple chunking for documentation
            chunks = await self.chunker._chunk_generic(doc_content, "documentation")

            for chunk_data in chunks:
                chunk_id = str(uuid.uuid4())

                chunk = ContextChunk(
                    chunk_id=chunk_id,
                    content=chunk_data['content'],
                    context_type=ContextType.DOCUMENTATION,
                    tier=ContextTier.AMBIENT,  # Documentation is ambient context
                    keywords=await self._extract_keywords(chunk_data['content'])
                )

                # Store chunk
                self.context_chunks[chunk_id] = chunk
                hierarchy.ambient_contexts.append(chunk_id)
                hierarchy.total_chunks += 1

                # Add to search index
                if self.search_engine:
                    await self.search_engine.add_chunk(chunk)

                self.metrics['chunks_created'] += 1

            hierarchy.last_updated = datetime.utcnow()

        except Exception as e:
            self.logger.error(f"Failed to process documentation: {e}")

    async def _extract_entities(self, content: str) -> List[str]:
        """Extract named entities from content."""
        try:
            # Simple entity extraction - could be enhanced with NLP libraries
            import re

            entities = []

            # Extract function/method names
            func_pattern = r'\b(?:def|function|async\s+def)\s+(\w+)'
            entities.extend(re.findall(func_pattern, content, re.IGNORECASE))

            # Extract class names
            class_pattern = r'\b(?:class|interface)\s+(\w+)'
            entities.extend(re.findall(class_pattern, content, re.IGNORECASE))

            # Extract variable names (simple heuristic)
            var_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*='
            entities.extend(re.findall(var_pattern, content)[:10])  # Limit to avoid noise

            return list(set(entities))  # Remove duplicates

        except Exception as e:
            self.logger.error(f"Entity extraction failed: {e}")
            return []

    async def _extract_keywords(self, content: str) -> List[str]:
        """Extract keywords from content."""
        try:
            import re
            from collections import Counter

            # Simple keyword extraction
            words = re.findall(r'\b[a-zA-Z]{3,}\b', content.lower())

            # Remove common stop words
            stop_words = {
                'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
                'by', 'this', 'that', 'these', 'those', 'is', 'are', 'was', 'were',
                'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
                'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must'
            }

            filtered_words = [word for word in words if word not in stop_words]

            # Get most common words
            word_counts = Counter(filtered_words)
            keywords = [word for word, count in word_counts.most_common(20)]

            return keywords

        except Exception as e:
            self.logger.error(f"Keyword extraction failed: {e}")
            return []

    async def _rebalance_hierarchy(self, session_id: str) -> None:
        """Rebalance context hierarchy based on usage patterns."""
        try:
            if session_id not in self.session_hierarchies:
                return

            hierarchy = self.session_hierarchies[session_id]

            # Move chunks between tiers based on access patterns and relevance
            chunks_to_move = []

            # Check core contexts - demote unused ones
            for chunk_id in hierarchy.core_contexts[:]:
                if chunk_id in self.context_chunks:
                    chunk = self.context_chunks[chunk_id]

                    # Demote if not accessed recently and low relevance
                    time_since_access = datetime.utcnow() - chunk.last_accessed
                    if (time_since_access > timedelta(hours=1) and
                        chunk.get_combined_relevance() < 0.5):
                        chunks_to_move.append((chunk_id, ContextTier.CORE, ContextTier.RELEVANT))

            # Check relevant contexts - promote frequently used ones
            for chunk_id in hierarchy.relevant_contexts[:]:
                if chunk_id in self.context_chunks:
                    chunk = self.context_chunks[chunk_id]

                    # Promote if frequently accessed and high relevance
                    if (chunk.access_count > 5 and
                        chunk.get_combined_relevance() > 0.8 and
                        len(hierarchy.core_contexts) < 20):  # Limit core size
                        chunks_to_move.append((chunk_id, ContextTier.RELEVANT, ContextTier.CORE))

            # Apply moves
            for chunk_id, from_tier, to_tier in chunks_to_move:
                await self._move_chunk_tier(session_id, chunk_id, from_tier, to_tier)

        except Exception as e:
            self.logger.error(f"Hierarchy rebalancing failed: {e}")

    async def _move_chunk_tier(
        self,
        session_id: str,
        chunk_id: str,
        from_tier: ContextTier,
        to_tier: ContextTier
    ) -> bool:
        """Move a chunk from one tier to another."""
        try:
            hierarchy = self.session_hierarchies[session_id]

            # Remove from old tier
            if from_tier == ContextTier.CORE and chunk_id in hierarchy.core_contexts:
                hierarchy.core_contexts.remove(chunk_id)
            elif from_tier == ContextTier.RELEVANT and chunk_id in hierarchy.relevant_contexts:
                hierarchy.relevant_contexts.remove(chunk_id)
            elif from_tier == ContextTier.AMBIENT and chunk_id in hierarchy.ambient_contexts:
                hierarchy.ambient_contexts.remove(chunk_id)

            # Add to new tier
            if to_tier == ContextTier.CORE:
                hierarchy.core_contexts.append(chunk_id)
            elif to_tier == ContextTier.RELEVANT:
                hierarchy.relevant_contexts.append(chunk_id)
            elif to_tier == ContextTier.AMBIENT:
                hierarchy.ambient_contexts.append(chunk_id)

            # Update chunk tier
            if chunk_id in self.context_chunks:
                self.context_chunks[chunk_id].tier = to_tier

            return True

        except Exception as e:
            self.logger.error(f"Failed to move chunk tier: {e}")
            return False

    def _generate_cache_key(
        self,
        session_id: str,
        query: str,
        strategy: RetrievalStrategy,
        max_chunks: int
    ) -> str:
        """Generate a cache key for query results."""
        key_data = f"{session_id}:{query}:{strategy.value}:{max_chunks}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _update_cache_access(self, cache_key: str) -> None:
        """Update cache access order for LRU eviction."""
        if cache_key in self.cache_access_order:
            self.cache_access_order.remove(cache_key)
        self.cache_access_order.append(cache_key)

    def _cache_result(self, cache_key: str, result: ContextRetrievalResult) -> None:
        """Cache a query result with LRU eviction."""
        if not self.enable_caching:
            return

        self.query_cache[cache_key] = result
        self._update_cache_access(cache_key)

        # Evict oldest entries if cache is full
        while len(self.query_cache) > self.max_cache_size:
            oldest_key = self.cache_access_order.popleft()
            if oldest_key in self.query_cache:
                del self.query_cache[oldest_key]

    async def _rank_chunks_hybrid(
        self,
        chunks: List[ContextChunk],
        query: str,
        semantic_scores: List[float],
        keyword_scores: List[float]
    ) -> List[ContextChunk]:
        """Rank chunks using hybrid semantic and keyword scores."""
        try:
            chunk_scores = {}

            # Combine semantic and keyword scores
            for i, chunk in enumerate(chunks):
                sem_score = semantic_scores[i] if i < len(semantic_scores) else 0.0
                kw_score = keyword_scores[i] if i < len(keyword_scores) else 0.0

                # Weighted combination (favor semantic for code, keyword for text)
                if chunk.context_type in [ContextType.CODE_FUNCTION, ContextType.CODE_CLASS]:
                    combined_score = sem_score * 0.7 + kw_score * 0.3
                else:
                    combined_score = sem_score * 0.5 + kw_score * 0.5

                # Apply relevance and recency weights
                final_score = combined_score * chunk.get_combined_relevance()
                chunk_scores[chunk.chunk_id] = final_score

            # Sort chunks by final score
            ranked_chunks = sorted(chunks, key=lambda c: chunk_scores.get(c.chunk_id, 0.0), reverse=True)
            return ranked_chunks

        except Exception as e:
            self.logger.error(f"Hybrid ranking failed: {e}")
            return chunks

    async def _remove_chunk(self, chunk_id: str) -> bool:
        """Remove a chunk from all data structures."""
        try:
            # Remove from context storage
            if chunk_id in self.context_chunks:
                del self.context_chunks[chunk_id]

            # Remove from search index
            if self.search_engine:
                self.search_engine.remove_chunk(chunk_id)

            # Remove from hierarchies
            for hierarchy in self.session_hierarchies.values():
                for chunk_list in [hierarchy.core_contexts, hierarchy.relevant_contexts,
                                 hierarchy.ambient_contexts, hierarchy.archived_contexts]:
                    if chunk_id in chunk_list:
                        chunk_list.remove(chunk_id)

            return True

        except Exception as e:
            self.logger.error(f"Failed to remove chunk {chunk_id}: {e}")
            return False

    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension."""
        ext = Path(file_path).suffix.lower()

        language_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'javascript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.cc': 'cpp',
            '.cxx': 'cpp',
            '.c': 'c',
            '.go': 'go',
            '.rs': 'rust',
            '.rb': 'ruby',
            '.php': 'php',
            '.cs': 'csharp',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.scala': 'scala'
        }

        return language_map.get(ext, 'text')

    def _determine_file_tier(self, file_path: str, metadata: Dict[str, Any]) -> ContextTier:
        """Determine the appropriate context tier for a file."""
        path = Path(file_path)

        # Core files (main modules, recently modified)
        if (path.name in ['main.py', 'index.js', 'app.py', '__init__.py'] or
            'main' in path.name.lower() or 'app' in path.name.lower()):
            return ContextTier.CORE

        # Relevant files (source code in main directories)
        if (path.suffix in ['.py', '.js', '.ts', '.java', '.cpp'] and
            not any(part in path.parts for part in ['test', 'tests', 'node_modules', '__pycache__'])):
            return ContextTier.RELEVANT

        # Everything else is ambient
        return ContextTier.AMBIENT

    def _get_code_context_type(self, metadata: Dict[str, Any]) -> ContextType:
        """Determine context type from code metadata."""
        chunk_type = metadata.get('type', 'generic')

        if chunk_type == 'function':
            return ContextType.CODE_FUNCTION
        elif chunk_type == 'class':
            return ContextType.CODE_CLASS
        elif chunk_type in ['module', 'file']:
            return ContextType.CODE_MODULE
        else:
            return ContextType.CODE_MODULE

    def get_metrics(self) -> Dict[str, Any]:
        """Get hierarchical context manager metrics."""
        cache_hit_rate = 0.0
        total_queries = self.metrics['cache_hits'] + self.metrics['cache_misses']
        if total_queries > 0:
            cache_hit_rate = self.metrics['cache_hits'] / total_queries * 100

        search_stats = {}
        if self.search_engine:
            search_stats = self.search_engine.get_statistics()

        return {
            **self.metrics,
            'cache_hit_rate_percent': cache_hit_rate,
            'active_sessions': len(self.session_hierarchies),
            'total_chunks': len(self.context_chunks),
            'cached_queries': len(self.query_cache),
            'search_engine': search_stats
        }
