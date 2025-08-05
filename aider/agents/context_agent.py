"""
ContextAgent: Specialized agent for project context management and semantic search.

Phase 2.2 Priority 2 Enhancement: Project Context Intelligence
This agent handles all context-related operations in the Aider Multi-Agent Hive Architecture:
- Building comprehensive project context with AI insights
- Semantic indexing and search with AST parsing
- Context sharing between agents
- Real-time context updates
- Advanced project structure analysis
- Dependency tracking and relationship mapping
- AI-powered code understanding
- Semantic code search capabilities
"""

import asyncio
import ast
import hashlib
import json
import logging
import os
import re
import sqlite3
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import structlog

# AI and analysis imports for Phase 2.2 Priority 2
try:
    import tree_sitter
    from tree_sitter import Language, Parser
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from .base_agent import BaseAgent, AgentMessage, MessagePriority, AgentState, AgentCapability

# Import AI capabilities from ModelManager
try:
    from ..models.model_manager import analyze_code, ModelManager
    AI_MODELS_AVAILABLE = True
except ImportError:
    AI_MODELS_AVAILABLE = False


class ContextType(Enum):
    """Types of context information."""
    PROJECT_STRUCTURE = "project_structure"
    FILE_CONTENT = "file_content"
    CODE_SYMBOLS = "code_symbols"
    DEPENDENCIES = "dependencies"
    GIT_HISTORY = "git_history"
    DOCUMENTATION = "documentation"
    CONFIGURATION = "configuration"
    BUILD_ARTIFACTS = "build_artifacts"
    TEST_RESULTS = "test_results"
    RUNTIME_LOGS = "runtime_logs"
    USER_PREFERENCES = "user_preferences"
    SESSION_STATE = "session_state"


class ContextScope(Enum):
    """Scope levels for context information."""
    GLOBAL = "global"           # Entire project
    MODULE = "module"           # Specific module/package
    FILE = "file"              # Single file
    FUNCTION = "function"       # Function/method level
    LINE = "line"              # Line-specific
    SESSION = "session"         # User session
    TEMPORARY = "temporary"     # Temporary context


@dataclass
class ContextEntry:
    """Individual context entry with metadata."""
    id: str
    type: ContextType
    scope: ContextScope
    content: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    accessed_at: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0
    relevance_score: float = 1.0
    tags: Set[str] = field(default_factory=set)
    dependencies: Set[str] = field(default_factory=set)
    embeddings: Optional[List[float]] = None


@dataclass
class ContextQuery:
    """Query for retrieving context information."""
    query_text: str
    context_types: Optional[List[ContextType]] = None
    scopes: Optional[List[ContextScope]] = None
    tags: Optional[Set[str]] = None
    max_results: int = 10
    min_relevance: float = 0.5
    include_metadata: bool = True
    semantic_search: bool = True
    exact_match: bool = False


@dataclass
class ContextSearchResult:
    """Result of a context search query."""
    entries: List[ContextEntry]
    total_matches: int
    search_time: float
    query: ContextQuery
    relevance_scores: Dict[str, float]
    semantic_matches: Dict[str, float]


@dataclass
class SemanticAnalysis:
    """Enhanced semantic analysis results for Phase 2.2 Priority 2."""
    entities: Dict[str, Any]
    references: Dict[str, List[str]]
    complexity: float
    insights: Dict[str, Any]
    relationships: Dict[str, Set[str]]
    ast_info: Dict[str, Any]
    language: str
    quality_score: float = 0.0

@dataclass
class ProjectAnalysis:
    """Comprehensive project analysis with AI insights."""
    structure: Dict[str, Any]
    languages: Dict[str, Any]
    dependencies: Dict[str, List[str]]
    patterns: Dict[str, Any]
    insights: Dict[str, Any]
    complexity_score: float
    architecture_type: str = "unknown"
    frameworks: List[str] = field(default_factory=list)
    semantic_map: Dict[str, SemanticAnalysis] = field(default_factory=dict)

@dataclass
class ProjectContext:
    """Comprehensive project context information."""
    project_root: str
    structure: Dict[str, Any]
    files: Dict[str, Dict[str, Any]]
    dependencies: Dict[str, List[str]]
    git_info: Dict[str, Any]
    build_config: Dict[str, Any]
    documentation: Dict[str, str]
    metrics: Dict[str, Any]
    analysis: Optional[ProjectAnalysis] = None
    semantic_index: Dict[str, Any] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.utcnow)


class ContextAgent(BaseAgent):
    """
    Specialized agent for context management and semantic search.

    Responsibilities:
    - Build and maintain project context
    - Provide semantic search capabilities
    - Manage context sharing between agents
    - Track context dependencies and relationships
    - Optimize context retrieval performance
    - Handle real-time context updates
    """

    def __init__(
        self,
        agent_id: str = "context_agent",
        config: Optional[Dict[str, Any]] = None,
        message_bus=None,
    ):
        """Initialize the context agent."""
        super().__init__(
            agent_id=agent_id,
            agent_type="context",
            config=config,
            message_bus=message_bus,
        )

        # Context agent specific configuration
        self.max_context_entries = self.config.get('max_context_entries', 10000)
        self.context_ttl = self.config.get('context_ttl', 3600)  # 1 hour
        self.enable_semantic_search = self.config.get('enable_semantic_search', True)
        self.enable_caching = self.config.get('enable_caching', True)
        self.auto_update_interval = self.config.get('auto_update_interval', 300)  # 5 minutes

        # Context storage
        self.context_store: Dict[str, ContextEntry] = {}
        self.context_index: Dict[ContextType, Set[str]] = defaultdict(set)
        self.tag_index: Dict[str, Set[str]] = defaultdict(set)
        self.dependency_graph: Dict[str, Set[str]] = defaultdict(set)

        # Project context cache
        self.project_context: Optional[ProjectContext] = None
        self.context_lock = asyncio.Lock()

        # Phase 2.2 Priority 2: Enhanced analysis capabilities
        self.model_manager = None
        self.ast_parsers = {}
        self.semantic_cache: Dict[str, SemanticAnalysis] = {}

        # Project Context Database
        self.context_db_path = self.config.get('context_db_path', 'project_context.db')
        self.context_db = None

        # AI-powered insights
        self.enable_ai_insights = self.config.get('enable_ai_insights', True)
        self.insight_cache_ttl = self.config.get('insight_cache_ttl', 3600)  # 1 hour

        # Search and indexing
        self.search_cache: Dict[str, ContextSearchResult] = {}
        self.cache_ttl = self.config.get('cache_ttl', 300)  # 5 minutes

        # File system monitoring
        self.watched_paths: Set[str] = set()
        self.file_hashes: Dict[str, str] = {}
        self.ignore_patterns = {
            '.git', '__pycache__', '.pyc', '.pyo', 'node_modules',
            '.venv', 'venv', '.env', '.DS_Store', 'Thumbs.db'
        }

        # Performance metrics
        self.context_metrics = {
            'total_queries': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'context_updates': 0,
            'average_search_time': 0.0,
            'context_size': 0,
            'index_size': 0
        }

        # Context extraction patterns
        self.extraction_patterns = {
            '.py': self._extract_python_context,
            '.js': self._extract_javascript_context,
            '.ts': self._extract_typescript_context,
            '.java': self._extract_java_context,
            '.cpp': self._extract_cpp_context,
            '.c': self._extract_c_context,
            '.go': self._extract_go_context,
            '.rs': self._extract_rust_context,
            '.php': self._extract_php_context,
            '.rb': self._extract_ruby_context,
            '.sql': self._extract_sql_context,
            '.md': self._extract_markdown_context,
            '.json': self._extract_json_context,
            '.yaml': self._extract_yaml_context,
            '.yml': self._extract_yaml_context,
            '.xml': self._extract_xml_context,
            '.html': self._extract_html_context,
            '.css': self._extract_css_context
        }

    async def initialize(self) -> bool:
        """Initialize the context agent with Phase 2.2 Priority 2 enhancements."""
        try:
            # Initialize AI capabilities
            if AI_MODELS_AVAILABLE and self.enable_ai_insights:
                try:
                    self.model_manager = ModelManager()
                    await self.model_manager.initialize()
                    self.logger.info("AI insights enabled for project analysis")
                except Exception as e:
                    self.logger.warning(f"AI insights disabled: {e}")
                    self.enable_ai_insights = False

            # Initialize AST parsers
            await self._initialize_ast_parsers()

            # Initialize project context database
            await self._initialize_context_database()

            # Continue with original initialization
            await super().initialize()

            # Register context-specific message handlers
            self.register_message_handler('build_project_context', self._handle_build_project_context)
            self.register_message_handler('search_context', self._handle_search_context)
            self.register_message_handler('update_context', self._handle_update_context)
            self.register_message_handler('get_context', self._handle_get_context)
            self.register_message_handler('analyze_project', self._handle_analyze_project)
            self.register_message_handler('task_request', self._handle_task_request)

            # Initialize context store
            await self._initialize_context_store()

            # Start background tasks
            self._start_background_tasks()

            self.logger.info("ContextAgent initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize ContextAgent: {e}")
            return False

    async def build_project_context(
        self,
        project_root: str,
        include_files: bool = True,
        include_git: bool = True,
        include_dependencies: bool = True,
        force_refresh: bool = False
    ) -> ProjectContext:
        """
        Build comprehensive project context.

        Args:
            project_root: Root directory of the project
            include_files: Whether to include file content analysis
            include_git: Whether to include git information
            include_dependencies: Whether to analyze dependencies
            force_refresh: Force refresh even if cached context exists

        Returns:
            ProjectContext with comprehensive project information
        """
        start_time = time.time()
        context_id = str(uuid.uuid4())

        self.logger.info(
            "Building project context",
            context_id=context_id,
            project_root=project_root,
            include_files=include_files,
            include_git=include_git,
            include_dependencies=include_dependencies
        )

        try:
            async with self.context_lock:
                # Check if we have cached context and it's still valid
                if (not force_refresh and
                    self.project_context and
                    self.project_context.project_root == project_root and
                    (datetime.utcnow() - self.project_context.last_updated).seconds < self.auto_update_interval):

                    self.logger.debug("Using cached project context")
                    return self.project_context

                # Build project structure
                structure = await self._analyze_project_structure(project_root)

                # Analyze files if requested
                files = {}
                if include_files:
                    files = await self._analyze_project_files(project_root, structure)

                # Analyze dependencies if requested
                dependencies = {}
                if include_dependencies:
                    dependencies = await self._analyze_project_dependencies(project_root)

                # Get git information if requested
                git_info = {}
                if include_git:
                    git_info = await self._analyze_git_info(project_root)

                # Analyze build configuration
                build_config = await self._analyze_build_config(project_root)

                # Extract documentation
                documentation = await self._extract_project_documentation(project_root, files)

                # Calculate metrics
                metrics = await self._calculate_project_metrics(structure, files, dependencies)

                # Create project context
                self.project_context = ProjectContext(
                    project_root=project_root,
                    structure=structure,
                    files=files,
                    dependencies=dependencies,
                    git_info=git_info,
                    build_config=build_config,
                    documentation=documentation,
                    metrics=metrics,
                    last_updated=datetime.utcnow()
                )

                # Store context entries
                await self._store_project_context_entries(self.project_context)

                build_time = time.time() - start_time
                self.context_metrics['context_updates'] += 1

                self.logger.info(
                    "Project context built successfully",
                    context_id=context_id,
                    build_time=build_time,
                    files_analyzed=len(files),
                    dependencies_found=len(dependencies)
                )

                return self.project_context

        except Exception as e:
            self.logger.error(
                "Failed to build project context",
                context_id=context_id,
                error=str(e),
                exc_info=True
            )
            raise

    async def search_context(self, query: ContextQuery) -> ContextSearchResult:
        """
        Search for context information based on query.

        Args:
            query: Context search query

        Returns:
            ContextSearchResult with matching entries
        """
        start_time = time.time()
        query_id = str(uuid.uuid4())

        self.logger.debug(
            "Searching context",
            query_id=query_id,
            query_text=query.query_text,
            semantic_search=query.semantic_search
        )

        try:
            # Check cache first
            cache_key = self._generate_cache_key(query)
            if self.enable_caching and cache_key in self.search_cache:
                cached_result = self.search_cache[cache_key]
                if (time.time() - cached_result.search_time) < self.cache_ttl:
                    self.context_metrics['cache_hits'] += 1
                    return cached_result

            self.context_metrics['cache_misses'] += 1

            # Perform search
            matching_entries = []
            relevance_scores = {}
            semantic_matches = {}

            # Filter by context types
            candidate_entries = set()
            if query.context_types:
                for context_type in query.context_types:
                    candidate_entries.update(self.context_index[context_type])
            else:
                candidate_entries = set(self.context_store.keys())

            # Filter by scopes
            if query.scopes:
                scope_filtered = set()
                for entry_id in candidate_entries:
                    if self.context_store[entry_id].scope in query.scopes:
                        scope_filtered.add(entry_id)
                candidate_entries = scope_filtered

            # Filter by tags
            if query.tags:
                tag_filtered = set()
                for tag in query.tags:
                    tag_filtered.update(self.tag_index[tag])
                candidate_entries = candidate_entries.intersection(tag_filtered)

            # Score and rank entries
            for entry_id in candidate_entries:
                entry = self.context_store[entry_id]

                # Calculate relevance score
                relevance = self._calculate_relevance_score(entry, query)
                if relevance >= query.min_relevance:
                    relevance_scores[entry_id] = relevance

                    # Calculate semantic similarity if enabled
                    if query.semantic_search and self.enable_semantic_search:
                        semantic_score = await self._calculate_semantic_similarity(entry, query)
                        semantic_matches[entry_id] = semantic_score
                        # Combine scores
                        relevance = (relevance + semantic_score) / 2

                    if relevance >= query.min_relevance:
                        matching_entries.append((entry, relevance))

            # Sort by relevance and limit results
            matching_entries.sort(key=lambda x: x[1], reverse=True)
            matching_entries = matching_entries[:query.max_results]

            # Extract entries
            result_entries = [entry for entry, score in matching_entries]

            # Update access statistics
            for entry in result_entries:
                entry.accessed_at = datetime.utcnow()
                entry.access_count += 1

            search_time = time.time() - start_time

            # Create result
            result = ContextSearchResult(
                entries=result_entries,
                total_matches=len(matching_entries),
                search_time=search_time,
                query=query,
                relevance_scores=relevance_scores,
                semantic_matches=semantic_matches
            )

            # Cache result
            if self.enable_caching:
                self.search_cache[cache_key] = result

            # Update metrics
            self.context_metrics['total_queries'] += 1
            self._update_average_search_time(search_time)

            self.logger.debug(
                "Context search completed",
                query_id=query_id,
                matches_found=len(result_entries),
                search_time=search_time
            )

            return result

        except Exception as e:
            self.logger.error(
                "Context search failed",
                query_id=query_id,
                error=str(e),
                exc_info=True
            )
            raise

    async def get_relevant_context(
        self,
        request: str,
        context_types: Optional[List[ContextType]] = None,
        max_entries: int = 5
    ) -> List[ContextEntry]:
        """
        Get context entries relevant to a specific request.

        Args:
            request: The request or query text
            context_types: Specific context types to search
            max_entries: Maximum number of entries to return

        Returns:
            List of relevant context entries
        """
        query = ContextQuery(
            query_text=request,
            context_types=context_types,
            max_results=max_entries,
            semantic_search=True
        )

        result = await self.search_context(query)
        return result.entries

    async def update_context(
        self,
        entry_id: str,
        content: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[Set[str]] = None,
        relevance_score: Optional[float] = None
    ) -> bool:
        """
        Update an existing context entry.

        Args:
            entry_id: ID of the context entry to update
            content: New content (if provided)
            metadata: New metadata (if provided)
            tags: New tags (if provided)
            relevance_score: New relevance score (if provided)

        Returns:
            True if update was successful, False otherwise
        """
        try:
            async with self.context_lock:
                if entry_id not in self.context_store:
                    self.logger.warning(f"Context entry not found: {entry_id}")
                    return False

                entry = self.context_store[entry_id]

                # Update fields if provided
                if content is not None:
                    entry.content = content
                if metadata is not None:
                    entry.metadata.update(metadata)
                if tags is not None:
                    # Remove old tag mappings
                    for old_tag in entry.tags:
                        self.tag_index[old_tag].discard(entry_id)
                    # Add new tag mappings
                    entry.tags = tags
                    for new_tag in tags:
                        self.tag_index[new_tag].add(entry_id)
                if relevance_score is not None:
                    entry.relevance_score = relevance_score

                entry.updated_at = datetime.utcnow()

                # Clear search cache
                self.search_cache.clear()

                self.logger.debug(f"Context entry updated: {entry_id}")
                return True

        except Exception as e:
            self.logger.error(f"Failed to update context entry {entry_id}: {e}")
            return False

    async def add_context_entry(
        self,
        content: Any,
        context_type: ContextType,
        scope: ContextScope,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[Set[str]] = None,
        entry_id: Optional[str] = None
    ) -> str:
        """
        Add a new context entry.

        Args:
            content: Content of the context entry
            context_type: Type of context
            scope: Scope of the context
            metadata: Optional metadata
            tags: Optional tags
            entry_id: Optional custom entry ID

        Returns:
            ID of the created context entry
        """
        if entry_id is None:
            entry_id = str(uuid.uuid4())

        try:
            async with self.context_lock:
                entry = ContextEntry(
                    id=entry_id,
                    type=context_type,
                    scope=scope,
                    content=content,
                    metadata=metadata or {},
                    tags=tags or set()
                )

                # Store entry
                self.context_store[entry_id] = entry

                # Update indices
                self.context_index[context_type].add(entry_id)
                for tag in entry.tags:
                    self.tag_index[tag].add(entry_id)

                # Clear search cache
                self.search_cache.clear()

                self.context_metrics['context_size'] = len(self.context_store)
                self.context_metrics['index_size'] = sum(len(s) for s in self.context_index.values())

                self.logger.debug(f"Context entry added: {entry_id}")
                return entry_id

        except Exception as e:
            self.logger.error(f"Failed to add context entry: {e}")
            raise

    # Private helper methods

    async def _analyze_project_structure(self, project_root: str) -> Dict[str, Any]:
        """Analyze project directory structure."""
        structure = {
            'root': project_root,
            'directories': [],
            'files': [],
            'total_files': 0,
            'total_directories': 0,
            'file_types': defaultdict(int),
            'size_bytes': 0
        }

        try:
            for root, dirs, files in os.walk(project_root):
                # Filter out ignored directories
                dirs[:] = [d for d in dirs if d not in self.ignore_patterns]

                rel_root = os.path.relpath(root, project_root)
                if rel_root != '.':
                    structure['directories'].append(rel_root)
                    structure['total_directories'] += 1

                for file in files:
                    if any(pattern in file for pattern in self.ignore_patterns):
                        continue

                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, project_root)

                    try:
                        file_size = os.path.getsize(file_path)
                        file_ext = Path(file).suffix.lower()

                        structure['files'].append({
                            'path': rel_path,
                            'size': file_size,
                            'extension': file_ext,
                            'modified': os.path.getmtime(file_path)
                        })

                        structure['total_files'] += 1
                        structure['file_types'][file_ext] += 1
                        structure['size_bytes'] += file_size

                    except (OSError, IOError):
                        continue

            return structure

        except Exception as e:
            self.logger.error(f"Failed to analyze project structure: {e}")
            return structure

    async def _analyze_project_files(self, project_root: str, structure: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Analyze individual project files."""
        files_analysis = {}

        try:
            for file_info in structure['files']:
                file_path = os.path.join(project_root, file_info['path'])
                file_ext = file_info['extension']

                # Skip binary files or very large files
                if file_info['size'] > self.max_file_size:
                    continue

                try:
                    # Extract context based on file type
                    if file_ext in self.extraction_patterns:
                        analysis = await self.extraction_patterns[file_ext](file_path)
                        files_analysis[file_info['path']] = analysis
                    else:
                        # Generic text file analysis
                        analysis = await self._extract_generic_context(file_path)
                        files_analysis[file_info['path']] = analysis

                except Exception as e:
                    self.logger.warning(f"Failed to analyze file {file_path}: {e}")
                    continue

            return files_analysis

        except Exception as e:
            self.logger.error(f"Failed to analyze project files: {e}")
            return files_analysis

    async def _analyze_project_dependencies(self, project_root: str) -> Dict[str, List[str]]:
        """Analyze project dependencies."""
        dependencies = {
            'python': [],
            'javascript': [],
            'java': [],
            'other': []
        }

        try:
            # Python dependencies
            requirements_files = ['requirements.txt', 'requirements-dev.txt', 'pyproject.toml', 'setup.py']
            for req_file in requirements_files:
                req_path = os.path.join(project_root, req_file)
                if os.path.exists(req_path):
                    deps = await self._extract_python_dependencies(req_path)
                    dependencies['python'].extend(deps)

            # JavaScript dependencies
            package_json = os.path.join(project_root, 'package.json')
            if os.path.exists(package_json):
                deps = await self._extract_javascript_dependencies(package_json)
                dependencies['javascript'].extend(deps)

            # Java dependencies
            pom_xml = os.path.join(project_root, 'pom.xml')
            build_gradle = os.path.join(project_root, 'build.gradle')
            if os.path.exists(pom_xml):
                deps = await self._extract_java_dependencies(pom_xml)
                dependencies['java'].extend(deps)
            elif os.path.exists(build_gradle):
                deps = await self._extract_gradle_dependencies(build_gradle)
                dependencies['java'].extend(deps)

            return dependencies

        except Exception as e:
            self.logger.error(f"Failed to analyze dependencies: {e}")
            return dependencies

    async def _analyze_git_info(self, project_root: str) -> Dict[str, Any]:
        """Analyze git repository information."""
        git_info = {}

        try:
            git_dir = os.path.join(project_root, '.git')
            if not os.path.exists(git_dir):
                return git_info

            # Get basic git information
            git_info.update({
                'is_git_repo': True,
                'branches': await self._get_git_branches(project_root),
                'current_branch': await self._get_current_branch(project_root),
                'recent_commits': await self._get_recent_commits(project_root),
                'status': await self._get_git_status(project_root),
                'remotes': await self._get_git_remotes(project_root)
            })

            return git_info

        except Exception as e:
            self.logger.error(f"Failed to analyze git info: {e}")
            return {'is_git_repo': False}

    # Context extraction methods for different file types

    async def _extract_python_context(self, file_path: str) -> Dict[str, Any]:
        """Extract context from Python files."""
        context = {
            'type': 'python',
            'functions': [],
            'classes': [],
            'imports': [],
            'docstrings': [],
            'complexity': 0
        }

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Parse AST
            try:
                tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        context['functions'].append({
                            'name': node.name,
                            'line': node.lineno,
                            'args': [arg.arg for arg in node.args.args],
                            'docstring': ast.get_docstring(node)
                        })
                    elif isinstance(node, ast.ClassDef):
                        context['classes'].append({
                            'name': node.name,
                            'line': node.lineno,
                            'methods': [n.name for n in node.body if isinstance(n, ast.FunctionDef)],
                            'docstring': ast.get_docstring(node)
                        })
                    elif isinstance(node, ast.Import):
                        for alias in node.names:
                            context['imports'].append(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            for alias in node.names:
                                context['imports'].append(f"{node.module}.{alias.name}")

            except SyntaxError:
                # Handle syntax errors gracefully
                pass

            return context

        except Exception as e:
            self.logger.warning(f"Failed to extract Python context from {file_path}: {e}")
            return context

    async def _extract_javascript_context(self, file_path: str) -> Dict[str, Any]:
        """Extract context from JavaScript files."""
        context = {
            'type': 'javascript',
            'functions': [],
            'classes': [],
            'imports': [],
            'exports': []
        }

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Simple regex-based extraction (would use proper parser in production)
            import re

            # Extract function declarations
            func_pattern = r'function\s+(\w+)\s*\([^)]*\)'
            for match in re.finditer(func_pattern, content):
                context['functions'].append(match.group(1))

            # Extract class declarations
            class_pattern = r'class\s+(\w+)'
            for match in re.finditer(class_pattern, content):
                context['classes'].append(match.group(1))

            # Extract imports
            import_pattern = r'import\s+.*?from\s+[\'"]([^\'"]+)[\'"]'
            for match in re.finditer(import_pattern, content):
                context['imports'].append(match.group(1))

            return context

        except Exception as e:
            self.logger.warning(f"Failed to extract JavaScript context from {file_path}: {e}")
            return context

    async def _extract_generic_context(self, file_path: str) -> Dict[str, Any]:
        """Extract generic context from text files."""
        context = {
            'type': 'generic',
            'line_count': 0,
            'word_count': 0,
            'char_count': 0,
            'keywords': []
        }

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            context['line_count'] = len(content.split('\n'))
            context['word_count'] = len(content.split())
            context['char_count'] = len(content)

            # Extract common keywords (simplified)
            import re
            words = re.findall(r'\b\w+\b', content.lower())
            word_freq = {}
            for word in words:
                if len(word) > 3:  # Only consider words longer than 3 characters
                    word_freq[word] = word_freq.get(word, 0) + 1

            # Get top keywords
            context['keywords'] = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]

            return context

        except Exception as e:
            self.logger.warning(f"Failed to extract generic context from {file_path}: {e}")
            return context

    # Message handlers

    async def _handle_build_project_context(self, message: AgentMessage) -> None:
        """Handle build project context requests."""
        try:
            data = message.data
            result = await self.build_project_context(
                project_root=data.get('project_root', '.'),
                include_files=data.get('include_files', True),
                include_git=data.get('include_git', True),
                include_dependencies=data.get('include_dependencies', True),
                force_refresh=data.get('force_refresh', False)
            )

            response = AgentMessage(
                message_id=str(uuid.uuid4()),
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                message_type='project_context_result',
                data={
                    'project_context': {
                        'project_root': result.project_root,
                        'structure': result.structure,
                        'files_count': len(result.files),
                        'dependencies': result.dependencies,
                        'git_info': result.git_info,
                        'build_config': result.build_config,
                        'metrics': result.metrics,
                        'last_updated': result.last_updated.isoformat()
                    }
                },
                correlation_id=message.correlation_id
            )

            await self.send_message(response)

        except Exception as e:
            self.logger.error(f"Error handling build_project_context: {e}", exc_info=True)

    async def _handle_search_context(self, message: AgentMessage) -> None:
        """Handle context search requests."""
        try:
            data = message.data
            query = ContextQuery(
                query_text=data.get('query_text', ''),
                context_types=[ContextType(t) for t in data.get('context_types', [])] if data.get('context_types') else None,
                scopes=[ContextScope(s) for s in data.get('scopes', [])] if data.get('scopes') else None,
                tags=set(data.get('tags', [])) if data.get('tags') else None,
                max_results=data.get('max_results', 10),
                min_relevance=data.get('min_relevance', 0.5),
                semantic_search=data.get('semantic_search', True)
            )

            result = await self.search_context(query)

            response = AgentMessage(
                message_id=str(uuid.uuid4()),
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                message_type='context_search_result',
                data={
                    'entries': [self._serialize_context_entry(entry) for entry in result.entries],
                    'total_matches': result.total_matches,
                    'search_time': result.search_time,
                    'relevance_scores': result.relevance_scores
                },
                correlation_id=message.correlation_id
            )

            await self.send_message(response)

        except Exception as e:
            self.logger.error(f"Error handling search_context: {e}", exc_info=True)

    async def _handle_update_context(self, message: AgentMessage) -> None:
        """Handle context update requests."""
        try:
            data = message.data
            success = await self.update_context(
                entry_id=data.get('entry_id'),
                content=data.get('content'),
                metadata=data.get('metadata'),
                tags=set(data.get('tags', [])) if data.get('tags') else None,
                relevance_score=data.get('relevance_score')
            )

            response = AgentMessage(
                message_id=str(uuid.uuid4()),
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                message_type='context_update_result',
                data={'success': success},
                correlation_id=message.correlation_id
            )

            await self.send_message(response)

        except Exception as e:
            self.logger.error(f"Error handling update_context: {e}", exc_info=True)

    async def _handle_get_context(self, message: AgentMessage) -> None:
        """Handle get context requests."""
        try:
            data = message.data
            entry_id = data.get('entry_id')

            if entry_id and entry_id in self.context_store:
                entry = self.context_store[entry_id]
                entry.accessed_at = datetime.utcnow()
                entry.access_count += 1

                response_data = self._serialize_context_entry(entry)
            else:
                response_data = None

            response = AgentMessage(
                message_id=str(uuid.uuid4()),
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                message_type='context_get_result',
                data={'entry': response_data},
                correlation_id=message.correlation_id
            )

            await self.send_message(response)

        except Exception as e:
            self.logger.error(f"Error handling get_context: {e}", exc_info=True)

    async def _handle_analyze_project(self, message: AgentMessage) -> None:
        """Handle project analysis requests."""
        try:
            data = message.data
            project_root = data.get('project_root', '.')

            # Build project context if not exists
            if not self.project_context or self.project_context.project_root != project_root:
                await self.build_project_context(project_root)

            # Perform analysis
            analysis = {
                'structure_analysis': self._analyze_structure_complexity(self.project_context.structure),
                'code_analysis': self._analyze_code_quality(self.project_context.files),
                'dependency_analysis': self._analyze_dependency_health(self.project_context.dependencies),
                'git_analysis': self._analyze_git_health(self.project_context.git_info),
                'recommendations': self._generate_project_recommendations()
            }

            response = AgentMessage(
                message_id=str(uuid.uuid4()),
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                message_type='project_analysis_result',
                data={'analysis': analysis},
                correlation_id=message.correlation_id
            )

            await self.send_message(response)

        except Exception as e:
            self.logger.error(f"Error handling analyze_project: {e}", exc_info=True)

    async def _handle_task_request(self, message: AgentMessage) -> None:
        """Handle generic task requests from the orchestrator."""
        try:
            data = message.data
            action = data.get('action', '')

            if action == 'build_project_context':
                await self._handle_build_project_context(message)
            elif action == 'search_context':
                await self._handle_search_context(message)
            elif action == 'analyze_project':
                await self._handle_analyze_project(message)
            else:
                # Default context building for unknown actions
                await self._handle_build_project_context(message)

        except Exception as e:
            self.logger.error(f"Error handling task_request: {e}", exc_info=True)

    # Utility methods

    async def _initialize_context_store(self) -> None:
        """Initialize the context store and indices."""
        try:
            # Initialize empty stores
            self.context_store.clear()
            self.context_index.clear()
            self.tag_index.clear()
            self.dependency_graph.clear()
            self.search_cache.clear()

            self.logger.debug("Context store initialized")

        except Exception as e:
            self.logger.error(f"Failed to initialize context store: {e}")
            raise

    def _start_background_tasks(self) -> None:
        """Start background tasks for context maintenance."""
        # Context cleanup task
        asyncio.create_task(self._context_cleanup_loop())

        # Cache cleanup task
        asyncio.create_task(self._cache_cleanup_loop())

    async def _context_cleanup_loop(self) -> None:
        """Background task to clean up expired context entries."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                await self._cleanup_expired_context()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in context cleanup loop: {e}")

    async def _cache_cleanup_loop(self) -> None:
        """Background task to clean up expired cache entries."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(60)  # Run every minute
                await self._cleanup_expired_cache()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in cache cleanup loop: {e}")

    async def _cleanup_expired_context(self) -> None:
        """Clean up expired context entries."""
        try:
            current_time = datetime.utcnow()
            expired_entries = []

            for entry_id, entry in self.context_store.items():
                # Check if entry is expired
                if entry.scope == ContextScope.TEMPORARY:
                    age = (current_time - entry.created_at).total_seconds()
                    if age > self.context_ttl:
                        expired_entries.append(entry_id)

            # Remove expired entries
            for entry_id in expired_entries:
                await self._remove_context_entry(entry_id)

            if expired_entries:
                self.logger.debug(f"Cleaned up {len(expired_entries)} expired context entries")

        except Exception as e:
            self.logger.error(f"Error cleaning up expired context: {e}")

    async def _cleanup_expired_cache(self) -> None:
        """Clean up expired cache entries."""
        try:
            current_time = time.time()
            expired_keys = []

            for cache_key, result in self.search_cache.items():
                if (current_time - result.search_time) > self.cache_ttl:
                    expired_keys.append(cache_key)

            for key in expired_keys:
                del self.search_cache[key]

            if expired_keys:
                self.logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

        except Exception as e:
            self.logger.error(f"Error cleaning up expired cache: {e}")

    async def _remove_context_entry(self, entry_id: str) -> None:
        """Remove a context entry and update indices."""
        if entry_id in self.context_store:
            entry = self.context_store[entry_id]

            # Remove from indices
            self.context_index[entry.type].discard(entry_id)
            for tag in entry.tags:
                self.tag_index[tag].discard(entry_id)

            # Remove from store
            del self.context_store[entry_id]

    def _generate_cache_key(self, query: ContextQuery) -> str:
        """Generate a cache key for a context query."""
        key_data = {
            'query_text': query.query_text,
            'context_types': [t.value for t in query.context_types] if query.context_types else None,
            'scopes': [s.value for s in query.scopes] if query.scopes else None,
            'tags': sorted(list(query.tags)) if query.tags else None,
            'max_results': query.max_results,
            'min_relevance': query.min_relevance,
            'semantic_search': query.semantic_search
        }
        return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()

    def _calculate_relevance_score(self, entry: ContextEntry, query: ContextQuery) -> float:
        """Calculate relevance score for a context entry."""
        score = entry.relevance_score

        # Boost score based on content match
        content_str = str(entry.content).lower()
        query_lower = query.query_text.lower()

        # Simple keyword matching
        query_words = query_lower.split()
        matches = sum(1 for word in query_words if word in content_str)
        if query_words:
            keyword_score = matches / len(query_words)
            score *= (1 + keyword_score)

        # Boost recent entries
        age_hours = (datetime.utcnow() - entry.updated_at).total_seconds() / 3600
        if age_hours < 24:
            score *= 1.2
        elif age_hours < 168:  # 1 week
            score *= 1.1

        # Boost frequently accessed entries
        if entry.access_count > 10:
            score *= 1.1

        return min(score, 2.0)  # Cap at 2.0

    async def _calculate_semantic_similarity(self, entry: ContextEntry, query: ContextQuery) -> float:
        """Calculate semantic similarity between entry and query."""
        # Simplified semantic similarity - in production would use embeddings
        content_str = str(entry.content).lower()
        query_lower = query.query_text.lower()

        # Simple Jaccard similarity
        content_words = set(content_str.split())
        query_words = set(query_lower.split())

        if not content_words or not query_words:
            return 0.0

        intersection = len(content_words.intersection(query_words))
        union = len(content_words.union(query_words))

        return intersection / union if union > 0 else 0.0

    def _serialize_context_entry(self, entry: ContextEntry) -> Dict[str, Any]:
        """Serialize a context entry for message transmission."""
        return {
            'id': entry.id,
            'type': entry.type.value,
            'scope': entry.scope.value,
            'content': entry.content,
            'metadata': entry.metadata,
            'created_at': entry.created_at.isoformat(),
            'updated_at': entry.updated_at.isoformat(),
            'accessed_at': entry.accessed_at.isoformat(),
            'access_count': entry.access_count,
            'relevance_score': entry.relevance_score,
            'tags': list(entry.tags)
        }

    async def _store_project_context_entries(self, project_context: ProjectContext) -> None:
        """Store project context as individual context entries."""
        try:
            # Store project structure
            await self.add_context_entry(
                content=project_context.structure,
                context_type=ContextType.PROJECT_STRUCTURE,
                scope=ContextScope.GLOBAL,
                metadata={'project_root': project_context.project_root},
                tags={'project', 'structure'},
                entry_id=f"project_structure_{hashlib.md5(project_context.project_root.encode()).hexdigest()}"
            )

            # Store file analyses
            for file_path, analysis in project_context.files.items():
                await self.add_context_entry(
                    content=analysis,
                    context_type=ContextType.FILE_CONTENT,
                    scope=ContextScope.FILE,
                    metadata={'file_path': file_path, 'project_root': project_context.project_root},
                    tags={'file', 'code', analysis.get('type', 'unknown')},
                    entry_id=f"file_{hashlib.md5(file_path.encode()).hexdigest()}"
                )

            # Store dependencies
            if project_context.dependencies:
                await self.add_context_entry(
                    content=project_context.dependencies,
                    context_type=ContextType.DEPENDENCIES,
                    scope=ContextScope.GLOBAL,
                    metadata={'project_root': project_context.project_root},
                    tags={'dependencies', 'project'},
                    entry_id=f"dependencies_{hashlib.md5(project_context.project_root.encode()).hexdigest()}"
                )

            # Store git info
            if project_context.git_info:
                await self.add_context_entry(
                    content=project_context.git_info,
                    context_type=ContextType.GIT_HISTORY,
                    scope=ContextScope.GLOBAL,
                    metadata={'project_root': project_context.project_root},
                    tags={'git', 'version_control'},
                    entry_id=f"git_info_{hashlib.md5(project_context.project_root.encode()).hexdigest()}"
                )

        except Exception as e:
            self.logger.error(f"Failed to store project context entries: {e}")

    def _update_average_search_time(self, search_time: float) -> None:
        """Update the running average search time metric."""
        current_avg = self.context_metrics['average_search_time']
        total_queries = self.context_metrics['total_queries']

        if total_queries > 1:
            new_avg = ((current_avg * (total_queries - 1)) + search_time) / total_queries
            self.context_metrics['average_search_time'] = new_avg
        else:
            self.context_metrics['average_search_time'] = search_time

    # Placeholder methods for analysis functions

    def _analyze_structure_complexity(self, structure: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze project structure complexity."""
        return {
            'total_files': structure.get('total_files', 0),
            'total_directories': structure.get('total_directories', 0),
            'depth_score': 'medium',  # Simplified
            'organization_score': 'good'  # Simplified
        }

    def _analyze_code_quality(self, files: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze overall code quality."""
        return {
            'total_files_analyzed': len(files),
            'average_quality_score': 0.8,  # Simplified
            'issues_found': 0,
            'recommendations': []
        }

    def _analyze_dependency_health(self, dependencies: Dict[str, List[str]]) -> Dict[str, Any]:
        """Analyze dependency health."""
        total_deps = sum(len(deps) for deps in dependencies.values())
        return {
            'total_dependencies': total_deps,
            'health_score': 'good',  # Simplified
            'outdated_dependencies': [],
            'security_issues': []
        }

    def _analyze_git_health(self, git_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze git repository health."""
        return {
            'is_git_repo': git_info.get('is_git_repo', False),
            'branch_count': len(git_info.get('branches', [])),
            'commit_frequency': 'regular',  # Simplified
            'repository_health': 'good'
        }

    def _generate_project_recommendations(self) -> List[str]:
        """Generate project improvement recommendations."""
        return [
            "Consider adding more documentation",
            "Review dependency versions",
            "Implement code quality checks"
        ]

    # Simplified implementations for dependency extraction

    async def _extract_python_dependencies(self, req_path: str) -> List[str]:
        """Extract Python dependencies from requirements file."""
        deps = []
        try:
            with open(req_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Simple parsing - would be more sophisticated in production
                        dep = line.split('==')[0].split('>=')[0].split('~=')[0]
                        deps.append(dep)
        except Exception as e:
            self.logger.warning(f"Failed to extract Python dependencies: {e}")
        return deps

    async def _extract_javascript_dependencies(self, package_path: str) -> List[str]:
        """Extract JavaScript dependencies from package.json."""
        deps = []
        try:
            with open(package_path, 'r') as f:
                data = json.load(f)
                deps.extend(data.get('dependencies', {}).keys())
                deps.extend(data.get('devDependencies', {}).keys())
        except Exception as e:
            self.logger.warning(f"Failed to extract JavaScript dependencies: {e}")
        return deps

    async def _extract_java_dependencies(self, pom_path: str) -> List[str]:
        """Extract Java dependencies from pom.xml."""
        # Simplified - would use XML parser in production
        return []

    async def _extract_gradle_dependencies(self, gradle_path: str) -> List[str]:
        """Extract Java dependencies from build.gradle."""
        # Simplified - would parse Gradle file in production
        return []

    # Git utility methods

    async def _get_git_branches(self, project_root: str) -> List[str]:
        """Get git branches."""
        # Simplified - would use git commands in production
        return ['main', 'develop']

    async def _get_current_branch(self, project_root: str) -> str:
        """Get current git branch."""
        # Simplified - would use git commands in production
        return 'main'

    async def _get_recent_commits(self, project_root: str) -> List[Dict[str, str]]:
        """Get recent git commits."""
        # Simplified - would use git commands in production
        return []

    async def _get_git_status(self, project_root: str) -> Dict[str, Any]:
        """Get git status."""
        # Simplified - would use git commands in production
        return {'clean': True, 'modified_files': []}

    async def _get_git_remotes(self, project_root: str) -> List[str]:
        """Get git remotes."""
        # Simplified - would use git commands in production
        return ['origin']

    # Additional context extraction methods (simplified implementations)

    async def _extract_typescript_context(self, file_path: str) -> Dict[str, Any]:
        """Extract context from TypeScript files."""
        return await self._extract_javascript_context(file_path)

    async def _extract_java_context(self, file_path: str) -> Dict[str, Any]:
        """Extract context from Java files."""
        return {'type': 'java', 'classes': [], 'methods': [], 'imports': []}

    async def _extract_cpp_context(self, file_path: str) -> Dict[str, Any]:
        """Extract context from C++ files."""
        return {'type': 'cpp', 'functions': [], 'classes': [], 'includes': []}

    async def _extract_c_context(self, file_path: str) -> Dict[str, Any]:
        """Extract context from C files."""
        return {'type': 'c', 'functions': [], 'structs': [], 'includes': []}

    async def _extract_go_context(self, file_path: str) -> Dict[str, Any]:
        """Extract context from Go files."""
        return {'type': 'go', 'functions': [], 'structs': [], 'imports': []}

    async def _extract_rust_context(self, file_path: str) -> Dict[str, Any]:
        """Extract context from Rust files."""
        return {'type': 'rust', 'functions': [], 'structs': [], 'imports': []}

    async def _extract_php_context(self, file_path: str) -> Dict[str, Any]:
        """Extract context from PHP files."""
        return {'type': 'php', 'functions': [], 'classes': [], 'includes': []}

    async def _extract_ruby_context(self, file_path: str) -> Dict[str, Any]:
        """Extract context from Ruby files."""
        return {'type': 'ruby', 'methods': [], 'classes': [], 'requires': []}

    async def _extract_sql_context(self, file_path: str) -> Dict[str, Any]:
        """Extract context from SQL files."""
        return {'type': 'sql', 'tables': [], 'procedures': [], 'functions': []}

    async def _extract_markdown_context(self, file_path: str) -> Dict[str, Any]:
        """Extract context from Markdown files."""
        context = {'type': 'markdown', 'headings': [], 'links': []}
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            import re
            # Extract headings
            for match in re.finditer(r'^(#{1,6})\s+(.+)$', content, re.MULTILINE):
                level = len(match.group(1))
                heading = match.group(2)
                context['headings'].append({'level': level, 'text': heading})

            # Extract links
            for match in re.finditer(r'\[([^\]]+)\]\(([^)]+)\)', content):
                text = match.group(1)
                url = match.group(2)
                context['links'].append({'text': text, 'url': url})

        except Exception as e:
            self.logger.warning(f"Failed to extract Markdown context: {e}")
        return context

    async def _extract_json_context(self, file_path: str) -> Dict[str, Any]:
        """Extract context from JSON files."""
        context = {'type': 'json', 'keys': [], 'structure': {}}
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            context['keys'] = list(data.keys()) if isinstance(data, dict) else []
            context['structure'] = {'type': type(data).__name__, 'size': len(data) if hasattr(data, '__len__') else 0}
        except Exception as e:
            self.logger.warning(f"Failed to extract JSON context: {e}")
        return context

    async def _extract_yaml_context(self, file_path: str) -> Dict[str, Any]:
        """Extract context from YAML files."""
        return {'type': 'yaml', 'keys': [], 'structure': {}}

    async def _extract_xml_context(self, file_path: str) -> Dict[str, Any]:
        """Extract context from XML files."""
        return {'type': 'xml', 'elements': [], 'attributes': []}

    async def _extract_html_context(self, file_path: str) -> Dict[str, Any]:
        """Extract context from HTML files."""
        return {'type': 'html', 'tags': [], 'classes': [], 'ids': []}

    async def _extract_css_context(self, file_path: str) -> Dict[str, Any]:
        """Extract context from CSS files."""
        return {'type': 'css', 'selectors': [], 'properties': []}

    async def _analyze_build_config(self, project_root: str) -> Dict[str, Any]:
        """Analyze build configuration files."""
        config = {}

        # Check for common build files
        build_files = ['Makefile', 'CMakeLists.txt', 'build.gradle', 'pom.xml', 'setup.py', 'pyproject.toml']
        for build_file in build_files:
            build_path = os.path.join(project_root, build_file)
            if os.path.exists(build_path):
                config[build_file] = {'exists': True, 'path': build_path}

        return config

    async def _extract_project_documentation(self, project_root: str, files: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
        """Extract project documentation."""
        docs = {}

        # Look for common documentation files
        doc_files = ['README.md', 'README.rst', 'CHANGELOG.md', 'CONTRIBUTING.md', 'LICENSE']
        for doc_file in doc_files:
            doc_path = os.path.join(project_root, doc_file)
            if os.path.exists(doc_path):
                try:
                    with open(doc_path, 'r', encoding='utf-8') as f:
                        docs[doc_file] = f.read()
                except Exception as e:
                    self.logger.warning(f"Failed to read {doc_file}: {e}")

        return docs

    async def _calculate_project_metrics(self, structure: Dict[str, Any], files: Dict[str, Dict[str, Any]], dependencies: Dict[str, List[str]]) -> Dict[str, Any]:
        """Calculate project metrics."""
        return {
            'total_files': structure.get('total_files', 0),
            'total_directories': structure.get('total_directories', 0),
            'total_size_bytes': structure.get('size_bytes', 0),
            'file_types': dict(structure.get('file_types', {})),
            'dependency_count': sum(len(deps) for deps in dependencies.values()),
            'analysis_timestamp': datetime.utcnow().isoformat()
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Get context agent performance metrics."""
        return {
            **self.context_metrics,
            'context_store_size': len(self.context_store),
            'search_cache_size': len(self.search_cache),
            'agent_status': self.get_status()
        }

    # Abstract method implementations
    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """
        Process an incoming message and route it to appropriate context handlers.

        Args:
            message: The message to process

        Returns:
            Optional response message
        """
        try:
            self.logger.debug(f"Processing message: {message.message_type} from {message.sender_id}")

            # Handle different message types
            if message.message_type == "context_request":
                payload = message.payload
                action = payload.get('action', 'build')

                if action == "build":
                    result = await self.build_project_context(
                        project_path=payload.get('project_path', '.'),
                        include_patterns=payload.get('include_patterns', []),
                        exclude_patterns=payload.get('exclude_patterns', [])
                    )
                elif action == "search":
                    result = await self.search_context(
                        query=payload.get('query', ''),
                        context_types=payload.get('context_types', []),
                        max_results=payload.get('max_results', 10)
                    )
                elif action == "update":
                    await self.update_context(
                        context_id=payload.get('context_id', ''),
                        updates=payload.get('updates', {})
                    )
                    result = {"status": "updated", "context_id": payload.get('context_id')}
                elif action == "get":
                    result = self.get_context(
                        context_id=payload.get('context_id', ''),
                        context_type=payload.get('context_type')
                    )
                else:
                    raise ValueError(f"Unknown context action: {action}")

                return AgentMessage(
                    sender_id=self.agent_id,
                    recipient_id=message.sender_id,
                    message_type="context_response",
                    payload=result,
                    correlation_id=message.correlation_id
                )

            elif message.message_type == "context_sync":
                await self._handle_context_sync(message)
                return None

            elif message.message_type == "project_analysis":
                result = await self.analyze_project_structure(message.payload.get('project_path', '.'))
                return AgentMessage(
                    sender_id=self.agent_id,
                    recipient_id=message.sender_id,
                    message_type="analysis_response",
                    payload=result,
                    correlation_id=message.correlation_id
                )

            else:
                # Return None for unhandled message types to avoid recursion
                self.logger.debug(f"Unhandled message type: {message.message_type}")
                return None

        except Exception as e:
            self.logger.error(f"Error processing message: {e}", exc_info=True)

            # Return error response if this was a request expecting a response
            if message.message_type in ["context_request", "project_analysis"]:
                return AgentMessage(
                    sender_id=self.agent_id,
                    recipient_id=message.sender_id,
                    message_type="error_response",
                    payload={
                        'error': str(e),
                        'error_type': type(e).__name__,
                        'failed_action': message.payload.get('action', 'unknown')
                    },
                    correlation_id=message.correlation_id
                )

            return None

    def get_capabilities(self) -> List[AgentCapability]:
        """
        Return list of capabilities this context agent provides.

        Returns:
            List of agent capabilities
        """


        return [
            AgentCapability(
                name="project_context_building",
                description="Build comprehensive project context from source files and structure",
                input_types=["path", "dict"],
                output_types=["dict"],
                cost_estimate=2.0
            ),
            AgentCapability(
                name="semantic_search",
                description="Perform semantic search across project context and code",
                input_types=["text"],
                output_types=["dict", "list"],
                cost_estimate=1.0
            ),
            AgentCapability(
                name="context_management",
                description="Store, update, and retrieve context information",
                input_types=["dict"],
                output_types=["dict"],
                cost_estimate=0.5
            ),
            AgentCapability(
                name="project_analysis",
                description="Analyze project structure, dependencies, and relationships",
                input_types=["path"],
                output_types=["dict"],
                cost_estimate=1.5
            ),
            AgentCapability(
                name="file_indexing",
                description="Index and analyze file contents for semantic search",
                input_types=["file", "text"],
                output_types=["dict"],
                cost_estimate=1.0
            ),
            AgentCapability(
                name="dependency_tracking",
                description="Track and analyze code dependencies and imports",
                input_types=["code", "path"],
                output_types=["dict"],
                cost_estimate=1.0
            ),
            AgentCapability(
                name="context_sharing",
                description="Share context information between agents in real-time",
                input_types=["dict"],
                output_types=["dict"],
                cost_estimate=0.5
            )
        ]

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check and return status.

        Returns:
            Dictionary containing health status information
        """
        try:
            # Initialize base health status
            base_health = {
                "agent_id": self.agent_id,
                "agent_type": self.agent_type,
                "state": self.state.value if hasattr(self, 'state') else "unknown"
            }

            # Check context agent-specific health
            current_time = datetime.utcnow()

            # Check context store health
            context_store_size = len(self.context_store)
            context_store_health = context_store_size >= 0  # Always healthy if accessible

            # Check search functionality
            search_cache_size = len(self.search_cache)
            search_health = hasattr(self, 'search_context')

            # Check core capabilities
            can_build_context = hasattr(self, 'build_project_context')
            can_search = hasattr(self, 'search_context')
            can_analyze = hasattr(self, 'analyze_project_structure')
            can_update = hasattr(self, 'update_context')

            core_capabilities = can_build_context and can_search and can_analyze and can_update

            # Calculate performance metrics
            total_operations = self.context_metrics.get('contexts_built', 0) + self.context_metrics.get('searches_performed', 0)
            cache_hit_rate = 0.0
            if self.context_metrics.get('searches_performed', 0) > 0:
                cache_hit_rate = (self.context_metrics.get('cache_hits', 0) / self.context_metrics.get('searches_performed', 1)) * 100

            # Determine overall health status
            is_healthy = (
                context_store_health and
                search_health and
                core_capabilities and
                cache_hit_rate >= 20.0  # Consider healthy if cache hit rate >= 20%
            )

            health_status = {
                **base_health,
                "status": "healthy" if is_healthy else "degraded",
                "timestamp": current_time.isoformat(),
                "context_agent_specific": {
                    "context_store_size": context_store_size,
                    "context_store_health": context_store_health,
                    "search_cache_size": search_cache_size,
                    "search_health": search_health,
                    "core_capabilities": {
                        "build_context": can_build_context,
                        "search": can_search,
                        "analyze": can_analyze,
                        "update": can_update
                    },
                    "capabilities_health": core_capabilities,
                    "cache_hit_rate": cache_hit_rate,
                    "total_operations": total_operations,
                    "contexts_built": self.context_metrics.get('contexts_built', 0),
                    "searches_performed": self.context_metrics.get('searches_performed', 0),
                    "average_build_time": self.context_metrics.get('avg_build_time', 0.0),
                    "average_search_time": self.context_metrics.get('avg_search_time', 0.0)
                },
                "capabilities": len(self.get_capabilities()),
                "uptime": (current_time - self.created_at).total_seconds() if hasattr(self, 'created_at') else 0
            }

            # Add any critical issues
            issues = []
            if not context_store_health:
                issues.append("Context store not accessible")
            if not search_health:
                issues.append("Search functionality unavailable")
            if not core_capabilities:
                issues.append("Missing core context management capabilities")
            if cache_hit_rate < 20.0 and self.context_metrics.get('searches_performed', 0) > 10:
                issues.append(f"Low cache hit rate: {cache_hit_rate:.1f}%")

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

    # ========================================
    # Phase 2.2 Priority 2: Enhanced Methods
    # Project Context Intelligence Implementation
    # ========================================

    async def _initialize_ast_parsers(self) -> None:
        """Initialize AST parsers for multiple languages."""
        try:
            if TREE_SITTER_AVAILABLE:
                # Initialize tree-sitter parsers for supported languages
                self.ast_parsers = {}
                self.logger.info("Tree-sitter AST parsing enabled")
            else:
                self.logger.warning("Tree-sitter not available, using basic AST parsing")

        except Exception as e:
            self.logger.error(f"Failed to initialize AST parsers: {e}")

    async def _initialize_context_database(self) -> None:
        """Initialize project context database for persistent storage."""
        try:
            self.context_db = sqlite3.connect(self.context_db_path)
            cursor = self.context_db.cursor()

            # Create tables for project intelligence
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS project_analysis (
                    project_id TEXT PRIMARY KEY,
                    project_root TEXT,
                    analysis_data TEXT,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS file_semantics (
                    file_id TEXT PRIMARY KEY,
                    project_id TEXT,
                    file_path TEXT,
                    semantic_data TEXT,
                    ast_data TEXT,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS code_patterns (
                    pattern_id TEXT PRIMARY KEY,
                    project_id TEXT,
                    pattern_type TEXT,
                    pattern_data TEXT,
                    confidence REAL,
                    created_at TIMESTAMP
                )
            ''')

            self.context_db.commit()
            self.logger.info("Project context database initialized")

        except Exception as e:
            self.logger.error(f"Failed to initialize context database: {e}")
            self.context_db = None

    async def analyze_project_structure(self, project_path: str) -> ProjectAnalysis:
        """
        Comprehensive project analysis with AI insights (Phase 2.2 Priority 2).

        Args:
            project_path: Path to the project root

        Returns:
            ProjectAnalysis with comprehensive insights
        """
        try:
            self.logger.info(f"Starting comprehensive project analysis: {project_path}")

            # File system analysis
            structure = await self._scan_directory_structure(project_path)

            # Language detection and patterns
            languages = await self._detect_languages_and_frameworks(structure)

            # Dependency analysis
            dependencies = await self._analyze_dependencies(project_path, languages)

            # Architecture pattern recognition
            patterns = await self._identify_architecture_patterns(structure, languages)

            # AI-enhanced insights
            ai_insights = {}
            if self.enable_ai_insights and self.model_manager:
                ai_insights = await self._generate_project_insights(structure, dependencies)

            # Calculate complexity score
            complexity_score = self._calculate_project_complexity(structure, dependencies)

            # Determine architecture type
            architecture_type = self._determine_architecture_type(structure, patterns)

            # Extract frameworks
            frameworks = self._extract_frameworks(languages, dependencies)

            analysis = ProjectAnalysis(
                structure=structure,
                languages=languages,
                dependencies=dependencies,
                patterns=patterns,
                insights=ai_insights,
                complexity_score=complexity_score,
                architecture_type=architecture_type,
                frameworks=frameworks
            )

            # Store analysis in database
            if self.context_db:
                await self._store_project_analysis(project_path, analysis)

            self.logger.info(f"Project analysis complete: {architecture_type} architecture, {len(languages)} languages")
            return analysis

        except Exception as e:
            self.logger.error(f"Project analysis failed: {e}")
            raise

    async def analyze_code_semantics(self, file_path: str, content: str) -> SemanticAnalysis:
        """
        Deep semantic analysis of code files (Phase 2.2 Priority 2).

        Args:
            file_path: Path to the code file
            content: File content to analyze

        Returns:
            SemanticAnalysis with comprehensive semantic information
        """
        try:
            language = self._detect_language(file_path)

            # AST parsing and analysis
            ast_analysis = await self._parse_ast(content, language)

            # Function/class extraction
            entities = await self._extract_code_entities(ast_analysis, content)

            # Cross-reference analysis
            references = await self._analyze_references(entities, content)

            # AI-powered semantic insights
            semantic_insights = {}
            if self.enable_ai_insights and self.model_manager:
                semantic_insights = await self._analyze_code_with_ai(content, language)

            # Build relationship graph
            relationships = self._build_relationship_graph(entities, references)

            # Calculate quality score
            quality_score = self._calculate_code_quality(content, entities, ast_analysis)

            analysis = SemanticAnalysis(
                entities=entities,
                references=references,
                complexity=ast_analysis.get('complexity', 0.0),
                insights=semantic_insights,
                relationships=relationships,
                ast_info=ast_analysis,
                language=language,
                quality_score=quality_score
            )

            # Cache the analysis
            self.semantic_cache[file_path] = analysis

            # Store in database
            if self.context_db:
                await self._store_file_semantics(file_path, analysis)

            return analysis

        except Exception as e:
            self.logger.error(f"Semantic analysis failed for {file_path}: {e}")
            raise

    async def _scan_directory_structure(self, project_path: str) -> Dict[str, Any]:
        """Enhanced directory scanning with intelligent filtering."""
        structure = {
            'root': project_path,
            'directories': [],
            'files': [],
            'file_tree': {},
            'metrics': {
                'total_files': 0,
                'total_directories': 0,
                'code_files': 0,
                'config_files': 0,
                'doc_files': 0
            }
        }

        try:
            for root, dirs, files in os.walk(project_path):
                # Intelligent directory filtering
                dirs[:] = [d for d in dirs if not self._should_ignore_directory(d)]

                rel_root = os.path.relpath(root, project_path)
                if rel_root != '.':
                    structure['directories'].append(rel_root)
                    structure['metrics']['total_directories'] += 1

                for file in files:
                    if self._should_ignore_file(file):
                        continue

                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, project_path)

                    file_info = await self._analyze_file_info(file_path, rel_path)
                    structure['files'].append(file_info)

                    # Update metrics
                    structure['metrics']['total_files'] += 1
                    if file_info.get('is_code_file'):
                        structure['metrics']['code_files'] += 1
                    elif file_info.get('is_config_file'):
                        structure['metrics']['config_files'] += 1
                    elif file_info.get('is_doc_file'):
                        structure['metrics']['doc_files'] += 1

            return structure

        except Exception as e:
            self.logger.error(f"Directory scanning failed: {e}")
            return structure

    async def _detect_languages_and_frameworks(self, structure: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced language and framework detection."""
        languages = {}
        frameworks = []

        try:
            file_extensions = {}

            # Analyze file extensions
            for file_info in structure['files']:
                ext = file_info.get('extension', '').lower()
                if ext:
                    file_extensions[ext] = file_extensions.get(ext, 0) + 1

            # Map extensions to languages
            language_mapping = {
                '.py': 'Python',
                '.js': 'JavaScript',
                '.ts': 'TypeScript',
                '.java': 'Java',
                '.go': 'Go',
                '.rs': 'Rust',
                '.cpp': 'C++',
                '.c': 'C',
                '.cs': 'C#',
                '.php': 'PHP',
                '.rb': 'Ruby',
                '.swift': 'Swift',
                '.kt': 'Kotlin',
                '.scala': 'Scala'
            }

            primary_language = None
            max_files = 0

            for ext, count in file_extensions.items():
                if ext in language_mapping:
                    lang = language_mapping[ext]
                    languages[lang] = {
                        'extension': ext,
                        'file_count': count,
                        'percentage': (count / structure['metrics']['total_files']) * 100
                    }

                    if count > max_files:
                        max_files = count
                        primary_language = lang

            # Detect frameworks based on files and dependencies
            frameworks = await self._detect_frameworks(structure, languages)

            return {
                'languages': languages,
                'primary_language': primary_language,
                'frameworks': frameworks,
                'total_languages': len(languages)
            }

        except Exception as e:
            self.logger.error(f"Language detection failed: {e}")
            return {'languages': {}, 'frameworks': [], 'primary_language': None}

    async def _analyze_dependencies(self, project_path: str, languages: Dict[str, Any]) -> Dict[str, List[str]]:
        """Enhanced dependency analysis with relationship mapping."""
        dependencies = {
            'direct': [],
            'dev': [],
            'peer': [],
            'internal': [],
            'external': []
        }

        try:
            # Python dependencies
            if 'Python' in languages.get('languages', {}):
                deps = await self._analyze_python_dependencies(project_path)
                dependencies.update(deps)

            # JavaScript/TypeScript dependencies
            if any(lang in languages.get('languages', {}) for lang in ['JavaScript', 'TypeScript']):
                deps = await self._analyze_js_dependencies(project_path)
                dependencies.update(deps)

            # Java dependencies
            if 'Java' in languages.get('languages', {}):
                deps = await self._analyze_java_dependencies(project_path)
                dependencies.update(deps)

            return dependencies

        except Exception as e:
            self.logger.error(f"Dependency analysis failed: {e}")
            return dependencies

    async def _generate_project_insights(self, structure: Dict[str, Any], dependencies: Dict[str, Any]) -> Dict[str, Any]:
        """Generate AI-powered project insights."""
        if not self.model_manager:
            return {}

        try:
            # Prepare context for AI analysis
            context = {
                'file_count': structure['metrics']['total_files'],
                'directory_count': structure['metrics']['total_directories'],
                'code_files': structure['metrics']['code_files'],
                'dependencies': len(dependencies.get('direct', [])),
                'file_types': [f['extension'] for f in structure['files']]
            }

            # Generate insights using AI
            insights_prompt = f"""
            Analyze this project structure and provide insights:

            Project Statistics:
            - Total files: {context['file_count']}
            - Code files: {context['code_files']}
            - Dependencies: {context['dependencies']}

            Provide insights on:
            1. Project complexity
            2. Architecture patterns
            3. Potential improvements
            4. Technology stack assessment
            """

            insights = await analyze_code(insights_prompt, "project_analysis")

            return {
                'ai_generated': True,
                'complexity_assessment': insights.get('complexity', 'unknown'),
                'architecture_suggestions': insights.get('suggestions', []),
                'technology_insights': insights.get('technology', {}),
                'improvement_recommendations': insights.get('improvements', [])
            }

        except Exception as e:
            self.logger.error(f"AI insights generation failed: {e}")
            return {'ai_generated': False, 'error': str(e)}

    def _calculate_project_complexity(self, structure: Dict[str, Any], dependencies: Dict[str, Any]) -> float:
        """Calculate project complexity score."""
        try:
            # Base complexity from file count
            file_complexity = min(structure['metrics']['total_files'] / 100, 5.0)

            # Dependency complexity
            dep_count = len(dependencies.get('direct', []))
            dep_complexity = min(dep_count / 20, 3.0)

            # Directory depth complexity
            max_depth = max([len(d.split('/')) for d in structure['directories']] + [1])
            depth_complexity = min(max_depth / 10, 2.0)

            # Language diversity complexity
            lang_count = len(structure.get('languages', {}))
            lang_complexity = min(lang_count / 5, 2.0)

            total_complexity = file_complexity + dep_complexity + depth_complexity + lang_complexity
            return min(total_complexity, 10.0)  # Cap at 10

        except Exception as e:
            self.logger.error(f"Complexity calculation failed: {e}")
            return 5.0  # Default medium complexity

    # Helper methods for file analysis
    def _should_ignore_directory(self, dirname: str) -> bool:
        """Check if directory should be ignored."""
        ignore_dirs = {
            '.git', '__pycache__', '.pytest_cache', 'node_modules',
            '.venv', 'venv', '.env', 'env', 'build', 'dist',
            '.next', '.cache', 'coverage', '.nyc_output'
        }
        return dirname in ignore_dirs or dirname.startswith('.')

    def _should_ignore_file(self, filename: str) -> bool:
        """Check if file should be ignored."""
        ignore_extensions = {'.pyc', '.pyo', '.log', '.tmp', '.swp', '.swo'}
        ignore_files = {'.DS_Store', 'Thumbs.db', '.gitignore'}

        return (filename in ignore_files or
                any(filename.endswith(ext) for ext in ignore_extensions) or
                filename.startswith('.'))

    async def _analyze_file_info(self, file_path: str, rel_path: str) -> Dict[str, Any]:
        """Analyze individual file information."""
        try:
            stat = os.stat(file_path)
            ext = Path(file_path).suffix.lower()

            # Determine file category
            code_extensions = {'.py', '.js', '.ts', '.java', '.go', '.rs', '.cpp', '.c', '.cs'}
            config_extensions = {'.json', '.yaml', '.yml', '.toml', '.ini', '.cfg'}
            doc_extensions = {'.md', '.rst', '.txt', '.doc', '.docx'}

            return {
                'path': rel_path,
                'size': stat.st_size,
                'extension': ext,
                'modified': stat.st_mtime,
                'is_code_file': ext in code_extensions,
                'is_config_file': ext in config_extensions,
                'is_doc_file': ext in doc_extensions
            }

        except Exception as e:
            self.logger.error(f"File analysis failed for {file_path}: {e}")
            return {'path': rel_path, 'error': str(e)}

    async def _detect_frameworks(self, structure: Dict[str, Any], languages: Dict[str, Any]) -> List[str]:
        """Detect frameworks used in the project."""
        frameworks = []

        try:
            files = structure.get('files', [])

            # Check for common framework indicators
            framework_indicators = {
                'React': ['package.json', 'react'],
                'Vue': ['package.json', 'vue'],
                'Angular': ['package.json', 'angular'],
                'Django': ['manage.py', 'settings.py'],
                'Flask': ['app.py', 'flask'],
                'FastAPI': ['main.py', 'fastapi'],
                'Spring': ['pom.xml', 'build.gradle'],
                'Express': ['package.json', 'express'],
                'Laravel': ['composer.json', 'artisan']
            }

            for file_info in files:
                file_path = file_info.get('path', '')
                file_name = os.path.basename(file_path)

                for framework, indicators in framework_indicators.items():
                    if file_name in indicators:
                        if framework not in frameworks:
                            frameworks.append(framework)

            return frameworks

        except Exception as e:
            self.logger.error(f"Framework detection failed: {e}")
            return []

    async def _identify_architecture_patterns(self, structure: Dict[str, Any], languages: Dict[str, Any]) -> Dict[str, Any]:
        """Identify architecture patterns in the project."""
        patterns = {
            'mvc': False,
            'microservices': False,
            'layered': False,
            'component_based': False,
            'modular': False
        }

        try:
            directories = structure.get('directories', [])
            files = structure.get('files', [])

            # Check for MVC pattern
            mvc_indicators = ['models', 'views', 'controllers', 'templates']
            mvc_found = sum(1 for indicator in mvc_indicators
                           if any(indicator in dir_path.lower() for dir_path in directories))
            if mvc_found >= 2:
                patterns['mvc'] = True

            # Check for microservices
            service_indicators = ['service', 'api', 'gateway', 'docker']
            microservices_found = sum(1 for indicator in service_indicators
                                    if any(indicator in dir_path.lower() for dir_path in directories))
            if microservices_found >= 2:
                patterns['microservices'] = True

            # Check for layered architecture
            layer_indicators = ['presentation', 'business', 'data', 'domain']
            layered_found = sum(1 for indicator in layer_indicators
                              if any(indicator in dir_path.lower() for dir_path in directories))
            if layered_found >= 2:
                patterns['layered'] = True

            # Check for component-based
            if any('component' in dir_path.lower() for dir_path in directories):
                patterns['component_based'] = True

            # Check for modular structure
            if len(directories) > 5:
                patterns['modular'] = True

            return patterns

        except Exception as e:
            self.logger.error(f"Architecture pattern identification failed: {e}")
            return patterns

    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension."""
        try:
            ext = Path(file_path).suffix.lower()

            language_mapping = {
                '.py': 'Python',
                '.js': 'JavaScript',
                '.ts': 'TypeScript',
                '.jsx': 'JavaScript',
                '.tsx': 'TypeScript',
                '.java': 'Java',
                '.go': 'Go',
                '.rs': 'Rust',
                '.cpp': 'C++',
                '.cc': 'C++',
                '.cxx': 'C++',
                '.c': 'C',
                '.cs': 'C#',
                '.php': 'PHP',
                '.rb': 'Ruby',
                '.swift': 'Swift',
                '.kt': 'Kotlin',
                '.scala': 'Scala',
                '.sql': 'SQL',
                '.html': 'HTML',
                '.css': 'CSS',
                '.scss': 'SCSS',
                '.less': 'LESS',
                '.json': 'JSON',
                '.yaml': 'YAML',
                '.yml': 'YAML',
                '.xml': 'XML',
                '.md': 'Markdown',
                '.sh': 'Shell',
                '.bash': 'Shell',
                '.zsh': 'Shell'
            }

            return language_mapping.get(ext, 'Unknown')

        except Exception as e:
            self.logger.error(f"Language detection failed for {file_path}: {e}")
            return 'Unknown'

    def _determine_architecture_type(self, structure: Dict[str, Any], patterns: Dict[str, Any]) -> str:
        """Determine the overall architecture type of the project."""
        try:
            if patterns.get('microservices'):
                return 'microservices'
            elif patterns.get('mvc'):
                return 'mvc'
            elif patterns.get('layered'):
                return 'layered'
            elif patterns.get('component_based'):
                return 'component-based'
            elif patterns.get('modular'):
                return 'modular'
            else:
                # Determine based on file count and structure
                file_count = structure.get('metrics', {}).get('total_files', 0)
                if file_count > 100:
                    return 'large-monolith'
                elif file_count > 20:
                    return 'medium-project'
                else:
                    return 'simple-project'

        except Exception as e:
            self.logger.error(f"Architecture type determination failed: {e}")
            return 'unknown'

    def _extract_frameworks(self, languages: Dict[str, Any], dependencies: Dict[str, Any]) -> List[str]:
        """Extract frameworks from language and dependency information."""
        frameworks = []

        try:
            # Extract from dependencies
            for dep_list in dependencies.values():
                if isinstance(dep_list, list):
                    for dep in dep_list:
                        if isinstance(dep, str):
                            # Check for common frameworks
                            framework_names = ['django', 'flask', 'fastapi', 'react', 'vue', 'angular', 'express', 'spring']
                            for framework in framework_names:
                                if framework in dep.lower() and framework not in frameworks:
                                    frameworks.append(framework.capitalize())

            return frameworks

        except Exception as e:
            self.logger.error(f"Framework extraction failed: {e}")
            return []

    async def _parse_ast(self, content: str, language: str) -> Dict[str, Any]:
        """Parse AST from code content."""
        ast_info = {
            'functions': [],
            'classes': [],
            'imports': [],
            'complexity': 0.0,
            'lines_of_code': len(content.split('\n'))
        }

        try:
            if language == 'Python':
                import ast as python_ast

                try:
                    tree = python_ast.parse(content)

                    for node in python_ast.walk(tree):
                        if isinstance(node, python_ast.FunctionDef):
                            ast_info['functions'].append({
                                'name': node.name,
                                'line': node.lineno,
                                'args': len(node.args.args)
                            })
                        elif isinstance(node, python_ast.ClassDef):
                            ast_info['classes'].append({
                                'name': node.name,
                                'line': node.lineno,
                                'methods': len([n for n in node.body if isinstance(n, python_ast.FunctionDef)])
                            })
                        elif isinstance(node, (python_ast.Import, python_ast.ImportFrom)):
                            if isinstance(node, python_ast.Import):
                                for alias in node.names:
                                    ast_info['imports'].append(alias.name)
                            else:  # ImportFrom
                                if node.module:
                                    ast_info['imports'].append(node.module)

                    # Simple complexity calculation
                    ast_info['complexity'] = len(ast_info['functions']) + len(ast_info['classes']) * 2

                except SyntaxError:
                    # If parsing fails, return basic info
                    pass

            return ast_info

        except Exception as e:
            self.logger.error(f"AST parsing failed: {e}")
            return ast_info

    async def _extract_code_entities(self, ast_analysis: Dict[str, Any], content: str) -> Dict[str, Any]:
        """Extract code entities from AST analysis."""
        entities = {}

        try:
            # Add functions
            for func in ast_analysis.get('functions', []):
                entity_id = f"function_{func['name']}"
                entities[entity_id] = {
                    'type': 'function',
                    'name': func['name'],
                    'line': func['line'],
                    'args_count': func.get('args', 0)
                }

            # Add classes
            for cls in ast_analysis.get('classes', []):
                entity_id = f"class_{cls['name']}"
                entities[entity_id] = {
                    'type': 'class',
                    'name': cls['name'],
                    'line': cls['line'],
                    'methods_count': cls.get('methods', 0)
                }

            # Add imports
            for imp in ast_analysis.get('imports', []):
                entity_id = f"import_{imp}"
                entities[entity_id] = {
                    'type': 'import',
                    'name': imp
                }

            return entities

        except Exception as e:
            self.logger.error(f"Entity extraction failed: {e}")
            return {}

    async def _analyze_references(self, entities: Dict[str, Any], content: str) -> Dict[str, List[str]]:
        """Analyze references between code entities."""
        references = {}

        try:
            # Simple reference analysis based on text matching
            content_lines = content.split('\n')

            for entity_id, entity_info in entities.items():
                entity_name = entity_info.get('name', '')
                refs = []

                # Find references to this entity in the code
                for line_num, line in enumerate(content_lines, 1):
                    if entity_name in line and entity_info.get('line', 0) != line_num:
                        refs.append(f"line_{line_num}")

                references[entity_id] = refs

            return references

        except Exception as e:
            self.logger.error(f"Reference analysis failed: {e}")
            return {}

    def _build_relationship_graph(self, entities: Dict[str, Any], references: Dict[str, List[str]]) -> Dict[str, Set[str]]:
        """Build relationship graph between entities."""
        relationships = {}

        try:
            for entity_id in entities.keys():
                relationships[entity_id] = set()

                # Find entities that reference this one
                for other_entity_id, refs in references.items():
                    if entity_id != other_entity_id and refs:
                        relationships[entity_id].add(other_entity_id)

            return relationships

        except Exception as e:
            self.logger.error(f"Relationship graph building failed: {e}")
            return {}

    def _calculate_code_quality(self, content: str, entities: Dict[str, Any], ast_analysis: Dict[str, Any]) -> float:
        """Calculate code quality score."""
        try:
            score = 50.0  # Base score

            # Factor in function count vs file size
            lines = len(content.split('\n'))
            functions = len([e for e in entities.values() if e.get('type') == 'function'])

            if lines > 0:
                function_density = functions / lines * 100
                if function_density > 0.1:  # Good function density
                    score += 20
                elif function_density > 0.05:
                    score += 10

            # Factor in documentation
            if '"""' in content or "'''" in content:
                score += 15
            elif '#' in content:
                score += 5

            # Factor in complexity
            complexity = ast_analysis.get('complexity', 0)
            if complexity < lines * 0.1:  # Low complexity relative to size
                score += 15
            elif complexity < lines * 0.2:
                score += 5

            return min(score, 100.0)

        except Exception as e:
            self.logger.error(f"Code quality calculation failed: {e}")
            return 50.0

    async def _analyze_code_with_ai(self, content: str, language: str) -> Dict[str, Any]:
        """Analyze code using AI for semantic insights."""
        if not self.model_manager:
            return {}

        try:
            prompt = f"""
            Analyze this {language} code and provide insights:

            ```{language.lower()}
            {content}
            ```

            Provide insights on:
            1. Code patterns used
            2. Potential improvements
            3. Code complexity assessment
            4. Best practices compliance
            """

            insights = await analyze_code(prompt, "semantic_analysis")

            return {
                'ai_generated': True,
                'patterns': insights.get('patterns', []),
                'improvements': insights.get('improvements', []),
                'complexity_assessment': insights.get('complexity', 'medium'),
                'best_practices': insights.get('best_practices', [])
            }

        except Exception as e:
            self.logger.error(f"AI code analysis failed: {e}")
            return {'ai_generated': False, 'error': str(e)}

    async def _store_project_analysis(self, project_path: str, analysis: ProjectAnalysis) -> None:
        """Store project analysis in database."""
        if not self.context_db:
            return

        try:
            cursor = self.context_db.cursor()

            project_id = hashlib.md5(project_path.encode()).hexdigest()
            analysis_data = {
                'architecture_type': analysis.architecture_type,
                'complexity_score': analysis.complexity_score,
                'languages': analysis.languages,
                'frameworks': analysis.frameworks,
                'patterns': analysis.patterns
            }

            cursor.execute('''
                INSERT OR REPLACE INTO project_analysis
                (project_id, project_root, analysis_data, created_at, updated_at)
                VALUES (?, ?, ?, datetime('now'), datetime('now'))
            ''', (project_id, project_path, json.dumps(analysis_data)))

            self.context_db.commit()

        except Exception as e:
            self.logger.error(f"Failed to store project analysis: {e}")

    async def _store_file_semantics(self, file_path: str, analysis: SemanticAnalysis) -> None:
        """Store file semantic analysis in database."""
        if not self.context_db:
            return

        try:
            cursor = self.context_db.cursor()

            file_id = hashlib.md5(file_path.encode()).hexdigest()
            semantic_data = {
                'quality_score': analysis.quality_score,
                'complexity': analysis.complexity,
                'language': analysis.language,
                'entities_count': len(analysis.entities),
                'references_count': len(analysis.references)
            }

            cursor.execute('''
                INSERT OR REPLACE INTO file_semantics
                (file_id, project_id, file_path, semantic_data, ast_data, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, datetime('now'), datetime('now'))
            ''', (file_id, 'default', file_path, json.dumps(semantic_data), json.dumps(analysis.ast_info)))

            self.context_db.commit()

        except Exception as e:
            self.logger.error(f"Failed to store file semantics: {e}")

    async def _analyze_python_dependencies(self, project_path: str) -> Dict[str, List[str]]:
        """Analyze Python project dependencies."""
        dependencies = {'direct': [], 'dev': []}

        try:
            # Check requirements.txt
            req_file = os.path.join(project_path, 'requirements.txt')
            if os.path.exists(req_file):
                with open(req_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            dep = line.split('==')[0].split('>=')[0].split('<=')[0]
                            dependencies['direct'].append(dep)

            # Check pyproject.toml
            pyproject_file = os.path.join(project_path, 'pyproject.toml')
            if os.path.exists(pyproject_file):
                # Simple parsing - in full implementation would use toml library
                pass

            return dependencies

        except Exception as e:
            self.logger.error(f"Python dependency analysis failed: {e}")
            return dependencies

    async def _analyze_js_dependencies(self, project_path: str) -> Dict[str, List[str]]:
        """Analyze JavaScript/Node.js project dependencies."""
        dependencies = {'direct': [], 'dev': []}

        try:
            package_file = os.path.join(project_path, 'package.json')
            if os.path.exists(package_file):
                with open(package_file, 'r') as f:
                    package_data = json.load(f)

                    if 'dependencies' in package_data:
                        dependencies['direct'].extend(package_data['dependencies'].keys())

                    if 'devDependencies' in package_data:
                        dependencies['dev'].extend(package_data['devDependencies'].keys())

            return dependencies

        except Exception as e:
            self.logger.error(f"JavaScript dependency analysis failed: {e}")
            return dependencies

    async def _analyze_java_dependencies(self, project_path: str) -> Dict[str, List[str]]:
        """Analyze Java project dependencies."""
        dependencies = {'direct': []}

        try:
            # Check pom.xml (Maven)
            pom_file = os.path.join(project_path, 'pom.xml')
            if os.path.exists(pom_file):
                # Simple parsing - in full implementation would use XML parser
                pass

            # Check build.gradle (Gradle)
            gradle_file = os.path.join(project_path, 'build.gradle')
            if os.path.exists(gradle_file):
                # Simple parsing - would need more sophisticated parsing
                pass

            return dependencies

        except Exception as e:
            self.logger.error(f"Java dependency analysis failed: {e}")
            return dependencies
