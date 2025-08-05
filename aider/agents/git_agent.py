"""
GitAgent: Specialized agent for intelligent git operations and version control.

Phase 2.2 Priority 3 Enhancement: AI-Powered Git Operations Intelligence
This agent handles all git-related operations in the Aider Multi-Agent Hive Architecture:
- AI-powered intelligent commit message generation
- Smart branch management and operations
- AI-assisted merge conflict resolution
- Advanced change analysis and impact tracking
- Repository health monitoring with insights
- Automated git workflow intelligence
- Cross-agent integration with code changes
"""

import asyncio
import json
import logging
import os
import re
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import structlog

# AI and analysis imports for Phase 2.2 Priority 3
try:
    from ..models.model_manager import analyze_code, ModelManager, generate_code
    AI_MODELS_AVAILABLE = True
except ImportError:
    AI_MODELS_AVAILABLE = False

try:
    import difflib
    DIFFLIB_AVAILABLE = True
except ImportError:
    DIFFLIB_AVAILABLE = False

from .base_agent import BaseAgent, AgentMessage, MessagePriority, AgentState, AgentCapability


class GitOperation(Enum):
    """Types of git operations."""
    COMMIT = "commit"
    BRANCH = "branch"
    MERGE = "merge"
    REBASE = "rebase"
    CHECKOUT = "checkout"
    PUSH = "push"
    PULL = "pull"
    FETCH = "fetch"
    STATUS = "status"
    LOG = "log"
    DIFF = "diff"
    ADD = "add"
    RESET = "reset"
    STASH = "stash"
    TAG = "tag"


class ChangeType(Enum):
    """Types of changes in git."""
    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"
    RENAMED = "renamed"
    COPIED = "copied"
    UNTRACKED = "untracked"
    STAGED = "staged"
    UNSTAGED = "unstaged"


class ConflictResolutionStrategy(Enum):
    """Strategies for resolving merge conflicts."""
    MANUAL = "manual"
    AUTOMATIC = "automatic"
    OURS = "ours"
    THEIRS = "theirs"
    SMART_MERGE = "smart_merge"
    INTERACTIVE = "interactive"


@dataclass
class GitFile:
    """Represents a file in git with change information."""
    path: str
    change_type: ChangeType
    additions: int = 0
    deletions: int = 0
    is_binary: bool = False
    old_path: Optional[str] = None
    diff_content: Optional[str] = None


@dataclass
class GitCommit:
    """Represents a git commit."""
    hash: str
    author: str
    email: str
    date: datetime
    message: str
    files_changed: List[GitFile]
    additions: int = 0
    deletions: int = 0
    is_merge: bool = False


@dataclass
class GitBranch:
    """Represents a git branch."""
    name: str
    is_current: bool = False
    is_remote: bool = False
    last_commit: Optional[str] = None
    commit_count: int = 0
    behind_count: int = 0
    ahead_count: int = 0


@dataclass
class GitStatus:
    """Represents current git repository status."""
    current_branch: str
    staged_files: List[GitFile]
    unstaged_files: List[GitFile]
    untracked_files: List[str]
    is_clean: bool
    has_conflicts: bool = False
    conflicted_files: List[str] = field(default_factory=list)
    ahead_count: int = 0
    behind_count: int = 0


@dataclass
class ConflictInfo:
    """Information about a merge conflict."""
    file_path: str
    conflict_markers: List[Dict[str, Any]]
    our_content: str
    their_content: str
    base_content: Optional[str] = None
    resolution_suggestions: List[str] = field(default_factory=list)


@dataclass
class CommitSuggestion:
    """AI-enhanced suggestion for commit message and structure."""
    message: str
    description: Optional[str] = None
    type: str = "feat"  # conventional commit type
    scope: Optional[str] = None
    breaking_change: bool = False
    files_to_stage: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    ai_generated: bool = False
    impact_analysis: Dict[str, Any] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)

@dataclass
class ChangeImpactAnalysis:
    """Analysis of change impact across the repository."""
    affected_components: List[str]
    risk_level: str  # low, medium, high, critical
    breaking_changes: List[str]
    test_recommendations: List[str]
    deployment_notes: List[str]
    rollback_strategy: Optional[str] = None
    confidence_score: float = 0.0


class GitAgent(BaseAgent):
    """
    AI-Enhanced agent for intelligent git operations and version control.

    Phase 2.2 Priority 3 Responsibilities:
    - Generate AI-powered intelligent commit messages
    - Smart branch management with strategy recommendations
    - AI-assisted merge conflict resolution
    - Advanced repository health and change impact analysis
    - Automated git workflow intelligence
    - Cross-agent integration with intelligent change tracking
    """

    def __init__(
        self,
        agent_id: str = "git_agent",
        config: Optional[Dict[str, Any]] = None,
        message_bus=None,
    ):
        """Initialize the git agent."""
        super().__init__(
            agent_id=agent_id,
            agent_type="git",
            config=config,
            message_bus=message_bus,
        )

        # Git agent specific configuration
        self.repository_path = self.config.get('repository_path', '.')
        self.auto_stage_changes = self.config.get('auto_stage_changes', True)
        self.commit_message_style = self.config.get('commit_message_style', 'conventional')
        self.max_commit_message_length = self.config.get('max_commit_message_length', 72)
        self.enable_conflict_resolution = self.config.get('enable_conflict_resolution', True)
        self.default_branch = self.config.get('default_branch', 'main')

        # Phase 2.2 Priority 3: AI-enhanced capabilities
        self.enable_ai_commit_messages = self.config.get('enable_ai_commit_messages', True)
        self.enable_change_impact_analysis = self.config.get('enable_change_impact_analysis', True)
        self.enable_smart_conflict_resolution = self.config.get('enable_smart_conflict_resolution', True)
        self.model_manager = None

        # Git command configuration
        self.git_timeout = self.config.get('git_timeout', 30.0)
        self.git_encoding = self.config.get('git_encoding', 'utf-8')

        # Current repository state
        self.current_status: Optional[GitStatus] = None
        self.repository_info: Dict[str, Any] = {}
        self.status_lock = asyncio.Lock()

        # Change tracking
        self.tracked_changes: Dict[str, GitFile] = {}
        self.pending_commits: List[CommitSuggestion] = []

        # Conflict resolution
        self.active_conflicts: Dict[str, ConflictInfo] = {}
        self.resolution_history: List[Dict[str, Any]] = []

        # Phase 2.2 Priority 3: Enhanced tracking
        self.change_impact_cache: Dict[str, ChangeImpactAnalysis] = {}
        self.ai_insights_cache: Dict[str, Dict[str, Any]] = {}
        self.smart_suggestions: List[Dict[str, Any]] = []

        # Performance metrics
        self.git_metrics = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'commits_created': 0,
            'conflicts_resolved': 0,
            'average_operation_time': 0.0,
            'operation_types': {}
        }

        # Commit message templates
        self.commit_templates = {
            'conventional': {
                'feat': 'feat({scope}): {description}',
                'fix': 'fix({scope}): {description}',
                'docs': 'docs({scope}): {description}',
                'style': 'style({scope}): {description}',
                'refactor': 'refactor({scope}): {description}',
                'test': 'test({scope}): {description}',
                'chore': 'chore({scope}): {description}'
            },
            'simple': '{type}: {description}',
            'detailed': '{type}({scope}): {description}\n\n{body}'
        }

        # Git command patterns
        self.git_patterns = {
            'status_pattern': re.compile(r'^(.)(.) (.+)$'),
            'branch_pattern': re.compile(r'^(\*?)(\s+)(.+?)(\s+\w+)?$'),
            'commit_pattern': re.compile(r'^([a-f0-9]+)\s+(.+)$'),
            'conflict_marker': re.compile(r'^<{7}|={7}|>{7}')
        }

    async def initialize(self) -> bool:
        """Initialize the AI-enhanced git agent."""
        try:
            # Initialize AI capabilities
            if AI_MODELS_AVAILABLE and self.enable_ai_commit_messages:
                try:
                    self.model_manager = ModelManager()
                    await self.model_manager.initialize()
                    self.logger.info("AI-powered git operations enabled")
                except Exception as e:
                    self.logger.warning(f"AI git capabilities disabled: {e}")
                    self.enable_ai_commit_messages = False
            await super().initialize()

            # Register git-specific message handlers
            self.register_message_handler('intelligent_commit', self._handle_intelligent_commit)
            self.register_message_handler('resolve_conflicts', self._handle_resolve_conflicts)
            self.register_message_handler('branch_operation', self._handle_branch_operation)
            self.register_message_handler('git_status', self._handle_git_status)
            self.register_message_handler('analyze_changes', self._handle_analyze_changes)
            self.register_message_handler('perform_git_operation', self._handle_perform_git_operation)
            self.register_message_handler('task_request', self._handle_task_request)

            # Validate git repository
            await self._validate_git_repository()

            # Initialize repository info
            await self._initialize_repository_info()

            self.logger.info("GitAgent initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize GitAgent: {e}")
            return False

    async def intelligent_commit(
        self,
        files_to_commit: Optional[List[str]] = None,
        commit_message: Optional[str] = None,
        auto_stage: bool = True,
        generate_message: bool = True
    ) -> Dict[str, Any]:
        """
        Create an intelligent commit with auto-generated message and staging.

        Args:
            files_to_commit: Specific files to commit (None for all staged)
            commit_message: Custom commit message (None to auto-generate)
            auto_stage: Whether to automatically stage changed files
            generate_message: Whether to generate commit message automatically

        Returns:
            Dictionary with commit results and metadata
        """
        start_time = time.time()
        operation_id = str(uuid.uuid4())

        self.logger.info(
            "Starting intelligent commit",
            operation_id=operation_id,
            files_to_commit=len(files_to_commit) if files_to_commit else "all",
            auto_stage=auto_stage,
            generate_message=generate_message
        )

        try:
            # Get current status
            status = await self.get_git_status()

            # Stage files if requested
            if auto_stage:
                if files_to_commit:
                    for file_path in files_to_commit:
                        await self._run_git_command(['add', file_path])
                else:
                    # Stage all modified and new files
                    modified_files = [f.path for f in status.unstaged_files if f.change_type != ChangeType.DELETED]
                    for file_path in modified_files:
                        await self._run_git_command(['add', file_path])

            # Get updated status after staging
            status = await self.get_git_status()

            if not status.staged_files:
                return {
                    'success': False,
                    'message': 'No files staged for commit',
                    'commit_hash': None
                }

            # Generate commit message if needed
            if generate_message and not commit_message:
                suggestion = await self._generate_commit_message(status.staged_files)
                commit_message = suggestion.message

            if not commit_message:
                commit_message = "Update files"  # Fallback

            # Create the commit
            result = await self._run_git_command(['commit', '-m', commit_message])

            # Extract commit hash
            commit_hash = await self._get_latest_commit_hash()

            # Analyze the commit
            commit_analysis = await self._analyze_commit(commit_hash)

            # Update metrics
            operation_time = time.time() - start_time
            self._update_operation_metrics(GitOperation.COMMIT, operation_time, True)
            self.git_metrics['commits_created'] += 1

            result_data = {
                'success': True,
                'commit_hash': commit_hash,
                'commit_message': commit_message,
                'files_committed': len(status.staged_files),
                'analysis': commit_analysis,
                'operation_time': operation_time
            }

            self.logger.info(
                "Intelligent commit completed",
                operation_id=operation_id,
                commit_hash=commit_hash,
                files_committed=len(status.staged_files),
                operation_time=operation_time
            )

            return result_data

        except Exception as e:
            self.logger.error(
                "Intelligent commit failed",
                operation_id=operation_id,
                error=str(e),
                exc_info=True
            )
            self._update_operation_metrics(GitOperation.COMMIT, time.time() - start_time, False)
            return {
                'success': False,
                'error': str(e),
                'commit_hash': None
            }

    async def resolve_conflicts(
        self,
        strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.SMART_MERGE,
        files_to_resolve: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Resolve merge conflicts intelligently.

        Args:
            strategy: Resolution strategy to use
            files_to_resolve: Specific files to resolve (None for all)

        Returns:
            Dictionary with resolution results
        """
        start_time = time.time()
        operation_id = str(uuid.uuid4())

        self.logger.info(
            "Starting conflict resolution",
            operation_id=operation_id,
            strategy=strategy.value,
            files_to_resolve=files_to_resolve
        )

        try:
            # Get current status and identify conflicts
            status = await self.get_git_status()

            if not status.has_conflicts:
                return {
                    'success': True,
                    'message': 'No conflicts to resolve',
                    'resolved_files': []
                }

            conflicts_to_resolve = files_to_resolve or status.conflicted_files
            resolved_files = []
            resolution_details = []

            for file_path in conflicts_to_resolve:
                if file_path in status.conflicted_files:
                    try:
                        # Analyze the conflict
                        conflict_info = await self._analyze_conflict(file_path)

                        # Apply resolution strategy
                        resolution_result = await self._apply_resolution_strategy(
                            file_path, conflict_info, strategy
                        )

                        if resolution_result['success']:
                            resolved_files.append(file_path)
                            resolution_details.append({
                                'file': file_path,
                                'strategy_used': strategy.value,
                                'changes_made': resolution_result.get('changes_made', []),
                                'confidence': resolution_result.get('confidence', 0.0)
                            })

                            # Mark as resolved
                            await self._run_git_command(['add', file_path])

                    except Exception as e:
                        self.logger.warning(f"Failed to resolve conflict in {file_path}: {e}")
                        continue

            # Update metrics
            operation_time = time.time() - start_time
            self._update_operation_metrics(GitOperation.MERGE, operation_time, len(resolved_files) > 0)
            self.git_metrics['conflicts_resolved'] += len(resolved_files)

            result_data = {
                'success': len(resolved_files) > 0,
                'resolved_files': resolved_files,
                'resolution_details': resolution_details,
                'remaining_conflicts': [f for f in status.conflicted_files if f not in resolved_files],
                'operation_time': operation_time
            }

            self.logger.info(
                "Conflict resolution completed",
                operation_id=operation_id,
                resolved_files=len(resolved_files),
                remaining_conflicts=len(result_data['remaining_conflicts']),
                operation_time=operation_time
            )

            return result_data

        except Exception as e:
            self.logger.error(
                "Conflict resolution failed",
                operation_id=operation_id,
                error=str(e),
                exc_info=True
            )
            self._update_operation_metrics(GitOperation.MERGE, time.time() - start_time, False)
            return {
                'success': False,
                'error': str(e),
                'resolved_files': []
            }

    async def get_git_status(self, refresh: bool = True) -> GitStatus:
        """
        Get current git repository status.

        Args:
            refresh: Whether to refresh the status from git

        Returns:
            GitStatus object with current repository state
        """
        if not refresh and self.current_status:
            return self.current_status

        try:
            async with self.status_lock:
                # Get current branch
                current_branch = await self._get_current_branch()

                # Get status information
                status_output = await self._run_git_command(['status', '--porcelain=v1'])

                staged_files = []
                unstaged_files = []
                untracked_files = []

                for line in status_output.split('\n'):
                    if not line.strip():
                        continue

                    match = self.git_patterns['status_pattern'].match(line)
                    if match:
                        staged_char, unstaged_char, file_path = match.groups()

                        # Handle staged changes
                        if staged_char and staged_char != ' ':
                            change_type = self._parse_status_char(staged_char)
                            staged_files.append(GitFile(path=file_path, change_type=change_type))

                        # Handle unstaged changes
                        if unstaged_char and unstaged_char != ' ':
                            change_type = self._parse_status_char(unstaged_char)
                            unstaged_files.append(GitFile(path=file_path, change_type=change_type))

                        # Handle untracked files
                        if staged_char == '?' and unstaged_char == '?':
                            untracked_files.append(file_path)

                # Check for conflicts
                conflicted_files = []
                has_conflicts = False

                try:
                    merge_head_path = os.path.join(self.repository_path, '.git', 'MERGE_HEAD')
                    if os.path.exists(merge_head_path):
                        has_conflicts = True
                        # Find conflicted files
                        for git_file in unstaged_files:
                            if await self._has_conflict_markers(git_file.path):
                                conflicted_files.append(git_file.path)
                except:
                    pass

                # Get ahead/behind counts
                ahead_count, behind_count = await self._get_ahead_behind_counts(current_branch)

                is_clean = (len(staged_files) == 0 and len(unstaged_files) == 0 and
                           len(untracked_files) == 0 and not has_conflicts)

                self.current_status = GitStatus(
                    current_branch=current_branch,
                    staged_files=staged_files,
                    unstaged_files=unstaged_files,
                    untracked_files=untracked_files,
                    is_clean=is_clean,
                    has_conflicts=has_conflicts,
                    conflicted_files=conflicted_files,
                    ahead_count=ahead_count,
                    behind_count=behind_count
                )

                return self.current_status

        except Exception as e:
            self.logger.error(f"Failed to get git status: {e}")
            # Return empty status on error
            return GitStatus(
                current_branch="unknown",
                staged_files=[],
                unstaged_files=[],
                untracked_files=[],
                is_clean=True
            )

    async def analyze_changes(self, since: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze changes in the repository.

        Args:
            since: Commit hash or reference to analyze changes since

        Returns:
            Dictionary with change analysis results
        """
        try:
            # Get current status
            status = await self.get_git_status()

            # Analyze staged changes
            staged_analysis = await self._analyze_file_changes(status.staged_files)

            # Analyze unstaged changes
            unstaged_analysis = await self._analyze_file_changes(status.unstaged_files)

            # Get recent commits if since is specified
            recent_commits = []
            if since:
                recent_commits = await self._get_commits_since(since)

            # Calculate change statistics
            total_files_changed = len(status.staged_files) + len(status.unstaged_files)
            total_additions = (staged_analysis.get('total_additions', 0) +
                             unstaged_analysis.get('total_additions', 0))
            total_deletions = (staged_analysis.get('total_deletions', 0) +
                             unstaged_analysis.get('total_deletions', 0))

            # Detect change patterns
            change_patterns = await self._detect_change_patterns(status.staged_files + status.unstaged_files)

            analysis_result = {
                'status': {
                    'current_branch': status.current_branch,
                    'is_clean': status.is_clean,
                    'has_conflicts': status.has_conflicts,
                    'ahead_count': status.ahead_count,
                    'behind_count': status.behind_count
                },
                'staged_changes': staged_analysis,
                'unstaged_changes': unstaged_analysis,
                'statistics': {
                    'total_files_changed': total_files_changed,
                    'total_additions': total_additions,
                    'total_deletions': total_deletions,
                    'untracked_files': len(status.untracked_files)
                },
                'change_patterns': change_patterns,
                'recent_commits': recent_commits,
                'recommendations': await self._generate_change_recommendations(status)
            }

            return analysis_result

        except Exception as e:
            self.logger.error(f"Failed to analyze changes: {e}")
            return {'error': str(e)}

    async def create_branch(self, branch_name: str, source_branch: Optional[str] = None) -> Dict[str, Any]:
        """Create a new git branch."""
        try:
            # Validate branch name
            if not self._is_valid_branch_name(branch_name):
                return {'success': False, 'error': 'Invalid branch name'}

            # Check if branch already exists
            existing_branches = await self._get_branch_list()
            if branch_name in [b.name for b in existing_branches]:
                return {'success': False, 'error': f'Branch {branch_name} already exists'}

            # Create branch
            if source_branch:
                await self._run_git_command(['checkout', '-b', branch_name, source_branch])
            else:
                await self._run_git_command(['checkout', '-b', branch_name])

            return {
                'success': True,
                'branch_name': branch_name,
                'source_branch': source_branch or 'current'
            }

        except Exception as e:
            self.logger.error(f"Failed to create branch {branch_name}: {e}")
            return {'success': False, 'error': str(e)}

    async def switch_branch(self, branch_name: str) -> Dict[str, Any]:
        """Switch to a different git branch."""
        try:
            # Check if branch exists
            existing_branches = await self._get_branch_list()
            if branch_name not in [b.name for b in existing_branches]:
                return {'success': False, 'error': f'Branch {branch_name} does not exist'}

            # Switch branch
            await self._run_git_command(['checkout', branch_name])

            # Update current status
            await self.get_git_status(refresh=True)

            return {
                'success': True,
                'previous_branch': self.current_status.current_branch if self.current_status else 'unknown',
                'current_branch': branch_name
            }

        except Exception as e:
            self.logger.error(f"Failed to switch to branch {branch_name}: {e}")
            return {'success': False, 'error': str(e)}

    # Private helper methods

    async def _validate_git_repository(self) -> None:
        """Validate that we're in a git repository."""
        try:
            await self._run_git_command(['rev-parse', '--git-dir'])
        except Exception as e:
            raise Exception(f"Not a git repository: {e}")

    async def _initialize_repository_info(self) -> None:
        """Initialize repository information."""
        try:
            # Get remote information
            remotes = await self._get_remote_info()

            # Get repository root
            repo_root = await self._run_git_command(['rev-parse', '--show-toplevel'])

            # Get current user configuration
            try:
                user_name = await self._run_git_command(['config', 'user.name'])
                user_email = await self._run_git_command(['config', 'user.email'])
            except:
                user_name = "Unknown"
                user_email = "unknown@example.com"

            self.repository_info = {
                'root_path': repo_root.strip(),
                'remotes': remotes,
                'user_name': user_name.strip(),
                'user_email': user_email.strip(),
                'initialized_at': datetime.utcnow()
            }

        except Exception as e:
            self.logger.warning(f"Failed to initialize repository info: {e}")
            self.repository_info = {}

    async def _run_git_command(self, args: List[str], cwd: Optional[str] = None) -> str:
        """Run a git command and return the output."""
        cmd = ['git'] + args
        working_dir = cwd or self.repository_path

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=working_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=dict(os.environ, LC_ALL='C')
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.git_timeout
            )

            if process.returncode != 0:
                error_msg = stderr.decode(self.git_encoding, errors='replace').strip()
                raise subprocess.CalledProcessError(process.returncode, cmd, error_msg)

            return stdout.decode(self.git_encoding, errors='replace').strip()

        except asyncio.TimeoutError:
            raise Exception(f"Git command timed out: {' '.join(cmd)}")
        except Exception as e:
            raise Exception(f"Git command failed: {' '.join(cmd)}: {str(e)}")

    async def _generate_commit_message(self, staged_files: List[GitFile]) -> CommitSuggestion:
        """Generate AI-powered intelligent commit message based on staged files."""
        try:
            # Try AI-powered generation first
            if self.enable_ai_commit_messages and self.model_manager:
                ai_suggestion = await self._generate_ai_commit_message(staged_files)
                if ai_suggestion:
                    return ai_suggestion

            # Fallback to rule-based generation
            # Analyze the changes
            file_analysis = await self._analyze_file_changes(staged_files)

            # Determine commit type and scope
            commit_type = self._determine_commit_type(staged_files, file_analysis)
            scope = self._determine_commit_scope(staged_files)

            # Generate description
            description = self._generate_commit_description(staged_files, file_analysis, commit_type)

            # Check for breaking changes
            breaking_change = await self._detect_breaking_changes(staged_files)

            # Build commit message
            if self.commit_message_style == 'conventional':
                if scope:
                    message = f"{commit_type}({scope}): {description}"
                else:
                    message = f"{commit_type}: {description}"
            else:
                message = f"{commit_type}: {description}"

            # Add breaking change indicator
            if breaking_change:
                message = message.replace(':', '!:', 1)

            # Truncate if too long
            if len(message) > self.max_commit_message_length:
                message = message[:self.max_commit_message_length - 3] + "..."

            # Calculate confidence score
            confidence = self._calculate_commit_confidence(staged_files, file_analysis)

            return CommitSuggestion(
                message=message,
                type=commit_type,
                scope=scope,
                breaking_change=breaking_change,
                files_to_stage=[f.path for f in staged_files],
                confidence_score=confidence
            )

        except Exception as e:
            self.logger.error(f"Failed to generate commit message: {e}")
            return CommitSuggestion(
                message="Update files",
                type="chore",
                confidence_score=0.1
            )

    def _determine_commit_type(self, staged_files: List[GitFile], analysis: Dict[str, Any]) -> str:
        """Determine the type of commit based on files and changes."""
        # Count different types of changes
        new_files = sum(1 for f in staged_files if f.change_type == ChangeType.ADDED)
        modified_files = sum(1 for f in staged_files if f.change_type == ChangeType.MODIFIED)
        deleted_files = sum(1 for f in staged_files if f.change_type == ChangeType.DELETED)

        # Check file types
        has_tests = any('test' in f.path.lower() for f in staged_files)
        has_docs = any(f.path.endswith(('.md', '.rst', '.txt')) for f in staged_files)
        has_config = any(f.path.endswith(('.json', '.yaml', '.yml', '.toml', '.ini')) for f in staged_files)

        # Determine type based on patterns
        if has_tests and len(staged_files) == sum([new_files, modified_files]) and 'test' in str(staged_files):
            return 'test'
        elif has_docs and not any(f.path.endswith(('.py', '.js', '.java', '.cpp')) for f in staged_files):
            return 'docs'
        elif has_config and len(staged_files) <= 3:
            return 'chore'
        elif new_files > modified_files and new_files > 0:
            return 'feat'
        elif any('fix' in f.path.lower() or 'bug' in f.path.lower() for f in staged_files):
            return 'fix'
        elif modified_files > 0 and new_files == 0:
            # Could be fix or refactor - need more analysis
            return 'fix' if analysis.get('likely_bug_fix', False) else 'refactor'
        else:
            return 'feat'

    def _determine_commit_scope(self, staged_files: List[GitFile]) -> Optional[str]:
        """Determine the scope of the commit."""
        if not staged_files:
            return None

        # Extract common directory/module
        paths = [f.path for f in staged_files]
        common_parts = []

        if len(paths) == 1:
            # Single file - use parent directory
            path_parts = Path(paths[0]).parent.parts
            if path_parts:
                return path_parts[-1]
        else:
            # Multiple files - find common path
            path_parts_list = [Path(p).parts for p in paths]
            if path_parts_list:
                min_length = min(len(parts) for parts in path_parts_list)
                for i in range(min_length):
                    parts_at_i = [parts[i] for parts in path_parts_list]
                    if len(set(parts_at_i)) == 1:
                        common_parts.append(parts_at_i[0])
                    else:
                        break

        # Return the most specific common part
        if common_parts:
            return common_parts[-1]

        return None

    def _generate_commit_description(self, staged_files: List[GitFile], analysis: Dict[str, Any], commit_type: str) -> str:
        """Generate a description for the commit."""
        file_count = len(staged_files)

        if file_count == 1:
            file_path = staged_files[0].path
            file_name = Path(file_path).name

            if commit_type == 'feat':
                return f"add {file_name}"
            elif commit_type == 'fix':
                return f"fix issue in {file_name}"
            elif commit_type == 'docs':
                return f"update {file_name}"
            elif commit_type == 'test':
                return f"add tests for {file_name}"
            else:
                return f"update {file_name}"
        else:
            if commit_type == 'feat':
                return f"add {file_count} new files"
            elif commit_type == 'fix':
                return f"fix issues in {file_count} files"
            elif commit_type == 'docs':
                return f"update documentation"
            elif commit_type == 'test':
                return f"add tests"
            else:
                return f"update {file_count} files"

    async def _detect_breaking_changes(self, staged_files: List[GitFile]) -> bool:
        """Detect if changes include breaking changes."""
        # Simple heuristics for breaking changes
        for git_file in staged_files:
            if git_file.change_type == ChangeType.DELETED:
                # Deleted files might be breaking
                if git_file.path.endswith(('.py', '.js', '.java')):
                    return True

            # Check for API changes (simplified)
            if git_file.diff_content:
                if 'def ' in git_file.diff_content and '-def ' in git_file.diff_content:
                    return True
                if 'class ' in git_file.diff_content and '-class ' in git_file.diff_content:
                    return True

        return False

    def _calculate_commit_confidence(self, staged_files: List[GitFile], analysis: Dict[str, Any]) -> float:
        """Calculate confidence score for the generated commit message."""
        base_confidence = 0.7

        # Increase confidence based on file patterns
        if len(staged_files) == 1:
            base_confidence += 0.2

        # Decrease confidence for many files
        if len(staged_files) > 10:
            base_confidence -= 0.3

        # Increase confidence for common patterns
        common_extensions = ['.py', '.js', '.java', '.cpp', '.md']
        if any(f.path.endswith(ext) for f in staged_files for ext in common_extensions):
            base_confidence += 0.1

        return max(0.1, min(1.0, base_confidence))

    async def _analyze_conflict(self, file_path: str) -> ConflictInfo:
        """Analyze a conflict in a specific file."""
        try:
            conflict_markers = []
            our_content = ""
            their_content = ""
            base_content = None

            with open(os.path.join(self.repository_path, file_path), 'r', encoding='utf-8') as f:
                content = f.read()

            # Find conflict markers
            lines = content.split('\n')
            in_conflict = False
            current_section = None

            for i, line in enumerate(lines):
                if line.startswith('<<<<<<<'):
                    in_conflict = True
                    current_section = 'ours'
                    conflict_markers.append({'type': 'start', 'line': i, 'content': line})
                elif line.startswith('======='):
                    current_section = 'theirs'
                    conflict_markers.append({'type': 'middle', 'line': i, 'content': line})
                elif line.startswith('>>>>>>>'):
                    in_conflict = False
                    current_section = None
                    conflict_markers.append({'type': 'end', 'line': i, 'content': line})
                elif in_conflict:
                    if current_section == 'ours':
                        our_content += line + '\n'
                    elif current_section == 'theirs':
                        their_content += line + '\n'

            # Generate resolution suggestions
            suggestions = self._generate_conflict_suggestions(our_content, their_content)

            return ConflictInfo(
                file_path=file_path,
                conflict_markers=conflict_markers,
                our_content=our_content.strip(),
                their_content=their_content.strip(),
                base_content=base_content,
                resolution_suggestions=suggestions
            )

        except Exception as e:
            self.logger.error(f"Failed to analyze conflict in {file_path}: {e}")
            return ConflictInfo(
                file_path=file_path,
                conflict_markers=[],
                our_content="",
                their_content="",
                resolution_suggestions=[]
            )

    def _generate_conflict_suggestions(self, our_content: str, their_content: str) -> List[str]:
        """Generate suggestions for resolving conflicts."""
        suggestions = []

        if not our_content.strip():
            suggestions.append("Accept their changes (our version is empty)")
        elif not their_content.strip():
            suggestions.append("Accept our changes (their version is empty)")
        elif our_content.strip() == their_content.strip():
            suggestions.append("Both versions are identical, accept either")
        else:
            suggestions.append("Review and merge both versions manually")
            suggestions.append("Accept our version")
            suggestions.append("Accept their version")

            # Check for simple cases
            if len(our_content.split('\n')) == 1 and len(their_content.split('\n')) == 1:
                suggestions.append("Single line conflict - choose the correct value")

        return suggestions

    async def _apply_resolution_strategy(
        self,
        file_path: str,
        conflict_info: ConflictInfo,
        strategy: ConflictResolutionStrategy
    ) -> Dict[str, Any]:
        """Apply a conflict resolution strategy to a file."""
        try:
            full_path = os.path.join(self.repository_path, file_path)

            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()

            changes_made = []

            if strategy == ConflictResolutionStrategy.OURS:
                # Keep our version
                resolved_content = self._resolve_conflicts_ours(content)
                changes_made.append("Resolved all conflicts by keeping our version")

            elif strategy == ConflictResolutionStrategy.THEIRS:
                # Keep their version
                resolved_content = self._resolve_conflicts_theirs(content)
                changes_made.append("Resolved all conflicts by keeping their version")

            elif strategy == ConflictResolutionStrategy.SMART_MERGE:
                # Attempt smart resolution
                resolved_content, smart_changes = self._resolve_conflicts_smart(content, conflict_info)
                changes_made.extend(smart_changes)

            else:
                # Default to manual resolution guidance
                return {
                    'success': False,
                    'message': 'Manual resolution required',
                    'changes_made': []
                }

            # Write resolved content
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(resolved_content)

            return {
                'success': True,
                'changes_made': changes_made,
                'confidence': 0.8 if strategy == ConflictResolutionStrategy.SMART_MERGE else 0.9
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'changes_made': []
            }

    def _resolve_conflicts_ours(self, content: str) -> str:
        """Resolve conflicts by keeping our version."""
        lines = content.split('\n')
        resolved_lines = []
        in_conflict = False

        for line in lines:
            if line.startswith('<<<<<<<'):
                in_conflict = True
                continue
            elif line.startswith('======='):
                continue
            elif line.startswith('>>>>>>>'):
                in_conflict = False
                continue
            elif not in_conflict or not line.startswith('======='):
                if in_conflict and line.startswith('======='):
                    continue
                if not in_conflict:
                    resolved_lines.append(line)
                elif in_conflict:
                    # We're in the "ours" section before =======
                    resolved_lines.append(line)

        return '\n'.join(resolved_lines)

    def _resolve_conflicts_theirs(self, content: str) -> str:
        """Resolve conflicts by keeping their version."""
        lines = content.split('\n')
        resolved_lines = []
        in_conflict = False
        in_theirs_section = False

        for line in lines:
            if line.startswith('<<<<<<<'):
                in_conflict = True
                in_theirs_section = False
                continue
            elif line.startswith('======='):
                in_theirs_section = True
                continue
            elif line.startswith('>>>>>>>'):
                in_conflict = False
                in_theirs_section = False
                continue
            elif not in_conflict:
                resolved_lines.append(line)
            elif in_conflict and in_theirs_section:
                resolved_lines.append(line)

        return '\n'.join(resolved_lines)

    def _resolve_conflicts_smart(self, content: str, conflict_info: ConflictInfo) -> Tuple[str, List[str]]:
        """Attempt smart conflict resolution."""
        changes_made = []

        # For now, use simple heuristics
        our_content = conflict_info.our_content
        their_content = conflict_info.their_content

        # If one side is empty, choose the non-empty one
        if not our_content.strip() and their_content.strip():
            resolved_content = self._resolve_conflicts_theirs(content)
            changes_made.append("Chose their version (our version was empty)")
        elif not their_content.strip() and our_content.strip():
            resolved_content = self._resolve_conflicts_ours(content)
            changes_made.append("Chose our version (their version was empty)")
        elif our_content.strip() == their_content.strip():
            resolved_content = self._resolve_conflicts_ours(content)
            changes_made.append("Both versions were identical")
        else:
            # Default to ours for now - more sophisticated logic could be added
            resolved_content = self._resolve_conflicts_ours(content)
            changes_made.append("Applied our version (manual review recommended)")

        return resolved_content, changes_made

    # Message handlers

    async def _handle_intelligent_commit(self, message: AgentMessage) -> None:
        """Handle intelligent commit requests."""
        try:
            data = message.data
            result = await self.intelligent_commit(
                files_to_commit=data.get('files_to_commit'),
                commit_message=data.get('commit_message'),
                auto_stage=data.get('auto_stage', True),
                generate_message=data.get('generate_message', True)
            )

            response = AgentMessage(
                message_id=str(uuid.uuid4()),
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                message_type='intelligent_commit_result',
                data=result,
                correlation_id=message.correlation_id
            )

            await self.send_message(response)

        except Exception as e:
            self.logger.error(f"Error handling intelligent_commit: {e}", exc_info=True)

    async def _handle_resolve_conflicts(self, message: AgentMessage) -> None:
        """Handle conflict resolution requests."""
        try:
            data = message.data
            strategy_name = data.get('strategy', 'smart_merge')
            strategy = ConflictResolutionStrategy(strategy_name)

            result = await self.resolve_conflicts(
                strategy=strategy,
                files_to_resolve=data.get('files_to_resolve')
            )

            response = AgentMessage(
                message_id=str(uuid.uuid4()),
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                message_type='resolve_conflicts_result',
                data=result,
                correlation_id=message.correlation_id
            )

            await self.send_message(response)

        except Exception as e:
            self.logger.error(f"Error handling resolve_conflicts: {e}", exc_info=True)

    async def _handle_branch_operation(self, message: AgentMessage) -> None:
        """Handle branch operation requests."""
        try:
            data = message.data
            operation = data.get('operation', 'status')

            if operation == 'create':
                result = await self.create_branch(
                    branch_name=data.get('branch_name'),
                    source_branch=data.get('source_branch')
                )
            elif operation == 'switch':
                result = await self.switch_branch(data.get('branch_name'))
            elif operation == 'list':
                branches = await self._get_branch_list()
                result = {'success': True, 'branches': [b.__dict__ for b in branches]}
            else:
                result = {'success': False, 'error': f'Unknown operation: {operation}'}

            response = AgentMessage(
                message_id=str(uuid.uuid4()),
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                message_type='branch_operation_result',
                data=result,
                correlation_id=message.correlation_id
            )

            await self.send_message(response)

        except Exception as e:
            self.logger.error(f"Error handling branch_operation: {e}", exc_info=True)

    async def _handle_git_status(self, message: AgentMessage) -> None:
        """Handle git status requests."""
        try:
            data = message.data
            refresh = data.get('refresh', True)

            status = await self.get_git_status(refresh=refresh)

            result = {
                'success': True,
                'status': {
                    'current_branch': status.current_branch,
                    'staged_files': [f.__dict__ for f in status.staged_files],
                    'unstaged_files': [f.__dict__ for f in status.unstaged_files],
                    'untracked_files': status.untracked_files,
                    'is_clean': status.is_clean,
                    'has_conflicts': status.has_conflicts,
                    'conflicted_files': status.conflicted_files,
                    'ahead_count': status.ahead_count,
                    'behind_count': status.behind_count
                }
            }

            response = AgentMessage(
                message_id=str(uuid.uuid4()),
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                message_type='git_status_result',
                data=result,
                correlation_id=message.correlation_id
            )

            await self.send_message(response)

        except Exception as e:
            self.logger.error(f"Error handling git_status: {e}", exc_info=True)

    async def _handle_analyze_changes(self, message: AgentMessage) -> None:
        """Handle analyze changes requests."""
        try:
            data = message.data
            result = await self.analyze_changes(since=data.get('since'))

            response = AgentMessage(
                message_id=str(uuid.uuid4()),
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                message_type='analyze_changes_result',
                data={'success': True, 'analysis': result},
                correlation_id=message.correlation_id
            )

            await self.send_message(response)

        except Exception as e:
            self.logger.error(f"Error handling analyze_changes: {e}", exc_info=True)

    async def _handle_perform_git_operation(self, message: AgentMessage) -> None:
        """Handle generic git operation requests."""
        try:
            data = message.data
            operation = data.get('operation', '')

            if operation == 'intelligent_commit':
                await self._handle_intelligent_commit(message)
            elif operation == 'resolve_conflicts':
                await self._handle_resolve_conflicts(message)
            elif operation == 'status':
                await self._handle_git_status(message)
            elif operation == 'analyze_changes':
                await self._handle_analyze_changes(message)
            else:
                # Generic git command execution
                result = await self._execute_git_operation(data)

                response = AgentMessage(
                    message_id=str(uuid.uuid4()),
                    sender_id=self.agent_id,
                    recipient_id=message.sender_id,
                    message_type='git_operation_result',
                    data=result,
                    correlation_id=message.correlation_id
                )

                await self.send_message(response)

        except Exception as e:
            self.logger.error(f"Error handling perform_git_operation: {e}", exc_info=True)

    # ========================================
    # Phase 2.2 Priority 3: AI-Enhanced Methods
    # Git Operations Intelligence Implementation
    # ========================================

    async def _generate_ai_commit_message(self, staged_files: List[GitFile]) -> Optional[CommitSuggestion]:
        """Generate AI-powered commit message with advanced analysis."""
        if not self.model_manager:
            return None

        try:
            # Prepare context for AI analysis
            file_changes = []
            total_additions = 0
            total_deletions = 0

            for file in staged_files:
                file_info = {
                    'path': file.path,
                    'change_type': file.change_type.value,
                    'additions': file.additions,
                    'deletions': file.deletions,
                    'is_binary': file.is_binary
                }

                if file.diff_content and not file.is_binary:
                    file_info['diff_sample'] = file.diff_content[:500]  # First 500 chars

                file_changes.append(file_info)
                total_additions += file.additions
                total_deletions += file.deletions

            # Create AI prompt for commit message generation
            prompt = f"""
            Analyze these code changes and generate an intelligent commit message:

            Files changed: {len(staged_files)}
            Total additions: {total_additions}
            Total deletions: {total_deletions}

            File changes:
            {json.dumps(file_changes, indent=2)}

            Generate a commit message following conventional commit format:
            - Use appropriate type (feat, fix, docs, style, refactor, test, chore)
            - Include clear, concise description
            - Mention breaking changes if any
            - Keep first line under 72 characters
            - Provide detailed description if needed

            Also analyze:
            1. Change impact and risk level
            2. Suggested testing approach
            3. Any deployment considerations
            """

            # Generate AI analysis
            ai_response = await analyze_code(prompt, "commit_analysis")

            if not ai_response:
                return None

            # Extract commit message components
            commit_type = ai_response.get('type', 'feat')
            scope = ai_response.get('scope')
            description = ai_response.get('description', 'Update files')
            breaking_change = ai_response.get('breaking_change', False)
            detailed_description = ai_response.get('detailed_description')

            # Format commit message
            if scope:
                message = f"{commit_type}({scope}): {description}"
            else:
                message = f"{commit_type}: {description}"

            if breaking_change:
                message += "!"

            # Create impact analysis
            impact_analysis = {
                'risk_level': ai_response.get('risk_level', 'low'),
                'affected_components': ai_response.get('affected_components', []),
                'testing_suggestions': ai_response.get('testing_suggestions', []),
                'deployment_notes': ai_response.get('deployment_notes', [])
            }

            return CommitSuggestion(
                message=message,
                description=detailed_description,
                type=commit_type,
                scope=scope,
                breaking_change=breaking_change,
                files_to_stage=[f.path for f in staged_files],
                confidence_score=ai_response.get('confidence', 0.8),
                ai_generated=True,
                impact_analysis=impact_analysis,
                suggestions=ai_response.get('suggestions', [])
            )

        except Exception as e:
            self.logger.error(f"AI commit message generation failed: {e}")
            return None

    async def analyze_change_impact(self, files: List[str] = None) -> ChangeImpactAnalysis:
        """Analyze the impact of changes across the repository."""
        try:
            if not files:
                # Get all staged/modified files
                status = await self.get_git_status()
                files = [f.path for f in status.staged_files + status.unstaged_files]

            if not files:
                return ChangeImpactAnalysis(
                    affected_components=[],
                    risk_level="low",
                    breaking_changes=[],
                    test_recommendations=[],
                    deployment_notes=[]
                )

            # Check cache first
            cache_key = "|".join(sorted(files))
            if cache_key in self.change_impact_cache:
                return self.change_impact_cache[cache_key]

            # Analyze file changes
            affected_components = set()
            breaking_changes = []
            risk_factors = []

            for file_path in files:
                # Determine component from file path
                path_parts = Path(file_path).parts
                if path_parts:
                    component = path_parts[0]
                    affected_components.add(component)

                # Check for potential breaking changes
                if await self._is_breaking_change_file(file_path):
                    breaking_changes.append(file_path)
                    risk_factors.append(f"Breaking change in {file_path}")

                # Check for critical files
                if await self._is_critical_file(file_path):
                    risk_factors.append(f"Critical file modified: {file_path}")

            # Determine risk level
            risk_level = "low"
            if breaking_changes or len(risk_factors) > 3:
                risk_level = "high"
            elif len(risk_factors) > 1 or len(affected_components) > 2:
                risk_level = "medium"

            # Generate recommendations
            test_recommendations = await self._generate_test_recommendations(files, risk_level)
            deployment_notes = await self._generate_deployment_notes(files, risk_level)

            analysis = ChangeImpactAnalysis(
                affected_components=list(affected_components),
                risk_level=risk_level,
                breaking_changes=breaking_changes,
                test_recommendations=test_recommendations,
                deployment_notes=deployment_notes,
                confidence_score=0.8
            )

            # Cache the analysis
            self.change_impact_cache[cache_key] = analysis
            return analysis

        except Exception as e:
            self.logger.error(f"Change impact analysis failed: {e}")
            return ChangeImpactAnalysis(
                affected_components=[],
                risk_level="unknown",
                breaking_changes=[],
                test_recommendations=[],
                deployment_notes=[]
            )

    async def smart_conflict_resolution(self, file_path: str) -> Dict[str, Any]:
        """AI-powered smart conflict resolution."""
        try:
            if not self.enable_smart_conflict_resolution:
                return {'strategy': 'manual', 'reason': 'Smart resolution disabled'}

            conflict_info = await self._analyze_conflict(file_path)

            if not self.model_manager:
                return {'strategy': 'manual', 'reason': 'AI not available'}

            # Prepare conflict context for AI analysis
            prompt = f"""
            Analyze this merge conflict and suggest resolution strategy:

            File: {file_path}
            Our content:
            {conflict_info.our_content}

            Their content:
            {conflict_info.their_content}

            Conflict markers: {len(conflict_info.conflict_markers)}

            Suggest:
            1. Best resolution strategy (merge, ours, theirs, manual)
            2. Specific resolution if possible
            3. Risk assessment
            4. Explanation of the conflict
            """

            ai_response = await analyze_code(prompt, "conflict_resolution")

            strategy = ai_response.get('strategy', 'manual')
            explanation = ai_response.get('explanation', 'Complex conflict requiring manual resolution')
            risk_level = ai_response.get('risk_level', 'medium')
            resolution = ai_response.get('resolution')

            return {
                'strategy': strategy,
                'explanation': explanation,
                'risk_level': risk_level,
                'resolution': resolution,
                'confidence': ai_response.get('confidence', 0.6),
                'ai_generated': True
            }

        except Exception as e:
            self.logger.error(f"Smart conflict resolution failed: {e}")
            return {'strategy': 'manual', 'error': str(e)}

    async def suggest_branch_strategy(self, feature_description: str) -> Dict[str, Any]:
        """AI-powered branch strategy suggestions."""
        try:
            if not self.model_manager:
                return {'strategy': 'feature', 'reason': 'AI not available'}

            # Get current repository state
            status = await self.get_git_status()
            current_branch = status.current_branch

            prompt = f"""
            Suggest optimal git branch strategy for this development task:

            Feature description: {feature_description}
            Current branch: {current_branch}
            Repository has conflicts: {status.has_conflicts}
            Staged files: {len(status.staged_files)}
            Unstaged files: {len(status.unstaged_files)}

            Recommend:
            1. Branch naming strategy
            2. Branching approach (feature, hotfix, release)
            3. Workflow recommendations
            4. Merge strategy suggestions
            """

            ai_response = await analyze_code(prompt, "branch_strategy")

            return {
                'branch_name': ai_response.get('branch_name', 'feature/new-feature'),
                'branch_type': ai_response.get('branch_type', 'feature'),
                'workflow': ai_response.get('workflow', 'gitflow'),
                'merge_strategy': ai_response.get('merge_strategy', 'merge'),
                'recommendations': ai_response.get('recommendations', []),
                'confidence': ai_response.get('confidence', 0.7),
                'ai_generated': True
            }

        except Exception as e:
            self.logger.error(f"Branch strategy suggestion failed: {e}")
            return {'strategy': 'feature', 'error': str(e)}

    async def _is_breaking_change_file(self, file_path: str) -> bool:
        """Check if file changes could be breaking."""
        breaking_indicators = [
            'api', 'interface', 'contract', 'schema',
            'migration', 'config', 'settings'
        ]

        file_lower = file_path.lower()
        return any(indicator in file_lower for indicator in breaking_indicators)

    async def _is_critical_file(self, file_path: str) -> bool:
        """Check if file is critical to system operation."""
        critical_files = [
            'package.json', 'requirements.txt', 'Dockerfile',
            'docker-compose.yml', 'Makefile', 'setup.py'
        ]

        file_name = Path(file_path).name.lower()
        return file_name in critical_files

    async def _generate_test_recommendations(self, files: List[str], risk_level: str) -> List[str]:
        """Generate testing recommendations based on changed files."""
        recommendations = []

        if risk_level == "high":
            recommendations.extend([
                "Run full test suite",
                "Perform integration testing",
                "Manual testing required"
            ])
        elif risk_level == "medium":
            recommendations.extend([
                "Run related unit tests",
                "Check integration points"
            ])
        else:
            recommendations.append("Run basic unit tests")

        # File-specific recommendations
        for file_path in files:
            if 'test' in file_path.lower():
                recommendations.append(f"Validate test coverage for {file_path}")
            elif file_path.endswith('.py'):
                recommendations.append(f"Run Python tests for {file_path}")
            elif file_path.endswith(('.js', '.ts')):
                recommendations.append(f"Run JavaScript tests for {file_path}")

        return list(set(recommendations))  # Remove duplicates

    async def _generate_deployment_notes(self, files: List[str], risk_level: str) -> List[str]:
        """Generate deployment notes based on changed files."""
        notes = []

        if risk_level == "high":
            notes.extend([
                "Deploy during maintenance window",
                "Prepare rollback plan",
                "Monitor system closely after deployment"
            ])
        elif risk_level == "medium":
            notes.append("Monitor for issues post-deployment")

        # File-specific notes
        for file_path in files:
            if 'config' in file_path.lower():
                notes.append(f"Configuration change in {file_path} - verify settings")
            elif 'migration' in file_path.lower():
                notes.append(f"Database migration required: {file_path}")
            elif file_path.endswith('.sql'):
                notes.append(f"Database changes detected: {file_path}")

        return notes

    async def _handle_task_request(self, message: AgentMessage) -> None:
        """Handle generic task requests from the orchestrator."""
        try:
            data = message.data
            action = data.get('action', '')

            if action == 'intelligent_commit':
                await self._handle_intelligent_commit(message)
            elif action == 'perform_git_operation':
                await self._handle_perform_git_operation(message)
            else:
                # Default to git status
                await self._handle_git_status(message)

        except Exception as e:
            self.logger.error(f"Error handling task_request: {e}", exc_info=True)

    # Utility methods

    async def _execute_git_operation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a generic git operation."""
        try:
            command = data.get('command', [])
            if not command:
                return {'success': False, 'error': 'No command provided'}

            output = await self._run_git_command(command)
            return {'success': True, 'output': output}

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _parse_status_char(self, char: str) -> ChangeType:
        """Parse git status character to ChangeType."""
        mapping = {
            'A': ChangeType.ADDED,
            'M': ChangeType.MODIFIED,
            'D': ChangeType.DELETED,
            'R': ChangeType.RENAMED,
            'C': ChangeType.COPIED,
            '?': ChangeType.UNTRACKED
        }
        return mapping.get(char, ChangeType.MODIFIED)

    async def _has_conflict_markers(self, file_path: str) -> bool:
        """Check if a file has conflict markers."""
        try:
            full_path = os.path.join(self.repository_path, file_path)
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()

            return any(marker in content for marker in ['<<<<<<<', '=======', '>>>>>>>'])
        except:
            return False

    async def _get_current_branch(self) -> str:
        """Get the current git branch."""
        try:
            return await self._run_git_command(['branch', '--show-current'])
        except:
            return 'unknown'

    async def _get_ahead_behind_counts(self, branch: str) -> Tuple[int, int]:
        """Get ahead/behind counts for a branch."""
        try:
            output = await self._run_git_command(['status', '--porcelain=v1', '--branch'])
            for line in output.split('\n'):
                if line.startswith('##'):
                    # Parse branch info line
                    if '[ahead' in line:
                        ahead_match = re.search(r'ahead (\d+)', line)
                        ahead = int(ahead_match.group(1)) if ahead_match else 0
                    else:
                        ahead = 0

                    if '[behind' in line:
                        behind_match = re.search(r'behind (\d+)', line)
                        behind = int(behind_match.group(1)) if behind_match else 0
                    else:
                        behind = 0

                    return ahead, behind

            return 0, 0
        except:
            return 0, 0

    async def _get_latest_commit_hash(self) -> str:
        """Get the hash of the latest commit."""
        try:
            return await self._run_git_command(['rev-parse', 'HEAD'])
        except:
            return 'unknown'

    async def _analyze_commit(self, commit_hash: str) -> Dict[str, Any]:
        """Analyze a specific commit."""
        try:
            # Get commit info
            commit_info = await self._run_git_command(['show', '--stat', commit_hash])

            # Get file changes
            files_changed = await self._run_git_command(['diff-tree', '--no-commit-id', '--name-only', '-r', commit_hash])

            return {
                'commit_hash': commit_hash,
                'files_changed': files_changed.split('\n') if files_changed else [],
                'commit_info': commit_info
            }
        except Exception as e:
            return {'error': str(e)}

    async def _analyze_file_changes(self, files: List[GitFile]) -> Dict[str, Any]:
        """Analyze changes in a list of files."""
        analysis = {
            'total_files': len(files),
            'added_files': 0,
            'modified_files': 0,
            'deleted_files': 0,
            'total_additions': 0,
            'total_deletions': 0,
            'file_types': {},
            'likely_bug_fix': False
        }

        for git_file in files:
            if git_file.change_type == ChangeType.ADDED:
                analysis['added_files'] += 1
            elif git_file.change_type == ChangeType.MODIFIED:
                analysis['modified_files'] += 1
            elif git_file.change_type == ChangeType.DELETED:
                analysis['deleted_files'] += 1

            analysis['total_additions'] += git_file.additions
            analysis['total_deletions'] += git_file.deletions

            # Track file types
            ext = Path(git_file.path).suffix
            analysis['file_types'][ext] = analysis['file_types'].get(ext, 0) + 1

            # Check for likely bug fixes
            if any(keyword in git_file.path.lower() for keyword in ['fix', 'bug', 'error']):
                analysis['likely_bug_fix'] = True

        return analysis

    async def _get_commits_since(self, since: str) -> List[Dict[str, Any]]:
        """Get commits since a specific reference."""
        try:
            output = await self._run_git_command(['log', '--oneline', f'{since}..HEAD'])
            commits = []

            for line in output.split('\n'):
                if line.strip():
                    match = self.git_patterns['commit_pattern'].match(line.strip())
                    if match:
                        commit_hash, message = match.groups()
                        commits.append({
                            'hash': commit_hash,
                            'message': message
                        })

            return commits
        except:
            return []

    async def _detect_change_patterns(self, files: List[GitFile]) -> Dict[str, Any]:
        """Detect patterns in file changes."""
        patterns = {
            'bulk_rename': False,
            'new_feature': False,
            'bug_fix': False,
            'documentation_update': False,
            'test_changes': False,
            'configuration_changes': False
        }

        # Analyze file paths and types
        paths = [f.path for f in files]

        # Check for bulk renames
        renamed_files = [f for f in files if f.change_type == ChangeType.RENAMED]
        if len(renamed_files) > 3:
            patterns['bulk_rename'] = True

        # Check for new features
        new_files = [f for f in files if f.change_type == ChangeType.ADDED]
        if len(new_files) > 1 and any('.py' in f.path or '.js' in f.path for f in new_files):
            patterns['new_feature'] = True

        # Check for bug fixes
        if any('fix' in f.path.lower() or 'bug' in f.path.lower() for f in files):
            patterns['bug_fix'] = True

        # Check for documentation
        if any(f.path.endswith(('.md', '.rst', '.txt')) for f in files):
            patterns['documentation_update'] = True

        # Check for tests
        if any('test' in f.path.lower() for f in files):
            patterns['test_changes'] = True

        # Check for configuration
        if any(f.path.endswith(('.json', '.yaml', '.yml', '.toml', '.ini')) for f in files):
            patterns['configuration_changes'] = True

        return patterns

    async def _generate_change_recommendations(self, status: GitStatus) -> List[str]:
        """Generate recommendations based on current git status."""
        recommendations = []

        if status.unstaged_files:
            recommendations.append("Stage changes before committing")

        if status.untracked_files:
            recommendations.append("Consider adding new files to git")

        if status.has_conflicts:
            recommendations.append("Resolve merge conflicts before proceeding")

        if status.behind_count > 0:
            recommendations.append(f"Pull {status.behind_count} commits from remote")

        if status.ahead_count > 0:
            recommendations.append(f"Push {status.ahead_count} commits to remote")

        if status.is_clean:
            recommendations.append("Repository is clean - no changes to commit")

        return recommendations

    async def _get_branch_list(self) -> List[GitBranch]:
        """Get list of all branches."""
        try:
            output = await self._run_git_command(['branch', '-a'])
            branches = []

            for line in output.split('\n'):
                line = line.strip()
                if not line:
                    continue

                is_current = line.startswith('*')
                if is_current:
                    line = line[1:].strip()

                is_remote = line.startswith('remotes/')
                if is_remote:
                    line = line.replace('remotes/', '')

                branches.append(GitBranch(
                    name=line,
                    is_current=is_current,
                    is_remote=is_remote
                ))

            return branches
        except:
            return []

    async def _get_remote_info(self) -> Dict[str, str]:
        """Get information about git remotes."""
        try:
            output = await self._run_git_command(['remote', '-v'])
            remotes = {}

            for line in output.split('\n'):
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 2:
                        remote_name = parts[0]
                        remote_url = parts[1]
                        remotes[remote_name] = remote_url

            return remotes
        except:
            return {}

    def _is_valid_branch_name(self, branch_name: str) -> bool:
        """Check if a branch name is valid."""
        # Basic validation - git has more complex rules
        if not branch_name or branch_name.startswith('-'):
            return False

        invalid_chars = [' ', '~', '^', ':', '?', '*', '[', '\\']
        if any(char in branch_name for char in invalid_chars):
            return False

        if branch_name.endswith('.') or '..' in branch_name:
            return False

        return True

    def _update_operation_metrics(self, operation: GitOperation, duration: float, success: bool) -> None:
        """Update metrics for git operations."""
        self.git_metrics['total_operations'] += 1

        if success:
            self.git_metrics['successful_operations'] += 1
        else:
            self.git_metrics['failed_operations'] += 1

        # Update operation type metrics
        op_name = operation.value
        if op_name not in self.git_metrics['operation_types']:
            self.git_metrics['operation_types'][op_name] = {'count': 0, 'total_time': 0.0}

        self.git_metrics['operation_types'][op_name]['count'] += 1
        self.git_metrics['operation_types'][op_name]['total_time'] += duration

        # Update average operation time
        total_ops = self.git_metrics['total_operations']
        current_avg = self.git_metrics['average_operation_time']

        if total_ops > 1:
            new_avg = ((current_avg * (total_ops - 1)) + duration) / total_ops
            self.git_metrics['average_operation_time'] = new_avg
        else:
            self.git_metrics['average_operation_time'] = duration

    def get_metrics(self) -> Dict[str, Any]:
        """Get git agent performance metrics."""
        return {
            **self.git_metrics,
            'repository_info': self.repository_info,
            'current_status': self.current_status.__dict__ if self.current_status else None,
            'agent_status': self.get_status()
        }

    # Abstract method implementations
    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """
        Process an incoming message and route it to appropriate git handlers.

        Args:
            message: The message to process

        Returns:
            Optional response message
        """
        try:
            self.logger.debug(f"Processing message: {message.message_type} from {message.sender_id}")

            # Handle different message types
            if message.message_type == "git_request":
                payload = message.payload
                action = payload.get('action', 'status')

                if action == "commit":
                    result = await self.create_intelligent_commit(
                        files=payload.get('files', []),
                        message_context=payload.get('message_context', ''),
                        commit_type=payload.get('commit_type', 'feat')
                    )
                elif action == "status":
                    result = await self.get_repository_status()
                elif action == "branch":
                    result = await self.create_branch(
                        branch_name=payload.get('branch_name', ''),
                        base_branch=payload.get('base_branch', 'main')
                    )
                elif action == "merge":
                    result = await self.merge_branch(
                        source_branch=payload.get('source_branch', ''),
                        target_branch=payload.get('target_branch', 'main'),
                        strategy=payload.get('strategy', 'auto')
                    )
                elif action == "resolve_conflicts":
                    result = await self.resolve_conflicts(
                        files=payload.get('files', []),
                        resolution_strategy=payload.get('strategy', 'manual')
                    )
                else:
                    raise ValueError(f"Unknown git action: {action}")

                return AgentMessage(
                    sender_id=self.agent_id,
                    recipient_id=message.sender_id,
                    message_type="git_response",
                    payload=result,
                    correlation_id=message.correlation_id
                )

            elif message.message_type == "git_operation":
                await self._handle_git_operation(message)
                return None

            else:
                # Return None for unhandled message types to avoid recursion
                self.logger.debug(f"Unhandled message type: {message.message_type}")
                return None

        except Exception as e:
            self.logger.error(f"Error processing message: {e}", exc_info=True)

            # Return error response if this was a request expecting a response
            if message.message_type in ["git_request"]:
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
        Return list of capabilities this git agent provides.

        Returns:
            List of agent capabilities
        """


        return [
            AgentCapability(
                name="intelligent_commits",
                description="Generate intelligent commit messages and manage commits",
                input_types=["list", "text"],
                output_types=["dict"],
                cost_estimate=1.5
            ),
            AgentCapability(
                name="branch_management",
                description="Create, switch, and manage git branches",
                input_types=["text"],
                output_types=["dict"],
                cost_estimate=1.0
            ),
            AgentCapability(
                name="merge_operations",
                description="Merge branches with conflict resolution",
                input_types=["text"],
                output_types=["dict"],
                cost_estimate=2.0
            ),
            AgentCapability(
                name="conflict_resolution",
                description="Automatically resolve git merge conflicts",
                input_types=["list", "text"],
                output_types=["dict"],
                cost_estimate=2.5
            ),
            AgentCapability(
                name="repository_analysis",
                description="Analyze repository status, history, and structure",
                input_types=["path"],
                output_types=["dict"],
                cost_estimate=1.0
            ),
            AgentCapability(
                name="git_operations",
                description="Execute standard git operations (add, commit, push, pull, etc.)",
                input_types=["text", "list"],
                output_types=["dict"],
                cost_estimate=0.5
            ),
            AgentCapability(
                name="history_management",
                description="Manage git history, rebase, and advanced git operations",
                input_types=["text", "dict"],
                output_types=["dict"],
                cost_estimate=2.0
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

            # Check git agent-specific health
            current_time = datetime.utcnow()

            # Check git availability
            git_available = await self._check_git_availability()

            # Check repository status
            repo_valid = self.repository_info.get('is_git_repo', False)
            repo_path = self.repository_info.get('repo_path', '')

            # Check core capabilities
            can_commit = hasattr(self, 'create_intelligent_commit')
            can_branch = hasattr(self, 'create_branch')
            can_merge = hasattr(self, 'merge_branch')
            can_resolve = hasattr(self, 'resolve_conflicts')

            core_capabilities = can_commit and can_branch and can_merge and can_resolve

            # Calculate performance metrics
            total_operations = self.git_metrics.get('total_operations', 0)
            successful_operations = self.git_metrics.get('successful_operations', 0)
            success_rate = 0.0
            if total_operations > 0:
                success_rate = (successful_operations / total_operations) * 100

            # Determine overall health status
            is_healthy = (
                git_available and
                repo_valid and
                core_capabilities and
                success_rate >= 85.0  # Consider healthy if success rate >= 85%
            )

            health_status = {
                **base_health,
                "status": "healthy" if is_healthy else "degraded",
                "timestamp": current_time.isoformat(),
                "git_agent_specific": {
                    "git_available": git_available,
                    "repository_valid": repo_valid,
                    "repository_path": repo_path,
                    "core_capabilities": {
                        "commit": can_commit,
                        "branch": can_branch,
                        "merge": can_merge,
                        "resolve_conflicts": can_resolve
                    },
                    "capabilities_health": core_capabilities,
                    "success_rate": success_rate,
                    "total_operations": total_operations,
                    "successful_operations": successful_operations,
                    "failed_operations": self.git_metrics.get('failed_operations', 0),
                    "average_operation_time": self.git_metrics.get('average_operation_time', 0.0)
                },
                "capabilities": len(self.get_capabilities()),
                "uptime": (current_time - self.created_at).total_seconds() if hasattr(self, 'created_at') else 0
            }

            # Add any critical issues
            issues = []
            if not git_available:
                issues.append("Git command not available")
            if not repo_valid:
                issues.append("Not in a valid git repository")
            if not core_capabilities:
                issues.append("Missing core git capabilities")
            if success_rate < 85.0 and total_operations > 0:
                issues.append(f"Low success rate: {success_rate:.1f}%")

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

    async def _check_git_availability(self) -> bool:
        """Check if git command is available."""
        try:
            result = await self._run_git_command(['--version'])
            return result.returncode == 0
        except Exception:
            return False
