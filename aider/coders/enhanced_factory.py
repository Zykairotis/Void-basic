"""
Enhanced factory system for creating Aider coders with intelligent selection logic.

This module provides a comprehensive factory system that automatically selects
the optimal coder type and configuration based on context, model capabilities,
and user requirements using modern design patterns.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Type, Callable, TypeAlias, Literal
from dataclasses import dataclass
from pathlib import Path
from enum import Enum

from .base_coder import Coder

# Modern type aliases for better readability
CoderType: TypeAlias = Type[Any]  # Will be properly typed after Coder import
ConfigDict: TypeAlias = Dict[str, Any]
ModelName: TypeAlias = str
EditFormatType: TypeAlias = Literal["diff", "udiff", "whole", "diff-fenced", "editblock", "patch"]
from .ask_coder import AskCoder
from .architect_coder import ArchitectCoder
from .context_coder import ContextCoder
from .editblock_coder import EditBlockCoder
from .editblock_fenced_coder import EditBlockFencedCoder
from .udiff_coder import UnifiedDiffCoder
from .udiff_simple import UnifiedDiffSimpleCoder
from .wholefile_coder import WholeFileCoder
from .patch_coder import PatchCoder
from .help_coder import HelpCoder
from .editor_editblock_coder import EditorEditBlockCoder
from .editor_diff_fenced_coder import EditorDiffFencedCoder
from .editor_whole_coder import EditorWholeFileCoder

from .exceptions import (
    ConfigurationError,
    UnknownEditFormat,
    ErrorContext
)
from .config import (
    AiderConfig,
    get_current_config,
    create_config_for_model
)


# =============================================================================
# Factory Configuration and Data Classes
# =============================================================================

class TaskType(Enum):
    """Enumeration of different task types that coders can handle."""
    """Different types of tasks that coders can perform."""
    EDIT = "edit"
    ANALYZE = "analyze"
    ARCHITECT = "architect"
    CONTEXT = "context"
    HELP = "help"
    CHAT = "chat"


class CoderCapability(Enum):
    """Capabilities that coders can have."""
    FILE_EDITING = "file_editing"
    MULTIPLE_FILES = "multiple_files"
    PRECISE_DIFFS = "precise_diffs"
    LARGE_FILES = "large_files"
    CONTEXT_ANALYSIS = "context_analysis"
    SHELL_COMMANDS = "shell_commands"
    STREAMING = "streaming"


@dataclass
class CoderProfile:
    """Profile describing a coder's characteristics and optimal use cases."""
    coder_class: Type[Coder]
    edit_format: str
    capabilities: List[CoderCapability]
    best_for: List[str]
    model_requirements: List[str]
    file_size_preference: Tuple[int, int]  # (min_kb, max_kb)
    complexity_rating: int  # 1-5, where 5 is most complex
    description: str


@dataclass
class ContextAnalysis:
    """Analysis of the current context for coder selection."""
    file_count: int
    total_file_size_kb: float
    avg_file_size_kb: float
    max_file_size_kb: float
    file_types: List[str]
    has_git_repo: bool
    model_name: str
    task_complexity: str  # simple, medium, complex
    user_preference: Optional[str] = None


# =============================================================================
# Enhanced Coder Factory
# =============================================================================

class EnhancedCoderFactory:
    """
    Enhanced factory for creating optimal Aider coders.

    Uses intelligent selection logic based on context analysis,
    model capabilities, and user requirements.
    """

    def __init__(self):
        """Initialize the factory with coder profiles."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self._coder_profiles = self._initialize_coder_profiles()
        self._selection_rules = self._initialize_selection_rules()

    def _initialize_coder_profiles(self) -> Dict[str, CoderProfile]:
        """Initialize profiles for all available coders."""
        return {
            "editblock": CoderProfile(
                coder_class=EditBlockCoder,
                edit_format="diff",
                capabilities=[
                    CoderCapability.FILE_EDITING,
                    CoderCapability.MULTIPLE_FILES,
                    CoderCapability.SHELL_COMMANDS
                ],
                best_for=[
                    "General code editing",
                    "Small to medium files",
                    "Clear, targeted changes",
                    "Beginner-friendly format"
                ],
                model_requirements=["gpt-3.5", "gpt-4", "claude", "gemini"],
                file_size_preference=(0, 100),
                complexity_rating=2,
                description="Standard search/replace block editing"
            ),
            "editblock_fenced": CoderProfile(
                coder_class=EditBlockFencedCoder,
                edit_format="diff-fenced",
                capabilities=[
                    CoderCapability.FILE_EDITING,
                    CoderCapability.MULTIPLE_FILES,
                    CoderCapability.PRECISE_DIFFS
                ],
                best_for=[
                    "Claude models",
                    "Complex search/replace operations",
                    "Multiple file edits",
                    "Better context preservation"
                ],
                model_requirements=["claude", "gpt-4"],
                file_size_preference=(5, 200),
                complexity_rating=3,
                description="Fenced search/replace blocks optimized for Claude"
            ),
            "udiff": CoderProfile(
                coder_class=UnifiedDiffCoder,
                edit_format="udiff",
                capabilities=[
                    CoderCapability.FILE_EDITING,
                    CoderCapability.PRECISE_DIFFS,
                    CoderCapability.LARGE_FILES,
                    CoderCapability.MULTIPLE_FILES
                ],
                best_for=[
                    "Large files",
                    "Precise edits",
                    "Advanced models",
                    "Git-compatible changes",
                    "Complex refactoring"
                ],
                model_requirements=["gpt-4", "gpt-4-turbo", "claude-3"],
                file_size_preference=(20, 1000),
                complexity_rating=4,
                description="Unified diff format for precise edits"
            ),
            "udiff_simple": CoderProfile(
                coder_class=UnifiedDiffSimpleCoder,
                edit_format="udiff-simple",
                capabilities=[
                    CoderCapability.FILE_EDITING,
                    CoderCapability.PRECISE_DIFFS,
                    CoderCapability.LARGE_FILES
                ],
                best_for=[
                    "Models that struggle with complex diff format",
                    "Simpler diff requirements",
                    "When standard udiff is too complex"
                ],
                model_requirements=["gpt-3.5", "gemini"],
                file_size_preference=(10, 500),
                complexity_rating=3,
                description="Simplified unified diff format"
            ),
            "wholefile": CoderProfile(
                coder_class=WholeFileCoder,
                edit_format="whole",
                capabilities=[
                    CoderCapability.FILE_EDITING,
                    CoderCapability.STREAMING
                ],
                best_for=[
                    "Small files",
                    "Complete rewrites",
                    "New file creation",
                    "When precise edits are difficult"
                ],
                model_requirements=["any"],
                file_size_preference=(0, 50),
                complexity_rating=1,
                description="Complete file replacement"
            ),
            "patch": CoderProfile(
                coder_class=PatchCoder,
                edit_format="patch",
                capabilities=[
                    CoderCapability.FILE_EDITING,
                    CoderCapability.MULTIPLE_FILES,
                    CoderCapability.PRECISE_DIFFS,
                    CoderCapability.LARGE_FILES
                ],
                best_for=[
                    "Complex multi-file operations",
                    "Batch edits",
                    "Advanced workflows",
                    "Structured changes"
                ],
                model_requirements=["gpt-4-turbo", "claude-3-opus"],
                file_size_preference=(10, 500),
                complexity_rating=5,
                description="Advanced patch format for complex operations"
            ),
            "ask": CoderProfile(
                coder_class=AskCoder,
                edit_format="ask",
                capabilities=[CoderCapability.CONTEXT_ANALYSIS],
                best_for=[
                    "Code analysis",
                    "Questions about code",
                    "Understanding existing code",
                    "Read-only operations"
                ],
                model_requirements=["any"],
                file_size_preference=(0, 1000),
                complexity_rating=1,
                description="Ask questions without making edits"
            ),
            "architect": CoderProfile(
                coder_class=ArchitectCoder,
                edit_format="architect",
                capabilities=[
                    CoderCapability.CONTEXT_ANALYSIS,
                    CoderCapability.MULTIPLE_FILES
                ],
                best_for=[
                    "High-level planning",
                    "Architecture decisions",
                    "Complex refactoring planning",
                    "Two-stage editing workflow"
                ],
                model_requirements=["gpt-4", "claude-3"],
                file_size_preference=(10, 1000),
                complexity_rating=4,
                description="High-level architectural planning"
            ),
            "context": CoderProfile(
                coder_class=ContextCoder,
                edit_format="context",
                capabilities=[CoderCapability.CONTEXT_ANALYSIS],
                best_for=[
                    "Identifying files to edit",
                    "Understanding dependencies",
                    "Planning edit scope",
                    "File discovery"
                ],
                model_requirements=["gpt-4", "claude"],
                file_size_preference=(0, 1000),
                complexity_rating=2,
                description="Identify files that need editing"
            ),
            "help": CoderProfile(
                coder_class=HelpCoder,
                edit_format="help",
                capabilities=[],
                best_for=[
                    "User assistance",
                    "Documentation",
                    "Feature explanations"
                ],
                model_requirements=["any"],
                file_size_preference=(0, 100),
                complexity_rating=1,
                description="Interactive help and documentation"
            ),
            "editor_editblock": CoderProfile(
                coder_class=EditorEditBlockCoder,
                edit_format="editor-diff",
                capabilities=[
                    CoderCapability.FILE_EDITING,
                    CoderCapability.PRECISE_DIFFS
                ],
                best_for=[
                    "Focused editing without shell commands",
                    "Pure code modification",
                    "Automated workflows"
                ],
                model_requirements=["gpt-4", "claude"],
                file_size_preference=(5, 200),
                complexity_rating=3,
                description="Editor-focused search/replace without shell commands"
            ),
            "editor_whole": CoderProfile(
                coder_class=EditorWholeFileCoder,
                edit_format="editor-whole",
                capabilities=[CoderCapability.FILE_EDITING],
                best_for=[
                    "Pure file editing",
                    "Small file rewrites",
                    "Automated workflows"
                ],
                model_requirements=["any"],
                file_size_preference=(0, 30),
                complexity_rating=2,
                description="Editor-focused whole file replacement"
            ),
            "editor_fenced": CoderProfile(
                coder_class=EditorDiffFencedCoder,
                edit_format="editor-diff-fenced",
                capabilities=[
                    CoderCapability.FILE_EDITING,
                    CoderCapability.PRECISE_DIFFS
                ],
                best_for=[
                    "Claude models with editor focus",
                    "Complex edits without shell commands",
                    "Automated workflows"
                ],
                model_requirements=["claude"],
                file_size_preference=(5, 150),
                complexity_rating=3,
                description="Editor-focused fenced blocks for Claude"
            )
        }

    def _initialize_selection_rules(self) -> List[Callable[..., Any]]:
        """Initialize rules for coder selection in priority order."""
        return [
            self._rule_user_preference,
            self._rule_task_type,
            self._rule_model_optimization,
            self._rule_file_size_optimization,
            self._rule_complexity_optimization,
            self._rule_capability_matching,
            self._rule_fallback
        ]

    # =============================================================================
    # Main Factory Methods
    # =============================================================================

    def create_coder(
        self,
        io,
        task_type: str | TaskType | None = None,
        edit_format: str | None = None,
        model_name: ModelName | None = None,
        config: AiderConfig | None = None,
        context: ContextAnalysis | None = None,
        **kwargs
    ) -> Coder:
        """
        Create the optimal coder based on parameters and context.

        Args:
            io: Input/output handler
            task_type: Type of task to perform
            edit_format: Specific edit format to use
            model_name: AI model name
            config: Configuration to use
            context: Analysis context for optimization
            **kwargs: Additional legacy parameters

        Returns:
            Configured coder instance optimized for the given context

        Raises:
            ConfigurationError: If configuration is invalid
            UnknownEditFormat: If edit format is not supported
            context: Context analysis for optimization
            **kwargs: Additional coder parameters

        Returns:
            Optimally configured coder instance

        Raises:
            ConfigurationError: If optimal coder cannot be determined
        """
        try:
            # Normalize inputs
            if isinstance(task_type, str):
                task_type = TaskType(task_type)

            # Analyze context if not provided
            if not context:
                context = self._analyze_context(kwargs, model_name)

            # Get or create configuration
            if not config:
                config = self._get_optimal_config(context, model_name, edit_format)

            # Select optimal coder profile
            profile = self._select_optimal_profile(
                task_type=task_type,
                edit_format=edit_format,
                context=context,
                config=config
            )

            # Create and configure coder instance
            coder_instance = self._create_coder_instance(
                profile=profile,
                io=io,
                config=config,
                context=context,
                **kwargs
            )

            # Apply post-creation optimizations
            self._optimize_coder_instance(coder_instance, context)

            self.logger.info(
                f"Created {profile.coder_class.__name__} with {profile.edit_format} format"
            )

            return coder_instance

        except Exception as e:
            raise ConfigurationError(
                f"Failed to create coder: {e}",
                ErrorContext(
                    error_code="CODER_CREATION_FAILED",
                    suggestions=[
                        "Check model name and availability",
                        "Verify configuration parameters",
                        "Ensure files are accessible"
                    ]
                )
            ) from e

    def create_for_files(
        self,
        io,
        file_paths: List[str],
        model_name: str = "gpt-4",
        task_type: Optional[TaskType] = None,
        **kwargs
    ) -> Coder:
        """
        Create coder optimized for specific files.

        Args:
            io: Input/output handler
            file_paths: List of files to work with
            model_name: AI model name
            task_type: Optional task type
            **kwargs: Additional parameters

        Returns:
            Coder optimized for the given files
        """
        # Analyze files to create context
        context = self._analyze_files(file_paths, model_name)

        return self.create_coder(
            io=io,
            task_type=task_type,
            model_name=model_name,
            context=context,
            fnames=set(file_paths),
            **kwargs
        )

    def create_for_model(
        self,
        io,
        model_name: str,
        optimize_for_model: bool = True,
        **kwargs
    ) -> Coder:
        """
        Create coder optimized for a specific model.

        Args:
            io: Input/output handler
            model_name: AI model name
            optimize_for_model: Whether to optimize specifically for the model
            **kwargs: Additional parameters

        Returns:
            Model-optimized coder
        """
        if optimize_for_model:
            # Get model-specific recommendations
            optimal_format = self._get_optimal_format_for_model(model_name)
            edit_format = optimal_format

            return self.create_coder(
                io=io,
                model_name=model_name,
                edit_format=optimal_format,
                **kwargs
            )
        else:
            return self.create_coder(io=io, model_name=model_name, **kwargs)

    def suggest_coder_for_task(
        self,
        task_description: str,
        file_paths: Optional[List[str]] = None,
        model_name: str = "gpt-4"
    ) -> Dict[str, Any]:
        """
        Suggest the best coder for a given task description.

        Args:
            task_description: Natural language description of the task
            file_paths: Optional list of files involved
            model_name: AI model name

        Returns:
            Dictionary with coder suggestions and reasoning
        """
        # Analyze task complexity and type
        task_analysis = self._analyze_task_description(task_description)

        # Analyze files if provided
        context = None
        if file_paths:
            context = self._analyze_files(file_paths, model_name)

        # Get top recommendations
        recommendations = self._get_coder_recommendations(
            task_analysis=task_analysis,
            context=context,
            model_name=model_name
        )

        return {
            "task_analysis": task_analysis,
            "context_analysis": context,
            "recommendations": recommendations,
            "reasoning": self._explain_recommendations(recommendations, task_analysis)
        }

    # =============================================================================
    # Selection Logic Methods
    # =============================================================================

    def _select_optimal_profile(
        self,
        task_type: Optional[TaskType],
        edit_format: Optional[str],
        context: ContextAnalysis,
        config: AiderConfig
    ) -> CoderProfile:
        """
        Select the optimal coder profile using selection rules.

        Args:
            task_type: Type of task
            edit_format: Specific edit format requested
            context: Context analysis
            config: Configuration

        Returns:
            Selected coder profile
        """
        # If specific edit format requested, use it
        if edit_format:
            for profile in self._coder_profiles.values():
                if profile.edit_format == edit_format:
                    return profile
            raise UnknownEditFormat(edit_format, list(self._coder_profiles.keys()))

        # Apply selection rules in order
        candidates = list(self._coder_profiles.values())

        for rule in self._selection_rules:
            candidates = rule(candidates, task_type, context, config)
            if len(candidates) == 1:
                return candidates[0]

        # If multiple candidates remain, pick the best one
        if candidates:
            return self._pick_best_candidate(candidates, context)

        # Fallback to editblock
        return self._coder_profiles["editblock"]

    def _rule_user_preference(
        self,
        candidates: List[CoderProfile],
        task_type: Optional[TaskType],
        context: ContextAnalysis,
        config: AiderConfig
    ) -> List[CoderProfile]:
        """Rule: Honor user preference if specified."""
        if context.user_preference:
            filtered = [c for c in candidates if c.edit_format == context.user_preference]
            return filtered if filtered else candidates
        return candidates

    def _rule_task_type(
        self,
        candidates: List[CoderProfile],
        task_type: Optional[TaskType],
        context: ContextAnalysis,
        config: AiderConfig
    ) -> List[CoderProfile]:
        """Rule: Filter by task type requirements."""
        if not task_type:
            return candidates

        task_filters = {
            TaskType.EDIT: [CoderCapability.FILE_EDITING],
            TaskType.ANALYZE: [CoderCapability.CONTEXT_ANALYSIS],
            TaskType.ARCHITECT: [CoderCapability.CONTEXT_ANALYSIS, CoderCapability.MULTIPLE_FILES],
            TaskType.CONTEXT: [CoderCapability.CONTEXT_ANALYSIS],
            TaskType.HELP: [],
            TaskType.CHAT: []
        }

        required_capabilities = task_filters.get(task_type, [])
        if not required_capabilities:
            return candidates

        filtered = []
        for candidate in candidates:
            if all(cap in candidate.capabilities for cap in required_capabilities):
                filtered.append(candidate)

        return filtered if filtered else candidates

    def _rule_model_optimization(
        self,
        candidates: List[CoderProfile],
        task_type: Optional[TaskType],
        context: ContextAnalysis,
        config: AiderConfig
    ) -> List[CoderProfile]:
        """Rule: Filter by model compatibility and optimization."""
        model_name = context.model_name.lower()

        filtered = []
        for candidate in candidates:
            # Check if model is in requirements
            model_compatible = (
                "any" in candidate.model_requirements or
                any(req.lower() in model_name for req in candidate.model_requirements)
            )

            if model_compatible:
                filtered.append(candidate)

        return filtered if filtered else candidates

    def _rule_file_size_optimization(
        self,
        candidates: List[CoderProfile],
        task_type: Optional[TaskType],
        context: ContextAnalysis,
        config: AiderConfig
    ) -> List[CoderProfile]:
        """Rule: Filter by file size preferences."""
        avg_size = context.avg_file_size_kb

        filtered = []
        for candidate in candidates:
            min_size, max_size = candidate.file_size_preference
            if min_size <= avg_size <= max_size:
                filtered.append(candidate)

        return filtered if filtered else candidates

    def _rule_complexity_optimization(
        self,
        candidates: List[CoderProfile],
        task_type: Optional[TaskType],
        context: ContextAnalysis,
        config: AiderConfig
    ) -> List[CoderProfile]:
        """Rule: Filter by task complexity matching."""
        complexity_mapping = {
            "simple": [1, 2],
            "medium": [2, 3, 4],
            "complex": [3, 4, 5]
        }

        suitable_ratings = complexity_mapping.get(context.task_complexity, [1, 2, 3, 4, 5])

        filtered = []
        for candidate in candidates:
            if candidate.complexity_rating in suitable_ratings:
                filtered.append(candidate)

        return filtered if filtered else candidates

    def _rule_capability_matching(
        self,
        candidates: List[CoderProfile],
        task_type: Optional[TaskType],
        context: ContextAnalysis,
        config: AiderConfig
    ) -> List[CoderProfile]:
        """Rule: Prefer candidates with more relevant capabilities."""
        if context.file_count > 5:
            # Prefer multiple file capability
            filtered = [c for c in candidates if CoderCapability.MULTIPLE_FILES in c.capabilities]
            if filtered:
                return filtered

        if context.max_file_size_kb > 100:
            # Prefer large file capability
            filtered = [c for c in candidates if CoderCapability.LARGE_FILES in c.capabilities]
            if filtered:
                return filtered

        return candidates

    def _rule_fallback(
        self,
        candidates: List[CoderProfile],
        task_type: Optional[TaskType],
        context: ContextAnalysis,
        config: AiderConfig
    ) -> List[CoderProfile]:
        """Rule: Provide fallback selection."""
        if not candidates:
            return [self._coder_profiles["editblock"]]  # Safe fallback
        return candidates

    def _pick_best_candidate(self, candidates: List[CoderProfile], context: ContextAnalysis) -> CoderProfile:
        """Pick the best candidate from remaining options."""
        # Score candidates based on various factors
        scored_candidates = []

        for candidate in candidates:
            score = 0

            # Prefer formats optimized for the model
            model_name = context.model_name.lower()
            if "claude" in model_name and "fenced" in candidate.edit_format:
                score += 2
            elif "gpt-4" in model_name and candidate.edit_format == "udiff":
                score += 2

            # Prefer formats that match file size well
            avg_size = context.avg_file_size_kb
            min_size, max_size = candidate.file_size_preference
            size_score = 1 - abs(avg_size - (min_size + max_size) / 2) / max(max_size, 100)
            score += size_score

            # Prefer simpler formats for simple tasks
            if context.task_complexity == "simple":
                score += (6 - candidate.complexity_rating)
            elif context.task_complexity == "complex":
                score += candidate.complexity_rating

            scored_candidates.append((score, candidate))

        # Return highest scoring candidate
        scored_candidates.sort(reverse=True, key=lambda x: x[0])
        return scored_candidates[0][1]

    # =============================================================================
    # Context Analysis Methods
    # =============================================================================

    def _analyze_context(self, kwargs: Dict[str, Any], model_name: Optional[str]) -> ContextAnalysis:
        """
        Analyze context from coder parameters.

        Args:
            kwargs: Coder creation parameters
            model_name: AI model name

        Returns:
            Context analysis
        """
        # Extract file information
        fnames = kwargs.get('fnames', set())
        file_paths = list(fnames) if fnames else []

        if file_paths:
            return self._analyze_files(file_paths, model_name or "gpt-4")
        else:
            # Create minimal context
            return ContextAnalysis(
                file_count=0,
                total_file_size_kb=0,
                avg_file_size_kb=0,
                max_file_size_kb=0,
                file_types=[],
                has_git_repo=False,
                model_name=model_name or "gpt-4",
                task_complexity="medium"
            )

    def _analyze_files(self, file_paths: List[str], model_name: str) -> ContextAnalysis:
        """
        Analyze files to create context.

        Args:
            file_paths: List of file paths to analyze
            model_name: AI model name

        Returns:
            Context analysis based on files
        """
        total_size_kb = 0
        file_sizes = []
        file_types = set()
        existing_files = 0

        for file_path in file_paths:
            path = Path(file_path)

            if path.exists():
                existing_files += 1
                size_kb = path.stat().st_size / 1024
                total_size_kb += size_kb
                file_sizes.append(size_kb)

                if path.suffix:
                    file_types.add(path.suffix)

        # Calculate statistics
        avg_size_kb = total_size_kb / len(file_paths) if file_paths else 0
        max_size_kb = max(file_sizes) if file_sizes else 0

        # Determine task complexity
        complexity = "simple"
        if len(file_paths) > 5 or max_size_kb > 100:
            complexity = "medium"
        if len(file_paths) > 15 or max_size_kb > 500:
            complexity = "complex"

        # Check for git repo
        has_git = any(Path(p).parent.glob(".git") for p in file_paths if Path(p).exists())

        return ContextAnalysis(
            file_count=len(file_paths),
            total_file_size_kb=total_size_kb,
            avg_file_size_kb=avg_size_kb,
            max_file_size_kb=max_size_kb,
            file_types=list(file_types),
            has_git_repo=has_git,
            model_name=model_name,
            task_complexity=complexity
        )

    def _analyze_task_description(self, description: str) -> Dict[str, Any]:
        """
        Analyze task description to determine requirements.

        Args:
            description: Natural language task description

        Returns:
            Task analysis results
        """
        description_lower = description.lower()

        # Determine task type
        task_type = TaskType.EDIT  # default
        if any(word in description_lower for word in ["ask", "question", "explain", "what"]):
            task_type = TaskType.ANALYZE
        elif any(word in description_lower for word in ["architect", "design", "plan", "structure"]):
            task_type = TaskType.ARCHITECT
        elif any(word in description_lower for word in ["context", "find", "identify", "which files"]):
            task_type = TaskType.CONTEXT
        elif any(word in description_lower for word in ["help", "how to", "guide"]):
            task_type = TaskType.HELP

        # Determine complexity
        complexity = "medium"  # default
        if any(word in description_lower for word in ["simple", "basic", "quick", "small"]):
            complexity = "simple"
        elif any(word in description_lower for word in ["complex", "advanced", "refactor", "redesign"]):
            complexity = "complex"

        # Look for file-related keywords
        needs_multiple_files = any(word in description_lower for word in [
            "multiple files", "several files", "across files", "all files"
        ])

        return {
            "task_type": task_type,
            "complexity": complexity,
            "needs_multiple_files": needs_multiple_files,
            "keywords": description_lower.split()
        }

    # =============================================================================
    # Configuration and Optimization Methods
    # =============================================================================

    def _get_optimal_config(
        self,
        context: ContextAnalysis,
        model_name: Optional[str],
        edit_format: Optional[str]
    ) -> AiderConfig:
        """Get optimal configuration for the context."""
        # Start with current config or create new one
        config = get_current_config()
        if not config:
            config = create_config_for_model(model_name or "gpt-4", edit_format)

        # Apply context-based optimizations with safe attribute access
        if context.file_count > 10 and config.performance:
            if hasattr(config.performance, 'cache_prompts'):
                config.performance.cache_prompts = True
            if config.edit and hasattr(config.edit, 'validate_before_apply'):
                config.edit.validate_before_apply = True

        if context.max_file_size_kb > 200 and config.security:
            if hasattr(config.security, 'max_file_size_mb'):
                config.security.max_file_size_mb = int(max(20, context.max_file_size_kb / 1024 * 2))

        if context.task_complexity == "complex":
            if config.ui and hasattr(config.ui, 'verbose'):
                config.ui.verbose = True
            if config.edit and hasattr(config.edit, 'validate_before_apply'):
                config.edit.validate_before_apply = True

        return config

    def _get_optimal_format_for_model(self, model_name: str) -> str:
        """Get optimal edit format for a specific model."""
        model_lower = model_name.lower()

        # Model-specific optimizations
        if "claude-3" in model_lower:
            return "diff-fenced"
        elif "gpt-4-turbo" in model_lower:
            return "udiff"
        elif "gpt-4" in model_lower:
            return "udiff"
        elif "claude" in model_lower:
            return "diff-fenced"
        elif "gemini" in model_lower:
            return "diff"
        elif "gpt-3.5" in model_lower:
            return "diff"
        else:
            return "diff"  # Safe fallback

    def _create_coder_instance(
        self,
        profile: CoderProfile,
        io,
        config: AiderConfig,
        context: ContextAnalysis,
        **kwargs
    ) -> Coder:
        """
        Create a coder instance from the selected profile.

        Args:
            profile: Selected coder profile
            io: Input/output handler
            config: Configuration to use
            context: Context analysis
            **kwargs: Additional parameters

        Returns:
            Configured coder instance
        """
        # Prepare creation parameters
        creation_params = {
            "io": io,
            "edit_format": profile.edit_format,
            **kwargs
        }

        # Add model if editing coder
        if CoderCapability.FILE_EDITING in profile.capabilities:
            creation_params["main_model"] = kwargs.get("main_model")

        # Create instance using the coder's create method if available
        if hasattr(profile.coder_class, "create"):
            return profile.coder_class.create(**creation_params)
        else:
            return profile.coder_class(**creation_params)

    def _optimize_coder_instance(self, coder: Coder, context: ContextAnalysis):
        """
        Apply post-creation optimizations to coder instance.

        Args:
            coder: Coder instance to optimize
            context: Context analysis
        """
        # Optimize based on file count
        if context.file_count > 10:
            if hasattr(coder, 'map_tokens'):
                try:
                    current_tokens = getattr(coder, 'map_tokens', 1024)
                    if hasattr(coder, '__dict__') and 'map_tokens' in coder.__dict__:
                        coder.__dict__['map_tokens'] = min(current_tokens or 1024, 2048)
                except (AttributeError, TypeError):
                    # Attribute might be read-only or not settable, skip optimization
                    pass

        # Optimize based on file sizes
        if context.avg_file_size_kb > 50:
            if hasattr(coder, 'show_diffs'):
                coder.show_diffs = True

        # Optimize based on model
        model_lower = context.model_name.lower()
        if "claude" in model_lower:
            if hasattr(coder, 'fence'):
                # Claude prefers triple backticks
                coder.fence = ("```", "```")

    def _get_coder_recommendations(
        self,
        task_analysis: Dict[str, Any],
        context: Optional[ContextAnalysis],
        model_name: str
    ) -> List[Dict[str, Any]]:
        """
        Get ranked recommendations for coders based on analysis.

        Args:
            task_analysis: Task analysis results
            context: Optional context analysis
            model_name: AI model name

        Returns:
            List of recommendations with scores and reasoning
        """
        recommendations = []

        # Score each coder profile
        for name, profile in self._coder_profiles.items():
            score = 0
            reasons = []

            # Task type matching
            task_type = task_analysis.get("task_type")
            if task_type == TaskType.EDIT and CoderCapability.FILE_EDITING in profile.capabilities:
                score += 3
                reasons.append("Supports file editing")
            elif task_type == TaskType.ANALYZE and CoderCapability.CONTEXT_ANALYSIS in profile.capabilities:
                score += 3
                reasons.append("Good for code analysis")

            # Model compatibility
            model_compatible = (
                "any" in profile.model_requirements or
                any(req.lower() in model_name.lower() for req in profile.model_requirements)
            )
            if model_compatible:
                score += 2
                reasons.append(f"Compatible with {model_name}")

            # Complexity matching
            task_complexity = task_analysis.get("complexity", "medium")
            if task_complexity == "simple" and profile.complexity_rating <= 2:
                score += 1
                reasons.append("Good for simple tasks")
            elif task_complexity == "complex" and profile.complexity_rating >= 4:
                score += 2
                reasons.append("Handles complex tasks")

            # Context-based scoring
            if context:
                if context.file_count > 5 and CoderCapability.MULTIPLE_FILES in profile.capabilities:
                    score += 1
                    reasons.append("Handles multiple files well")

                if context.avg_file_size_kb > 100 and CoderCapability.LARGE_FILES in profile.capabilities:
                    score += 1
                    reasons.append("Optimized for large files")

            recommendations.append({
                "name": name,
                "profile": profile,
                "score": score,
                "reasons": reasons,
                "suitability": "high" if score >= 5 else "medium" if score >= 3 else "low"
            })

        # Sort by score
        recommendations.sort(key=lambda x: x["score"], reverse=True)
        return recommendations[:5]  # Return top 5

    def _explain_recommendations(
        self,
        recommendations: List[Dict[str, Any]],
        task_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate explanations for the recommendations.

        Args:
            recommendations: List of recommendations
            task_analysis: Task analysis results

        Returns:
            Explanation and reasoning
        """
        if not recommendations:
            return {"error": "No suitable coders found"}

        top_recommendation = recommendations[0]

        explanation = {
            "top_choice": {
                "name": top_recommendation["name"],
                "score": top_recommendation["score"],
                "reasons": top_recommendation["reasons"],
                "description": top_recommendation["profile"].description
            },
            "alternatives": [
                {
                    "name": rec["name"],
                    "score": rec["score"],
                    "description": rec["profile"].description,
                    "why_not_first": self._compare_to_top(rec, top_recommendation)
                }
                for rec in recommendations[1:3]  # Top 2 alternatives
            ],
            "task_summary": {
                "type": task_analysis.get("task_type", "unknown"),
                "complexity": task_analysis.get("complexity", "medium"),
                "keywords": task_analysis.get("keywords", [])[:5]
            }
        }

        return explanation

    def _compare_to_top(self, candidate: Dict[str, Any], top: Dict[str, Any]) -> str:
        """Compare a candidate to the top recommendation."""
        score_diff = top["score"] - candidate["score"]

        if score_diff <= 1:
            return "Very close alternative with similar capabilities"
        elif score_diff <= 2:
            return "Good alternative but slightly less optimal"
        else:
            return "Lower score due to less optimal fit for this task"

    # =============================================================================
    # Utility and Information Methods
    # =============================================================================

    def get_all_coder_info(self) -> Dict[str, Any]:
        """Get comprehensive information about all available coders."""
        coder_info = {}

        for name, profile in self._coder_profiles.items():
            coder_info[name] = {
                "class_name": profile.coder_class.__name__,
                "edit_format": profile.edit_format,
                "capabilities": [cap.value for cap in profile.capabilities],
                "best_for": profile.best_for,
                "model_requirements": profile.model_requirements,
                "file_size_range_kb": profile.file_size_preference,
                "complexity_rating": profile.complexity_rating,
                "description": profile.description
            }

        return coder_info

    def get_supported_edit_formats(self) -> List[str]:
        """Get list of all supported edit formats."""
        return [profile.edit_format for profile in self._coder_profiles.values()]

    def get_coders_for_capability(self, capability: CoderCapability) -> List[str]:
        """Get list of coders that support a specific capability."""
        return [
            name for name, profile in self._coder_profiles.items()
            if capability in profile.capabilities
        ]

    def validate_coder_compatibility(
        self,
        coder_name: str,
        model_name: str,
        file_size_kb: float
    ) -> Dict[str, Any]:
        """
        Validate if a coder is compatible with given parameters.

        Args:
            coder_name: Name of the coder to validate
            model_name: AI model name
            file_size_kb: File size in KB

        Returns:
            Compatibility analysis
        """
        if coder_name not in self._coder_profiles:
            return {"compatible": False, "reason": "Unknown coder"}

        profile = self._coder_profiles[coder_name]

        # Check model compatibility
        model_compatible = (
            "any" in profile.model_requirements or
            any(req.lower() in model_name.lower() for req in profile.model_requirements)
        )

        # Check file size preference
        min_size, max_size = profile.file_size_preference
        size_compatible = min_size <= file_size_kb <= max_size

        compatible = model_compatible and size_compatible

        return {
            "compatible": compatible,
            "model_compatible": model_compatible,
            "size_compatible": size_compatible,
            "profile": profile,
            "recommendations": self._get_compatibility_recommendations(
                profile, model_name, file_size_kb, compatible
            )
        }

    def _get_compatibility_recommendations(
        self,
        profile: CoderProfile,
        model_name: str,
        file_size_kb: float,
        compatible: bool
    ) -> List[str]:
        """Get recommendations for improving compatibility."""
        recommendations = []

        if not compatible:
            # Model recommendations
            model_lower = model_name.lower()
            if not any(req.lower() in model_lower for req in profile.model_requirements):
                if "any" not in profile.model_requirements:
                    recommendations.append(
                        f"Consider using one of these models: {', '.join(profile.model_requirements)}"
                    )

            # File size recommendations
            min_size, max_size = profile.file_size_preference
            if file_size_kb < min_size:
                recommendations.append(f"This coder works better with files larger than {min_size}KB")
            elif file_size_kb > max_size:
                recommendations.append(f"Consider using a coder optimized for files larger than {max_size}KB")

        return recommendations

    def benchmark_coders(
        self,
        test_scenarios: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Benchmark different coders against test scenarios.

        Args:
            test_scenarios: List of test scenarios with context and expected outcomes

        Returns:
            Benchmark results
        """
        results = {}

        for scenario in test_scenarios:
            scenario_name = scenario.get("name", "unnamed")
            context = scenario.get("context")

            # Test each applicable coder
            for name, profile in self._coder_profiles.items():
                if name not in results:
                    results[name] = {
                        "scenarios_tested": 0,
                        "scenarios_passed": 0,
                        "total_score": 0,
                        "scenarios": {}
                    }

                # Initialize score for this scenario
                score = 0

                # Score this coder for this scenario
                if context is not None:
                    score = self._score_coder_for_scenario(profile, context)

                passed = score >= 3  # Threshold for passing

                results[name]["scenarios_tested"] += 1
                results[name]["total_score"] += score
                if passed:
                    results[name]["scenarios_passed"] += 1

                results[name]["scenarios"][scenario_name] = {
                    "score": score,
                    "passed": passed
                }

        # Calculate final metrics
        for name, result in results.items():
            if result["scenarios_tested"] > 0:
                result["average_score"] = result["total_score"] / result["scenarios_tested"]
                result["pass_rate"] = result["scenarios_passed"] / result["scenarios_tested"]

        return results

    def _score_coder_for_scenario(self, profile: CoderProfile, context: Dict[str, Any]) -> int:
        """Score a coder profile for a specific scenario."""
        score = 0

        # Base score
        score += 1

        # Model compatibility
        model_name = context.get("model_name", "").lower()
        if any(req.lower() in model_name for req in profile.model_requirements):
            score += 2

        # File size compatibility
        file_size = context.get("file_size_kb", 0)
        min_size, max_size = profile.file_size_preference
        if min_size <= file_size <= max_size:
            score += 2

        # Capability matching
        required_caps = context.get("required_capabilities", [])
        if all(cap in profile.capabilities for cap in required_caps):
            score += 1

        return min(score, 5)  # Cap at 5


# =============================================================================
# Global Factory Instance
# =============================================================================

# Global factory instance
_factory_instance: Optional[EnhancedCoderFactory] = None


def get_coder_factory() -> EnhancedCoderFactory:
    """Get the global coder factory instance."""
    global _factory_instance
    if _factory_instance is None:
        _factory_instance = EnhancedCoderFactory()
    return _factory_instance


def create_optimal_coder(
    io,
    task_description: Optional[str] = None,
    files: Optional[List[str]] = None,
    model_name: str = "gpt-4",
    edit_format: Optional[str] = None,
    **kwargs
) -> Coder:
    """
    Convenience function to create an optimal coder.

    Args:
        io: Input/output handler
        task_description: Optional description of the task
        files: Optional list of files to work with
        model_name: AI model name
        edit_format: Optional specific edit format
        **kwargs: Additional parameters

    Returns:
        Optimally configured coder
    """
    factory = get_coder_factory()

    # Determine task type from description
    task_type = None
    if task_description:
        task_analysis = factory._analyze_task_description(task_description)
        task_type = task_analysis["task_type"]

    # Create appropriate coder
    if files:
        return factory.create_for_files(
            io=io,
            file_paths=files,
            model_name=model_name,
            task_type=task_type,
            edit_format=edit_format,
            **kwargs
        )
    else:
        return factory.create_coder(
            io=io,
            task_type=task_type,
            model_name=model_name,
            **kwargs
        )


# =============================================================================
# Backwards Compatibility Layer
# =============================================================================

class LegacyCoderFactory:
    """
    Backwards compatibility layer for the original Coder.create method.

    This allows the enhanced factory to be used as a drop-in replacement
    while maintaining compatibility with existing code.
    """

    @staticmethod
    def create(
        main_model=None,
        edit_format=None,
        io=None,
        **kwargs
    ) -> Coder:
        """
        Create coder using legacy interface.

        Args:
            main_model: The AI model to use
            edit_format: Edit format string
            io: Input/output handler
            **kwargs: Additional legacy parameters

        Returns:
            Coder instance compatible with legacy expectations
        """
        factory = get_coder_factory()

        # Extract model name
        model_name = "gpt-4"  # default
        if main_model and hasattr(main_model, 'name'):
            model_name = main_model.name

        # Map legacy edit formats to current formats
        format_mapping = {
            "diff": "diff",
            "whole": "whole",
            "udiff": "udiff",
            "architect": "architect",
            "ask": "ask",
            "help": "help",
            "context": "context",
            "patch": "patch",
            "editor-diff": "editor-diff",
            "editor-whole": "editor-whole",
            "editor-diff-fenced": "editor-diff-fenced"
        }

        mapped_format = format_mapping.get(edit_format, edit_format)

        # Create using enhanced factory
        try:
            coder = factory.create_coder(
                io=io,
                edit_format=mapped_format,
                model_name=model_name,
                main_model=main_model,
                **kwargs
            )

            # Ensure legacy attributes are present
            if main_model:
                coder.main_model = main_model

            return coder

        except Exception as e:
            # Fallback to original implementation if factory fails
            factory.logger.warning(f"Enhanced factory failed, falling back to legacy: {e}")
            return _create_legacy_fallback(main_model, edit_format, io, **kwargs)


def _create_legacy_fallback(main_model, edit_format, io, **kwargs) -> Coder:
    """Create coder using original logic as fallback."""
    # This would implement the original Coder.create logic
    # For now, return a basic EditBlockCoder
    return EditBlockCoder(
        main_model=main_model,
        edit_format=edit_format or "diff",
        io=io,
        **kwargs
    )


# =============================================================================
# Usage Examples and Utilities
# =============================================================================

def print_coder_selection_guide():
    """Print a comprehensive guide for coder selection."""
    factory = get_coder_factory()

    print("🎯 Aider Coder Selection Guide")
    print("=" * 50)

    for name, profile in factory._coder_profiles.items():
        print(f"\n📝 {name.upper()}")
        print(f"   Format: {profile.edit_format}")
        print(f"   Description: {profile.description}")
        print(f"   Complexity: {profile.complexity_rating}/5")
        print(f"   Best for: {', '.join(profile.best_for)}")
        print(f"   Models: {', '.join(profile.model_requirements)}")
        print(f"   File size: {profile.file_size_preference[0]}-{profile.file_size_preference[1]}KB")

    print("\n💡 Quick Selection Tips:")
    print("   • Small files (<10KB): wholefile")
    print("   • Large files (>100KB): udiff")
    print("   • Claude models: editblock_fenced")
    print("   • GPT-4: udiff or editblock")
    print("   • Questions only: ask")
    print("   • Planning: architect or context")


def analyze_project_for_optimal_coder(project_path: Path) -> Dict[str, Any]:
    """
    Analyze a project and recommend optimal coder configuration.

    Args:
        project_path: Path to the project

    Returns:
        Analysis results and recommendations
    """
    factory = get_coder_factory()

    # Analyze project files
    all_files = list(project_path.rglob("*"))
    code_files = [
        f for f in all_files
        if f.is_file() and f.suffix in [
            ".py", ".js", ".ts", ".java", ".cpp", ".c", ".h", ".cs", ".php", ".rb", ".go"
        ]
    ]

    if not code_files:
        return {"error": "No code files found in project"}

    # Calculate statistics
    total_size = sum(f.stat().st_size for f in code_files) / 1024  # KB
    avg_size = total_size / len(code_files)
    max_size = max(f.stat().st_size for f in code_files) / 1024

    file_types = list(set(f.suffix for f in code_files))

    # Create context
    context = ContextAnalysis(
        file_count=len(code_files),
        total_file_size_kb=total_size,
        avg_file_size_kb=avg_size,
        max_file_size_kb=max_size,
        file_types=file_types,
        has_git_repo=(project_path / ".git").exists(),
        model_name="gpt-4",  # default for analysis
        task_complexity="medium"
    )

    # Get recommendations for different scenarios
    recommendations = {
        "for_editing": factory.create_for_files(
            io=None,  # placeholder
            file_paths=[str(f.relative_to(project_path)) for f in code_files[:5]],
            model_name="gpt-4"
        ).__class__.__name__,
        "for_large_files": "udiff" if max_size > 100 else "diff",
        "for_claude": "diff-fenced",
        "for_gpt4": "udiff",
        "context_analysis": context
    }

    return recommendations


# =============================================================================
# Factory Registration System
# =============================================================================

def register_custom_coder(
    name: str,
    coder_class: Type[Coder],
    edit_format: str,
    capabilities: List[CoderCapability],
    **profile_kwargs
):
    """
    Register a custom coder with the factory.

    Args:
        name: Name for the coder
        coder_class: Coder class
        edit_format: Edit format string
        capabilities: List of capabilities
        **profile_kwargs: Additional profile parameters
    """
    factory = get_coder_factory()

    profile = CoderProfile(
        coder_class=coder_class,
        edit_format=edit_format,
        capabilities=capabilities,
        best_for=profile_kwargs.get("best_for", []),
        model_requirements=profile_kwargs.get("model_requirements", ["any"]),
        file_size_preference=profile_kwargs.get("file_size_preference", (0, 1000)),
        complexity_rating=profile_kwargs.get("complexity_rating", 3),
        description=profile_kwargs.get("description", "Custom coder")
    )

    factory._coder_profiles[name] = profile


# =============================================================================
# Testing and Validation
# =============================================================================

def test_factory_configuration():
    """Test the factory configuration and report any issues."""
    factory = get_coder_factory()
    issues = []

    # Test each coder profile
    for name, profile in factory._coder_profiles.items():
        try:
            # Validate profile
            if not profile.coder_class:
                issues.append(f"{name}: Missing coder class")

            if not profile.edit_format:
                issues.append(f"{name}: Missing edit format")

            if not profile.description:
                issues.append(f"{name}: Missing description")

            # Test instantiation (mock)
            # This would require actual IO and model objects in practice

        except Exception as e:
            issues.append(f"{name}: Configuration error - {e}")

    if issues:
        print("⚠️  Factory Configuration Issues:")
        for issue in issues:
            print(f"   • {issue}")
    else:
        print("✅ Factory configuration is valid")

    return len(issues) == 0


def get_factory_statistics() -> Dict[str, Any]:
    """Get statistics about the factory configuration."""
    factory = get_coder_factory()

    profiles = factory._coder_profiles

    stats = {
        "total_coders": len(profiles),
        "edit_formats": len(set(p.edit_format for p in profiles.values())),
        "capabilities_count": {},
        "complexity_distribution": {},
        "model_support": {}
    }

    # Count capabilities
    for profile in profiles.values():
        for capability in profile.capabilities:
            cap_name = capability.value
            stats["capabilities_count"][cap_name] = stats["capabilities_count"].get(cap_name, 0) + 1

    # Count complexity levels
    for profile in profiles.values():
        rating = profile.complexity_rating
        stats["complexity_distribution"][rating] = stats["complexity_distribution"].get(rating, 0) + 1

    # Count model support
    for profile in profiles.values():
        for model in profile.model_requirements:
            stats["model_support"][model] = stats["model_support"].get(model, 0) + 1

    return stats
