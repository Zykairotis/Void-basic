# Core Legacy Coders (for backwards compatibility)
from .architect_coder import ArchitectCoder
from .ask_coder import AskCoder
from .base_coder import Coder
from .context_coder import ContextCoder
from .editblock_coder import EditBlockCoder
from .editblock_fenced_coder import EditBlockFencedCoder
from .editor_diff_fenced_coder import EditorDiffFencedCoder
from .editor_editblock_coder import EditorEditBlockCoder
from .editor_whole_coder import EditorWholeFileCoder
from .help_coder import HelpCoder
from .patch_coder import PatchCoder
from .udiff_coder import UnifiedDiffCoder
from .udiff_simple import UnifiedDiffSimpleCoder
from .wholefile_coder import WholeFileCoder

# Enhanced Framework Components
from .exceptions import (
    AiderCoderError,
    ConfigurationError,
    ValidationError,
    EditOperationError,
    FileNotFoundError,
    FileNotEditableError,
    SearchTextNotFoundError,
    SearchTextNotUniqueError,
    DiffApplicationError,
    MalformedEditError,
    TokenLimitExceededError,
    ModelResponseError,
    PartialEditError,
    UnknownEditFormat,
    MissingAPIKeyError,
    FinishReasonLength,
    ErrorContext,
    handle_edit_errors,
    format_error_for_user
)

from .edit_strategies import (
    EditStrategy,
    EditStrategyFactory,
    EditStrategyCoordinator,
    SearchReplaceStrategy,
    UnifiedDiffStrategy,
    WholeFileStrategy,
    PatchStrategy,
    NoOpStrategy,
    EditResult,
    EditInstruction,
    EditMetrics,
    PerformanceTracker
)

from .config import (
    AiderConfig,
    ModelConfig,
    EditConfig,
    SecurityConfig,
    PerformanceConfig,
    UIConfig,
    LoggingConfig,
    EditFormat,
    ModelProvider,
    ConfigManager,
    ConfigBuilder,
    ConfigValidator,
    get_config_manager,
    get_current_config,
    load_config,
    create_config_for_model
)

from .enhanced_base_coder import (
    EnhancedCoder,
    EnhancedEditBlockCoder,
    EnhancedUnifiedDiffCoder,
    EnhancedWholeFileCoder,
    EnhancedAskCoder,
    migrate_legacy_coder,
    create_coder_for_task,
    get_optimal_coder_for_context,
    create_safe_coder
)

from .enhanced_factory import (
    EnhancedCoderFactory,
    LegacyCoderFactory,
    TaskType,
    CoderCapability,
    CoderProfile,
    ContextAnalysis,
    get_coder_factory,
    create_optimal_coder,
    register_custom_coder,
    print_coder_selection_guide,
    analyze_project_for_optimal_coder
)

from .performance_optimizer import (
    PerformanceMetrics,
    MetricsCollector,
    LRUCache,
    PerformanceOptimizer,
    ResourcePool,
    GlobalPerformanceManager,
    OptimizationStrategy,
    MemoryOptimizationStrategy,
    CpuOptimizationStrategy,
    performance_monitoring,
    async_performance_monitoring,
    performance_monitor,
    async_performance_monitor,
    cached_result,
    get_global_performance_manager,
    monitor_performance,
    optimize_object,
    get_performance_stats,
    CacheKey,
    CacheValue,
    MetricValue,
    OptimizationLevel,
    ResourceType
)

# from .single_wholefile_func_coder import SingleWholeFileFunctionCoder

__all__ = [
    # Legacy Coders (backwards compatibility)
    "HelpCoder",
    "AskCoder",
    "Coder",
    "EditBlockCoder",
    "EditBlockFencedCoder",
    "WholeFileCoder",
    "PatchCoder",
    "UnifiedDiffCoder",
    "UnifiedDiffSimpleCoder",
    "ArchitectCoder",
    "EditorEditBlockCoder",
    "EditorWholeFileCoder",
    "EditorDiffFencedCoder",
    "ContextCoder",

    # Enhanced Framework - Core Components
    "AiderCoderError",
    "ConfigurationError",
    "ValidationError",
    "EditOperationError",
    "UnknownEditFormat",
    "MissingAPIKeyError",
    "FinishReasonLength",
    "ErrorContext",
    "handle_edit_errors",
    "format_error_for_user",

    # Enhanced Framework - Edit Strategies
    "EditStrategy",
    "EditStrategyFactory",
    "EditStrategyCoordinator",
    "SearchReplaceStrategy",
    "UnifiedDiffStrategy",
    "WholeFileStrategy",
    "PatchStrategy",
    "NoOpStrategy",
    "EditResult",
    "EditInstruction",
    "EditMetrics",
    "PerformanceTracker",

    # Enhanced Framework - Configuration
    "AiderConfig",
    "ModelConfig",
    "EditConfig",
    "SecurityConfig",
    "PerformanceConfig",
    "UIConfig",
    "LoggingConfig",
    "EditFormat",
    "ModelProvider",
    "ConfigManager",
    "ConfigBuilder",
    "ConfigValidator",
    "get_config_manager",
    "get_current_config",
    "load_config",
    "create_config_for_model",

    # Enhanced Framework - Enhanced Coders
    "EnhancedCoder",
    "EnhancedEditBlockCoder",
    "EnhancedUnifiedDiffCoder",
    "EnhancedWholeFileCoder",
    "EnhancedAskCoder",
    "migrate_legacy_coder",
    "create_coder_for_task",
    "get_optimal_coder_for_context",
    "create_safe_coder",

    # Enhanced Framework - Factory System
    "EnhancedCoderFactory",
    "LegacyCoderFactory",
    "TaskType",
    "CoderCapability",
    "CoderProfile",
    "ContextAnalysis",
    "get_coder_factory",
    "create_optimal_coder",
    "register_custom_coder",
    "print_coder_selection_guide",
    "analyze_project_for_optimal_coder",

    # Enhanced Framework - Performance Optimization
    "PerformanceMetrics",
    "MetricsCollector",
    "LRUCache",
    "PerformanceOptimizer",
    "ResourcePool",
    "GlobalPerformanceManager",
    "OptimizationStrategy",
    "MemoryOptimizationStrategy",
    "CpuOptimizationStrategy",
    "performance_monitoring",
    "async_performance_monitoring",
    "performance_monitor",
    "async_performance_monitor",
    "cached_result",
    "get_global_performance_manager",
    "monitor_performance",
    "optimize_object",
    "get_performance_stats",
    "CacheKey",
    "CacheValue",
    "MetricValue",
    "OptimizationLevel",
    "ResourceType",
]
