"""
AI Models Module - Enterprise Model Management
Provides intelligent routing, multi-provider support, and production-grade AI model management.
"""

from .model_manager import (
    # Core Classes
    ModelManager,
    ModelProvider,
    ModelConfig,
    ModelRequest,
    ModelResponse,

    # Provider Implementations
    OpenAIProvider,
    AnthropicProvider,
    XAIProvider,

    # Enums
    TaskType,
    ComplexityLevel,
    Priority,

    # Convenience Functions
    generate_code,
    analyze_code,
    get_model_manager,

    # Utility Classes
    RateLimiter,
)

# Version info
__version__ = "2.1.0"
__author__ = "Aider AI Team"

# Default exports for common usage
__all__ = [
    # Core functionality
    "ModelManager",
    "ModelRequest",
    "ModelResponse",
    "get_model_manager",

    # Enums for request configuration
    "TaskType",
    "ComplexityLevel",
    "Priority",

    # Convenience functions
    "generate_code",
    "analyze_code",

    # Advanced usage
    "ModelProvider",
    "ModelConfig",
    "OpenAIProvider",
    "AnthropicProvider",
    "XAIProvider",
    "RateLimiter",
]

# Module-level documentation
"""
Usage Examples:

1. Quick Code Generation:
    ```python
    from aider.models import generate_code, ComplexityLevel

    code = await generate_code(
        "Create a REST API endpoint for user registration",
        language="python",
        complexity=ComplexityLevel.MEDIUM
    )
    ```

2. Advanced Model Management:
    ```python
    from aider.models import ModelManager, ModelRequest, TaskType, Priority

    manager = await get_model_manager()

    request = ModelRequest(
        prompt="Debug this function...",
        task_type=TaskType.DEBUGGING,
        complexity=ComplexityLevel.COMPLEX,
        priority=Priority.QUALITY
    )

    response = await manager.generate_response(request)
    ```

3. Health Monitoring:
    ```python
    from aider.models import get_model_manager

    manager = await get_model_manager()
    health = await manager.health_check()
    metrics = manager.get_performance_metrics()
    ```
"""
