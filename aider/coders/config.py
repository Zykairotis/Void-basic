"""
Enhanced configuration management system for Aider coders.

This module provides a comprehensive, type-safe configuration framework
that supports validation, environment variables, profiles, and extensibility.
"""

import os
import json
import yaml
from typing import Any, Dict, List, Optional, Union, Type, TypeVar, get_type_hints
from dataclasses import dataclass, field, fields
from pathlib import Path
from enum import Enum
import logging

from .exceptions import ConfigurationError, ValidationError, ErrorContext


# =============================================================================
# Configuration Data Classes
# =============================================================================

class EditFormat(Enum):
    """Supported edit formats."""
    DIFF = "diff"
    DIFF_FENCED = "diff-fenced"
    UDIFF = "udiff"
    UDIFF_SIMPLE = "udiff-simple"
    PATCH = "patch"
    WHOLE = "whole"
    EDITOR_DIFF = "editor-diff"
    EDITOR_DIFF_FENCED = "editor-diff-fenced"
    EDITOR_WHOLE = "editor-whole"
    ASK = "ask"
    HELP = "help"
    CONTEXT = "context"
    ARCHITECT = "architect"


class ModelProvider(Enum):
    """Supported model providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    COHERE = "cohere"
    LOCAL = "local"


@dataclass
class ModelConfig:
    """Configuration for AI models."""
    name: str
    provider: ModelProvider
    api_key_env_var: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: float = 0.0
    supports_functions: bool = True
    supports_streaming: bool = True
    context_window: Optional[int] = None
    cost_per_token: Optional[float] = None
    editor_model: Optional[str] = None
    editor_edit_format: Optional[EditFormat] = None

    def __post_init__(self):
        """Validate and normalize model configuration."""
        if isinstance(self.provider, str):
            self.provider = ModelProvider(self.provider)
        if isinstance(self.editor_edit_format, str):
            self.editor_edit_format = EditFormat(self.editor_edit_format)


@dataclass
class EditConfig:
    """Configuration for edit operations."""
    format: EditFormat = EditFormat.DIFF
    auto_commits: bool = True
    commit_message_template: str = "aider: {summary}"
    max_file_size_kb: int = 1000
    max_lines_per_edit: int = 1000
    validate_before_apply: bool = True
    backup_before_edit: bool = False
    dry_run_mode: bool = False
    suggest_shell_commands: bool = True

    def __post_init__(self):
        """Validate and normalize edit configuration."""
        if isinstance(self.format, str):
            self.format = EditFormat(self.format)


@dataclass
class SecurityConfig:
    """Security configuration for file operations."""
    allowed_file_extensions: List[str] = field(default_factory=lambda: [
        ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".cpp", ".c", ".h",
        ".cs", ".php", ".rb", ".go", ".rs", ".swift", ".kt", ".scala",
        ".html", ".css", ".scss", ".less", ".vue", ".svelte",
        ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".ini",
        ".sql", ".sh", ".bash", ".ps1", ".bat"
    ])
    blocked_file_patterns: List[str] = field(default_factory=lambda: [
        "*.exe", "*.dll", "*.so", "*.dylib", "*.bin",
        ".env", ".env.*", "*.key", "*.pem", "*.p12"
    ])
    allowed_directories: List[str] = field(default_factory=list)
    blocked_directories: List[str] = field(default_factory=lambda: [
        ".git", ".svn", ".hg", "__pycache__", "node_modules",
        ".venv", "venv", ".env", "build", "dist", "target"
    ])
    max_file_size_mb: int = 10
    allow_file_creation: bool = True
    allow_file_deletion: bool = False
    require_git_repo: bool = True


@dataclass
class PerformanceConfig:
    """Performance and optimization configuration."""
    cache_prompts: bool = True
    cache_size_mb: int = 100
    max_concurrent_requests: int = 3
    request_timeout_seconds: int = 120
    token_buffer_percent: int = 10  # Reserve 10% of context for response
    enable_streaming: bool = True
    warm_cache_on_start: bool = False
    num_cache_warming_pings: int = 0


@dataclass
class UIConfig:
    """User interface configuration."""
    show_diffs: bool = True
    show_tokens: bool = True
    show_costs: bool = True
    confirm_edits: bool = False
    auto_accept_architect: bool = False
    use_color: bool = True
    language: str = "english"
    verbose: bool = False
    progress_indicators: bool = True


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    file_path: Optional[str] = None
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    max_file_size_mb: int = 10
    backup_count: int = 5
    enable_metrics: bool = True
    log_api_calls: bool = False


@dataclass
class AiderConfig:
    """Main configuration class for Aider coders."""
    model: ModelConfig
    edit: EditConfig = field(default_factory=EditConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Additional settings
    workspace_path: Path | None = None
    profile_name: str = "default"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate the complete configuration."""
        if not self.profile_name:
            raise ValidationError("profile_name cannot be empty")
        if self.workspace_path is not None:
            if not isinstance(self.workspace_path, Path):
                self.workspace_path = Path(self.workspace_path)

        # Validate all sub-configurations
        for attr_name in ['model', 'edit', 'security', 'performance', 'ui', 'logging']:
            attr_value = getattr(self, attr_name)
            if attr_value is None:
                raise ValidationError(f"{attr_name} configuration cannot be None")


# =============================================================================
# Configuration Manager
# =============================================================================

class ConfigManager:
    """
    Centralized configuration management for Aider.

    Provides loading from files, environment variables, validation,
    and profile management.
    """

    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize configuration manager.

        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = config_dir or Path.home() / ".aider"
        self.config_dir.mkdir(exist_ok=True)

        self._current_config: Optional[AiderConfig] = None
        self._profiles: Dict[str, AiderConfig] = {}
        self._config_file_path = self.config_dir / "config.yaml"
        self._profiles_file_path = self.config_dir / "profiles.yaml"

        # Load default configurations
        self._load_default_profiles()

    def load_config(
        self,
        profile_name: str = "default",
        config_overrides: Optional[Dict[str, Any]] = None
    ) -> AiderConfig:
        """
        Load configuration for the specified profile.

        Args:
            profile_name: Name of the configuration profile
            config_overrides: Optional configuration overrides

        Returns:
            Loaded and validated configuration

        Raises:
            ConfigurationError: If configuration cannot be loaded
        """
        try:
            # Start with profile configuration
            if profile_name in self._profiles:
                config = self._profiles[profile_name]
            else:
                config = self._create_default_config()

            # Apply environment variable overrides
            config = self._apply_env_overrides(config)

            # Apply programmatic overrides
            if config_overrides:
                config = self._apply_dict_overrides(config, config_overrides)

            # Validate final configuration
            self._validate_config(config)

            self._current_config = config
            return config

        except Exception as e:
            raise ConfigurationError(
                f"Failed to load configuration for profile '{profile_name}'",
                ErrorContext(
                    error_code="CONFIG_LOAD_FAILED",
                    suggestions=[
                        "Check configuration file syntax",
                        "Verify profile exists",
                        "Check environment variables"
                    ]
                )
            ) from e

    def save_config(self, config: AiderConfig, profile_name: str = "default"):
        """
        Save configuration to file.

        Args:
            config: Configuration to save
            profile_name: Profile name to save under
        """
        try:
            self._profiles[profile_name] = config

            # Save to profiles file
            profiles_data = {}
            for name, prof_config in self._profiles.items():
                profiles_data[name] = self._config_to_dict(prof_config)

            with open(self._profiles_file_path, 'w') as f:
                yaml.dump(profiles_data, f, default_flow_style=False, indent=2)

        except Exception as e:
            raise ConfigurationError(
                f"Failed to save configuration for profile '{profile_name}'",
                ErrorContext(error_code="CONFIG_SAVE_FAILED")
            ) from e

    def create_profile(
        self,
        name: str,
        base_profile: str = "default",
        overrides: Optional[Dict[str, Any]] = None
    ) -> AiderConfig:
        """
        Create a new configuration profile.

        Args:
            name: Name for the new profile
            base_profile: Profile to base the new one on
            overrides: Optional configuration overrides

        Returns:
            The new configuration profile
        """
        if base_profile not in self._profiles:
            raise ConfigurationError(
                f"Base profile '{base_profile}' not found",
                ErrorContext(error_code="PROFILE_NOT_FOUND")
            )

        # Start with base profile
        base_config = self._profiles[base_profile]
        new_config = self._copy_config(base_config)

        # Apply overrides
        if overrides:
            new_config = self._apply_dict_overrides(new_config, overrides)

        # Save new profile
        new_config.profile_name = name
        self._profiles[name] = new_config

        return new_config

    def list_profiles(self) -> List[str]:
        """Get list of available configuration profiles."""
        return list(self._profiles.keys())

    def get_current_config(self) -> AiderConfig | None:
        """Get the current active configuration."""
        return self._current_config

    def _create_default_config(self) -> AiderConfig:
        """Create default configuration."""
        return AiderConfig(
            model=ModelConfig(
                name="gpt-4",
                provider=ModelProvider.OPENAI,
                api_key_env_var="OPENAI_API_KEY",
                max_tokens=4000,
                context_window=8000
            )
        )

    def _load_default_profiles(self):
        """Load default configuration profiles."""
        # Default profile
        self._profiles["default"] = self._create_default_config()

        # Development profile - more verbose, safer
        dev_config = self._create_default_config()
        dev_config.ui.verbose = True
        dev_config.ui.confirm_edits = True
        dev_config.edit.backup_before_edit = True
        dev_config.security.allow_file_deletion = False
        self._profiles["development"] = dev_config

        # Production profile - efficient, minimal
        prod_config = self._create_default_config()
        prod_config.ui.verbose = False
        prod_config.ui.confirm_edits = False
        prod_config.edit.auto_commits = True
        prod_config.performance.enable_streaming = True
        self._profiles["production"] = prod_config

        # Safe profile - maximum validation and confirmations
        safe_config = self._create_default_config()
        safe_config.ui.confirm_edits = True
        safe_config.edit.validate_before_apply = True
        safe_config.edit.backup_before_edit = True
        safe_config.edit.dry_run_mode = True
        safe_config.security.allow_file_deletion = False
        safe_config.security.max_file_size_mb = 5
        self._profiles["safe"] = safe_config

        # Fast profile - optimized for speed
        fast_config = self._create_default_config()
        fast_config.edit.format = EditFormat.DIFF
        fast_config.ui.confirm_edits = False
        fast_config.edit.validate_before_apply = False
        fast_config.performance.enable_streaming = True
        fast_config.performance.cache_prompts = True
        self._profiles["fast"] = fast_config

        # Load from files if they exist
        self._load_from_files()

    def _load_from_files(self):
        """Load configurations from files."""
        try:
            if self._profiles_file_path.exists():
                with open(self._profiles_file_path, 'r') as f:
                    profiles_data = yaml.safe_load(f)

                for name, profile_dict in profiles_data.items():
                    try:
                        config = self._dict_to_config(profile_dict)
                        self._profiles[name] = config
                    except Exception as e:
                        logging.warning(f"Failed to load profile '{name}': {e}")

        except Exception as e:
            logging.warning(f"Failed to load configuration files: {e}")

    def _apply_env_overrides(self, config: AiderConfig) -> AiderConfig:
        """Apply environment variable overrides to configuration."""
        # Model settings
        if os.getenv("AIDER_MODEL"):
            config.model.name = os.getenv("AIDER_MODEL")

        if os.getenv("AIDER_API_KEY"):
            config.model.api_key_env_var = "AIDER_API_KEY"

        # Edit format
        if os.getenv("AIDER_EDIT_FORMAT"):
            try:
                config.edit.format = EditFormat(os.getenv("AIDER_EDIT_FORMAT"))
            except ValueError:
                pass  # Invalid format, keep default

        # UI settings
        if os.getenv("AIDER_VERBOSE"):
            config.ui.verbose = os.getenv("AIDER_VERBOSE").lower() in ("true", "1", "yes")

        if os.getenv("AIDER_CONFIRM_EDITS"):
            config.ui.confirm_edits = os.getenv("AIDER_CONFIRM_EDITS").lower() in ("true", "1", "yes")

        # Performance settings
        if os.getenv("AIDER_CACHE_PROMPTS"):
            config.performance.cache_prompts = os.getenv("AIDER_CACHE_PROMPTS").lower() in ("true", "1", "yes")

        if os.getenv("AIDER_MAX_TOKENS"):
            try:
                config.model.max_tokens = int(os.getenv("AIDER_MAX_TOKENS"))
            except ValueError:
                pass

        # Security settings
        if os.getenv("AIDER_ALLOW_FILE_DELETION"):
            config.security.allow_file_deletion = os.getenv("AIDER_ALLOW_FILE_DELETION").lower() in ("true", "1", "yes")

        return config

    def _apply_dict_overrides(self, config: AiderConfig, overrides: Dict[str, Any]) -> AiderConfig:
        """Apply dictionary overrides to configuration."""
        config_dict = self._config_to_dict(config)

        # Deep merge the overrides
        merged_dict = self._deep_merge(config_dict, overrides)

        return self._dict_to_config(merged_dict)

    def _deep_merge(self, base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()

        for key, value in overrides.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def _config_to_dict(self, config: AiderConfig) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        def convert_value(value):
            if hasattr(value, '__dict__'):
                # Dataclass instance
                result = {}
                for field_info in fields(value):
                    field_value = getattr(value, field_info.name)
                    result[field_info.name] = convert_value(field_value)
                return result
            elif isinstance(value, Enum):
                return value.value
            elif isinstance(value, list):
                return [convert_value(item) for item in value]
            elif isinstance(value, dict):
                return {k: convert_value(v) for k, v in value.items()}
            else:
                return value

        return convert_value(config)

    def _dict_to_config(self, data: Dict[str, Any]) -> AiderConfig:
        """Convert dictionary to configuration."""
        # Extract model config
        model_data = data.get("model", {})
        model_config = ModelConfig(
            name=model_data.get("name", "gpt-4"),
            provider=ModelProvider(model_data.get("provider", "openai")),
            api_key_env_var=model_data.get("api_key_env_var"),
            max_tokens=model_data.get("max_tokens"),
            temperature=model_data.get("temperature", 0.0),
            supports_functions=model_data.get("supports_functions", True),
            supports_streaming=model_data.get("supports_streaming", True),
            context_window=model_data.get("context_window"),
            cost_per_token=model_data.get("cost_per_token"),
            editor_model=model_data.get("editor_model"),
            editor_edit_format=EditFormat(model_data["editor_edit_format"]) if model_data.get("editor_edit_format") else None
        )

        # Extract other configs
        edit_data = data.get("edit", {})
        edit_config = EditConfig(
            format=EditFormat(edit_data.get("format", "diff")),
            auto_commits=edit_data.get("auto_commits", True),
            commit_message_template=edit_data.get("commit_message_template", "aider: {summary}"),
            max_file_size_kb=edit_data.get("max_file_size_kb", 1000),
            max_lines_per_edit=edit_data.get("max_lines_per_edit", 1000),
            validate_before_apply=edit_data.get("validate_before_apply", True),
            backup_before_edit=edit_data.get("backup_before_edit", False),
            dry_run_mode=edit_data.get("dry_run_mode", False),
            suggest_shell_commands=edit_data.get("suggest_shell_commands", True)
        )

        # Security config
        security_data = data.get("security", {})
        security_config = SecurityConfig(
            allowed_file_extensions=security_data.get("allowed_file_extensions", SecurityConfig().allowed_file_extensions),
            blocked_file_patterns=security_data.get("blocked_file_patterns", SecurityConfig().blocked_file_patterns),
            allowed_directories=security_data.get("allowed_directories", []),
            blocked_directories=security_data.get("blocked_directories", SecurityConfig().blocked_directories),
            max_file_size_mb=security_data.get("max_file_size_mb", 10),
            allow_file_creation=security_data.get("allow_file_creation", True),
            allow_file_deletion=security_data.get("allow_file_deletion", False),
            require_git_repo=security_data.get("require_git_repo", True)
        )

        # Performance config
        perf_data = data.get("performance", {})
        performance_config = PerformanceConfig(
            cache_prompts=perf_data.get("cache_prompts", True),
            cache_size_mb=perf_data.get("cache_size_mb", 100),
            max_concurrent_requests=perf_data.get("max_concurrent_requests", 3),
            request_timeout_seconds=perf_data.get("request_timeout_seconds", 120),
            token_buffer_percent=perf_data.get("token_buffer_percent", 10),
            enable_streaming=perf_data.get("enable_streaming", True),
            warm_cache_on_start=perf_data.get("warm_cache_on_start", False),
            num_cache_warming_pings=perf_data.get("num_cache_warming_pings", 0)
        )

        # UI config
        ui_data = data.get("ui", {})
        ui_config = UIConfig(
            show_diffs=ui_data.get("show_diffs", True),
            show_tokens=ui_data.get("show_tokens", True),
            show_costs=ui_data.get("show_costs", True),
            confirm_edits=ui_data.get("confirm_edits", False),
            auto_accept_architect=ui_data.get("auto_accept_architect", False),
            use_color=ui_data.get("use_color", True),
            language=ui_data.get("language", "english"),
            verbose=ui_data.get("verbose", False),
            progress_indicators=ui_data.get("progress_indicators", True)
        )

        # Logging config
        log_data = data.get("logging", {})
        logging_config = LoggingConfig(
            level=log_data.get("level", "INFO"),
            file_path=log_data.get("file_path"),
            format=log_data.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
            max_file_size_mb=log_data.get("max_file_size_mb", 10),
            backup_count=log_data.get("backup_count", 5),
            enable_metrics=log_data.get("enable_metrics", True),
            log_api_calls=log_data.get("log_api_calls", False)
        )

        # Main config
        return AiderConfig(
            model=model_config,
            edit=edit_config,
            security=security_config,
            performance=performance_config,
            ui=ui_config,
            logging=logging_config,
            workspace_path=data.get("workspace_path"),
            profile_name=data.get("profile_name", "default"),
            custom_settings=data.get("custom_settings", {})
        )

    def _copy_config(self, config: AiderConfig) -> AiderConfig:
        """Create a deep copy of configuration."""
        config_dict = self._config_to_dict(config)
        return self._dict_to_config(config_dict)

    def _validate_config(self, config: AiderConfig):
        """Validate configuration values."""
        errors = []

        # Validate model configuration
        if not config.model.name:
            errors.append("Model name is required")

        if config.model.max_tokens and config.model.max_tokens <= 0:
            errors.append("Model max_tokens must be positive")

        if not (0 <= config.model.temperature <= 2):
            errors.append("Model temperature must be between 0 and 2")

        # Validate edit configuration
        if config.edit.max_file_size_kb <= 0:
            errors.append("Max file size must be positive")

        if config.edit.max_lines_per_edit <= 0:
            errors.append("Max lines per edit must be positive")

        # Validate security configuration
        if config.security.max_file_size_mb <= 0:
            errors.append("Max file size must be positive")

        # Validate performance configuration
        if config.performance.cache_size_mb <= 0:
            errors.append("Cache size must be positive")

        if config.performance.max_concurrent_requests <= 0:
            errors.append("Max concurrent requests must be positive")

        if config.performance.request_timeout_seconds <= 0:
            errors.append("Request timeout must be positive")

        if errors:
            raise ValidationError(
                f"Configuration validation failed: {'; '.join(errors)}",
                ErrorContext(
                    error_code="CONFIG_VALIDATION_FAILED",
                    suggestions=["Check configuration values", "Use valid ranges and types"]
                )
            )


# =============================================================================
# Configuration Builders
# =============================================================================

class ConfigBuilder:
    """Builder pattern for creating configurations."""

    def __init__(self):
        self._config_data = {}

    def with_model(self, name: str, provider: str, **kwargs) -> 'ConfigBuilder':
        """Configure model settings."""
        self._config_data["model"] = {
            "name": name,
            "provider": provider,
            **kwargs
        }
        return self

    def with_edit_format(self, format: Union[str, EditFormat]) -> 'ConfigBuilder':
        """Configure edit format."""
        if "edit" not in self._config_data:
            self._config_data["edit"] = {}

        if isinstance(format, EditFormat):
            format = format.value

        self._config_data["edit"]["format"] = format
        return self

    def with_security(self, **kwargs) -> 'ConfigBuilder':
        """Configure security settings."""
        if "security" not in self._config_data:
            self._config_data["security"] = {}

        self._config_data["security"].update(kwargs)
        return self

    def with_ui(self, **kwargs) -> 'ConfigBuilder':
        """Configure UI settings."""
        if "ui" not in self._config_data:
            self._config_data["ui"] = {}

        self._config_data["ui"].update(kwargs)
        return self

    def with_performance(self, **kwargs) -> 'ConfigBuilder':
        """Configure performance settings."""
        if "performance" not in self._config_data:
            self._config_data["performance"] = {}

        self._config_data["performance"].update(kwargs)
        return self

    def build(self, manager: ConfigManager) -> AiderConfig:
        """Build the configuration."""
        return manager._dict_to_config(self._config_data)


# =============================================================================
# Configuration Validation
# =============================================================================

class ConfigValidator:
    """Validates configuration values and provides suggestions."""

    @staticmethod
    def validate_model_config(model: ModelConfig) -> List[str]:
        """Validate model configuration."""
        issues = []

        if not model.name:
            issues.append("Model name is required")

        if model.api_key_env_var and not os.getenv(model.api_key_env_var):
            issues.append(f"API key environment variable {model.api_key_env_var} is not set")

        if model.max_tokens and model.max_tokens <= 0:
            issues.append("Max tokens must be positive")

        if not (0 <= model.temperature <= 2):
            issues.append("Temperature must be between 0 and 2")

        return issues

    @staticmethod
    def validate_security_config(security: SecurityConfig) -> List[str]:
        """Validate security configuration."""
        issues = []

        if security.max_file_size_mb <= 0:
            issues.append("Max file size must be positive")

        if not security.allowed_file_extensions:
            issues.append("At least one file extension must be allowed")

        return issues

    @staticmethod
    def get_optimization_suggestions(config: AiderConfig) -> List[str]:
        """Get optimization suggestions for configuration."""
        suggestions = []

        # Model optimization
        if "gpt-3.5" in config.model.name and config.edit.format == EditFormat.PATCH:
            suggestions.append("Consider using 'diff' format with GPT-3.5 for better results")

        # Performance optimization
        if config.performance.cache_prompts and config.performance.cache_size_mb < 50:
            suggestions.append("Consider increasing cache size for better performance")

        # Security optimization
        if config.security.allow_file_deletion and not config.ui.confirm_edits:
            suggestions.append("Enable edit confirmation when allowing file deletion")

        return suggestions


# =============================================================================
# Global Configuration Access
# =============================================================================

# Global configuration manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Get the global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def get_current_config() -> Optional[AiderConfig]:
    """Get the currently active configuration."""
    return get_config_manager().get_current_config()


def load_config(profile_name: str = "default", **overrides) -> AiderConfig:
    """Load configuration with the specified profile and overrides."""
    return get_config_manager().load_config(profile_name, overrides)


def create_config_for_model(
    model_name: str,
    edit_format: Optional[str] = None,
    **kwargs
) -> AiderConfig:
    """
    Create optimized configuration for a specific model.

    Args:
        model_name: Name of the AI model
        edit_format: Optional edit format override
        **kwargs: Additional configuration overrides

    Returns:
        Optimized configuration
    """
    # Determine provider from model name
    provider = ModelProvider.OPENAI  # default
    if "claude" in model_name.lower():
        provider = ModelProvider.ANTHROPIC
    elif "gemini" in model_name.lower() or "palm" in model_name.lower():
        provider = ModelProvider.GOOGLE
    elif "command" in model_name.lower():
        provider = ModelProvider.COHERE

    # Select optimal edit format if not specified
    if not edit_format:
        if "gpt-4" in model_name.lower():
            edit_format = "udiff"
        elif "claude" in model_name.lower():
            edit_format = "diff-fenced"
        else:
            edit_format = "diff"

    # Build configuration
    builder = ConfigBuilder()
    config = (builder
              .with_model(model_name, provider.value)
              .with_edit_format(edit_format)
              .build(get_config_manager()))

    # Apply additional overrides
    if kwargs:
        config = get_config_manager()._apply_dict_overrides(config, kwargs)

    return config


# =============================================================================
# Configuration Templates and Presets
# =============================================================================

def get_config_template_for_use_case(use_case: str) -> Dict[str, Any]:
    """
    Get configuration template optimized for specific use cases.

    Args:
        use_case: The use case (development, production, learning, etc.)

    Returns:
        Configuration template dictionary
    """
    templates = {
        "development": {
            "edit": {
                "validate_before_apply": True,
                "backup_before_edit": True,
                "dry_run_mode": False
            },
            "ui": {
                "verbose": True,
                "confirm_edits": True,
                "show_diffs": True,
                "show_tokens": True
            },
            "security": {
                "allow_file_deletion": False,
                "max_file_size_mb": 5
            },
            "performance": {
                "cache_prompts": True,
                "enable_streaming": True
            }
        },
        "production": {
            "edit": {
                "validate_before_apply": False,
                "backup_before_edit": False,
                "auto_commits": True
            },
            "ui": {
                "verbose": False,
                "confirm_edits": False,
                "show_tokens": False
            },
            "performance": {
                "cache_prompts": True,
                "enable_streaming": True,
                "max_concurrent_requests": 5
            }
        },
        "learning": {
            "edit": {
                "validate_before_apply": True,
                "backup_before_edit": True,
                "dry_run_mode": True
            },
            "ui": {
                "verbose": True,
                "confirm_edits": True,
                "show_diffs": True,
                "show_tokens": True,
                "show_costs": True
            },
            "security": {
                "allow_file_deletion": False,
                "allow_file_creation": True,
                "max_file_size_mb": 2
            }
        },
        "experimental": {
            "edit": {
                "format": "patch",
                "validate_before_apply": True
            },
            "ui": {
                "verbose": True,
                "confirm_edits": True
            },
            "performance": {
                "cache_prompts": False,  # Disable to see fresh responses
                "enable_streaming": True
            }
        }
    }

    return templates.get(use_case, templates["development"])


def create_minimal_config(model_name: str) -> AiderConfig:
    """
    Create a minimal configuration with just the essentials.

    Args:
        model_name: Name of the AI model to use

    Returns:
        Minimal configuration
    """
    # Determine provider
    provider = ModelProvider.OPENAI
    if "claude" in model_name.lower():
        provider = ModelProvider.ANTHROPIC
    elif "gemini" in model_name.lower():
        provider = ModelProvider.GOOGLE

    return AiderConfig(
        model=ModelConfig(
            name=model_name,
            provider=provider
        )
    )


# =============================================================================
# Configuration Migration and Compatibility
# =============================================================================

class ConfigMigrator:
    """Handles migration of old configuration formats."""

    @staticmethod
    def migrate_legacy_config(legacy_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Migrate legacy configuration format to current format.

        Args:
            legacy_data: Legacy configuration data

        Returns:
            Migrated configuration data
        """
        migrated = {}

        # Map legacy keys to new structure
        legacy_mappings = {
            "model_name": ("model", "name"),
            "edit_format": ("edit", "format"),
            "auto_commit": ("edit", "auto_commits"),
            "verbose": ("ui", "verbose"),
            "confirm_changes": ("ui", "confirm_edits"),
            "max_tokens": ("model", "max_tokens"),
            "temperature": ("model", "temperature")
        }

        for legacy_key, (section, new_key) in legacy_mappings.items():
            if legacy_key in legacy_data:
                if section not in migrated:
                    migrated[section] = {}
                migrated[section][new_key] = legacy_data[legacy_key]

        # Handle any unmapped keys in custom_settings
        custom_settings = {}
        for key, value in legacy_data.items():
            if key not in legacy_mappings:
                custom_settings[key] = value

        if custom_settings:
            migrated["custom_settings"] = custom_settings

        return migrated

    @staticmethod
    def get_migration_report(legacy_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a report of what would be migrated.

        Args:
            legacy_data: Legacy configuration data

        Returns:
            Migration report
        """
        migrated = ConfigMigrator.migrate_legacy_config(legacy_data)

        return {
            "legacy_keys_count": len(legacy_data),
            "migrated_keys_count": sum(len(section) for section in migrated.values() if isinstance(section, dict)),
            "unmapped_keys": list(migrated.get("custom_settings", {}).keys()),
            "migration_preview": migrated
        }


# =============================================================================
# Environment-Specific Helpers
# =============================================================================

def get_development_config(model_name: str = "gpt-4") -> AiderConfig:
    """Get configuration optimized for development work."""
    return ConfigBuilder() \
        .with_model(model_name, "openai") \
        .with_edit_format("diff-fenced") \
        .with_ui(verbose=True, confirm_edits=True, show_diffs=True) \
        .with_security(allow_file_deletion=False, backup_before_edit=True) \
        .build(get_config_manager())


def get_production_config(model_name: str = "gpt-4") -> AiderConfig:
    """Get configuration optimized for production use."""
    return ConfigBuilder() \
        .with_model(model_name, "openai") \
        .with_edit_format("udiff") \
        .with_ui(verbose=False, confirm_edits=False) \
        .with_performance(cache_prompts=True, enable_streaming=True) \
        .build(get_config_manager())


def get_safe_config(model_name: str = "gpt-4") -> AiderConfig:
    """Get configuration with maximum safety features."""
    return ConfigBuilder() \
        .with_model(model_name, "openai") \
        .with_edit_format("diff") \
        .with_ui(verbose=True, confirm_edits=True, show_diffs=True) \
        .with_security(
            allow_file_deletion=False,
            allow_file_creation=True,
            max_file_size_mb=5,
            backup_before_edit=True
        ) \
        .build(get_config_manager())


# =============================================================================
# Configuration Export and Import
# =============================================================================

def export_config_to_file(config: AiderConfig, file_path: Union[str, Path], format: str = "yaml"):
    """
    Export configuration to a file.

    Args:
        config: Configuration to export
        file_path: Path to save the configuration
        format: File format ("yaml" or "json")
    """
    config_dict = get_config_manager()._config_to_dict(config)

    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if format.lower() == "yaml":
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    elif format.lower() == "json":
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'yaml' or 'json'")


def import_config_from_file(file_path: Union[str, Path]) -> AiderConfig:
    """
    Import configuration from a file.

    Args:
        file_path: Path to the configuration file

    Returns:
        Loaded configuration

    Raises:
        ConfigurationError: If file cannot be loaded
    """
    path = Path(file_path)

    if not path.exists():
        raise ConfigurationError(
            f"Configuration file not found: {file_path}",
            ErrorContext(error_code="CONFIG_FILE_NOT_FOUND")
        )

    try:
        if path.suffix.lower() in [".yaml", ".yml"]:
            with open(path, 'r') as f:
                config_dict = yaml.safe_load(f)
        elif path.suffix.lower() == ".json":
            with open(path, 'r') as f:
                config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        return get_config_manager()._dict_to_config(config_dict)

    except Exception as e:
        raise ConfigurationError(
            f"Failed to load configuration from {file_path}",
            ErrorContext(
                error_code="CONFIG_IMPORT_FAILED",
                suggestions=[
                    "Check file format and syntax",
                    "Verify file permissions",
                    "Ensure file contains valid configuration data"
                ]
            )
        ) from e


# =============================================================================
# Configuration Utilities
# =============================================================================

def validate_model_availability(model_name: str, api_key: Optional[str] = None) -> bool:
    """
    Validate that a model is available and accessible.

    Args:
        model_name: Name of the model to validate
        api_key: Optional API key to use for validation

    Returns:
        True if model is available
    """
    # This would typically make an API call to validate the model
    # For now, return True as a placeholder
    return True


def get_recommended_config_for_project(project_path: Path) -> AiderConfig:
    """
    Analyze a project and recommend optimal configuration.

    Args:
        project_path: Path to the project to analyze

    Returns:
        Recommended configuration
    """
    # Analyze project characteristics
    total_files = len(list(project_path.rglob("*.*")))
    python_files = len(list(project_path.rglob("*.py")))
    js_files = len(list(project_path.rglob("*.js"))) + len(list(project_path.rglob("*.ts")))

    # Determine project type
    is_python_project = python_files > js_files and python_files > 5
    is_js_project = js_files > python_files and js_files > 5
    is_large_project = total_files > 1000

    # Start with base configuration
    config = get_development_config()

    # Optimize based on project characteristics
    if is_large_project:
        config.edit.format = EditFormat.UDIFF
        config.performance.cache_prompts = True
        config.security.max_file_size_mb = 20

    if is_python_project:
        config.edit.format = EditFormat.DIFF_FENCED
        config.security.allowed_file_extensions.extend([".pyi", ".pyx"])

    if is_js_project:
        config.edit.format = EditFormat.DIFF
        config.security.allowed_file_extensions.extend([".jsx", ".tsx", ".vue"])

    return config


def print_config_summary(config: AiderConfig):
    """Print a human-readable summary of the configuration."""
    print("🔧 Aider Configuration Summary")
    print("=" * 40)
    print(f"Profile: {config.profile_name}")
    print(f"Model: {config.model.name} ({config.model.provider.value})")
    print(f"Edit Format: {config.edit.format.value}")
    print(f"Auto Commits: {config.edit.auto_commits}")
    print(f"Validation: {config.edit.validate_before_apply}")
    print(f"Verbose Mode: {config.ui.verbose}")
    print(f"Confirm Edits: {config.ui.confirm_edits}")
    print(f"Cache Prompts: {config.performance.cache_prompts}")
    print(f"Max File Size: {config.security.max_file_size_mb}MB")
    print("=" * 40)


# =============================================================================
# Configuration Context Manager
# =============================================================================

class ConfigContext:
    """Context manager for temporary configuration changes."""

    def __init__(self, **overrides):
        """
        Initialize context with configuration overrides.

        Args:
            **overrides: Configuration values to temporarily override
        """
        self.overrides = overrides
        self.original_config = None

    def __enter__(self):
        """Enter the context and apply overrides."""
        self.original_config = get_current_config()
        if self.original_config:
            # Apply overrides temporarily
            manager = get_config_manager()
            temp_config = manager._apply_dict_overrides(self.original_config, self.overrides)
            manager._current_config = temp_config
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context and restore original configuration."""
        if self.original_config:
            get_config_manager()._current_config = self.original_config


# Example usage:
# with ConfigContext(ui={"verbose": True}, edit={"dry_run_mode": True}):
#     # Code here runs with temporary config overrides
#     pass
