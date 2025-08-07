"""
Enhanced base coder with improved architecture, error handling, and extensibility.

This module provides an improved base class for all Aider coders that incorporates
modern design patterns, comprehensive error handling, and better configuration management.
"""

import time
import traceback
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple, Callable, TypeAlias, override
from dataclasses import dataclass, field
from pathlib import Path
import logging
import asyncio
from contextlib import contextmanager, asynccontextmanager
from types import TracebackType
from typing import AsyncGenerator, Self

# Type aliases for better readability
FilePath: TypeAlias = str | Path
EditResults: TypeAlias = List[Any]
ConfigDict: TypeAlias = Dict[str, Any]
OperationResult: TypeAlias = bool
ResourceManager: TypeAlias = Any

from .exceptions import (
    AiderCoderError,
    ConfigurationError,
    ValidationError,
    EditOperationError,
    FileNotFoundError,
    FileNotEditableError,
    TokenLimitExceededError,
    ModelResponseError,
    PartialEditError,
    handle_edit_errors,
    format_error_for_user,
    log_error_for_debugging
)
from .edit_strategies import (
    EditStrategy,
    EditStrategyFactory,
    EditStrategyCoordinator,
    EditResult,
    EditInstruction,
    EditMetrics,
    PerformanceTracker
)
from .config import (
    AiderConfig,
    ConfigManager,
    get_current_config,
    ModelConfig,
    EditConfig,
    SecurityConfig
)
from .base_coder import Coder


# =============================================================================
# Enhanced Base Coder
# =============================================================================

class EnhancedCoder(ABC):
    """
    Enhanced base class for all Aider coders.

    Provides improved architecture with better error handling, configuration
    management, validation, and extensibility using modern design patterns.
    """

    def __init__(
        self,
        io,
        config: AiderConfig | None = None,
        main_model=None,
        edit_format: str | None = None,
        **kwargs
    ):
        """
        Initialize the enhanced coder.

        Args:
            io: Input/output handler
            config: Optional configuration (uses global if not provided)
            main_model: The AI model to use
            edit_format: Optional edit format override
            **kwargs: Additional legacy parameters for compatibility
        """
        # Core components
        self.io = io
        self.config = config or get_current_config()
        self.main_model = main_model

        # Validate configuration
        if not self.config:
            raise ConfigurationError("No configuration provided or available")

        # Initialize strategy system
        edit_format = edit_format or self.config.edit.format.value
        self.edit_strategy = EditStrategyFactory.create_strategy(
            edit_format,
            config=self._get_strategy_config(),
            model_name=self.main_model.name if self.main_model else None
        )

        self.strategy_coordinator = EditStrategyCoordinator(self.edit_strategy)

        # Performance and metrics
        self.performance_tracker = PerformanceTracker()
        self.session_start_time = time.time()

        # State management
        self.abs_fnames: set = set()
        self.abs_read_only_fnames: set = set()
        self.cur_messages: List[Dict] = []
        self.done_messages: List[Dict] = []
        self.partial_response_content: str = ""

        # Statistics
        self.total_cost: float = 0.0
        self.num_exhausted_context_windows: int = 0
        self.num_malformed_responses: int = 0
        self.aider_commit_hashes: List[str] = []

        # Setup logging
        self.logger = logging.getLogger(f"{self.__class__.__name__}")

        # Legacy compatibility - initialize from kwargs
        self._init_legacy_attributes(kwargs)

        # Initialize subclass-specific components
        self._init_subclass_components()

    def _init_legacy_attributes(self, kwargs: Dict[str, Any]):
        """Initialize legacy attributes for backwards compatibility."""
        # Extract common legacy parameters
        self.root = kwargs.get('root', '.')
        self.fnames = kwargs.get('fnames', set())
        self.read_only_fnames = kwargs.get('read_only_fnames', set())

        # Safe configuration access with defaults
        ui_config = self.config.ui if self.config and hasattr(self.config, 'ui') else None
        edit_config = self.config.edit if self.config and hasattr(self.config, 'edit') else None

        self.show_diffs = kwargs.get('show_diffs', getattr(ui_config, 'show_diffs', True) if ui_config else True)
        self.auto_commits = kwargs.get('auto_commits', getattr(edit_config, 'auto_commits', True) if edit_config else True)
        self.dirty_commits = kwargs.get('dirty_commits', True)
        self.pretty = kwargs.get('pretty', True)
        self.verbose = kwargs.get('verbose', getattr(ui_config, 'verbose', False) if ui_config else False)

        # Add repo attribute for compatibility
        self.repo = kwargs.get('repo', None)

        # Convert relative paths to absolute
        self._update_file_sets()

    def _update_file_sets(self):
        """Update absolute file name sets from relative file names."""
        for fname in self.fnames:
            abs_path = self.abs_root_path(fname)
            if abs_path:
                self.abs_fnames.add(abs_path)

        for fname in self.read_only_fnames:
            abs_path = self.abs_root_path(fname)
            if abs_path:
                self.abs_read_only_fnames.add(abs_path)

    def _get_strategy_config(self) -> Dict[str, Any]:
        """Get configuration for edit strategy."""
        # Safe configuration access with defaults
        edit_config = self.config.edit if self.config and self.config.edit else None

        return {
            "validate_before_apply": getattr(edit_config, 'validate_before_apply', True) if edit_config else True,
            "max_file_size_kb": getattr(edit_config, 'max_file_size_kb', 1024) if edit_config else 1024,
            "max_lines_per_edit": getattr(edit_config, 'max_lines_per_edit', 100) if edit_config else 100,
            "backup_before_edit": getattr(edit_config, 'backup_before_edit', False) if edit_config else False
        }

    @abstractmethod
    def _init_subclass_components(self):
        """Initialize subclass-specific components."""
        pass

    # =============================================================================
    # Public Interface Methods
    # =============================================================================

    def run(self, message: Optional[str] = None, preproc: bool = True) -> None:
        """
        Main entry point for running the coder.

        Args:
            message: Optional initial message
            preproc: Whether to preprocess the message
        """
        try:
            self._validate_preconditions()

            if message:
                self.run_one(message, preproc)
            else:
                self._run_interactive_loop()

        except KeyboardInterrupt:
            self._handle_keyboard_interrupt()
        except Exception as e:
            self._handle_unexpected_error(e)

    def run_one(self, user_message: str, preproc: bool = True) -> bool:
        """
        Process a single user message.

        Args:
            user_message: The user's input message
            preproc: Whether to preprocess the message

        Returns:
            True if processing was successful
        """
        start_time = time.time()

        try:
            # Preprocess message if requested
            if preproc:
                user_message = self.preprocess_user_input(user_message)

            # Initialize message processing
            self._init_before_message()

            # Send message and get response
            self._send_message_with_retry(user_message)

            # Process the response
            success = self._process_response()

            # Record metrics
            self._record_session_metrics(start_time, success)

            return success

        except Exception as e:
            self._handle_processing_error(e, user_message)
            return False

    @handle_edit_errors
    def apply_edits(self, edits: Optional[List[EditInstruction]] = None) -> List[EditResult]:
        """
        Apply edits using the configured strategy.

        Args:
            edits: Optional list of edit instructions. If None, parses from response.

        Returns:
            List of edit results
        """
        if edits is None:
            edits = self.get_edits()

        if not edits:
            return []

        # Validate edits before applying
        if self.config and self.config.edit and getattr(self.config.edit, 'validate_before_apply', True):
            self._validate_edits(edits)

        # Get file contents
        file_contents = self._get_file_contents_for_edits(edits)

        # Apply edits using strategy coordinator
        results = self.strategy_coordinator.process_edits(
            self.partial_response_content,
            strategy=self.edit_strategy,
            file_contents=file_contents
        )

        # Handle results
        successful, failed = self.strategy_coordinator.validate_results(results)

        if failed:
            self._handle_failed_edits(failed)

        if successful:
            self._handle_successful_edits(successful)

        return results

    def get_edits(self) -> List[EditInstruction]:
        """
        Extract edit instructions from AI response.

        Returns:
            List of edit instructions
        """
        if not self.partial_response_content:
            return []

        return self.edit_strategy.parse_edits(self.partial_response_content)

    def validate_file_access(self, file_path: str, operation: str = "read") -> bool:
        """
        Validate file access permissions and security constraints.

        Args:
            file_path: Path to the file
            operation: Type of operation (read, write, delete)

        Returns:
            True if access is allowed

        Raises:
            FileNotEditableError: If access is not allowed
        """
        try:
            path = Path(file_path)

            # Security checks
            if not self._is_file_allowed(path):
                raise FileNotEditableError(
                    file_path,
                    "File type or location not allowed by security policy"
                )

            # Get security config safely
            security_config = self.config.security if self.config and self.config.security else None

            # Operation-specific checks
            if operation == "write":
                allow_creation = getattr(security_config, 'allow_file_creation', True) if security_config else True
                if not allow_creation and not path.exists():
                    raise FileNotEditableError(file_path, "File creation not allowed")

            elif operation == "delete":
                allow_deletion = getattr(security_config, 'allow_file_deletion', False) if security_config else False
                if not allow_deletion:
                    raise FileNotEditableError(file_path, "File deletion not allowed")

            # Size checks
            if path.exists():
                size_mb = path.stat().st_size / (1024 * 1024)
                max_size = getattr(security_config, 'max_file_size_mb', 50) if security_config else 50
                if size_mb > max_size:
                    raise FileNotEditableError(
                        file_path,
                        f"File size ({size_mb:.1f}MB) exceeds limit ({max_size}MB)"
                    )

            return True

        except Exception as e:
            if isinstance(e, FileNotEditableError):
                raise
            raise FileNotEditableError(file_path, f"Access validation failed: {e}")

    # =============================================================================
    # File Management Methods
    # =============================================================================

    def abs_root_path(self, fname: str) -> Optional[Path]:
        """
        Convert relative filename to absolute path within project root.

        Args:
            fname: Relative filename

        Returns:
            Absolute Path object or None if invalid
        """
        try:
            if not fname:
                return None

            path = Path(self.root) / fname
            return path.resolve()

        except Exception as e:
            self.logger.warning(f"Failed to resolve path for {fname}: {e}")
            return None

    def add_file(self, fname: str, read_only: bool = False) -> bool:
        """
        Add a file to the coder's context.

        Args:
            fname: Filename to add
            read_only: Whether the file should be read-only

        Returns:
            True if file was added successfully
        """
        try:
            abs_path = self.abs_root_path(fname)
            if not abs_path:
                return False

            # Validate file access
            self.validate_file_access(str(abs_path), "read")

            # Add to appropriate set
            if read_only:
                self.abs_read_only_fnames.add(abs_path)
            else:
                self.abs_fnames.add(abs_path)
                # Validate write access for editable files
                self.validate_file_access(str(abs_path), "write")

            return True

        except Exception as e:
            self.logger.error(f"Failed to add file {fname}: {e}")
            return False

    def remove_file(self, fname: str) -> bool:
        """
        Remove a file from the coder's context.

        Args:
            fname: Filename to remove

        Returns:
            True if file was removed successfully
        """
        try:
            abs_path = self.abs_root_path(fname)
            if not abs_path:
                return False

            # Remove from both sets
            self.abs_fnames.discard(abs_path)
            self.abs_read_only_fnames.discard(abs_path)

            return True

        except Exception as e:
            self.logger.error(f"Failed to remove file {fname}: {e}")
            return False

    def get_file_content(self, fname: str) -> Optional[str]:
        """
        Get content of a file with proper error handling.

        Args:
            fname: Filename to read

        Returns:
            File content or None if read failed
        """
        try:
            abs_path = self.abs_root_path(fname)
            if not abs_path or not abs_path.exists():
                return None

            self.validate_file_access(str(abs_path), "read")
            return self.io.read_text(abs_path)

        except Exception as e:
            self.logger.error(f"Failed to read file {fname}: {e}")
            return None

    # =============================================================================
    # Enhanced Message Processing
    # =============================================================================

    def preprocess_user_input(self, message: str) -> str:
        """
        Preprocess user input with validation and enhancement.

        Args:
            message: Raw user input

        Returns:
            Processed message
        """
        if not message or not message.strip():
            raise ValidationError("Empty user message")

        # Basic sanitization
        message = message.strip()

        # Check for potential file mentions
        self._check_for_file_mentions(message)

        # Check for URLs
        self._check_and_open_urls(message)

        return message

    def _check_for_file_mentions(self, message: str):
        """Check for file mentions in the message and suggest adding them."""
        # This would implement file mention detection logic
        pass

    def _check_and_open_urls(self, message: str):
        """Check for URLs in the message and potentially fetch content."""
        # This would implement URL detection and fetching logic
        pass

    def _send_message_with_retry(self, message: str, max_retries: int = 3):
        """
        Send message with retry logic and error handling.

        Args:
            message: Message to send
            max_retries: Maximum number of retry attempts
        """
        last_error = None

        for attempt in range(max_retries):
            try:
                self._send_message(message)
                return

            except TokenLimitExceededError as e:
                # Try to reduce context
                if self._try_reduce_context():
                    continue
                else:
                    raise e

            except ModelResponseError as e:
                last_error = e
                if attempt < max_retries - 1:
                    self.logger.warning(f"Model response error (attempt {attempt + 1}): {e}")
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue

            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    self.logger.warning(f"Send error (attempt {attempt + 1}): {e}")
                    time.sleep(2 ** attempt)
                    continue

        if last_error:
            raise last_error

    @abstractmethod
    def _send_message(self, message: str):
        """Send message to AI model. Implemented by subclasses."""
        pass

    def _process_response(self) -> bool:
        """
        Process AI response and apply edits.

        Returns:
            True if processing was successful
        """
        try:
            if not self.partial_response_content:
                self.logger.warning("Empty response from AI model")
                return False

            # Let subclass handle response completion
            if hasattr(self, 'reply_completed'):
                self.reply_completed()

            # Apply edits if this is an editing coder
            if self.edit_strategy.supported_operations:
                results = self.apply_edits()
                return any(result.success for result in results)

            return True

        except Exception as e:
            self._handle_processing_error(e, "response processing")
            return False

    # =============================================================================
    # Validation Methods
    # =============================================================================

    def _validate_preconditions(self):
        """Validate that all preconditions for running are met."""
        if not self.io:
            raise ConfigurationError("IO handler not configured")

        if not self.main_model and self.edit_strategy.supported_operations:
            raise ConfigurationError("Main model required for editing operations")

        # Validate workspace
        if self.config and hasattr(self.config, 'workspace_path') and self.config.workspace_path:
            workspace = Path(self.config.workspace_path)
            if not workspace.exists():
                raise FileNotFoundError(str(workspace))

    def _validate_edits(self, edits: List[EditInstruction]):
        """
        Validate edit instructions before applying.

        Args:
            edits: List of edit instructions to validate
        """
        for edit in edits:
            # Validate file path
            if not edit.file_path:
                raise ValidationError("Edit instruction missing file path")

            # Validate content
            if not edit.content and edit.edit_type != "delete":
                raise ValidationError(f"Edit instruction for {edit.file_path} missing content")

            # Validate edit type
            if edit.edit_type not in self.edit_strategy.supported_operations:
                raise ValidationError(
                    f"Edit type '{edit.edit_type}' not supported by {self.edit_strategy.edit_format} strategy"
                )

            # Security validation
            self.validate_file_access(edit.file_path, "write")

    def _is_file_allowed(self, path: Path) -> bool:
        """
        Check if file is allowed by security policy.

        Args:
            path: Path to check

        Returns:
            True if file is allowed
        """
        # Get security config safely
        security_config = self.config.security if self.config and hasattr(self.config, 'security') else None

        # Check file extension
        allowed_extensions = getattr(security_config, 'allowed_file_extensions', set()) if security_config else set()
        if path.suffix and allowed_extensions and path.suffix not in allowed_extensions:
            return False

        # Check blocked patterns
        blocked_patterns = getattr(security_config, 'blocked_file_patterns', []) if security_config else []
        for pattern in blocked_patterns:
            if path.match(pattern):
                return False

        # Check directories
        blocked_dirs = getattr(security_config, 'blocked_directories', set()) if security_config else set()
        for part in path.parts:
            if part in blocked_dirs:
                return False

        return True

    # =============================================================================
    # Modern Context Management and Async Support
    # =============================================================================

    @contextmanager
    def editing_session(self, backup: bool = True):
        """Context manager for safe editing operations."""
        session_id = f"session_{int(time.time())}"
        backup_paths = []

        try:
            self.logger.info(f"Starting editing session {session_id}")

            # Create backups if requested
            if backup and self.abs_fnames:
                backup_paths = self._create_file_backups()

            yield session_id

        except Exception as e:
            self.logger.error(f"Error in editing session {session_id}: {e}")

            # Restore backups on error
            if backup_paths:
                self._restore_file_backups(backup_paths)

            raise
        finally:
            # Cleanup temporary resources
            self._cleanup_session_resources(session_id)
            self.logger.info(f"Completed editing session {session_id}")

    @asynccontextmanager
    async def async_editing_session(self, backup: bool = True) -> AsyncGenerator[str, None]:
        """Async context manager for concurrent editing operations."""
        session_id = f"async_session_{int(time.time())}"
        backup_paths = []

        try:
            self.logger.info(f"Starting async editing session {session_id}")

            if backup and self.abs_fnames:
                backup_paths = await self._create_file_backups_async()

            yield session_id

        except Exception as e:
            self.logger.error(f"Error in async editing session {session_id}: {e}")

            if backup_paths:
                await self._restore_file_backups_async(backup_paths)

            raise
        finally:
            await self._cleanup_session_resources_async(session_id)
            self.logger.info(f"Completed async editing session {session_id}")

    def _create_file_backups(self) -> List[Tuple[str, str]]:
        """Create backup copies of files before editing."""
        backup_paths = []
        for file_path in self.abs_fnames:
            if Path(file_path).exists():
                backup_path = f"{file_path}.bak.{int(time.time())}"
                try:
                    import shutil
                    shutil.copy2(file_path, backup_path)
                    backup_paths.append((file_path, backup_path))
                except OSError as e:
                    self.logger.warning(f"Failed to backup {file_path}: {e}")
        return backup_paths

    async def _create_file_backups_async(self) -> List[Tuple[str, str]]:
        """Async version of file backup creation."""
        backup_paths = []
        for file_path in self.abs_fnames:
            if Path(file_path).exists():
                backup_path = f"{file_path}.bak.{int(time.time())}"
                try:
                    # Use asyncio for file operations
                    await asyncio.to_thread(self._backup_single_file, file_path, backup_path)
                    backup_paths.append((file_path, backup_path))
                except OSError as e:
                    self.logger.warning(f"Failed to backup {file_path}: {e}")
        return backup_paths

    def _backup_single_file(self, source: str, destination: str) -> None:
        """Backup a single file synchronously."""
        import shutil
        shutil.copy2(source, destination)

    def _restore_file_backups(self, backup_paths: List[Tuple[str, str]]) -> None:
        """Restore files from backup copies."""
        for original, backup in backup_paths:
            try:
                import shutil
                shutil.move(backup, original)
                self.logger.info(f"Restored {original} from backup")
            except OSError as e:
                self.logger.error(f"Failed to restore {original}: {e}")

    async def _restore_file_backups_async(self, backup_paths: List[Tuple[str, str]]) -> None:
        """Async version of file restoration."""
        for original, backup in backup_paths:
            try:
                await asyncio.to_thread(self._restore_single_file, backup, original)
                self.logger.info(f"Restored {original} from backup")
            except OSError as e:
                self.logger.error(f"Failed to restore {original}: {e}")

    def _restore_single_file(self, backup: str, original: str) -> None:
        """Restore a single file synchronously."""
        import shutil
        shutil.move(backup, original)

    def _cleanup_session_resources(self, session_id: str) -> None:
        """Clean up resources from an editing session."""
        # Clean up temporary files, clear caches, etc.
        pass

    async def _cleanup_session_resources_async(self, session_id: str) -> None:
        """Async cleanup of session resources."""
        await asyncio.to_thread(self._cleanup_session_resources, session_id)

    # =============================================================================
    # Error Handling
    # =============================================================================

    def _handle_processing_error(self, error: Exception, context: str):
        """
        Handle errors during processing with proper logging and user feedback.

        Args:
            error: The exception that occurred
            context: Context where the error occurred
        """
        # Log for debugging
        error_data = {
            "context": context,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc()
        }
        self.logger.error(f"Processing error in {context}", extra=error_data)

        # Provide user feedback
        if isinstance(error, AiderCoderError):
            user_message = format_error_for_user(error)
        else:
            user_message = f"❌ Error during {context}: {error}"

        self.io.tool_error(user_message)

        # Try recovery if possible
        self._attempt_error_recovery(error, context)

    def _handle_failed_edits(self, failed_results: List[EditResult]):
        """
        Handle failed edit operations.

        Args:
            failed_results: List of failed edit results
        """
        error_summary = []

        for result in failed_results:
            error_summary.append(f"  - {result.file_path}: {result.error_message}")

        failed_message = f"❌ Failed to apply {len(failed_results)} edits:\n" + '\n'.join(error_summary)
        self.io.tool_error(failed_message)

        # Create partial edit error
        successful_files = [r.file_path for r in self._get_successful_results()]
        failed_files = {r.file_path: r.error_message or "Unknown error" for r in failed_results}

        raise PartialEditError(successful_files, failed_files)

    def _handle_successful_edits(self, successful_results: List[EditResult]):
        """
        Handle successful edit operations.

        Args:
            successful_results: List of successful edit results
        """
        if not successful_results:
            return

        # Show success message
        files_modified = [result.file_path for result in successful_results]
        self.io.tool_output(f"✅ Successfully modified {len(files_modified)} files: {', '.join(files_modified)}")

        # Auto-commit if enabled
        edit_config = self.config.edit if self.config and hasattr(self.config, 'edit') else None
        if edit_config and getattr(edit_config, 'auto_commits', True):
            self._auto_commit(successful_results)

    def _handle_keyboard_interrupt(self):
        """Handle keyboard interrupt gracefully."""
        self.io.tool_output("\n⚠️  Operation interrupted by user")

        # Save current state if possible
        try:
            self._save_session_state()
        except Exception as e:
            self.logger.warning(f"Failed to save session state: {e}")

    def _handle_unexpected_error(self, error: Exception):
        """
        Handle unexpected errors with comprehensive logging and user guidance.

        Args:
            error: The unexpected exception
        """
        # Log full error details
        self.logger.error(f"Unexpected error in {self.__class__.__name__}", exc_info=True)

        # Create user-friendly error message
        error_msg = f"❌ Unexpected error: {error}\n"
        error_msg += "📋 This error has been logged for debugging.\n"
        error_msg += "💡 Please consider reporting this issue with the log details."

        self.io.tool_error(error_msg)

    def _attempt_error_recovery(self, error: Exception, context: str):
        """
        Attempt to recover from errors using registered recovery strategies.

        Args:
            error: The error to recover from
            context: Context where error occurred
        """
        for recovery_strategy in self.strategy_coordinator.recovery_strategies:
            if recovery_strategy.can_recover(error):
                try:
                    recovery_strategy.recover(error, {"context": context})
                    self.io.tool_output("🔄 Attempted automatic error recovery")
                    return
                except Exception as recovery_error:
                    self.logger.warning(f"Recovery attempt failed: {recovery_error}")

    # =============================================================================
    # Performance and Metrics
    # =============================================================================

    def _record_session_metrics(self, start_time: float, success: bool):
        """Record performance metrics for the session."""
        try:
            end_time = time.time()

            # Count edits
            results = getattr(self, '_last_edit_results', [])
            successful_edits = sum(1 for r in results if r.success)
            failed_edits = len(results) - successful_edits

            metrics = EditMetrics(
                strategy_used=self.edit_strategy.edit_format,
                files_processed=len(self.abs_fnames),
                successful_edits=successful_edits,
                failed_edits=failed_edits,
                total_time_ms=(end_time - start_time) * 1000,
                tokens_used=getattr(self, 'last_token_count', None)
            )

            self.performance_tracker.record_edit_session(metrics)

        except Exception as e:
            self.logger.warning(f"Failed to record metrics: {e}")

    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance report for this coder instance."""
        return self.performance_tracker.get_overall_performance()

    # =============================================================================
    # Helper Methods
    # =============================================================================

    def _get_successful_results(self) -> List[EditResult]:
        """Get list of successful edit results from last operation."""
        return getattr(self, '_last_successful_results', [])

    def _get_file_contents_for_edits(self, edits: List[EditInstruction]) -> Dict[str, str]:
        """
        Get file contents for all files mentioned in edits.

        Args:
            edits: List of edit instructions

        Returns:
            Dictionary mapping file paths to their contents
        """
        file_contents = {}

        for edit in edits:
            content = self.get_file_content(edit.file_path)
            if content is not None:
                file_contents[edit.file_path] = content
            else:
                # File doesn't exist - that's okay for create operations
                file_contents[edit.file_path] = ""

        return file_contents

    def _try_reduce_context(self) -> bool:
        """
        Try to reduce context to fit within token limits.

        Returns:
            True if context was reduced
        """
        # Remove oldest messages first
        if len(self.done_messages) > 5:
            self.done_messages = self.done_messages[-5:]
            return True

        # Remove read-only files if necessary
        if self.abs_read_only_fnames:
            removed_file = self.abs_read_only_fnames.pop()
            self.logger.info(f"Removed read-only file from context: {removed_file}")
            return True

        return False

    def _auto_commit(self, results: List[EditResult]):
        """
        Automatically commit changes if enabled.

        Args:
            results: List of successful edit results
        """
        try:
            if hasattr(self, 'repo') and self.repo:
                return

            # Create commit message
            files = [result.file_path for result in results]
            summary = f"Modified {len(files)} files"
            if len(files) <= 3:
                summary = f"Modified {', '.join(files)}"

            edit_config = self.config.edit if self.config and hasattr(self.config, 'edit') else None
            template = getattr(edit_config, 'commit_message_template', "Auto-commit: {summary}") if edit_config else "Auto-commit: {summary}"
            commit_msg = template.format(summary=summary)

            # Perform commit if repo is available
            if hasattr(self, 'repo') and self.repo and hasattr(self.repo, 'commit'):
                commit_hash = self.repo.commit(
                    message=commit_msg,
                    fnames=files
                )
            else:
                commit_hash = None

            if commit_hash:
                self.aider_commit_hashes.append(commit_hash)
                self.io.tool_output(f"🔀 Auto-committed changes: {commit_hash[:8]}")

        except Exception as e:
            self.logger.warning(f"Auto-commit failed: {e}")

    def _save_session_state(self):
        """Save current session state for recovery."""
        state = {
            "timestamp": time.time(),
            "config_profile": getattr(self.config, 'profile_name', 'default') if self.config else 'default',
            "files_in_chat": [str(f) for f in self.abs_fnames],
            "readonly_files": [str(f) for f in self.abs_read_only_fnames],
            "total_cost": self.total_cost,
            "commit_hashes": self.aider_commit_hashes
        }

        # Save to config directory
        config_manager = ConfigManager()
        session_file = config_manager.config_dir / "last_session.json"

        import json
        with open(session_file, 'w') as f:
            json.dump(state, f, indent=2)

    def _run_interactive_loop(self):
        """Run interactive loop for continuous user input."""
        try:
            while True:
                try:
                    user_input = self.io.get_input()
                    if not user_input:
                        continue

                    success = self.run_one(user_input, preproc=True)

                    if not success:
                        self.io.tool_output("⚠️  Command failed. Try again or type 'help' for assistance.")

                except KeyboardInterrupt:
                    raise  # Let the outer handler deal with it
                except EOFError:
                    break
                except Exception as e:
                    self._handle_processing_error(e, "interactive loop")

        except KeyboardInterrupt:
            raise

    def _init_before_message(self):
        """Initialize state before processing a message."""
        # Clear previous response content
        self.partial_response_content = ""

        # Prepare for new message
        # Subclasses can override this for specific initialization

    # =============================================================================
    # Compatibility Methods
    # =============================================================================

    @property
    def edit_format(self) -> str:
        """Get current edit format for backwards compatibility."""
        return self.edit_strategy.edit_format

    def get_inchat_relative_files(self) -> List[str]:
        """Get list of relative file paths for files in chat."""
        root_path = Path(self.root)
        relative_files = []

        for abs_path in self.abs_fnames:
            try:
                rel_path = abs_path.relative_to(root_path)
                relative_files.append(str(rel_path))
            except ValueError:
                # File is outside project root
                relative_files.append(str(abs_path))

        return relative_files

    def allowed_to_edit(self, fname: str) -> Optional[str]:
        """
        Check if file is allowed to be edited (legacy compatibility).

        Args:
            fname: Filename to check

        Returns:
            Absolute path if allowed, None otherwise
        """
        try:
            abs_path = self.abs_root_path(fname)
            if not abs_path:
                return None

            self.validate_file_access(str(abs_path), "write")
            return str(abs_path)

        except Exception:
            return None

    # =============================================================================
    # Abstract Methods for Subclasses
    # =============================================================================

    @abstractmethod
    def get_announcements(self) -> List[str]:
        """Get announcements to show to the user."""
        pass

    def reply_completed(self):
        """Handle completion of AI reply. Override in subclasses."""
        pass

    # =============================================================================
    # Factory Methods
    # =============================================================================

    @classmethod
    def create(
        cls,
        io,
        config: Optional[AiderConfig] = None,
        edit_format: Optional[str] = None,
        **kwargs
    ) -> 'EnhancedCoder':
        """
        Factory method for creating coder instances.

        Args:
            io: Input/output handler
            config: Optional configuration
            edit_format: Optional edit format
            **kwargs: Additional parameters

        Returns:
            Configured coder instance
        """
        # Use current config if none provided
        if not config:
            config = get_current_config()

        # Create instance
        instance = cls(io=io, config=config, edit_format=edit_format, **kwargs)

        # Show announcements
        if config and config.ui.verbose:
            announcements = instance.get_announcements()
            for announcement in announcements:
                instance.io.tool_output(f"📢 {announcement}")

        return instance

    @classmethod
    def create_with_strategy(
        cls,
        io,
        strategy: EditStrategy,
        config: Optional[AiderConfig] = None,
        **kwargs
    ) -> 'EnhancedCoder':
        """
        Create coder instance with specific strategy.

        Args:
            io: Input/output handler
            strategy: Edit strategy to use
            config: Optional configuration
            **kwargs: Additional parameters

        Returns:
            Configured coder instance with specified strategy
        """
        instance = cls(io=io, config=config, **kwargs)
        instance.edit_strategy = strategy
        instance.strategy_coordinator = EditStrategyCoordinator(strategy)
        return instance


# =============================================================================
# Enhanced Coder Implementations
# =============================================================================

class EnhancedEditBlockCoder(EnhancedCoder):
    """Enhanced coder for search/replace block edits."""

    def _init_subclass_components(self):
        """Initialize components specific to edit block coder."""
        # Ensure we have the right strategy
        if self.edit_strategy.edit_format not in ["search-replace", "diff", "diff-fenced"]:
            self.edit_strategy = EditStrategyFactory.create_strategy("diff")

    def get_announcements(self) -> List[str]:
        """Get announcements for edit block coder."""
        announcements = []

        ui_config = self.config.ui if self.config and hasattr(self.config, 'ui') else None
        if ui_config and getattr(ui_config, 'verbose', False):
            announcements.append(f"Using {self.edit_strategy.edit_format} edit format")
            announcements.append("Search/replace blocks will be used for edits")

        edit_config = self.config.edit if self.config and hasattr(self.config, 'edit') else None
        validate_enabled = getattr(edit_config, 'validate_before_apply', True) if edit_config else True
        if validate_enabled:
            announcements.append("Edit validation is enabled")

        return announcements

    def _send_message(self, message: str):
        """Send message to AI model for edit block processing."""
        # This would integrate with the existing message sending logic
        # For now, this is a placeholder that would call the actual LLM
        self.partial_response_content = f"Processed: {message}"


class EnhancedUnifiedDiffCoder(EnhancedCoder):
    """Enhanced coder for unified diff edits."""

    def _init_subclass_components(self):
        """Initialize components specific to unified diff coder."""
        # Ensure we have the right strategy
        if self.edit_strategy.edit_format not in ["unified-diff", "udiff", "udiff-simple"]:
            self.edit_strategy = EditStrategyFactory.create_strategy("udiff")

    def get_announcements(self) -> List[str]:
        """Get announcements for unified diff coder."""
        announcements = []

        ui_config = self.config.ui if self.config and hasattr(self.config, 'ui') else None
        if ui_config and getattr(ui_config, 'verbose', False):
            announcements.append(f"Using {self.edit_strategy.edit_format} edit format")
            announcements.append("Unified diff format will be used for precise edits")

        return announcements

    def _send_message(self, message: str):
        """Send message to AI model for unified diff processing."""
        # This would integrate with the existing message sending logic
        self.partial_response_content = f"Processed diff: {message}"


class EnhancedWholeFileCoder(EnhancedCoder):
    """Enhanced coder for whole file edits."""

    def _init_subclass_components(self):
        """Initialize components specific to whole file coder."""
        # Ensure we have the right strategy
        if self.edit_strategy.edit_format not in ["whole-file", "whole", "editor-whole"]:
            self.edit_strategy = EditStrategyFactory.create_strategy("whole")

    def get_announcements(self) -> List[str]:
        """Get announcements for whole file coder."""
        announcements = []

        ui_config = self.config.ui if self.config and hasattr(self.config, 'ui') else None
        if ui_config and getattr(ui_config, 'verbose', False):
            announcements.append(f"Using {self.edit_strategy.edit_format} edit format")
            announcements.append("Entire files will be rewritten for edits")

        edit_config = self.config.edit if self.config and hasattr(self.config, 'edit') else None
        max_file_size = getattr(edit_config, 'max_file_size_kb', 100) if edit_config else 100
        if max_file_size < 50:
            announcements.append("⚠️  Whole file format works best with smaller files")

        return announcements

    def _send_message(self, message: str):
        """Send message to AI model for whole file processing."""
        # This would integrate with the existing message sending logic
        self.partial_response_content = f"Processed whole file: {message}"


class EnhancedAskCoder(EnhancedCoder):
    """Enhanced coder for asking questions without edits."""

    def _init_subclass_components(self):
        """Initialize components specific to ask coder."""
        # Use no-op strategy for read-only operations
        self.edit_strategy = EditStrategyFactory.create_strategy("ask")

    def get_announcements(self) -> List[str]:
        """Get announcements for ask coder."""
        return [
            "Ask mode: I will answer questions without making edits",
            "Use specific coder modes for file modifications"
        ]

    def _send_message(self, message: str):
        """Send message to AI model for question answering."""
        # This would integrate with the existing message sending logic
        self.partial_response_content = f"Answer: {message}"


# =============================================================================
# Migration Utilities
# =============================================================================

def migrate_legacy_coder(legacy_coder, config: Optional[AiderConfig] = None) -> EnhancedCoder:
    """
    Migrate a legacy coder instance to the enhanced version.

    Args:
        legacy_coder: Legacy coder instance
        config: Optional new configuration

    Returns:
        Enhanced coder instance
    """
    # Determine appropriate enhanced class based on edit format
    format_mapping = {
        "diff": EnhancedEditBlockCoder,
        "diff-fenced": EnhancedEditBlockCoder,
        "editor-diff": EnhancedEditBlockCoder,
        "editor-diff-fenced": EnhancedEditBlockCoder,
        "udiff": EnhancedUnifiedDiffCoder,
        "udiff-simple": EnhancedUnifiedDiffCoder,
        "whole": EnhancedWholeFileCoder,
        "editor-whole": EnhancedWholeFileCoder,
        "ask": EnhancedAskCoder,
        "help": EnhancedAskCoder,
        "context": EnhancedAskCoder,
        "architect": EnhancedAskCoder,
    }

    legacy_format = getattr(legacy_coder, 'edit_format', 'diff')
    enhanced_class = format_mapping.get(legacy_format, EnhancedEditBlockCoder)

    # Create enhanced instance
    enhanced_coder = enhanced_class(
        io=legacy_coder.io,
        config=config,
        main_model=getattr(legacy_coder, 'main_model', None),
        edit_format=legacy_format,
        root=getattr(legacy_coder, 'root', '.'),
        fnames=getattr(legacy_coder, 'abs_fnames', set()),
        read_only_fnames=getattr(legacy_coder, 'abs_read_only_fnames', set())
    )

    # Transfer state
    enhanced_coder.cur_messages = getattr(legacy_coder, 'cur_messages', [])
    enhanced_coder.done_messages = getattr(legacy_coder, 'done_messages', [])
    enhanced_coder.total_cost = getattr(legacy_coder, 'total_cost', 0.0)
    enhanced_coder.aider_commit_hashes = getattr(legacy_coder, 'aider_commit_hashes', [])

    return enhanced_coder


# =============================================================================
# Utility Functions
# =============================================================================

def create_coder_for_task(
    task_type: str,
    io,
    model_name: str = "gpt-4",
    **kwargs
) -> EnhancedCoder:
    """
    Create an appropriately configured coder for a specific task.

    Args:
        task_type: Type of task (edit, ask, architect, help)
        io: Input/output handler
        model_name: AI model to use
        **kwargs: Additional configuration

    Returns:
        Configured coder instance
    """
    # Map task types to coder classes and formats
    task_mapping = {
        "edit": (EnhancedEditBlockCoder, "diff-fenced"),
        "precise-edit": (EnhancedUnifiedDiffCoder, "udiff"),
        "rewrite": (EnhancedWholeFileCoder, "whole"),
        "ask": (EnhancedAskCoder, "ask"),
        "architect": (EnhancedAskCoder, "architect"),
        "help": (EnhancedAskCoder, "help"),
    }

    if task_type not in task_mapping:
        raise ValueError(f"Unknown task type: {task_type}")

    coder_class, edit_format = task_mapping[task_type]

    # Create configuration for the task
    from .config import create_config_for_model
    config = create_config_for_model(model_name, edit_format, **kwargs)

    return coder_class.create(io, config=config, edit_format=edit_format)


def get_optimal_coder_for_context(
    io,
    file_count: int,
    avg_file_size_kb: float,
    model_name: str = "gpt-4",
    **kwargs
) -> EnhancedCoder:
    """
    Get the optimal coder based on context characteristics.

    Args:
        io: Input/output handler
        file_count: Number of files to be edited
        avg_file_size_kb: Average file size in KB
        model_name: AI model to use
        **kwargs: Additional configuration

    Returns:
        Optimally configured coder instance
    """
    # Decision logic based on context
    if avg_file_size_kb > 100 or file_count > 10:
        # Large files or many files - use precise diffs
        return create_coder_for_task("precise-edit", io, model_name, **kwargs)
    elif avg_file_size_kb < 5:
        # Small files - whole file replacement is efficient
        return create_coder_for_task("rewrite", io, model_name, **kwargs)
    else:
        # Medium files - standard search/replace
        return create_coder_for_task("edit", io, model_name, **kwargs)


def create_safe_coder(io, model_name: str = "gpt-4", **kwargs) -> EnhancedCoder:
    """
    Create a coder with maximum safety features enabled.

    Args:
        io: Input/output handler
        model_name: AI model to use
        **kwargs: Additional configuration

    Returns:
        Safety-configured coder instance
    """
    from .config import get_safe_config

    config = get_safe_config(model_name)

    # Apply additional safety overrides
    config.edit.dry_run_mode = kwargs.get('dry_run', False)
    config.ui.confirm_edits = kwargs.get('confirm_edits', True)
    config.edit.backup_before_edit = kwargs.get('backup', True)

    return EnhancedEditBlockCoder.create(io, config=config, **kwargs)
