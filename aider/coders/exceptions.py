"""
Enhanced error handling and validation framework for Aider coders.

This module provides a comprehensive exception hierarchy that enables
better error reporting, validation, and user feedback throughout the
coding system.
"""

from typing import Any, Dict, List, Optional, TypeAlias, Union, override
from dataclasses import dataclass
from pathlib import Path

# Modern type aliases for better readability
ErrorMessage: TypeAlias = str
FilePath: TypeAlias = str | Path
ErrorCode: TypeAlias = str
ErrorMetadata: TypeAlias = Dict[str, Any]


@dataclass
class ErrorContext:
    """Rich context information for errors with modern typing."""
    file_path: str | None = None
    line_number: int | None = None
    column: int | None = None
    code_snippet: str | None = None
    suggestions: List[str] | None = None
    error_code: ErrorCode | None = None
    metadata: ErrorMetadata | None = None

    def __post_init__(self):
        """Validate error context after initialization."""
        if self.suggestions is None:
            self.suggestions = []
        if self.metadata is None:
            self.metadata = {}


class AiderCoderError(Exception):
    """Base exception for all Aider coder errors."""

    def __init__(
        self,
        message: ErrorMessage,
        context: ErrorContext | None = None,
        cause: Exception | None = None
    ):
        self.message = message
        self.context = context or ErrorContext()
        self.cause = cause
        super().__init__(self._format_message())

    def _format_message(self) -> ErrorMessage:
        """Format the error message with context using modern patterns."""
        parts = [self.message]

        if self.context and self.context.file_path:
            parts.append(f"File: {self.context.file_path}")

        if self.context and self.context.line_number:
            parts.append(f"Line: {self.context.line_number}")

        if self.context and self.context.error_code:
            parts.append(f"Code: {self.context.error_code}")

        if self.context.suggestions:
            suggestions = '\n'.join(f"  - {s}" for s in self.context.suggestions)
            parts.append(f"Suggestions:\n{suggestions}")

        return '\n'.join(parts)


# =============================================================================
# Configuration and Setup Errors
# =============================================================================

class ConfigurationError(AiderCoderError):
    """Errors related to coder configuration."""
    pass


class UnknownEditFormat(ConfigurationError):
    """Raised when an unsupported edit format is specified."""

    def __init__(self, edit_format: str, valid_formats: List[str]):
        self.edit_format = edit_format
        self.valid_formats = valid_formats

        context = ErrorContext(
            error_code="UNKNOWN_EDIT_FORMAT",
            suggestions=[
                f"Use one of the supported formats: {', '.join(valid_formats)}",
                "Check the documentation for format-specific requirements"
            ]
        )

        super().__init__(
            f"Unknown edit format '{edit_format}'. Valid formats are: {', '.join(valid_formats)}",
            context
        )


class MissingAPIKeyError(ConfigurationError):
    """Raised when required API credentials are missing."""

    def __init__(self, service: str, env_var: Optional[str] = None):
        self.service = service
        self.env_var = env_var

        suggestions = [f"Set up API credentials for {service}"]
        if env_var:
            suggestions.append(f"Set the {env_var} environment variable")

        context = ErrorContext(
            error_code="MISSING_API_KEY",
            suggestions=suggestions
        )

        super().__init__(
            f"Missing API key for {service}",
            context
        )


class ModelNotSupportedError(ConfigurationError):
    """Raised when a model doesn't support required features."""

    def __init__(self, model_name: str, required_feature: str):
        self.model_name = model_name
        self.required_feature = required_feature

        context = ErrorContext(
            error_code="MODEL_NOT_SUPPORTED",
            suggestions=[
                f"Use a model that supports {required_feature}",
                "Check the model compatibility documentation"
            ]
        )

        super().__init__(
            f"Model '{model_name}' does not support {required_feature}",
            context
        )


# =============================================================================
# Validation Errors
# =============================================================================

class ValidationError(AiderCoderError):
    """Base class for validation errors."""
    pass


class FileValidationError(ValidationError):
    """Errors related to file validation."""
    pass


class FileNotFoundError(FileValidationError):
    """Raised when a required file cannot be found."""

    def __init__(self, file_path: str, search_paths: Optional[List[str]] = None):
        self.file_path = file_path
        self.search_paths = search_paths or []

        suggestions = [
            "Check that the file path is correct",
            "Ensure the file exists in the project"
        ]

        if search_paths:
            suggestions.append(f"Searched in: {', '.join(search_paths)}")

        context = ErrorContext(
            file_path=file_path,
            error_code="FILE_NOT_FOUND",
            suggestions=suggestions
        )

        super().__init__(
            f"File not found: {file_path}",
            context
        )


class FileNotEditableError(FileValidationError):
    """Raised when a file cannot be edited due to permissions or other constraints."""

    def __init__(self, file_path: str, reason: str):
        self.file_path = file_path
        self.reason = reason

        context = ErrorContext(
            file_path=file_path,
            error_code="FILE_NOT_EDITABLE",
            suggestions=[
                "Check file permissions",
                "Ensure the file is not locked by another process",
                "Verify the file is not in .gitignore or excluded"
            ]
        )

        super().__init__(
            f"Cannot edit file '{file_path}': {reason}",
            context
        )


class ContentValidationError(ValidationError):
    """Errors related to content validation."""
    pass


class MalformedEditError(ContentValidationError):
    """Raised when an edit instruction is malformed."""

    def __init__(self, edit_content: str, expected_format: str, details: Optional[str] = None):
        self.edit_content = edit_content
        self.expected_format = expected_format
        self.details = details

        suggestions = [
            f"Ensure the edit follows the {expected_format} format",
            "Check for proper syntax and structure",
            "Review the documentation for format requirements"
        ]

        if details:
            suggestions.append(f"Specific issue: {details}")

        context = ErrorContext(
            error_code="MALFORMED_EDIT",
            suggestions=suggestions
        )

        super().__init__(
            f"Malformed edit in {expected_format} format",
            context
        )


# =============================================================================
# Edit Operation Errors
# =============================================================================

class EditOperationError(AiderCoderError):
    """Base class for errors during edit operations."""
    pass


class SearchTextNotUniqueError(EditOperationError):
    """Raised when search text matches multiple locations."""

    def __init__(self, search_text: str, file_path: str, match_count: int):
        self.search_text = search_text
        self.file_path = file_path
        self.match_count = match_count

        context = ErrorContext(
            file_path=file_path,
            error_code="SEARCH_NOT_UNIQUE",
            suggestions=[
                "Add more context lines to make the search unique",
                "Use line numbers or function names for specificity",
                "Break the edit into smaller, more specific chunks"
            ]
        )

        super().__init__(
            f"Search text found {match_count} times in {file_path}. Need unique match.",
            context
        )


class SearchTextNotFoundError(EditOperationError):
    """Raised when search text is not found in the target file."""

    def __init__(self, search_text: str, file_path: str):
        self.search_text = search_text
        self.file_path = file_path

        context = ErrorContext(
            file_path=file_path,
            code_snippet=search_text[:200] + "..." if len(search_text) > 200 else search_text,
            error_code="SEARCH_NOT_FOUND",
            suggestions=[
                "Check that the search text exactly matches the file content",
                "Ensure proper whitespace and indentation",
                "Verify the file hasn't been modified since the search text was generated"
            ]
        )

        super().__init__(
            f"Search text not found in {file_path}",
            context
        )


class DiffApplicationError(EditOperationError):
    """Raised when a diff cannot be applied."""

    def __init__(self, diff_content: str, file_path: str, reason: str):
        self.diff_content = diff_content
        self.file_path = file_path
        self.reason = reason

        context = ErrorContext(
            file_path=file_path,
            error_code="DIFF_APPLICATION_FAILED",
            suggestions=[
                "Check that the diff is properly formatted",
                "Ensure the original file hasn't changed",
                "Verify line endings and whitespace match"
            ]
        )

        super().__init__(
            f"Failed to apply diff to {file_path}: {reason}",
            context
        )


class PartialEditError(EditOperationError):
    """Raised when only some edits in a batch could be applied."""

    def __init__(self, successful_edits: List[str], failed_edits: Dict[str, str]):
        self.successful_edits = successful_edits
        self.failed_edits = failed_edits

        context = ErrorContext(
            error_code="PARTIAL_EDIT_FAILURE",
            suggestions=[
                "Review the failed edits and try them individually",
                "Check if failed files have conflicts",
                "Consider breaking complex edits into smaller chunks"
            ],
            metadata={
                "successful_count": len(successful_edits),
                "failed_count": len(failed_edits)
            }
        )

        failed_list = '\n'.join(f"  - {file}: {error}" for file, error in failed_edits.items())
        super().__init__(
            f"Applied {len(successful_edits)} edits successfully, but {len(failed_edits)} failed:\n{failed_list}",
            context
        )


# =============================================================================
# Runtime and Processing Errors
# =============================================================================

class RuntimeError(AiderCoderError):
    """Base class for runtime errors."""
    pass


class FinishReasonLength(RuntimeError):
    """Raised when AI response is truncated due to length limits."""

    def __init__(self, model_name: str, content_length: Optional[int] = None):
        self.model_name = model_name
        self.content_length = content_length

        context = ErrorContext(
            error_code="RESPONSE_TRUNCATED",
            suggestions=[
                "Break the request into smaller parts",
                "Use a model with higher output limits",
                "Simplify the complexity of the requested changes"
            ]
        )

        super().__init__(
            f"AI response was truncated due to length limits in model {model_name}",
            context
        )


class TokenLimitExceededError(RuntimeError):
    """Raised when token limits are exceeded."""

    def __init__(self, used_tokens: int, max_tokens: int, model_name: str):
        self.used_tokens = used_tokens
        self.max_tokens = max_tokens
        self.model_name = model_name

        context = ErrorContext(
            error_code="TOKEN_LIMIT_EXCEEDED",
            suggestions=[
                "Reduce the size of files in the chat",
                "Remove unnecessary context",
                "Use a model with higher token limits",
                "Break the task into smaller requests"
            ],
            metadata={
                "used_tokens": used_tokens,
                "max_tokens": max_tokens,
                "model": model_name
            }
        )

        super().__init__(
            f"Token limit exceeded: {used_tokens}/{max_tokens} for model {model_name}",
            context
        )


class ModelResponseError(RuntimeError):
    """Raised when AI model response is invalid or unexpected."""

    def __init__(self, model_name: str, response_content: str, expected_format: str):
        self.model_name = model_name
        self.response_content = response_content
        self.expected_format = expected_format

        context = ErrorContext(
            error_code="INVALID_MODEL_RESPONSE",
            suggestions=[
                "Retry the request with clearer instructions",
                "Check if the model supports the required format",
                "Try a different edit format or model"
            ]
        )

        super().__init__(
            f"Invalid response from {model_name}. Expected {expected_format} format",
            context
        )


# =============================================================================
# Git and Repository Errors
# =============================================================================

class RepositoryError(AiderCoderError):
    """Base class for repository-related errors."""
    pass


class GitOperationError(RepositoryError):
    """Raised when git operations fail."""

    def __init__(self, operation: str, exit_code: int, stderr: str):
        self.operation = operation
        self.exit_code = exit_code
        self.stderr = stderr

        context = ErrorContext(
            error_code="GIT_OPERATION_FAILED",
            suggestions=[
                "Check git repository status",
                "Ensure working directory is clean",
                "Verify git configuration is correct"
            ]
        )

        super().__init__(
            f"Git {operation} failed with exit code {exit_code}: {stderr}",
            context
        )


class DirtyRepositoryError(RepositoryError):
    """Raised when repository has uncommitted changes."""

    def __init__(self, dirty_files: List[str]):
        self.dirty_files = dirty_files

        context = ErrorContext(
            error_code="DIRTY_REPOSITORY",
            suggestions=[
                "Commit or stash your current changes",
                "Use --dirty-commits flag if appropriate",
                "Review the uncommitted files listed"
            ]
        )

        dirty_list = '\n'.join(f"  - {file}" for file in dirty_files)
        super().__init__(
            f"Repository has uncommitted changes:\n{dirty_list}",
            context
        )


# =============================================================================
# Prompt and Template Errors
# =============================================================================

class PromptError(AiderCoderError):
    """Base class for prompt-related errors."""
    pass


class TemplateRenderError(PromptError):
    """Raised when prompt template rendering fails."""

    def __init__(self, template_name: str, missing_vars: List[str]):
        self.template_name = template_name
        self.missing_vars = missing_vars

        context = ErrorContext(
            error_code="TEMPLATE_RENDER_FAILED",
            suggestions=[
                f"Provide values for missing variables: {', '.join(missing_vars)}",
                "Check template syntax and variable names",
                "Verify template configuration"
            ]
        )

        super().__init__(
            f"Failed to render template '{template_name}': missing variables {missing_vars}",
            context
        )


class PromptTooLongError(PromptError):
    """Raised when a prompt exceeds model limits."""

    def __init__(self, prompt_tokens: int, max_tokens: int, model_name: str):
        self.prompt_tokens = prompt_tokens
        self.max_tokens = max_tokens
        self.model_name = model_name

        context = ErrorContext(
            error_code="PROMPT_TOO_LONG",
            suggestions=[
                "Reduce the number of files in context",
                "Use a more efficient edit format",
                "Break the request into smaller parts",
                "Use a model with higher token limits"
            ]
        )

        super().__init__(
            f"Prompt too long: {prompt_tokens} tokens exceeds {max_tokens} limit for {model_name}",
            context
        )


# =============================================================================
# Edit Format Specific Errors
# =============================================================================

class EditFormatError(AiderCoderError):
    """Base class for edit format-specific errors."""
    pass


class SearchReplaceError(EditFormatError):
    """Errors specific to search/replace operations."""
    pass


class UnifiedDiffError(EditFormatError):
    """Errors specific to unified diff operations."""
    pass


class MalformedSearchBlockError(SearchReplaceError):
    """Raised when search/replace block is malformed."""

    def __init__(self, block_content: str, issue: str):
        self.block_content = block_content
        self.issue = issue

        context = ErrorContext(
            error_code="MALFORMED_SEARCH_BLOCK",
            code_snippet=block_content[:300] + "..." if len(block_content) > 300 else block_content,
            suggestions=[
                "Check search/replace block syntax",
                "Ensure proper fence markers are used",
                "Verify SEARCH and REPLACE sections are present"
            ]
        )

        super().__init__(
            f"Malformed search/replace block: {issue}",
            context
        )


class InvalidDiffFormatError(UnifiedDiffError):
    """Raised when diff format is invalid."""

    def __init__(self, diff_content: str, format_issue: str):
        self.diff_content = diff_content
        self.format_issue = format_issue

        context = ErrorContext(
            error_code="INVALID_DIFF_FORMAT",
            code_snippet=diff_content[:300] + "..." if len(diff_content) > 300 else diff_content,
            suggestions=[
                "Check diff header format (--- and +++)",
                "Ensure hunk headers are correct",
                "Verify line prefixes (+, -, space) are proper"
            ]
        )

        super().__init__(
            f"Invalid diff format: {format_issue}",
            context
        )


# =============================================================================
# Utility Functions
# =============================================================================

def validate_file_path(file_path: Union[str, Path], must_exist: bool = True) -> Path:
    """
    Validate and normalize a file path.

    Args:
        file_path: The file path to validate
        must_exist: Whether the file must exist

    Returns:
        Normalized Path object

    Raises:
        FileValidationError: If validation fails
    """
    try:
        path = Path(file_path)

        # Check for potentially dangerous paths
        if path.is_absolute():
            raise FileValidationError(
                f"Absolute paths not allowed: {file_path}",
                ErrorContext(
                    file_path=str(file_path),
                    error_code="ABSOLUTE_PATH_NOT_ALLOWED",
                    suggestions=["Use relative paths from project root"]
                )
            )

        # Check for path traversal attempts
        if ".." in path.parts:
            raise FileValidationError(
                f"Path traversal not allowed: {file_path}",
                ErrorContext(
                    file_path=str(file_path),
                    error_code="PATH_TRAVERSAL_NOT_ALLOWED",
                    suggestions=["Use paths within the project directory"]
                )
            )

        if must_exist and not path.exists():
            raise FileNotFoundError(str(file_path))

        return path

    except OSError as e:
        raise FileValidationError(
            f"Invalid file path: {file_path}",
            ErrorContext(
                file_path=str(file_path),
                error_code="INVALID_PATH",
                suggestions=["Check path format and characters"]
            )
        ) from e


def create_validation_error(
    message: str,
    file_path: Optional[str] = None,
    suggestions: Optional[List[str]] = None,
    error_code: Optional[str] = None
) -> ValidationError:
    """
    Factory function for creating validation errors with rich context.

    Args:
        message: The error message
        file_path: Optional file path context
        suggestions: Optional list of suggestions
        error_code: Optional error code

    Returns:
        ValidationError with proper context
    """
    context = ErrorContext(
        file_path=file_path,
        suggestions=suggestions or [],
        error_code=error_code
    )

    return ValidationError(message, context)


def handle_edit_errors(func):
    """
    Decorator for handling common edit operation errors.

    This decorator catches common exceptions and converts them
    to more informative AiderCoderError instances.
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except FileNotFoundError as e:
            raise FileNotFoundError(str(e.filename)) from e
        except PermissionError as e:
            raise FileNotEditableError(str(e.filename), "Permission denied") from e
        except UnicodeDecodeError as e:
            raise ContentValidationError(
                f"File encoding error: {e}",
                ErrorContext(
                    error_code="ENCODING_ERROR",
                    suggestions=[
                        "Check file encoding (should be UTF-8)",
                        "Verify file is a text file",
                        "Try opening file with different encoding"
                    ]
                )
            ) from e
        except Exception as e:
            # Re-raise if already an AiderCoderError
            if isinstance(e, AiderCoderError):
                raise
            # Wrap other exceptions
            raise AiderCoderError(f"Unexpected error in {func.__name__}: {e}") from e

    return wrapper


# =============================================================================
# Error Recovery Strategies
# =============================================================================

class ErrorRecoveryStrategy:
    """Base class for error recovery strategies."""

    def can_recover(self, error: AiderCoderError) -> bool:
        """Check if this strategy can recover from the given error."""
        return False

    def recover(self, error: AiderCoderError, context: Dict[str, Any]) -> Any:
        """Attempt to recover from the error."""
        raise NotImplementedError


class SearchTextRecoveryStrategy(ErrorRecoveryStrategy):
    """Recovery strategy for search text issues."""

    def can_recover(self, error: AiderCoderError) -> bool:
        return isinstance(error, (SearchTextNotFoundError, SearchTextNotUniqueError))

    def recover(self, error: AiderCoderError, context: Dict[str, Any]) -> str:
        """Attempt to fix search text issues by adding context."""
        if isinstance(error, SearchTextNotUniqueError):
            # Add more context lines to make search unique
            return self._add_context_lines(error.search_text, error.file_path)
        elif isinstance(error, SearchTextNotFoundError):
            # Try fuzzy matching to find similar content
            return self._fuzzy_search(error.search_text, error.file_path)

        raise error

    def _add_context_lines(self, search_text: str, file_path: str) -> str:
        """Add context lines to make search text unique."""
        # Implementation would analyze the file and add context
        pass

    def _fuzzy_search(self, search_text: str, file_path: str) -> str:
        """Find similar content using fuzzy matching."""
        # Implementation would use fuzzy string matching
        pass


# =============================================================================
# Error Reporting and Logging
# =============================================================================

def format_error_for_user(error: AiderCoderError) -> str:
    """
    Format an error message for user-friendly display.

    Args:
        error: The error to format

    Returns:
        Formatted error message
    """
    lines = [f"❌ {error.message}"]

    if error.context.file_path:
        lines.append(f"📁 File: {error.context.file_path}")

    if error.context.line_number:
        lines.append(f"📍 Line: {error.context.line_number}")

    if error.context.code_snippet:
        lines.append(f"📄 Code snippet:\n```\n{error.context.code_snippet}\n```")

    if error.context.suggestions:
        lines.append("💡 Suggestions:")
        lines.extend(f"  • {suggestion}" for suggestion in error.context.suggestions)

    return '\n'.join(lines)


def log_error_for_debugging(error: AiderCoderError) -> Dict[str, Any]:
    """
    Create a structured log entry for debugging purposes.

    Args:
        error: The error to log

    Returns:
        Structured error data
    """
    return {
        "error_type": type(error).__name__,
        "message": error.message,
        "error_code": error.context.error_code,
        "file_path": error.context.file_path,
        "line_number": error.context.line_number,
        "metadata": error.context.metadata,
        "suggestions": error.context.suggestions,
        "cause": str(error.cause) if error.cause else None,
        "stack_trace": error.__traceback__
    }


# =============================================================================
# Backwards Compatibility
# =============================================================================

# Keep the old exception names for backwards compatibility
UnknownEditFormat = UnknownEditFormat
MissingAPIKeyError = MissingAPIKeyError
FinishReasonLength = FinishReasonLength
