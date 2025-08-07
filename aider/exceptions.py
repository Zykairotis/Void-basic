import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Type, Union, Any

from aider.dump import dump  # noqa: F401

# Setup module logger
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExInfo:
    """Exception information with retry policy and description.

    Args:
        name: Exception class name
        retry: Whether this exception should trigger a retry
        description: Human-readable description of the error
    """
    name: str
    retry: bool
    description: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate the exception info after initialization."""
        if not self.name:
            raise ValueError("Exception name cannot be empty")
        if not isinstance(self.retry, bool):
            raise TypeError("retry must be a boolean")
        if self.description is not None and not isinstance(self.description, str):
            raise TypeError("description must be a string or None")


EXCEPTIONS = [
    ExInfo("APIConnectionError", True, None),
    ExInfo("APIError", True, None),
    ExInfo("APIResponseValidationError", True, None),
    ExInfo(
        "AuthenticationError",
        False,
        "The API provider is not able to authenticate you. Check your API key.",
    ),
    ExInfo("AzureOpenAIError", True, None),
    ExInfo("BadRequestError", False, None),
    ExInfo("BudgetExceededError", True, None),
    ExInfo(
        "ContentPolicyViolationError",
        True,
        "The API provider has refused the request due to a safety policy about the content.",
    ),
    ExInfo("ContextWindowExceededError", False, None),  # special case handled in base_coder
    ExInfo("InternalServerError", True, "The API provider's servers are down or overloaded."),
    ExInfo("InvalidRequestError", True, None),
    ExInfo("JSONSchemaValidationError", True, None),
    ExInfo("NotFoundError", False, None),
    ExInfo("OpenAIError", True, None),
    ExInfo(
        "RateLimitError",
        True,
        "The API provider has rate limited you. Try again later or check your quotas.",
    ),
    ExInfo("RouterRateLimitError", True, None),
    ExInfo("ServiceUnavailableError", True, "The API provider's servers are down or overloaded."),
    ExInfo("UnprocessableEntityError", True, None),
    ExInfo("UnsupportedParamsError", True, None),
    ExInfo(
        "Timeout",
        True,
        "The API provider timed out without returning a response. They may be down or overloaded.",
    ),
]


class LiteLLMExceptions:
    """Enhanced exception handler for LiteLLM with improved error handling."""

    def __init__(self) -> None:
        """Initialize the exception handler with improved error handling."""
        self.exceptions: Dict[Type[Exception], ExInfo] = {}
        self.exception_info: Dict[str, ExInfo] = {exi.name: exi for exi in EXCEPTIONS}
        self._loaded = False

        try:
            self._load()
        except Exception as e:
            logger.error(f"Failed to initialize LiteLLMExceptions: {e}")
            # Continue with empty exceptions dict for graceful degradation

    def _load(self, strict: bool = False) -> None:
        """Load LiteLLM exceptions with improved error handling.

        Args:
            strict: If True, raise errors for missing exceptions
        """
        if self._loaded:
            return

        try:
            import litellm
        except ImportError as e:
            logger.error(f"Failed to import litellm: {e}")
            if strict:
                raise
            return

        # Check for exceptions in litellm that we don't know about
        unknown_exceptions = []
        for var in dir(litellm):
            if var.endswith("Error") and hasattr(litellm, var):
                attr = getattr(litellm, var)
                # Only consider actual exception classes
                if (isinstance(attr, type) and
                    issubclass(attr, Exception) and
                    var not in self.exception_info):
                    unknown_exceptions.append(var)

        if unknown_exceptions:
            message = f"Unknown exceptions in litellm: {unknown_exceptions}"
            logger.warning(message)
            if strict:
                raise ValueError(message)

        # Map known exceptions to their info
        successful_mappings = 0
        for var in self.exception_info:
            try:
                ex_class = getattr(litellm, var, None)
                if ex_class is not None and isinstance(ex_class, type):
                    self.exceptions[ex_class] = self.exception_info[var]
                    successful_mappings += 1
                else:
                    logger.warning(f"Exception {var} not found in litellm or not a class")
            except Exception as e:
                logger.error(f"Error mapping exception {var}: {e}")
                if strict:
                    raise

        logger.debug(f"Successfully mapped {successful_mappings}/{len(self.exception_info)} exceptions")
        self._loaded = True

    def exceptions_tuple(self) -> Tuple[Type[Exception], ...]:
        """Return tuple of all known exception classes for use in except clauses."""
        if not self._loaded:
            self._load()
        return tuple(self.exceptions.keys())

    def get_ex_info(self, ex: Exception) -> ExInfo:
        """Return the ExInfo for a given exception instance with enhanced pattern matching.

        Args:
            ex: Exception instance to analyze

        Returns:
            ExInfo object with retry policy and description
        """
        if not self._loaded:
            self._load()

        if ex is None:
            return ExInfo("UnknownError", False, "No exception provided")

        try:
            import litellm
        except ImportError:
            logger.error("Cannot import litellm for exception analysis")
            return ExInfo("ImportError", False, "LiteLLM not available")

        ex_class = ex.__class__
        ex_str = str(ex).lower()

        # Enhanced pattern matching with specific error cases
        try:
            # APIConnectionError with specific dependency errors
            if ex_class is litellm.APIConnectionError:
                if "google.auth" in ex_str or "google-generativeai" in ex_str:
                    return ExInfo(
                        "APIConnectionError",
                        False,
                        "Missing Google AI dependency. Install with: pip install google-generativeai"
                    )
                if "boto3" in ex_str or "botocore" in ex_str:
                    return ExInfo(
                        "APIConnectionError",
                        False,
                        "Missing AWS dependency. Install with: pip install boto3"
                    )
                if "openrouter" in ex_str and ("choices" in ex_str or "rate" in ex_str):
                    return ExInfo(
                        "APIConnectionError",
                        True,
                        "OpenRouter or upstream API provider issue. This is usually temporary."
                    )
                if any(keyword in ex_str for keyword in ["timeout", "connection", "network"]):
                    return ExInfo(
                        "APIConnectionError",
                        True,
                        "Network connectivity issue. Check your internet connection."
                    )

            # APIError with specific business logic errors
            elif ex_class is litellm.APIError:
                if "insufficient credits" in ex_str or "quota exceeded" in ex_str:
                    return ExInfo(
                        "APIError",
                        False,
                        "Insufficient API credits. Please add credits or check your billing."
                    )
                if "model not found" in ex_str or "model does not exist" in ex_str:
                    return ExInfo(
                        "APIError",
                        False,
                        "The specified model is not available. Check the model name."
                    )
                if any(code in ex_str for code in ['"code":402', '"code":429']):
                    return ExInfo(
                        "APIError",
                        True,  # 429 is retryable, 402 usually isn't but we'll be optimistic
                        "API usage limit reached. Wait before retrying."
                    )

            # AuthenticationError patterns
            elif ex_class is litellm.AuthenticationError:
                if "api key" in ex_str or "invalid key" in ex_str:
                    return ExInfo(
                        "AuthenticationError",
                        False,
                        "Invalid API key. Please check your API key configuration."
                    )
                if "expired" in ex_str:
                    return ExInfo(
                        "AuthenticationError",
                        False,
                        "API key has expired. Please renew your API key."
                    )

            # Rate limit errors with backoff suggestion
            elif ex_class is litellm.RateLimitError:
                return ExInfo(
                    "RateLimitError",
                    True,
                    "Rate limit exceeded. Implement exponential backoff or upgrade your plan."
                )

            # Content policy violations
            elif ex_class is litellm.ContentPolicyViolationError:
                return ExInfo(
                    "ContentPolicyViolationError",
                    False,  # Usually not retryable without content changes
                    "Content violates API provider's usage policies. Modify your request."
                )

        except Exception as analysis_error:
            logger.error(f"Error during exception analysis: {analysis_error}")
            # Fall through to default handling

        # Default lookup in our exceptions mapping
        default_info = self.exceptions.get(ex_class)
        if default_info:
            return default_info

        # Ultimate fallback
        logger.warning(f"Unknown exception type: {ex_class.__name__}")
        return ExInfo(
            ex_class.__name__ if hasattr(ex_class, '__name__') else "UnknownError",
            False,  # Conservative default - don't retry unknown exceptions
            f"Unknown error type: {ex_class.__name__}. Check logs for details."
        )

    def should_retry(self, ex: Exception) -> bool:
        """Determine if an exception should trigger a retry.

        Args:
            ex: Exception to check

        Returns:
            True if the exception is retryable
        """
        try:
            info = self.get_ex_info(ex)
            return info.retry
        except Exception as e:
            logger.error(f"Error determining retry policy: {e}")
            return False  # Conservative default

    def get_error_message(self, ex: Exception) -> str:
        """Get a user-friendly error message for an exception.

        Args:
            ex: Exception to get message for

        Returns:
            User-friendly error message
        """
        try:
            info = self.get_ex_info(ex)
            if info.description:
                return info.description
            return f"{info.name}: {str(ex)}"
        except Exception as e:
            logger.error(f"Error getting error message: {e}")
            return f"An error occurred: {str(ex)}"
