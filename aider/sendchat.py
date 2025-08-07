import logging
from functools import lru_cache
from typing import Any, Dict, List, Optional, Set, Union

from aider.dump import dump  # noqa: F401
from aider.utils import format_messages

# Setup module logger
logger = logging.getLogger(__name__)

# Valid message roles
VALID_ROLES = {"system", "user", "assistant", "tool", "function"}

# Required keys for message validation
REQUIRED_MESSAGE_KEYS = {"role"}


class MessageValidationError(ValueError):
    """Exception raised when message validation fails."""
    pass


class MessageFormatError(ValueError):
    """Exception raised when message format is invalid."""
    pass


def sanity_check_messages(messages: List[Dict[str, Any]]) -> bool:
    """Check if messages alternate between user and assistant roles with comprehensive validation.

    System messages can be interspersed anywhere.
    Also verifies the last non-system message is from the user.

    Args:
        messages: List of message dictionaries to validate

    Returns:
        True if valid, False otherwise

    Raises:
        MessageValidationError: If messages don't properly alternate or have invalid structure
        MessageFormatError: If message format is invalid
    """
    if not isinstance(messages, list):
        raise MessageFormatError(f"Messages must be a list, got {type(messages)}")

    if not messages:
        logger.debug("Empty messages list provided")
        return True
    try:
        # Validate message structure first
        _validate_message_structure(messages)

        last_role = None
        last_non_system_role = None
        message_count = {"user": 0, "assistant": 0, "system": 0, "tool": 0}

        for i, msg in enumerate(messages):
            try:
                role = msg.get("role")
                if not role:
                    raise MessageFormatError(f"Message at index {i} missing required 'role' field")

                # Count message types
                message_count[role] = message_count.get(role, 0) + 1

                # System and tool messages can be interspersed anywhere
                if role in ("system", "tool"):
                    continue

                # Check for role alternation
                if last_role and role == last_role:
                    try:
                        turns = format_messages(messages)
                    except Exception as format_error:
                        logger.warning(f"Error formatting messages for error display: {format_error}")
                        turns = f"Messages at indices around {i} don't alternate properly"

                    error_msg = f"Messages don't properly alternate user/assistant:\n\n{turns}"
                    logger.error(error_msg)
                    raise MessageValidationError(error_msg)

                last_role = role
                last_non_system_role = role

            except Exception as e:
                if isinstance(e, (MessageValidationError, MessageFormatError)):
                    raise
                logger.error(f"Error processing message at index {i}: {e}")
                raise MessageFormatError(f"Invalid message at index {i}: {e}") from e

        # Log message statistics
        logger.debug(f"Message validation stats: {message_count}")

        # Ensure last non-system message is from user
        if last_non_system_role is None:
            logger.warning("No non-system messages found")
            return True

        is_valid = last_non_system_role == "user"
        if not is_valid:
            logger.warning(f"Last non-system message role is '{last_non_system_role}', expected 'user'")

        return is_valid

    except Exception as e:
        if isinstance(e, (MessageValidationError, MessageFormatError)):
            raise
        logger.error(f"Unexpected error in sanity_check_messages: {e}")
        raise MessageValidationError(f"Message validation failed: {e}") from e


def _validate_message_structure(messages: List[Dict[str, Any]]) -> None:
    """Validate the basic structure of messages.

    Args:
        messages: List of message dictionaries to validate

    Raises:
        MessageFormatError: If message structure is invalid
    """
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise MessageFormatError(f"Message at index {i} must be a dictionary, got {type(msg)}")

        # Check required keys
        for required_key in REQUIRED_MESSAGE_KEYS:
            if required_key not in msg:
                raise MessageFormatError(f"Message at index {i} missing required key '{required_key}'")

        # Validate role
        role = msg.get("role")
        if role not in VALID_ROLES:
            raise MessageFormatError(
                f"Message at index {i} has invalid role '{role}'. Valid roles: {VALID_ROLES}"
            )

        # Validate content exists for non-tool messages
        content = msg.get("content")
        if role != "tool" and content is None:
            logger.warning(f"Message at index {i} with role '{role}' has no content")


@lru_cache(maxsize=32)
def _get_opposite_role(role: str) -> str:
    """Get the opposite role for alternation with caching."""
    if role == "user":
        return "assistant"
    elif role == "assistant":
        return "user"
    else:
        return "user"  # Default fallback


def ensure_alternating_roles(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Ensure messages alternate between 'assistant' and 'user' roles with enhanced validation.

    Inserts empty messages of the opposite role when consecutive messages
    of the same role are found. Preserves system and tool messages.

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys

    Returns:
        List of messages with alternating roles

    Raises:
        MessageFormatError: If input messages have invalid format
    """
    if not isinstance(messages, list):
        raise MessageFormatError(f"Messages must be a list, got {type(messages)}")

    if not messages:
        logger.debug("Empty messages list provided to ensure_alternating_roles")
        return messages

    try:
        # Basic validation first
        _validate_message_structure(messages)

        fixed_messages = []
        prev_role = None
        insertions_made = 0

        for i, msg in enumerate(messages):
            current_role = msg.get("role")

            # System and tool messages don't affect alternation
            if current_role in ("system", "tool"):
                fixed_messages.append(msg.copy())  # Safe copy
                continue

            # If current role same as previous conversational role, insert opposite
            if (prev_role is not None and
                current_role == prev_role and
                current_role in ("user", "assistant")):

                opposite_role = _get_opposite_role(current_role)
                empty_message = {
                    "role": opposite_role,
                    "content": "",
                    "_inserted": True  # Mark as inserted for debugging
                }
                fixed_messages.append(empty_message)
                insertions_made += 1

                logger.debug(f"Inserted empty {opposite_role} message before index {i}")

            # Add the original message
            fixed_messages.append(msg.copy())

            # Update previous role only for conversational roles
            if current_role in ("user", "assistant"):
                prev_role = current_role

        if insertions_made > 0:
            logger.info(f"Inserted {insertions_made} empty messages to ensure role alternation")

        return fixed_messages

    except Exception as e:
        if isinstance(e, MessageFormatError):
            raise
        logger.error(f"Error ensuring alternating roles: {e}")
        raise MessageFormatError(f"Failed to ensure alternating roles: {e}") from e


def count_message_types(messages: List[Dict[str, Any]]) -> Dict[str, int]:
    """Count the number of messages by role with validation.

    Args:
        messages: List of message dictionaries

    Returns:
        Dictionary mapping role names to counts

    Raises:
        MessageFormatError: If messages have invalid format
    """
    if not isinstance(messages, list):
        raise MessageFormatError(f"Messages must be a list, got {type(messages)}")

    counts = {}

    try:
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                raise MessageFormatError(f"Message at index {i} must be a dictionary")

            role = msg.get("role")
            if not role:
                raise MessageFormatError(f"Message at index {i} missing 'role' field")

            counts[role] = counts.get(role, 0) + 1

        logger.debug(f"Message type counts: {counts}")
        return counts

    except Exception as e:
        if isinstance(e, MessageFormatError):
            raise
        logger.error(f"Error counting message types: {e}")
        raise MessageFormatError(f"Failed to count message types: {e}") from e


def validate_message_content(messages: List[Dict[str, Any]], strict: bool = False) -> List[str]:
    """Validate message content and return list of warnings.

    Args:
        messages: List of message dictionaries to validate
        strict: If True, raise exceptions for warnings

    Returns:
        List of validation warnings

    Raises:
        MessageValidationError: If strict=True and validation issues found
    """
    if not isinstance(messages, list):
        raise MessageFormatError(f"Messages must be a list, got {type(messages)}")

    warnings = []

    try:
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                warning = f"Message at index {i} is not a dictionary"
                warnings.append(warning)
                if strict:
                    raise MessageValidationError(warning)
                continue

            role = msg.get("role")
            content = msg.get("content")

            # Check for empty content in non-tool messages
            if role != "tool" and not content:
                warning = f"Message at index {i} with role '{role}' has empty content"
                warnings.append(warning)
                if strict:
                    raise MessageValidationError(warning)

            # Check for excessively long content
            if isinstance(content, str) and len(content) > 100000:
                warning = f"Message at index {i} has very long content ({len(content)} chars)"
                warnings.append(warning)
                if strict:
                    raise MessageValidationError(warning)

            # Check for suspicious content
            if isinstance(content, str):
                if content.count('\n') > 1000:
                    warning = f"Message at index {i} has excessive newlines ({content.count('\n')})"
                    warnings.append(warning)

        return warnings

    except Exception as e:
        if isinstance(e, (MessageValidationError, MessageFormatError)):
            raise
        logger.error(f"Error validating message content: {e}")
        raise MessageValidationError(f"Content validation failed: {e}") from e


def clean_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Clean and normalize messages for better processing.

    Args:
        messages: List of message dictionaries to clean

    Returns:
        List of cleaned message dictionaries
    """
    if not isinstance(messages, list):
        raise MessageFormatError(f"Messages must be a list, got {type(messages)}")

    cleaned = []

    try:
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                logger.warning(f"Skipping non-dictionary message at index {i}")
                continue

            # Create clean copy
            clean_msg = {}

            # Copy required fields
            for key in ["role", "content"]:
                if key in msg:
                    value = msg[key]
                    if isinstance(value, str):
                        # Clean up whitespace
                        value = value.strip()
                        # Normalize line endings
                        value = value.replace('\r\n', '\n').replace('\r', '\n')
                    clean_msg[key] = value

            # Copy optional fields
            for key in ["name", "function_call", "tool_calls", "tool_call_id"]:
                if key in msg:
                    clean_msg[key] = msg[key]

            # Only add if we have required fields
            if "role" in clean_msg:
                cleaned.append(clean_msg)
            else:
                logger.warning(f"Message at index {i} missing required 'role', skipping")

        logger.debug(f"Cleaned {len(cleaned)} messages from {len(messages)} input messages")
        return cleaned

    except Exception as e:
        logger.error(f"Error cleaning messages: {e}")
        raise MessageFormatError(f"Failed to clean messages: {e}") from e
