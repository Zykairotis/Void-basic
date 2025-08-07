#!/usr/bin/env python

import logging
import re
from typing import Optional

from aider.dump import dump  # noqa

# Setup module logger
logger = logging.getLogger(__name__)

# Standard tag identifier
REASONING_TAG = "thinking-content-" + "7bbeb8e1441453ad999a0bbba8a46d4b"
# Output formatting
REASONING_START = "--------------\n► **THINKING**"
REASONING_END = "------------\n► **ANSWER**"

# Compiled regex patterns for better performance
_compiled_patterns = {}

def _get_compiled_pattern(tag_name: str, pattern_type: str) -> re.Pattern:
    """Get or create compiled regex pattern for tag operations.

    Args:
        tag_name: The tag name to create pattern for
        pattern_type: Type of pattern ('complete', 'opening', 'closing')

    Returns:
        Compiled regex pattern
    """
    cache_key = (tag_name, pattern_type)
    if cache_key not in _compiled_patterns:
        if pattern_type == 'complete':
            pattern = f"<{re.escape(tag_name)}>.*?</{re.escape(tag_name)}>"
            _compiled_patterns[cache_key] = re.compile(pattern, re.DOTALL)
        elif pattern_type == 'opening':
            pattern = f"\\s*<{re.escape(tag_name)}>\\s*"
            _compiled_patterns[cache_key] = re.compile(pattern)
        elif pattern_type == 'closing':
            pattern = f"\\s*</{re.escape(tag_name)}>\\s*"
            _compiled_patterns[cache_key] = re.compile(pattern)
        else:
            raise ValueError(f"Unknown pattern type: {pattern_type}")

    return _compiled_patterns[cache_key]


def remove_reasoning_content(res: Optional[str], reasoning_tag: Optional[str]) -> str:
    """
    Remove reasoning content from text based on tags with enhanced error handling.

    Args:
        res: The text to process (can be None)
        reasoning_tag: The tag name to remove (can be None)

    Returns:
        Text with reasoning content removed, empty string if input is invalid

    Raises:
        TypeError: If inputs are not strings or None
    """
    # Input validation
    if res is None:
        logger.debug("remove_reasoning_content called with None text")
        return ""

    if reasoning_tag is None or not reasoning_tag.strip():
        logger.debug("remove_reasoning_content called with empty reasoning_tag")
        return res

    if not isinstance(res, str):
        raise TypeError(f"Expected str for res, got {type(res)}")

    if not isinstance(reasoning_tag, str):
        raise TypeError(f"Expected str for reasoning_tag, got {type(reasoning_tag)}")

    try:
        # Sanitize tag name to prevent regex injection
        sanitized_tag = reasoning_tag.strip()
        if not sanitized_tag:
            return res

        # Try to match the complete tag pattern first using compiled regex
        try:
            complete_pattern = _get_compiled_pattern(sanitized_tag, 'complete')
            res = complete_pattern.sub("", res).strip()
        except Exception as e:
            logger.error(f"Error applying complete pattern for tag '{sanitized_tag}': {e}")
            # Fallback to simple string replacement
            opening_tag = f"<{sanitized_tag}>"
            closing_tag = f"</{sanitized_tag}>"
            if opening_tag in res and closing_tag in res:
                start_idx = res.find(opening_tag)
                end_idx = res.find(closing_tag, start_idx) + len(closing_tag)
                res = (res[:start_idx] + res[end_idx:]).strip()

        # If closing tag exists but opening tag might be missing, remove everything before closing tag
        closing_tag = f"</{sanitized_tag}>"
        if closing_tag in res:
            try:
                parts = res.split(closing_tag, 1)
                if len(parts) > 1:
                    res = parts[1].strip()
            except Exception as e:
                logger.error(f"Error processing closing tag for '{sanitized_tag}': {e}")

        return res

    except Exception as e:
        logger.error(f"Unexpected error in remove_reasoning_content: {e}")
        return res  # Return original text on error


def replace_reasoning_tags(text: Optional[str], tag_name: Optional[str]) -> str:
    """
    Replace opening and closing reasoning tags with standard formatting.
    Ensures exactly one blank line before START and END markers.

    Args:
        text: The text containing the tags (can be None)
        tag_name: The name of the tag to replace (can be None)

    Returns:
        Text with reasoning tags replaced with standard format

    Raises:
        TypeError: If inputs are not strings or None
    """
    # Input validation
    if not text:
        logger.debug("replace_reasoning_tags called with empty text")
        return text or ""

    if not tag_name or not tag_name.strip():
        logger.debug("replace_reasoning_tags called with empty tag_name")
        return text

    if not isinstance(text, str):
        raise TypeError(f"Expected str for text, got {type(text)}")

    if not isinstance(tag_name, str):
        raise TypeError(f"Expected str for tag_name, got {type(tag_name)}")

    try:
        # Sanitize tag name
        sanitized_tag = tag_name.strip()
        if not sanitized_tag:
            return text

        # Replace opening tag with proper spacing using compiled regex
        try:
            opening_pattern = _get_compiled_pattern(sanitized_tag, 'opening')
            text = opening_pattern.sub(f"\n{REASONING_START}\n\n", text)
        except Exception as e:
            logger.error(f"Error replacing opening tag '{sanitized_tag}': {e}")
            # Fallback to simple string replacement
            opening_tag = f"<{sanitized_tag}>"
            text = text.replace(opening_tag, f"\n{REASONING_START}\n\n")

        # Replace closing tag with proper spacing using compiled regex
        try:
            closing_pattern = _get_compiled_pattern(sanitized_tag, 'closing')
            text = closing_pattern.sub(f"\n\n{REASONING_END}\n\n", text)
        except Exception as e:
            logger.error(f"Error replacing closing tag '{sanitized_tag}': {e}")
            # Fallback to simple string replacement
            closing_tag = f"</{sanitized_tag}>"
            text = text.replace(closing_tag, f"\n\n{REASONING_END}\n\n")

        return text

    except Exception as e:
        logger.error(f"Unexpected error in replace_reasoning_tags: {e}")
        return text  # Return original text on error


def format_reasoning_content(reasoning_content: Optional[str], tag_name: Optional[str]) -> str:
    """
    Format reasoning content with appropriate tags and enhanced validation.

    Args:
        reasoning_content: The content to format (can be None)
        tag_name: The tag name to use (can be None)

    Returns:
        Formatted reasoning content with tags, empty string if invalid input

    Raises:
        TypeError: If inputs are not strings or None
    """
    # Input validation
    if not reasoning_content or not reasoning_content.strip():
        logger.debug("format_reasoning_content called with empty content")
        return ""

    if not tag_name or not tag_name.strip():
        logger.warning("format_reasoning_content called with empty tag_name")
        return reasoning_content or ""

    if not isinstance(reasoning_content, str):
        raise TypeError(f"Expected str for reasoning_content, got {type(reasoning_content)}")

    if not isinstance(tag_name, str):
        raise TypeError(f"Expected str for tag_name, got {type(tag_name)}")

    try:
        # Sanitize inputs
        sanitized_content = reasoning_content.strip()
        sanitized_tag = tag_name.strip()

        if not sanitized_content or not sanitized_tag:
            return ""

        # Validate tag name doesn't contain problematic characters
        if '<' in sanitized_tag or '>' in sanitized_tag:
            logger.error(f"Invalid tag name contains angle brackets: {sanitized_tag}")
            return sanitized_content

        # Format with proper spacing and structure
        formatted = f"<{sanitized_tag}>\n\n{sanitized_content}\n\n</{sanitized_tag}>"

        logger.debug(f"Successfully formatted reasoning content with tag '{sanitized_tag}'")
        return formatted

    except Exception as e:
        logger.error(f"Unexpected error in format_reasoning_content: {e}")
        return reasoning_content or ""


def clear_compiled_patterns() -> None:
    """Clear the compiled regex pattern cache.

    Useful for testing or memory management in long-running processes.
    """
    global _compiled_patterns
    _compiled_patterns.clear()
    logger.debug("Cleared compiled regex pattern cache")
