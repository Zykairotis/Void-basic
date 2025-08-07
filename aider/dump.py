import json
import logging
import traceback
from typing import Any, List, Optional


def cvt(s: Any) -> str:
    """Convert any value to a string representation with improved error handling.

    Args:
        s: Value to convert to string

    Returns:
        String representation of the value
    """
    if isinstance(s, str):
        return s

    # Handle None explicitly
    if s is None:
        return "None"

    try:
        # Try JSON serialization first for better formatting
        return json.dumps(s, indent=4, ensure_ascii=False, default=str)
    except (TypeError, ValueError, OverflowError) as e:
        # Fallback to string representation
        try:
            return str(s)
        except Exception as str_error:
            # Last resort - return type info if str() fails
            return f"<{type(s).__name__} object - conversion failed: {str_error}>"


def dump(*vals: Any) -> None:
    """Debug utility to print variables with their names and values.

    Uses stack introspection to extract variable names from the calling code.
    Handles errors gracefully and provides informative output.

    Args:
        *vals: Variable number of values to dump
    """
    try:
        # http://docs.python.org/library/traceback.html
        stack = traceback.extract_stack()

        if len(stack) < 2:
            print("dump(): Unable to extract caller information")
            for i, val in enumerate(vals):
                print(f"  arg{i}: {cvt(val)}")
            return

        # Get the calling line
        caller_line = stack[-2][3]

        if not caller_line:
            print("dump(): No caller line available")
            for i, val in enumerate(vals):
                print(f"  arg{i}: {cvt(val)}")
            return

        # Extract variable names by parsing the function call
        # This is fragile but works for most cases
        vars_str = _extract_var_names(caller_line)

        # Convert all values to strings
        converted_vals: List[str] = []
        for val in vals:
            try:
                converted_vals.append(cvt(val))
            except Exception as e:
                converted_vals.append(f"<conversion error: {e}>")

        # Check if any value contains newlines
        has_newline = any("\n" in val for val in converted_vals)

        # Print with appropriate formatting
        if has_newline:
            print(f"{vars_str}:")
            for val in converted_vals:
                if "\n" in val:
                    print(val)
                else:
                    print(f"  {val}")
        else:
            print(f"{vars_str}: {', '.join(converted_vals)}")

    except Exception as e:
        # Fallback: just print the values with minimal info
        print(f"dump() error: {e}")
        for i, val in enumerate(vals):
            try:
                print(f"  arg{i}: {cvt(val)}")
            except Exception as cvt_error:
                print(f"  arg{i}: <error converting value: {cvt_error}>")


def _extract_var_names(caller_line: str) -> str:
    """Extract variable names from the dump() function call.

    Args:
        caller_line: The line of code that called dump()

    Returns:
        String representation of the variable names
    """
    try:
        # Find the dump() call and extract the arguments
        if "dump(" not in caller_line:
            return "unknown"

        # Extract content between parentheses
        start = caller_line.find("dump(") + 5

        # Find matching closing parenthesis
        paren_count = 1
        end = start

        for i, char in enumerate(caller_line[start:], start):
            if char == "(":
                paren_count += 1
            elif char == ")":
                paren_count -= 1
                if paren_count == 0:
                    end = i
                    break

        if end <= start:
            return "unknown"

        return caller_line[start:end].strip()

    except Exception:
        # Fallback to original method if new parsing fails
        try:
            vars_str = "(".join(caller_line.split("(")[1:])
            vars_str = ")".join(vars_str.split(")")[:-1])
            return vars_str.strip()
        except Exception:
            return "unknown"
