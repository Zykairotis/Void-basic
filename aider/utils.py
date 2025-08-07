import asyncio
import logging
import os
import platform
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import oslex

from aider.dump import dump  # noqa: F401
from aider.waiting import Spinner

# Setup module logger
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS: Set[str] = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp", ".pdf",
    ".svg", ".ico", ".tga", ".psd", ".ai", ".eps", ".raw"
}


class IgnorantTemporaryDirectory:
    """Temporary directory that ignores cleanup errors gracefully."""

    def __init__(self) -> None:
        """Initialize temporary directory with version-appropriate error handling."""
        try:
            if sys.version_info >= (3, 10):
                self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
            else:
                self.temp_dir = tempfile.TemporaryDirectory()
            logger.debug(f"Created temporary directory: {self.temp_dir.name}")
        except Exception as e:
            logger.error(f"Failed to create temporary directory: {e}")
            raise

    def __enter__(self) -> str:
        """Enter context manager."""
        try:
            return self.temp_dir.__enter__()
        except Exception as e:
            logger.error(f"Error entering temporary directory context: {e}")
            raise

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager with cleanup."""
        self.cleanup()

    def cleanup(self) -> None:
        """Clean up temporary directory, ignoring common errors."""
        try:
            self.temp_dir.cleanup()
            logger.debug("Temporary directory cleaned up successfully")
        except (OSError, PermissionError, RecursionError) as e:
            logger.debug(f"Ignoring cleanup error: {e}")
            # Ignore errors (Windows and potential recursion)
        except Exception as e:
            logger.warning(f"Unexpected error during cleanup: {e}")

    def __getattr__(self, item: str) -> Any:
        """Delegate attribute access to underlying temp_dir."""
        return getattr(self.temp_dir, item)


class ChdirTemporaryDirectory(IgnorantTemporaryDirectory):
    """Temporary directory that changes to itself on context entry."""

    def __init__(self) -> None:
        """Initialize with current working directory tracking."""
        try:
            self.cwd = os.getcwd()
            logger.debug(f"Saved current directory: {self.cwd}")
        except (FileNotFoundError, OSError) as e:
            logger.warning(f"Could not get current directory: {e}")
            self.cwd = None

        super().__init__()

    def __enter__(self) -> str:
        """Enter context and change to temporary directory."""
        res = super().__enter__()
        try:
            temp_path = Path(self.temp_dir.name).resolve()
            os.chdir(temp_path)
            logger.debug(f"Changed to temporary directory: {temp_path}")
            return res
        except (OSError, FileNotFoundError) as e:
            logger.error(f"Failed to change to temporary directory: {e}")
            raise

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context and restore original directory."""
        if self.cwd:
            try:
                os.chdir(self.cwd)
                logger.debug(f"Restored original directory: {self.cwd}")
            except (FileNotFoundError, OSError) as e:
                logger.warning(f"Could not restore original directory: {e}")

        super().__exit__(exc_type, exc_val, exc_tb)


class GitTemporaryDirectory(ChdirTemporaryDirectory):
    """Temporary directory with git repository initialization."""

    def __enter__(self) -> str:
        """Enter context and initialize git repository."""
        dname = super().__enter__()
        try:
            self.repo = make_repo(dname)
            logger.debug(f"Initialized git repository in: {dname}")
            return dname
        except Exception as e:
            logger.error(f"Failed to initialize git repository: {e}")
            # Clean up and re-raise
            super().__exit__(None, None, None)
            raise

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context and clean up git repository."""
        try:
            if hasattr(self, 'repo'):
                del self.repo
                logger.debug("Cleaned up git repository")
        except Exception as e:
            logger.warning(f"Error cleaning up git repository: {e}")

        super().__exit__(exc_type, exc_val, exc_tb)


def make_repo(path: Optional[Union[str, Path]] = None):
    """Create a git repository with test configuration.

    Args:
        path: Path to initialize repository in, defaults to current directory

    Returns:
        Initialized git repository object

    Raises:
        ImportError: If git module is not available
        Exception: If repository initialization fails
    """
    try:
        import git
    except ImportError as e:
        logger.error("GitPython not available for repository creation")
        raise ImportError("GitPython required for git repository operations") from e

    if not path:
        path = "."

    path_obj = Path(path)

    try:
        repo = git.Repo.init(str(path_obj))

        # Configure with test user safely
        with repo.config_writer() as config:
            config.set_value("user", "name", "Test User")
            config.set_value("user", "email", "testuser@example.com")
            config.set_value("init", "defaultBranch", "main")  # Modern default

        logger.debug(f"Initialized git repository at: {path_obj}")
        return repo

    except Exception as e:
        logger.error(f"Failed to initialize git repository at {path_obj}: {e}")
        raise


@lru_cache(maxsize=256)
def is_image_file(file_name: Union[str, Path, None]) -> bool:
    """
    Check if the given file name has an image file extension.
    Uses caching for better performance with repeated checks.

    Args:
        file_name: The name of the file to check

    Returns:
        True if the file is an image, False otherwise
    """
    if not file_name:
        return False

    try:
        file_str = str(file_name).lower()  # Convert to lowercase for comparison
        return any(file_str.endswith(ext.lower()) for ext in IMAGE_EXTENSIONS)
    except (AttributeError, TypeError) as e:
        logger.debug(f"Error checking if file is image: {e}")
        return False


def safe_abs_path(res: Union[str, Path]) -> str:
    """
    Get absolute path, safely handling Windows 8.3 paths and errors.

    Args:
        res: Path to resolve

    Returns:
        Absolute path as string

    Raises:
        ValueError: If path cannot be resolved
    """
    if not res:
        raise ValueError("Path cannot be empty")

    try:
        path_obj = Path(res).resolve()
        abs_path = str(path_obj)
        logger.debug(f"Resolved path: {res} -> {abs_path}")
        return abs_path
    except (OSError, ValueError) as e:
        logger.error(f"Failed to resolve path '{res}': {e}")
        raise ValueError(f"Cannot resolve path: {res}") from e


def format_content(role: str, content: str) -> str:
    """
    Format content with role prefix for each line.

    Args:
        role: Role prefix to add to each line
        content: Content to format

    Returns:
        Formatted content with role prefix
    """
    if not content:
        return ""

    if not role:
        return content

    try:
        formatted_lines = [f"{role} {line}" for line in content.splitlines()]
        return "\n".join(formatted_lines)
    except (AttributeError, TypeError) as e:
        logger.error(f"Error formatting content: {e}")
        return content


def format_messages(messages: List[Dict[str, Any]], title: Optional[str] = None) -> str:
    """
    Format chat messages for display with enhanced error handling.

    Args:
        messages: List of message dictionaries
        title: Optional title for the formatted output

    Returns:
        Formatted message string
    """
    if not messages:
        return title.upper() + " " + "*" * 50 if title else ""

    output = []

    try:
        if title:
            output.append(f"{title.upper()} {'*' * 50}")

        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                logger.warning(f"Invalid message format at index {i}: {type(msg)}")
                output.append(f"INVALID MESSAGE {i}: {str(msg)[:100]}")
                continue

            output.append("-------")

            role = msg.get("role", "UNKNOWN").upper()
            content = msg.get("content")

            if isinstance(content, list):  # Handle list content (e.g., image messages)
                for item in content:
                    try:
                        if isinstance(item, dict):
                            for key, value in item.items():
                                if isinstance(value, dict) and "url" in value:
                                    output.append(f"{role} {key.capitalize()} URL: {value['url']}")
                                else:
                                    output.append(f"{role} {key}: {value}")
                        else:
                            output.append(f"{role} {item}")
                    except Exception as e:
                        logger.warning(f"Error formatting list content item: {e}")
                        output.append(f"{role} ERROR_FORMATTING_ITEM: {str(item)[:50]}")

            elif isinstance(content, str):  # Handle string content
                try:
                    output.append(format_content(role, content))
                except Exception as e:
                    logger.warning(f"Error formatting string content: {e}")
                    output.append(f"{role} ERROR_FORMATTING_CONTENT")

            elif content is not None:
                output.append(f"{role} {str(content)}")

            # Handle function calls
            function_call = msg.get("function_call")
            if function_call:
                try:
                    output.append(f"{role} Function Call: {function_call}")
                except Exception as e:
                    logger.warning(f"Error formatting function call: {e}")
                    output.append(f"{role} Function Call: ERROR_FORMATTING")

        return "\n".join(output)

    except Exception as e:
        logger.error(f"Error formatting messages: {e}")
        return f"ERROR FORMATTING MESSAGES: {str(e)}"


def show_messages(
    messages: List[Dict[str, Any]],
    title: Optional[str] = None,
    functions: Optional[Any] = None
) -> None:
    """
    Display formatted messages with optional function information.

    Args:
        messages: List of message dictionaries to format and display
        title: Optional title for the output
        functions: Optional function information to dump
    """
    try:
        formatted_output = format_messages(messages, title)
        print(formatted_output)

        if functions:
            try:
                dump(functions)
            except Exception as e:
                logger.error(f"Error dumping functions: {e}")
                print(f"ERROR DUMPING FUNCTIONS: {e}")

    except Exception as e:
        logger.error(f"Error showing messages: {e}")
        print(f"ERROR SHOWING MESSAGES: {e}")


def split_chat_history_markdown(text: str, include_tool: bool = False) -> List[Dict[str, str]]:
    """
    Parse chat history markdown into structured messages with improved error handling.

    Args:
        text: Markdown text to parse
        include_tool: Whether to include tool messages in output

    Returns:
        List of message dictionaries with role and content
    """
    if not text or not isinstance(text, str):
        logger.warning("Invalid or empty text provided for markdown parsing")
        return []

    messages = []
    user = []
    assistant = []
    tool = []

    try:
        lines = text.splitlines(keepends=True)

        def append_msg(role: str, lines: List[str]) -> None:
            """Safely append message if content exists."""
            try:
                content = "".join(lines).strip()
                if content:
                    messages.append({"role": role, "content": content})
            except Exception as e:
                logger.warning(f"Error appending message for role {role}: {e}")

        for line_num, line in enumerate(lines, 1):
            try:
                if line.startswith("# "):
                    continue

                if line.startswith("> "):
                    append_msg("assistant", assistant)
                    assistant = []
                    append_msg("user", user)
                    user = []
                    tool.append(line[2:])
                    continue

                if line.startswith("#### "):
                    append_msg("assistant", assistant)
                    assistant = []
                    append_msg("tool", tool)
                    tool = []

                    content = line[5:]
                    user.append(content)
                    continue

                # Default: add to assistant after clearing user and tool
                append_msg("user", user)
                user = []
                append_msg("tool", tool)
                tool = []

                assistant.append(line)

            except Exception as e:
                logger.warning(f"Error processing line {line_num}: {e}")
                continue

        # Final cleanup
        append_msg("assistant", assistant)
        append_msg("user", user)
        append_msg("tool", tool)

        if not include_tool:
            messages = [m for m in messages if m["role"] != "tool"]

        logger.debug(f"Parsed {len(messages)} messages from markdown")
        return messages

    except Exception as e:
        logger.error(f"Error splitting chat history markdown: {e}")
        return []


def get_pip_install(args: List[str]) -> List[str]:
    """
    Build pip install command with safe defaults and validation.

    Args:
        args: Additional arguments for pip install

    Returns:
        Complete pip install command as list

    Raises:
        ValueError: If args are invalid
    """
    if not args:
        raise ValueError("No packages specified for installation")

    if not isinstance(args, list):
        raise ValueError("Arguments must be provided as a list")

    # Validate each argument for basic safety
    for arg in args:
        if not isinstance(arg, str):
            raise ValueError(f"All arguments must be strings, got: {type(arg)}")
        if arg.strip() != arg or not arg:
            raise ValueError(f"Invalid argument: '{arg}'")

    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--upgrade-strategy",
        "only-if-needed",
        "--no-warn-script-location",  # Reduce noise
    ]
    cmd.extend(args)

    logger.debug(f"Built pip install command: {' '.join(cmd)}")
    return cmd


def run_install(cmd: List[str]) -> Tuple[bool, str]:
    """
    Run package installation with enhanced error handling and progress indication.

    Args:
        cmd: Installation command as list of strings

    Returns:
        Tuple of (success, output_string)
    """
    if not cmd or not isinstance(cmd, list):
        error_msg = "Invalid command provided to run_install"
        logger.error(error_msg)
        return False, error_msg

    print()
    try:
        print("Installing:", printable_shell_command(cmd))
    except Exception as e:
        logger.warning(f"Error formatting command for display: {e}")
        print(f"Installing: {' '.join(cmd[:3])}...")

    # First ensure pip is available
    ensurepip_cmd = [sys.executable, "-m", "ensurepip", "--upgrade"]
    try:
        result = subprocess.run(
            ensurepip_cmd,
            capture_output=True,
            check=False,
            timeout=30,
            text=True
        )
        if result.returncode == 0:
            logger.debug("ensurepip completed successfully")
    except (subprocess.TimeoutExpired, Exception) as e:
        logger.debug(f"ensurepip step failed or timed out: {e}")
        # Continue anyway

    output_lines = []
    spinner = None
    process = None

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            encoding=sys.stdout.encoding or 'utf-8',
            errors="replace",
        )

        spinner = Spinner("Installing...")

        while True:
            try:
                char = process.stdout.read(1)
                if not char:
                    break

                output_lines.append(char)
                spinner.step()

            except Exception as e:
                logger.warning(f"Error reading installation output: {e}")
                break

        return_code = process.wait()
        output = "".join(output_lines)

        if return_code == 0:
            print("Installation complete.")
            print()
            logger.info("Package installation successful")
            return True, output
        else:
            logger.error(f"Installation failed with return code: {return_code}")

    except subprocess.CalledProcessError as e:
        error_msg = f"Error running pip install: {e}"
        logger.error(error_msg)
        print(f"\n{error_msg}")
        return False, str(e)

    except Exception as e:
        error_msg = f"Unexpected error during installation: {e}"
        logger.error(error_msg, exc_info=True)
        print(f"\n{error_msg}")
        return False, str(e)

    finally:
        if spinner:
            try:
                spinner.end()
            except Exception:
                pass

        if process:
            try:
                process.terminate()
            except Exception:
                pass

    print("\nInstallation failed.\n")
    return False, "".join(output_lines)


def find_common_root(abs_fnames: Union[Set[str], List[str], Tuple[str, ...]]) -> str:
    """
    Find common root path for a collection of absolute filenames.

    Args:
        abs_fnames: Collection of absolute file paths

    Returns:
        Common root path as string
    """
    if not abs_fnames:
        logger.debug("No filenames provided, using current directory")
        try:
            return safe_abs_path(os.getcwd())
        except (FileNotFoundError, ValueError):
            return "."

    try:
        fname_list = list(abs_fnames)

        # Validate all paths are strings
        for fname in fname_list:
            if not isinstance(fname, (str, Path)):
                logger.warning(f"Invalid filename type: {type(fname)}")
                fname_list = [str(f) for f in fname_list]
                break

        if len(fname_list) == 1:
            result = safe_abs_path(os.path.dirname(fname_list[0]))
            logger.debug(f"Single file, using parent directory: {result}")
            return result
        elif fname_list:
            result = safe_abs_path(os.path.commonpath(fname_list))
            logger.debug(f"Found common path for {len(fname_list)} files: {result}")
            return result

    except (OSError, ValueError) as e:
        logger.warning(f"Error finding common path: {e}")

    # Fallback to current directory
    try:
        result = safe_abs_path(os.getcwd())
        logger.debug(f"Using current directory as fallback: {result}")
        return result
    except (FileNotFoundError, ValueError):
        logger.warning("Cannot access current directory, using '.'")
        return "."


def format_tokens(count: Union[int, float]) -> str:
    """
    Format token count in a human-readable way with validation.

    Args:
        count: Number of tokens to format

    Returns:
        Formatted token count string

    Raises:
        ValueError: If count is negative or invalid
    """
    if not isinstance(count, (int, float)):
        try:
            count = float(count)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Token count must be a number, got: {type(count)}") from e

    if count < 0:
        raise ValueError(f"Token count cannot be negative: {count}")

    # Handle very large numbers
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    elif count >= 1_000:
        if count < 10_000:
            return f"{count / 1_000:.1f}k"
        else:
            return f"{round(count / 1_000)}k"
    else:
        return f"{int(count)}"


def touch_file(fname: Union[str, Path]) -> bool:
    """
    Create a file and its parent directories if they don't exist.

    Args:
        fname: File path to create

    Returns:
        True if successful, False otherwise
    """
    if not fname:
        logger.error("Empty filename provided to touch_file")
        return False

    try:
        fname_path = Path(fname)

        # Create parent directories
        fname_path.parent.mkdir(parents=True, exist_ok=True)

        # Touch the file
        fname_path.touch(exist_ok=True)

        logger.debug(f"Successfully touched file: {fname_path}")
        return True

    except (OSError, PermissionError) as e:
        logger.error(f"Failed to touch file '{fname}': {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error touching file '{fname}': {e}")
        return False


def check_pip_install_extra(
    io,
    module: Optional[str],
    prompt: Optional[str],
    pip_install_cmd: List[str],
    self_update: bool = False
) -> Optional[bool]:
    """
    Check if a module is available, and optionally install it with enhanced error handling.

    Args:
        io: InputOutput object for user interaction
        module: Module name to check for import
        prompt: User prompt message
        pip_install_cmd: Pip install command arguments
        self_update: Whether this is a self-update operation

    Returns:
        True if module is available/installed, False if failed, None if user declined
    """
    # Validate inputs
    if not pip_install_cmd or not isinstance(pip_install_cmd, list):
        logger.error("Invalid pip_install_cmd provided")
        io.tool_error("Invalid installation command provided")
        return False

    # Check if module is already available
    if module:
        try:
            __import__(module)
            logger.debug(f"Module {module} is already available")
            return True
        except (ImportError, ModuleNotFoundError, RuntimeError) as e:
            logger.debug(f"Module {module} not available: {e}")

    try:
        cmd = get_pip_install(pip_install_cmd)
    except (ValueError, TypeError) as e:
        logger.error(f"Failed to build pip install command: {e}")
        io.tool_error(f"Invalid installation command: {e}")
        return False

    if prompt:
        try:
            io.tool_warning(prompt)
        except Exception as e:
            logger.warning(f"Error showing prompt: {e}")

    # Handle Windows self-update case
    if self_update and platform.system() == "Windows":
        try:
            io.tool_output("Run this command to update:")
            print()
            print(printable_shell_command(cmd))
            return None
        except Exception as e:
            logger.error(f"Error showing Windows update command: {e}")
            return False

    # Ask user for confirmation
    try:
        cmd_display = printable_shell_command(cmd)
    except Exception as e:
        logger.warning(f"Error formatting command for display: {e}")
        cmd_display = " ".join(cmd[:3]) + "..."

    try:
        if not io.confirm_ask("Run pip install?", default="y", subject=cmd_display):
            logger.debug("User declined installation")
            return None
    except Exception as e:
        logger.error(f"Error getting user confirmation: {e}")
        io.tool_error("Failed to get user confirmation")
        return False

    # Perform installation
    try:
        success, output = run_install(cmd)
    except Exception as e:
        logger.error(f"Error during installation: {e}")
        io.tool_error(f"Installation error: {e}")
        return False

    if success:
        # Verify installation if module specified
        if module:
            try:
                __import__(module)
                logger.info(f"Successfully installed and verified module: {module}")
                return True
            except (ImportError, ModuleNotFoundError, RuntimeError) as err:
                logger.error(f"Module installation verification failed: {err}")
                io.tool_error(f"Installation verification failed: {err}")
                # Fall through to show manual command
            except Exception as e:
                logger.error(f"Unexpected error verifying installation: {e}")
                io.tool_error(f"Verification error: {e}")
        else:
            logger.info("Installation completed successfully")
            return True

    # Installation failed - show error and manual command
    try:
        io.tool_error(output)
        print()
        print("Install failed, try running this command manually:")
        print(printable_shell_command(cmd))
    except Exception as e:
        logger.error(f"Error showing installation failure message: {e}")

    return False


def printable_shell_command(cmd_list: List[str]) -> str:
    """
    Convert a list of command arguments to a properly shell-escaped string with validation.

    Args:
        cmd_list: List of command arguments

    Returns:
        Shell-escaped command string

    Raises:
        ValueError: If cmd_list is invalid
        TypeError: If cmd_list contains non-string elements
    """
    if not cmd_list:
        raise ValueError("Command list cannot be empty")

    if not isinstance(cmd_list, (list, tuple)):
        raise TypeError(f"Expected list or tuple, got {type(cmd_list)}")

    # Validate all elements are strings
    for i, arg in enumerate(cmd_list):
        if not isinstance(arg, str):
            raise TypeError(f"All command arguments must be strings, got {type(arg)} at index {i}")
        if not arg:  # Check for empty strings
            logger.warning(f"Empty string argument at index {i}")

    try:
        result = oslex.join(cmd_list)
        logger.debug(f"Formatted shell command: {result[:100]}{'...' if len(result) > 100 else ''}")
        return result
    except Exception as e:
        logger.error(f"Error formatting shell command: {e}")
        # Fallback to simple space-separated join
        return " ".join(f'"{arg}"' if " " in arg else arg for arg in cmd_list)


# Async utilities for future use
async def async_run_command(
    command: str,
    cwd: Optional[Union[str, Path]] = None,
    timeout: Optional[float] = None
) -> Tuple[int, str]:
    """
    Asynchronously run a command with timeout support.

    Args:
        command: Command to execute
        cwd: Working directory
        timeout: Timeout in seconds

    Returns:
        Tuple of (return_code, output)
    """
    try:
        logger.debug(f"Running async command: {command}")

        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(cwd) if cwd else None,
            env=os.environ.copy()
        )

        try:
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=timeout)
            output = stdout.decode('utf-8', errors='replace') if stdout else ""
            return_code = process.returncode or 0

            logger.debug(f"Async command completed with return code: {return_code}")
            return return_code, output

        except asyncio.TimeoutError:
            logger.warning(f"Async command timed out after {timeout} seconds")
            process.kill()
            await process.wait()
            return 124, f"Command timed out after {timeout} seconds"

    except Exception as e:
        logger.error(f"Error in async command execution: {e}")
        return 1, f"Error executing command: {e}"


@lru_cache(maxsize=32)
def get_file_extension(filename: Union[str, Path]) -> str:
    """
    Get file extension with caching for performance.

    Args:
        filename: File name or path

    Returns:
        File extension (including dot) or empty string
    """
    try:
        return Path(filename).suffix.lower()
    except Exception:
        return ""


def batch_file_operations(
    file_paths: List[Union[str, Path]],
    operation: Callable[[Path], bool],
    max_workers: Optional[int] = None
) -> List[Tuple[str, bool]]:
    """
    Perform file operations in parallel for better performance.

    Args:
        file_paths: List of file paths to process
        operation: Function to apply to each file path
        max_workers: Maximum number of worker threads

    Returns:
        List of (file_path, success) tuples
    """
    import concurrent.futures

    if not file_paths:
        return []

    results = []
    max_workers = max_workers or min(len(file_paths), os.cpu_count() or 1)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {
                executor.submit(operation, Path(path)): str(path)
                for path in file_paths
            }

            for future in concurrent.futures.as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    success = future.result()
                    results.append((path, success))
                except Exception as e:
                    logger.error(f"Error processing {path}: {e}")
                    results.append((path, False))

    except Exception as e:
        logger.error(f"Error in batch file operations: {e}")
        # Fallback to sequential processing
        for path in file_paths:
            try:
                success = operation(Path(path))
                results.append((str(path), success))
            except Exception as op_error:
                logger.error(f"Error processing {path}: {op_error}")
                results.append((str(path), False))

    return results
