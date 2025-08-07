import asyncio
import logging
import os
import platform
import shlex
import subprocess
import sys
import threading
import time
from io import BytesIO
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import pexpect
import psutil

# Setup module logger
logger = logging.getLogger(__name__)


def run_cmd(
    command: str,
    verbose: bool = False,
    error_print: Optional[Callable[[str], None]] = None,
    cwd: Optional[Union[str, Path]] = None,
    timeout: Optional[float] = None,
) -> Tuple[int, str]:
    """
    Run a command with improved error handling and platform detection.

    Args:
        command: Command to execute
        verbose: Enable verbose output
        error_print: Custom error printing function
        cwd: Working directory for command execution
        timeout: Timeout in seconds for command execution

    Returns:
        Tuple of (return_code, output)
    """
    if not command or not command.strip():
        error_message = "Empty command provided"
        logger.error(error_message)
        if error_print:
            error_print(error_message)
        else:
            print(error_message, file=sys.stderr)
        return 1, error_message

    # Validate working directory
    if cwd:
        cwd_path = Path(cwd)
        if not cwd_path.exists():
            error_message = f"Working directory does not exist: {cwd}"
            logger.error(error_message)
            if error_print:
                error_print(error_message)
            else:
                print(error_message, file=sys.stderr)
            return 1, error_message

    try:
        # Choose execution method based on platform and capabilities
        if (sys.stdin.isatty() and
            hasattr(pexpect, "spawn") and
            platform.system() != "Windows" and
            not timeout):  # pexpect doesn't handle timeouts well
            return run_cmd_pexpect(command, verbose, cwd)

        return run_cmd_subprocess(command, verbose, cwd, timeout=timeout)

    except OSError as e:
        error_message = f"System error running command '{command}': {e}"
        logger.error(error_message)
        if error_print:
            error_print(error_message)
        else:
            print(error_message, file=sys.stderr)
        return 1, error_message
    except Exception as e:
        error_message = f"Unexpected error running command '{command}': {e}"
        logger.error(error_message, exc_info=True)
        if error_print:
            error_print(error_message)
        else:
            print(error_message, file=sys.stderr)
        return 1, error_message


def get_windows_parent_process_name() -> Optional[str]:
    """Get the name of the parent Windows shell process.

    Returns:
        Name of parent process if it's a known shell, None otherwise
    """
    try:
        current_process = psutil.Process()
        max_iterations = 10  # Prevent infinite loops
        iterations = 0

        while iterations < max_iterations:
            parent = current_process.parent()
            if parent is None:
                break

            try:
                parent_name = parent.name().lower()
                if parent_name in ["powershell.exe", "cmd.exe", "pwsh.exe"]:
                    logger.debug(f"Found parent shell: {parent_name}")
                    return parent_name
                current_process = parent
                iterations += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break

        return None
    except Exception as e:
        logger.debug(f"Error getting parent process name: {e}")
        return None


def run_cmd_subprocess(
    command: str,
    verbose: bool = False,
    cwd: Optional[Union[str, Path]] = None,
    encoding: Optional[str] = None,
    timeout: Optional[float] = None,
) -> Tuple[int, str]:
    """
    Run command using subprocess with enhanced error handling and features.

    Args:
        command: Command to execute
        verbose: Enable verbose logging
        cwd: Working directory
        encoding: Text encoding for command output
        timeout: Timeout in seconds

    Returns:
        Tuple of (return_code, output)
    """
    # Use safe encoding default
    if encoding is None:
        encoding = sys.stdout.encoding or 'utf-8'

    if verbose:
        logger.info(f"Using run_cmd_subprocess: {command}")

    try:
        # Security: Validate command for basic safety
        if any(dangerous in command.lower() for dangerous in ['rm -rf /', 'del /f /q', 'format']):
            logger.warning(f"Potentially dangerous command detected: {command[:50]}...")

        shell = os.environ.get("SHELL", "/bin/sh")
        parent_process = None
        processed_command = command

        # Determine the appropriate shell and command format
        if platform.system() == "Windows":
            parent_process = get_windows_parent_process_name()
            if parent_process in ["powershell.exe", "pwsh.exe"]:
                # Use PowerShell with proper escaping
                processed_command = f"powershell -NoProfile -Command {shlex.quote(command)}"
            shell = None  # Let subprocess choose on Windows

        if verbose:
            logger.debug(f"Running command: {processed_command}")
            logger.debug(f"SHELL: {shell}")
            logger.debug(f"CWD: {cwd}")
            if platform.system() == "Windows":
                logger.debug(f"Parent process: {parent_process}")

        # Use more robust process creation
        process = subprocess.Popen(
            processed_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.PIPE,
            text=True,
            shell=True,
            encoding=encoding,
            errors="replace",
            bufsize=1,  # Line buffered for better real-time output
            universal_newlines=True,
            cwd=str(cwd) if cwd else None,
            env=os.environ.copy(),  # Inherit environment safely
        )

        # Capture output with timeout support
        output_chunks = []
        start_time = time.time()

        try:
            # Close stdin to avoid hanging on input-expecting commands
            if process.stdin:
                process.stdin.close()

            while True:
                # Check timeout
                if timeout and (time.time() - start_time) > timeout:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()

                    output_str = "".join(output_chunks)
                    error_msg = f"Command timed out after {timeout} seconds"
                    logger.error(error_msg)
                    return 124, output_str + f"\n{error_msg}"  # 124 is timeout exit code

                # Read output with small timeout to check for process completion
                try:
                    chunk = process.stdout.read(1)
                    if not chunk:
                        break

                    # Print in real-time for user feedback
                    print(chunk, end="", flush=True)
                    output_chunks.append(chunk)

                except Exception:
                    # Process might have ended
                    break

            # Wait for process completion
            return_code = process.wait()
            output_str = "".join(output_chunks)

            if verbose:
                logger.debug(f"Command completed with return code: {return_code}")

            return return_code, output_str

        except KeyboardInterrupt:
            logger.info("Command interrupted by user")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            return 130, "".join(output_chunks) + "\nCommand interrupted by user"

    except subprocess.SubprocessError as e:
        error_msg = f"Subprocess error: {e}"
        logger.error(error_msg)
        return 1, error_msg
    except OSError as e:
        error_msg = f"OS error: {e}"
        logger.error(error_msg)
        return 1, error_msg
    except Exception as e:
        error_msg = f"Unexpected error: {e}"
        logger.error(error_msg, exc_info=True)
        return 1, error_msg


def run_cmd_pexpect(
    command: str,
    verbose: bool = False,
    cwd: Optional[Union[str, Path]] = None,
) -> Tuple[int, str]:
    """
    Run a shell command interactively using pexpect with enhanced error handling.

    Args:
        command: The command to run as a string
        verbose: If True, enable verbose logging
        cwd: Working directory for the command

    Returns:
        A tuple containing (exit_status, output)
    """
    if verbose:
        logger.info(f"Using run_cmd_pexpect: {command}")

    output = BytesIO()

    def output_callback(data: bytes) -> bytes:
        """Callback to capture pexpect output."""
        try:
            output.write(data)
            return data
        except Exception as e:
            logger.error(f"Error in output callback: {e}")
            return data

    try:
        # Security validation
        if any(dangerous in command.lower() for dangerous in ['rm -rf /', 'del /f /q', 'format']):
            logger.warning(f"Potentially dangerous command detected: {command[:50]}...")

        # Use the SHELL environment variable, falling back to safe defaults
        shell = os.environ.get("SHELL", "/bin/bash")

        # Validate shell exists and is executable
        if not os.path.exists(shell) or not os.access(shell, os.X_OK):
            shell = "/bin/sh"  # Final fallback

        if not os.path.exists(shell):
            raise OSError(f"No suitable shell found. Tried: {os.environ.get('SHELL')}, /bin/bash, /bin/sh")

        if verbose:
            logger.debug(f"Using shell: {shell}")
            logger.debug(f"Working directory: {cwd}")

        child = None
        try:
            # Determine spawn method based on shell availability
            if os.path.exists(shell) and os.access(shell, os.X_OK):
                if verbose:
                    logger.debug("Running pexpect.spawn with shell")
                child = pexpect.spawn(
                    shell,
                    args=["-i", "-c", command],
                    encoding="utf-8",
                    cwd=str(cwd) if cwd else None,
                    timeout=300,  # 5 minute default timeout
                    env=os.environ.copy()
                )
            else:
                if verbose:
                    logger.debug("Running pexpect.spawn without shell")
                child = pexpect.spawn(
                    command,
                    encoding="utf-8",
                    cwd=str(cwd) if cwd else None,
                    timeout=300,
                    env=os.environ.copy()
                )

            # Set up proper terminal dimensions
            try:
                child.setwinsize(24, 80)  # Standard terminal size
            except Exception as e:
                logger.debug(f"Could not set window size: {e}")

            # Transfer control to the user, capturing output
            child.interact(output_filter=output_callback)

            # Wait for the command to finish and get the exit status
            child.close()

            exit_status = child.exitstatus if child.exitstatus is not None else child.signalstatus
            if exit_status is None:
                exit_status = 1  # Default error code if status unavailable

            output_str = output.getvalue().decode("utf-8", errors="replace")

            if verbose:
                logger.debug(f"pexpect command completed with exit status: {exit_status}")

            return exit_status, output_str

        finally:
            # Ensure child process cleanup
            if child and child.isalive():
                try:
                    child.terminate()
                    child.wait()
                except Exception as e:
                    logger.debug(f"Error cleaning up pexpect process: {e}")

    except pexpect.ExceptionPexpect as e:
        error_msg = f"Pexpect error running command '{command}': {e}"
        logger.error(error_msg)
        return 1, error_msg
    except (OSError, IOError) as e:
        error_msg = f"System error running command '{command}': {e}"
        logger.error(error_msg)
        return 1, error_msg
    except Exception as e:
        error_msg = f"Unexpected error running command '{command}': {e}"
        logger.error(error_msg, exc_info=True)
        return 1, error_msg
