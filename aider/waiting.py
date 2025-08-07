#!/usr/bin/env python

"""
Thread-based, killable spinner utility with enhanced error handling and performance.

Use it like:

    from aider.waiting import WaitingSpinner

    spinner = WaitingSpinner("Waiting for LLM")
    spinner.start()
    ...  # long task
    spinner.stop()
"""

import logging
import sys
import threading
import time
from typing import Optional, List

from rich.console import Console

# Setup module logger
logger = logging.getLogger(__name__)


class Spinner:
    """
    Minimal spinner that scans a single marker back and forth across a line.

    The animation is pre-rendered into a list of frames.  If the terminal
    cannot display unicode the frames are converted to plain ASCII.
    """

    last_frame_idx = 0  # Class variable to store the last frame index
    _unicode_support_cache: Optional[bool] = None  # Cache unicode support detection

    def __init__(self, text: str, width: int = 7):
        if not isinstance(text, str):
            raise TypeError(f"Text must be a string, got {type(text)}")

        self.text = text
        self.start_time = time.time()
        self.last_update = 0.0
        self.visible = False
        self._cleanup_done = False

        try:
            self.is_tty = sys.stdout.isatty()
            self.console = Console()
        except Exception as e:
            logger.warning(f"Error initializing console: {e}")
            self.is_tty = False
            self.console = Console(force_terminal=False)

        # Pre-render the animation frames using pure ASCII so they will
        # always display, even on very limited terminals.
        ascii_frames = [
            "#=        ",  # C1 C2 space(8)
            "=#        ",  # C2 C1 space(8)
            " =#       ",  # space(1) C2 C1 space(7)
            "  =#      ",  # space(2) C2 C1 space(6)
            "   =#     ",  # space(3) C2 C1 space(5)
            "    =#    ",  # space(4) C2 C1 space(4)
            "     =#   ",  # space(5) C2 C1 space(3)
            "      =#  ",  # space(6) C2 C1 space(2)
            "       =# ",  # space(7) C2 C1 space(1)
            "        =#",  # space(8) C2 C1
            "        #=",  # space(8) C1 C2
            "       #= ",  # space(7) C1 C2 space(1)
            "      #=  ",  # space(6) C1 C2 space(2)
            "     #=   ",  # space(5) C1 C2 space(3)
            "    #=    ",  # space(4) C1 C2 space(4)
            "   #=     ",  # space(3) C1 C2 space(5)
            "  #=      ",  # space(2) C1 C2 space(6)
            " #=       ",  # space(1) C1 C2 space(7)
        ]

        self.unicode_palette = "░█"
        xlate_from, xlate_to = ("=#", self.unicode_palette)

        # If unicode is supported, swap the ASCII chars for nicer glyphs.
        if self._supports_unicode():
            translation_table = str.maketrans(xlate_from, xlate_to)
            frames = [f.translate(translation_table) for f in ascii_frames]
            self.scan_char = xlate_to[xlate_from.find("#")]
        else:
            frames = ascii_frames
            self.scan_char = "#"

        # Bounce the scanner back and forth.
        self.frames = frames
        self.frame_idx = Spinner.last_frame_idx  # Initialize from class variable
        self.width = len(frames[0]) - 2  # number of chars between the brackets
        self.animation_len = len(frames[0])
        self.last_display_len = 0  # Length of the last spinner line (frame + text)

    def _supports_unicode(self) -> bool:
        # Use cached result if available
        if Spinner._unicode_support_cache is not None:
            return Spinner._unicode_support_cache

        if not self.is_tty:
            Spinner._unicode_support_cache = False
            return False

        try:
            out = self.unicode_palette
            out += "\b" * len(self.unicode_palette)
            out += " " * len(self.unicode_palette)
            out += "\b" * len(self.unicode_palette)
            sys.stdout.write(out)
            sys.stdout.flush()
            Spinner._unicode_support_cache = True
            return True
        except UnicodeEncodeError:
            Spinner._unicode_support_cache = False
            return False
        except Exception as e:
            logger.debug(f"Unicode support detection failed: {e}")
            Spinner._unicode_support_cache = False
            return False

    def _next_frame(self) -> str:
        try:
            frame = self.frames[self.frame_idx]
            self.frame_idx = (self.frame_idx + 1) % len(self.frames)
            Spinner.last_frame_idx = self.frame_idx  # Update class variable
            return frame
        except (IndexError, AttributeError) as e:
            logger.error(f"Error getting next frame: {e}")
            return "..."  # Fallback frame

    def step(self, text: Optional[str] = None) -> None:
        if text is not None:
            if not isinstance(text, str):
                logger.warning(f"Invalid text type: {type(text)}")
                text = str(text)
            self.text = text

        if not self.is_tty or self._cleanup_done:
            return

        try:
            now = time.time()
            if not self.visible and now - self.start_time >= 0.5:
                self.visible = True
                self.last_update = 0.0
                if self.is_tty:
                    try:
                        self.console.show_cursor(False)
                    except Exception as e:
                        logger.debug(f"Error hiding cursor: {e}")

            if not self.visible or now - self.last_update < 0.1:
                return

            self.last_update = now
            frame_str = self._next_frame()
        except Exception as e:
            logger.error(f"Error in spinner step: {e}")
            return

        # Determine the maximum width for the spinner line
        # Subtract 2 as requested, to leave a margin or prevent cursor wrapping issues
        max_spinner_width = self.console.width - 2
        if max_spinner_width < 0:  # Handle extremely narrow terminals
            max_spinner_width = 0

        current_text_payload = f" {self.text}"
        line_to_display = f"{frame_str}{current_text_payload}"

        # Truncate the line if it's too long for the console width
        if len(line_to_display) > max_spinner_width:
            line_to_display = line_to_display[:max_spinner_width]

        len_line_to_display = len(line_to_display)

        # Calculate padding to clear any remnants from a longer previous line
        padding_to_clear = " " * max(0, self.last_display_len - len_line_to_display)

        # Write the spinner frame, text, and any necessary clearing spaces
        sys.stdout.write(f"\r{line_to_display}{padding_to_clear}")
        self.last_display_len = len_line_to_display

        # Calculate number of backspaces to position cursor at the scanner character
        scan_char_abs_pos = frame_str.find(self.scan_char)

        # Total characters written to the line (frame + text + padding)
        total_chars_written_on_line = len_line_to_display + len(padding_to_clear)

        # num_backspaces will be non-positive if scan_char_abs_pos is beyond
        # total_chars_written_on_line (e.g., if the scan char itself was truncated).
        # (e.g., if the scan char itself was truncated).
        # In such cases, (effectively) 0 backspaces are written,
        # and the cursor stays at the end of the line.
        num_backspaces = total_chars_written_on_line - scan_char_abs_pos
        sys.stdout.write("\b" * num_backspaces)
        sys.stdout.flush()

    def end(self) -> None:
        if self._cleanup_done:
            return

        try:
            if self.visible and self.is_tty:
                clear_len = getattr(self, 'last_display_len', 0)
                if clear_len > 0:
                    try:
                        sys.stdout.write("\r" + " " * clear_len + "\r")
                        sys.stdout.flush()
                    except Exception as e:
                        logger.debug(f"Error clearing spinner output: {e}")

                try:
                    self.console.show_cursor(True)
                except Exception as e:
                    logger.debug(f"Error showing cursor: {e}")

            self.visible = False
            self._cleanup_done = True

        except Exception as e:
            logger.error(f"Error ending spinner: {e}")
            # Ensure cleanup is marked as done even if error occurs
            self._cleanup_done = True


class WaitingSpinner:
    """Background spinner that can be started/stopped safely with enhanced error handling."""

    def __init__(self, text: str = "Waiting for LLM", delay: float = 0.15):
        if not isinstance(text, str):
            raise TypeError(f"Text must be a string, got {type(text)}")
        if not isinstance(delay, (int, float)) or delay <= 0:
            raise ValueError(f"Delay must be a positive number, got {delay}")

        try:
            self.spinner = Spinner(text)
            self.delay = max(0.05, min(1.0, delay))  # Clamp delay to reasonable range
            self._stop_event = threading.Event()
            self._thread: Optional[threading.Thread] = None
            self._started = False
            self._lock = threading.Lock()  # For thread safety

        except Exception as e:
            logger.error(f"Error initializing WaitingSpinner: {e}")
            raise

    def _spin(self) -> None:
        """Main spinner loop with enhanced error handling."""
        try:
            while not self._stop_event.is_set():
                try:
                    self.spinner.step()
                except Exception as e:
                    logger.debug(f"Error in spinner step: {e}")
                    # Continue spinning even if step fails

                try:
                    self._stop_event.wait(timeout=self.delay)
                except Exception as e:
                    logger.debug(f"Error in spinner sleep: {e}")
                    time.sleep(self.delay)  # Fallback to regular sleep

        except Exception as e:
            logger.error(f"Error in spinner thread: {e}")
        finally:
            try:
                self.spinner.end()
            except Exception as e:
                logger.debug(f"Error ending spinner in thread: {e}")

    def start(self) -> None:
        """Start the spinner in a background thread with thread safety."""
        with self._lock:
            if self._started:
                logger.debug("Spinner already started")
                return

            try:
                # Create new thread if needed
                if self._thread is None or not self._thread.is_alive():
                    self._stop_event.clear()
                    self._thread = threading.Thread(target=self._spin, daemon=True)

                if not self._thread.is_alive():
                    self._thread.start()
                    self._started = True
                    logger.debug("Spinner started successfully")

            except Exception as e:
                logger.error(f"Error starting spinner: {e}")
                self._started = False

    def stop(self) -> None:
        """Request the spinner to stop and wait briefly for the thread to exit."""
        with self._lock:
            if not self._started:
                return

            try:
                self._stop_event.set()

                if self._thread and self._thread.is_alive():
                    # Wait for thread to finish with reasonable timeout
                    join_timeout = max(self.delay * 2, 0.5)
                    self._thread.join(timeout=join_timeout)

                    if self._thread.is_alive():
                        logger.warning("Spinner thread did not stop gracefully")

                # Ensure spinner is properly ended
                try:
                    self.spinner.end()
                except Exception as e:
                    logger.debug(f"Error ending spinner: {e}")

                self._started = False
                logger.debug("Spinner stopped successfully")

            except Exception as e:
                logger.error(f"Error stopping spinner: {e}")
                # Ensure we're marked as stopped even if cleanup fails
                self._started = False

    # Allow use as a context-manager
    def __enter__(self) -> 'WaitingSpinner':
        """Enter context manager and start spinner."""
        try:
            self.start()
            return self
        except Exception as e:
            logger.error(f"Error entering spinner context: {e}")
            # Try to clean up if start failed
            try:
                self.stop()
            except Exception:
                pass
            raise

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager and stop spinner."""
        try:
            self.stop()
        except Exception as e:
            logger.error(f"Error exiting spinner context: {e}")
            # Don't raise exception in __exit__ unless it's critical


def main():
    """Main function with enhanced error handling and user feedback."""
    spinner = None
    try:
        spinner = Spinner("Running spinner...")
        logger.info("Starting spinner demo")

        for i in range(100):
            try:
                time.sleep(0.15)
                spinner.step(f"Running spinner... {i+1}/100")
            except KeyboardInterrupt:
                print("\nInterrupted by user.")
                break
            except Exception as e:
                logger.warning(f"Error in spinner step {i}: {e}")
                continue

        print("Success!")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        logger.error(f"Error in spinner demo: {e}")
        print(f"Error: {e}")
    finally:
        if spinner:
            try:
                spinner.end()
            except Exception as e:
                logger.error(f"Error ending spinner: {e}")


if __name__ == "__main__":
    main()
