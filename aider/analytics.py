import json
import logging
import platform
import sys
import time
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Union, Any

from mixpanel import MixpanelException
from posthog import Posthog

from aider import __version__
from aider.dump import dump  # noqa: F401
from aider import models

PERCENT = 10

# Setup module logger
logger = logging.getLogger(__name__)

@lru_cache(maxsize=128)
def compute_hex_threshold(percent: float) -> str:
    """Convert percentage to 6-digit hex threshold.

    Args:
        percent: Percentage threshold (0-100)

    Returns:
        str: 6-digit hex threshold

    Raises:
        ValueError: If percent is not between 0 and 100
    """
    if not (0 <= percent <= 100):
        raise ValueError(f"Percentage must be between 0 and 100, got {percent}")

    return format(int(0xFFFFFF * percent / 100), "06x")


def is_uuid_in_percentage(uuid_str: Optional[str], percent: float) -> bool:
    """Check if a UUID string falls within the first X percent of the UUID space.

    Args:
        uuid_str: UUID string to test
        percent: Percentage threshold (0-100)

    Returns:
        bool: True if UUID falls within the first X percent

    Raises:
        ValueError: If percent is not between 0 and 100
    """
    if not (0 <= percent <= 100):
        raise ValueError(f"Percentage must be between 0 and 100, got {percent}")

    if not uuid_str or percent == 0:
        return False

    # Validate UUID format
    if len(uuid_str) < 6:
        logger.warning(f"UUID string too short: {uuid_str[:10]}...")
        return False

    try:
        threshold = compute_hex_threshold(percent)
        return uuid_str[:6].lower() <= threshold.lower()
    except (IndexError, TypeError) as e:
        logger.error(f"Error processing UUID {uuid_str[:10]}...: {e}")
        return False


mixpanel_project_token = "6da9a43058a5d1b9f3353153921fb04d"
posthog_project_api_key = "phc_99T7muzafUMMZX15H8XePbMSreEUzahHbtWjy3l5Qbv"
posthog_host = "https://us.i.posthog.com"


class Analytics:
    """Analytics handler with improved error handling and performance."""

    def __init__(
        self,
        logfile: Optional[str] = None,
        permanently_disable: bool = False,
        posthog_host: Optional[str] = None,
        posthog_project_api_key: Optional[str] = None,
    ) -> None:
        # providers
        self.mp = None
        self.ph = None

        # saved state
        self.user_id: Optional[str] = None
        self.permanently_disable: Optional[bool] = None
        self.asked_opt_in: Optional[bool] = None

        # ephemeral
        self.logfile = logfile
        self.custom_posthog_host = posthog_host
        self.custom_posthog_project_api_key = posthog_project_api_key

        # Cache for system info to avoid repeated computation
        self._system_info_cache: Optional[Dict[str, str]] = None

        try:
            self.get_or_create_uuid()
        except Exception as e:
            logger.error(f"Failed to get or create UUID: {e}")
            self.disable(permanently_disable)
            return

        if self.permanently_disable or permanently_disable or not self.asked_opt_in:
            self.disable(permanently_disable)

    def enable(self) -> None:
        """Enable analytics with improved error handling."""
        if not self.user_id:
            logger.debug("No user ID available, disabling analytics")
            self.disable(False)
            return

        if self.permanently_disable:
            logger.debug("Analytics permanently disabled")
            self.disable(True)
            return

        if not self.asked_opt_in:
            logger.debug("User has not opted in, disabling analytics")
            self.disable(False)
            return

        try:
            # Initialize PostHog with error handling
            self.ph = Posthog(
                project_api_key=self.custom_posthog_project_api_key or posthog_project_api_key,
                host=self.custom_posthog_host or posthog_host,
                on_error=self.posthog_error,
                enable_exception_autocapture=True,
                super_properties=self.get_system_info(),
            )
            logger.debug("Analytics enabled successfully")
        except Exception as e:
            logger.error(f"Failed to enable analytics: {e}")
            self.disable(False)

    def disable(self, permanently: bool) -> None:
        """Disable analytics with proper cleanup."""
        try:
            # Clean shutdown of analytics providers
            if self.ph:
                try:
                    self.ph.shutdown()
                except Exception as e:
                    logger.debug(f"Error shutting down PostHog: {e}")
                finally:
                    self.ph = None

            self.mp = None

            if permanently:
                self.asked_opt_in = True
                self.permanently_disable = True
                try:
                    self.save_data()
                except Exception as e:
                    logger.error(f"Failed to save analytics disable state: {e}")

        except Exception as e:
            logger.error(f"Error during analytics disable: {e}")
            # Ensure providers are None even if cleanup fails
            self.mp = None
            self.ph = None

    def need_to_ask(self, args_analytics: Optional[bool]) -> bool:
        """Determine if we need to ask user about analytics opt-in."""
        if args_analytics is False:
            return False

        could_ask = not self.asked_opt_in and not self.permanently_disable
        if not could_ask:
            return False

        if args_analytics is True:
            return True

        if args_analytics is not None:
            logger.warning(f"Unexpected args_analytics value: {args_analytics}")
            return False

        if not self.user_id:
            return False

        try:
            return is_uuid_in_percentage(self.user_id, PERCENT)
        except Exception as e:
            logger.error(f"Error checking UUID percentage: {e}")
            return False

    def get_data_file_path(self) -> Optional[Path]:
        """Get analytics data file path with improved error handling."""
        try:
            data_file = Path.home() / ".aider" / "analytics.json"
            data_file.parent.mkdir(parents=True, exist_ok=True)
            return data_file
        except (OSError, RuntimeError) as e:
            logger.error(f"Cannot access analytics data directory: {e}")
            # If we can't create/access the directory, disable analytics
            try:
                self.disable(permanently=False)
            except Exception as disable_error:
                logger.error(f"Error disabling analytics: {disable_error}")
            return None

    def get_or_create_uuid(self) -> None:
        """Get or create user UUID with error handling."""
        try:
            self.load_data()
            if self.user_id:
                return

            self.user_id = str(uuid.uuid4())
            self.save_data()
        except Exception as e:
            logger.error(f"Failed to get or create UUID: {e}")
            # Disable analytics if we can't get a UUID
            self.disable(permanently=False)

    def load_data(self) -> None:
        """Load analytics data with improved error handling."""
        data_file = self.get_data_file_path()
        if not data_file:
            return

        if not data_file.exists():
            return

        try:
            with data_file.open('r', encoding='utf-8') as f:
                data = json.load(f)

            self.permanently_disable = data.get("permanently_disable")
            self.user_id = data.get("uuid")
            self.asked_opt_in = data.get("asked_opt_in", False)

            logger.debug(f"Loaded analytics data: opt_in={self.asked_opt_in}, disabled={self.permanently_disable}")

        except (json.JSONDecodeError, OSError, KeyError) as e:
            logger.error(f"Failed to load analytics data: {e}")
            # Reset to defaults on corruption
            self.permanently_disable = None
            self.user_id = None
            self.asked_opt_in = None
            self.disable(permanently=False)

    def save_data(self) -> None:
        """Save analytics data with atomic write and error handling."""
        data_file = self.get_data_file_path()
        if not data_file:
            return

        data = {
            "uuid": self.user_id,
            "permanently_disable": self.permanently_disable,
            "asked_opt_in": self.asked_opt_in,
        }

        try:
            # Use atomic write by writing to temp file first
            temp_file = data_file.with_suffix('.tmp')
            with temp_file.open('w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)

            # Atomic rename
            temp_file.replace(data_file)
            logger.debug("Analytics data saved successfully")

        except (OSError, json.JSONEncodeError) as e:
            logger.error(f"Failed to save analytics data: {e}")
            # Clean up temp file if it exists
            try:
                temp_file = data_file.with_suffix('.tmp')
                if temp_file.exists():
                    temp_file.unlink()
            except:
                pass
            # If we can't write the file, disable analytics
            self.disable(permanently=False)

    def get_system_info(self) -> Dict[str, str]:
        """Get system information with caching and error handling."""
        if self._system_info_cache is not None:
            return self._system_info_cache

        try:
            self._system_info_cache = {
                "python_version": sys.version.split()[0] if sys.version else "unknown",
                "os_platform": platform.system(),
                "os_release": platform.release(),
                "machine": platform.machine(),
                "aider_version": __version__,
            }
        except Exception as e:
            logger.error(f"Error getting system info: {e}")
            # Fallback to minimal info
            self._system_info_cache = {
                "python_version": "unknown",
                "os_platform": "unknown",
                "os_release": "unknown",
                "machine": "unknown",
                "aider_version": __version__,
            }

        return self._system_info_cache

    def _redact_model_name(self, model) -> Optional[str]:
        """Redact model name for privacy with improved error handling."""
        if not model:
            return None

        try:
            if not hasattr(model, 'name'):
                logger.warning(f"Model object has no name attribute: {type(model)}")
                return None

            model_name = model.name
            if not model_name:
                return None

            info = models.model_info_manager.get_model_from_cached_json_db(model_name)
            if info:
                return model_name
            elif "/" in model_name:
                return model_name.split("/")[0] + "/REDACTED"
            return None

        except Exception as e:
            logger.error(f"Error redacting model name: {e}")
            return "REDACTED"

    def posthog_error(self, error=None, **kwargs) -> None:
        """Disable PostHog on error with proper logging."""
        try:
            logger.error(f"PostHog error occurred: {error}, kwargs: {kwargs}")
            # https://github.com/PostHog/posthog-python/blob/9e1bb8c58afaa229da24c4fb576c08bb88a75752/posthog/consumer.py#L86
            # https://github.com/Aider-AI/aider/issues/2532
            if self.ph:
                try:
                    self.ph.shutdown()
                except Exception as shutdown_error:
                    logger.debug(f"Error shutting down PostHog after error: {shutdown_error}")
                finally:
                    self.ph = None
        except Exception as e:
            logger.error(f"Error in posthog_error handler: {e}")
            self.ph = None

    def event(self, event_name: str, main_model=None, **kwargs) -> None:
        """Record analytics event with improved error handling and validation."""
        if not self.mp and not self.ph and not self.logfile:
            return

        if not event_name:
            logger.warning("Empty event name provided")
            return

        try:
            properties = {}

            # Process model information with error handling
            if main_model:
                try:
                    properties["main_model"] = self._redact_model_name(main_model)
                    if hasattr(main_model, 'weak_model'):
                        properties["weak_model"] = self._redact_model_name(main_model.weak_model)
                    if hasattr(main_model, 'editor_model'):
                        properties["editor_model"] = self._redact_model_name(main_model.editor_model)
                except Exception as e:
                    logger.error(f"Error processing model info for event {event_name}: {e}")

            # Add user-provided properties
            properties.update(kwargs)

            # Sanitize property values
            sanitized_properties = {}
            for key, value in properties.items():
                try:
                    if isinstance(value, (int, float, bool)):
                        sanitized_properties[key] = value
                    elif value is None:
                        sanitized_properties[key] = None
                    else:
                        # Convert to string and limit length to prevent memory issues
                        str_value = str(value)
                        sanitized_properties[key] = str_value[:1000] if len(str_value) > 1000 else str_value
                except Exception as e:
                    logger.warning(f"Error sanitizing property {key}: {e}")
                    sanitized_properties[key] = "ERROR_SANITIZING"

            # Send to analytics providers
            if self.mp:
                try:
                    self.mp.track(self.user_id, event_name, dict(sanitized_properties))
                except MixpanelException as e:
                    logger.error(f"Mixpanel error for event {event_name}: {e}")
                    self.mp = None
                except Exception as e:
                    logger.error(f"Unexpected error sending to Mixpanel: {e}")
                    self.mp = None

            if self.ph:
                try:
                    self.ph.capture(self.user_id, event_name, dict(sanitized_properties))
                except Exception as e:
                    logger.error(f"PostHog error for event {event_name}: {e}")
                    # Don't disable PostHog here as posthog_error will handle it

            # Log to file if specified
            if self.logfile:
                self._log_event_to_file(event_name, sanitized_properties)

        except Exception as e:
            logger.error(f"Unexpected error in event tracking for {event_name}: {e}")

    def _log_event_to_file(self, event_name: str, properties: Dict[str, Any]) -> None:
        """Log event to file with proper error handling."""
        try:
            log_entry = {
                "event": event_name,
                "properties": properties,
                "user_id": self.user_id,
                "timestamp": time.time(),
                "iso_timestamp": time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime()),
            }

            with open(self.logfile, "a", encoding='utf-8') as f:
                json.dump(log_entry, f, ensure_ascii=False)
                f.write("\n")

        except (OSError, json.JSONEncodeError) as e:
            logger.error(f"Failed to log event {event_name} to file: {e}")
        except Exception as e:
            logger.error(f"Unexpected error logging event {event_name}: {e}")


if __name__ == "__main__":
    dump(compute_hex_threshold(PERCENT))
