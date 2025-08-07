import asyncio
import logging
import math
import os
import queue
import tempfile
import time
import warnings
from pathlib import Path
from typing import Optional, Tuple

from prompt_toolkit.shortcuts import prompt

from aider.llm import litellm

from .dump import dump  # noqa: F401

# Setup module logger
logger = logging.getLogger(__name__)

warnings.filterwarnings(
    "ignore", message="Couldn't find ffmpeg or avconv - defaulting to ffmpeg, but may not work"
)
warnings.filterwarnings("ignore", category=SyntaxWarning)

# Import audio libraries with better error handling
try:
    from pydub import AudioSegment  # noqa
    from pydub.exceptions import CouldntDecodeError, CouldntEncodeError  # noqa
    PYDUB_AVAILABLE = True
except ImportError as e:
    logger.warning(f"pydub not available: {e}")
    AudioSegment = None
    CouldntDecodeError = Exception
    CouldntEncodeError = Exception
    PYDUB_AVAILABLE = False

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except (OSError, ModuleNotFoundError, ImportError) as e:
    logger.warning(f"soundfile not available: {e}")
    sf = None
    SOUNDFILE_AVAILABLE = False


class SoundDeviceError(Exception):
    """Exception raised when sound device operations fail."""
    pass


class AudioProcessingError(Exception):
    """Exception raised when audio processing fails."""
    pass


class TranscriptionError(Exception):
    """Exception raised when transcription fails."""
    pass


class Voice:
    """Enhanced voice recording and transcription with improved error handling."""

    def __init__(self, audio_format: str = "wav", device_name: Optional[str] = None):
        """
        Initialize Voice recorder with enhanced validation and error handling.

        Args:
            audio_format: Audio format for recording ("wav", "mp3", "webm")
            device_name: Optional specific audio device name to use

        Raises:
            SoundDeviceError: If sound device initialization fails
            ValueError: If audio format is unsupported
        """
        # Validate dependencies
        if not SOUNDFILE_AVAILABLE:
            raise SoundDeviceError("soundfile library is required but not available")

        # Validate audio format
        supported_formats = ["wav", "mp3", "webm"]
        if audio_format not in supported_formats:
            raise ValueError(f"Unsupported audio format: {audio_format}. Supported: {supported_formats}")

        self.audio_format = audio_format
        self.max_rms = 0
        self.min_rms = 1e5
        self.pct = 0
        self.threshold = 0.15

        # Initialize sound device
        try:
            logger.info("Initializing sound device...")
            import sounddevice as sd
            self.sd = sd

            # Query available devices with error handling
            try:
                devices = sd.query_devices()
            except Exception as e:
                logger.error(f"Failed to query audio devices: {e}")
                raise SoundDeviceError(f"Cannot access audio devices: {e}") from e

            self.device_id = self._find_audio_device(devices, device_name)

            logger.info(f"Voice recorder initialized with format: {audio_format}, device: {device_name or 'default'}")

        except (OSError, ModuleNotFoundError, ImportError) as e:
            logger.error(f"Failed to initialize sound device: {e}")
            raise SoundDeviceError(f"Sound device initialization failed: {e}") from e

    def _find_audio_device(self, devices, device_name: Optional[str]) -> Optional[int]:
        """Find and validate audio input device."""
        if not device_name:
            logger.debug("Using default audio device")
            return None

        # Find device by name
        device_id = None
        available_inputs = []

        for i, device in enumerate(devices):
            try:
                if device["max_input_channels"] > 0:
                    available_inputs.append(device["name"])
                    if device_name in device["name"]:
                        device_id = i
                        break
            except (KeyError, TypeError) as e:
                logger.warning(f"Invalid device info at index {i}: {e}")
                continue

        if device_id is None:
            error_msg = (f"Device '{device_name}' not found. "
                        f"Available input devices: {available_inputs}")
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(f"Using input device: {device_name} (ID: {device_id})")
        return device_id

    def callback(self, indata, frames, time, status):
        """Audio callback function with enhanced error handling."""
        try:
            import numpy as np

            if status:
                logger.warning(f"Audio callback status: {status}")

            # Calculate RMS with error handling
            try:
                rms = np.sqrt(np.mean(indata**2))

                # Update RMS statistics
                self.max_rms = max(self.max_rms, rms)
                self.min_rms = min(self.min_rms, rms)

                # Calculate percentage with bounds checking
                rng = self.max_rms - self.min_rms
                if rng > 0.001:
                    self.pct = max(0.0, min(1.0, (rms - self.min_rms) / rng))
                else:
                    self.pct = 0.5

                # Store audio data
                if hasattr(self, 'q') and self.q:
                    self.q.put(indata.copy())

            except Exception as e:
                logger.error(f"Error processing audio data: {e}")
                self.pct = 0.0  # Safe fallback

        except ImportError as e:
            logger.error(f"NumPy not available in callback: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in audio callback: {e}")

    def get_prompt(self) -> str:
        """Generate recording prompt with progress bar and error handling."""
        try:
            num = 10

            # Safely handle percentage calculation
            if math.isnan(self.pct) or self.pct < self.threshold:
                cnt = 0
            else:
                # Ensure count is within valid range
                cnt = max(0, min(num, int(self.pct * num)))

            # Build progress bar with unicode fallback
            try:
                bar = "░" * cnt + "█" * (num - cnt)
                bar = bar[:num]
            except UnicodeEncodeError:
                # ASCII fallback for terminals without unicode support
                bar = "." * cnt + "#" * (num - cnt)
                bar = bar[:num]

            # Calculate duration safely
            try:
                dur = time.time() - self.start_time
                dur = max(0, dur)  # Ensure non-negative
            except (AttributeError, TypeError):
                dur = 0.0

            return f"Recording, press ENTER when done... {dur:.1f}sec {bar}"

        except Exception as e:
            logger.error(f"Error generating prompt: {e}")
            return "Recording, press ENTER when done..."

    def record_and_transcribe(self, history: Optional[str] = None, language: Optional[str] = None) -> Optional[str]:
        """
        Record audio and transcribe it with comprehensive error handling.

        Args:
            history: Previous conversation history for context
            language: Language code for transcription

        Returns:
            Transcribed text or None if failed/cancelled
        """
        try:
            logger.info("Starting audio recording and transcription")
            return self.raw_record_and_transcribe(history, language)

        except KeyboardInterrupt:
            logger.info("Recording cancelled by user")
            print("\nRecording cancelled.")
            return None

        except SoundDeviceError as e:
            logger.error(f"Sound device error: {e}")
            print(f"Audio device error: {e}")
            print("Please ensure you have a working audio input device connected and try again.")
            return None

        except AudioProcessingError as e:
            logger.error(f"Audio processing error: {e}")
            print(f"Audio processing failed: {e}")
            return None

        except TranscriptionError as e:
            logger.error(f"Transcription error: {e}")
            print(f"Transcription failed: {e}")
            return None

        except Exception as e:
            logger.error(f"Unexpected error in record_and_transcribe: {e}", exc_info=True)
            print(f"Recording failed due to unexpected error: {e}")
            return None

    def raw_record_and_transcribe(self, history: Optional[str], language: Optional[str]) -> Optional[str]:
        """Core recording and transcription logic with enhanced error handling."""
        # Initialize recording queue
        self.q = queue.Queue()
        temp_files_to_cleanup = []

        try:
            # Create temporary file with proper cleanup
            temp_wav = tempfile.mktemp(suffix=".wav")
            temp_files_to_cleanup.append(temp_wav)

            # Get sample rate with fallbacks
            sample_rate = self._get_sample_rate()
            logger.debug(f"Using sample rate: {sample_rate}")

            # Record audio
            self.start_time = time.time()
            self._record_audio(sample_rate, temp_wav)

            # Convert to target format if needed
            final_file = self._convert_audio_format(temp_wav, temp_files_to_cleanup)

            # Transcribe audio
            return self._transcribe_audio(final_file, history, language)

        except Exception as e:
            logger.error(f"Error in raw_record_and_transcribe: {e}")
            raise

        finally:
            # Clean up temporary files
            self._cleanup_temp_files(temp_files_to_cleanup)

    def _get_sample_rate(self) -> int:
        """Get sample rate with proper fallbacks."""
        try:
            device_info = self.sd.query_devices(self.device_id, "input")
            sample_rate = int(device_info["default_samplerate"])
            logger.debug(f"Device sample rate: {sample_rate}")
            return sample_rate

        except (TypeError, ValueError, KeyError) as e:
            logger.warning(f"Cannot get device sample rate, using fallback: {e}")
            return 16000  # Standard fallback

        except self.sd.PortAudioError as e:
            logger.error(f"PortAudio error getting sample rate: {e}")
            raise SoundDeviceError("No audio input device detected. Please check your audio settings.") from e

    def _record_audio(self, sample_rate: int, output_file: str) -> None:
        """Record audio to file with error handling."""
        try:
            with self.sd.InputStream(
                samplerate=sample_rate,
                channels=1,
                callback=self.callback,
                device=self.device_id,
                dtype='float32'  # Explicit dtype for better compatibility
            ):
                logger.debug("Started audio recording")
                prompt(self.get_prompt, refresh_interval=0.1)

        except self.sd.PortAudioError as err:
            logger.error(f"PortAudio error during recording: {err}")
            raise SoundDeviceError(f"Error accessing audio input device: {err}") from err

        except Exception as e:
            logger.error(f"Unexpected error during recording: {e}")
            raise AudioProcessingError(f"Recording failed: {e}") from e

        # Write recorded data to file
        try:
            with sf.SoundFile(output_file, mode="x", samplerate=sample_rate, channels=1) as file:
                audio_data_count = 0
                while not self.q.empty():
                    try:
                        data = self.q.get_nowait()
                        file.write(data)
                        audio_data_count += len(data)
                    except queue.Empty:
                        break
                    except Exception as e:
                        logger.warning(f"Error writing audio chunk: {e}")

                logger.debug(f"Wrote {audio_data_count} audio samples to {output_file}")

        except Exception as e:
            logger.error(f"Error writing audio file: {e}")
            raise AudioProcessingError(f"Failed to save recording: {e}") from e

    def _convert_audio_format(self, temp_wav: str, temp_files: list) -> str:
        """Convert audio to target format with size optimization."""
        use_audio_format = self.audio_format

        # Check file size and auto-convert large files
        try:
            file_size = os.path.getsize(temp_wav)
            size_mb = file_size / (1024 * 1024)

            if size_mb > 24.9 and self.audio_format == "wav":
                logger.warning(f"Audio file is {size_mb:.1f}MB, switching to mp3 format")
                print(f"\nWarning: Audio file is too large ({size_mb:.1f}MB), switching to mp3 format.")
                use_audio_format = "mp3"

        except OSError as e:
            logger.warning(f"Cannot check file size: {e}")

        # Convert format if needed
        if use_audio_format != "wav":
            if not PYDUB_AVAILABLE:
                logger.warning("pydub not available, cannot convert audio format")
                print("Warning: Cannot convert audio format, using original WAV file")
                return temp_wav

            try:
                new_filename = tempfile.mktemp(suffix=f".{use_audio_format}")
                temp_files.append(new_filename)

                audio = AudioSegment.from_wav(temp_wav)

                # Optimize export settings based on format
                export_params = {"format": use_audio_format}
                if use_audio_format == "mp3":
                    export_params.update({"bitrate": "128k", "parameters": ["-q:a", "2"]})

                audio.export(new_filename, **export_params)

                logger.info(f"Converted audio from WAV to {use_audio_format}")
                return new_filename

            except (CouldntDecodeError, CouldntEncodeError) as e:
                logger.error(f"Audio format conversion error: {e}")
                print(f"Error converting audio format: {e}")
                return temp_wav  # Fall back to original

            except (OSError, FileNotFoundError) as e:
                logger.error(f"File system error during conversion: {e}")
                print(f"File system error during conversion: {e}")
                return temp_wav

            except Exception as e:
                logger.error(f"Unexpected error during audio conversion: {e}")
                print(f"Unexpected error during audio conversion: {e}")
                return temp_wav

        return temp_wav

    def _transcribe_audio(self, audio_file: str, history: Optional[str], language: Optional[str]) -> Optional[str]:
        """Transcribe audio file with enhanced error handling."""
        try:
            logger.info(f"Starting transcription of {audio_file}")

            with open(audio_file, "rb") as fh:
                # Prepare transcription parameters
                transcribe_params = {
                    "model": "whisper-1",
                    "file": fh
                }

                if history:
                    transcribe_params["prompt"] = history
                if language:
                    transcribe_params["language"] = language

                # Perform transcription
                transcript = litellm.transcription(**transcribe_params)

            if not transcript or not hasattr(transcript, 'text'):
                raise TranscriptionError("Empty or invalid transcript response")

            text = transcript.text.strip()
            if not text:
                logger.warning("Transcription returned empty text")
                return None

            logger.info(f"Transcription successful, {len(text)} characters")
            return text

        except FileNotFoundError as e:
            logger.error(f"Audio file not found for transcription: {e}")
            raise TranscriptionError(f"Audio file not found: {audio_file}") from e

        except Exception as err:
            logger.error(f"Transcription failed: {err}")
            raise TranscriptionError(f"Unable to transcribe {audio_file}: {err}") from err

    def _cleanup_temp_files(self, temp_files: list) -> None:
        """Clean up temporary files safely."""
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    logger.debug(f"Cleaned up temporary file: {temp_file}")
            except OSError as e:
                logger.warning(f"Could not remove temporary file {temp_file}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error removing {temp_file}: {e}")


def main():
    """Main function for testing voice functionality."""
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Error: Please set the OPENAI_API_KEY environment variable.")
            return 1

        logger.info("Starting voice test")
        voice = Voice()
        result = voice.record_and_transcribe()

        if result:
            print(f"Transcription: {result}")
            return 0
        else:
            print("No transcription result")
            return 1

    except Exception as e:
        logger.error(f"Error in voice test: {e}", exc_info=True)
        print(f"Voice test failed: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
