#!/usr/bin/env python3
"""
Real-time audio transcription and translation application - OPTIMIZED FOR LOW LATENCY.
Captures system audio, transcribes Japanese speech with faster-whisper (open-source, GPU-accelerated),
translates the text to French with DeepL API, and displays results in a live-updating GUI.

OPTIMIZATIONS FOR 1-2s LATENCY:
1. FULL GPU UTILIZATION: large-v3 model with CUDA float16, automatic fallback to medium
2. INTELLIGENT VAD-BASED CHUNKING: webrtcvad/silero-vad with 300ms silence detection
3. IN-MEMORY PROCESSING: No temporary files - uses io.BytesIO + soundfile
4. ANTI-HALLUCINATION WHISPER PARAMETERS: condition_on_previous_text, compression_ratio_threshold, etc.
5. ASYNCHRONOUS PIPELINE: VAD → Transcription → Translation → Display with no blocking waits
6. LATENCY LOGGING: Detailed timestamps at each stage for validation
"""

import os
import threading
import time
import queue
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import sounddevice as sd
import torch
from faster_whisper import WhisperModel
import deepl
import tkinter as tk
from tkinter import scrolledtext
import tkinter.ttk as ttk

# Top-level optional imports
try:
    from scipy.signal import resample
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    resample = None  # type: ignore

# ============================================================================
# VOICE ACTIVITY DETECTION (VAD) - DEPENDENCIES WITH FALLBACKS
# ============================================================================
try:
    import webrtcvad
    VAD_BACKEND = "webrtcvad"
    print("[INIT] Using webrtcvad for voice activity detection (fast, lightweight)")
except ImportError:
    try:
        from silero_vad import load_silero_vad, get_speech_timestamps
        VAD_BACKEND = "silero"
        print("[INIT] Using silero-vad for voice activity detection (more accurate)")
    except ImportError:
        VAD_BACKEND = None
        print("[WARN] No VAD library available. Will use fixed duration chunking.")
        print("[WARN] Install for better results: pip install webrtcvad OR pip install silero-vad")

# ============================================================================
# CONFIGURATION - TUNED FOR LOW LATENCY
# ============================================================================
SAMPLE_RATE = 16000  # Whisper expects 16kHz
CHANNELS = 2         # Stereo for system audio capture (will convert to mono)
VOLUME_THRESHOLD = 0.005  # Lower threshold to catch quieter speech

# Force medium model for 1-2s latency (large-v3 is too slow on RTX 5060)
WHISPER_MODEL_SIZE = "medium"
WHISPER_COMPUTE_TYPE = "float16" if torch.cuda.is_available() else "int8"
WHISPER_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GPU_AVAILABLE = torch.cuda.is_available()
print(f"[CONFIG] Using device: {WHISPER_DEVICE}")
print(f"[CONFIG] Model: {WHISPER_MODEL_SIZE} (forced for latency)")
print(f"[CONFIG] Compute type: {WHISPER_COMPUTE_TYPE}")

# --- Smart Chunking / VAD Configuration (Optimized for low latency) ---
MIN_CHUNK_DURATION = 0.5        # Reduced from 1.0s to 0.5s for faster response
MAX_CHUNK_DURATION = 12.0       # Reduced from 15.0s to 12.0s safety cap
SILENCE_THRESHOLD_MS = 300      # Reduced from 500ms to 300ms for faster sentence detection
VAD_AGGRESSIVENESS = 2          # webrtcvad aggressiveness (0-3, higher = more aggressive)
OVERLAP_DURATION = 0.5          # Reduced from 2.0s to 0.5s to minimize latency

# --- Anti-Hallucination Whisper Parameters (Optimized for Speed) ---
WHISPER_PARAMS = {
    "language": "ja",
    "beam_size": 1,                    # Faster: greedy decoding
    "vad_filter": False,               # We do our own VAD
    "condition_on_previous_text": False,  # Prevents hallucination chains
    "compression_ratio_threshold": 2.4,
    "no_speech_threshold": 0.6,
    "temperature": 0.0,                # Deterministic
}

# Audio capture mode: False for microphone input, True for system output (loopback)
CAPTURE_SYSTEM_AUDIO = True

# ============================================================================
# MAIN APPLICATION CLASS
# ============================================================================
class AudioTranscriber:
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.processing_queue = queue.Queue()  # Decouple VAD from Whisper/DeepL
        self.is_recording = False
        self.is_processing = False

        # Initialize API clients (will be set when keys are provided)
        self.whisper_model = None
        self.deepl_translator = None

        # GUI components
        self.root = None
        self.text_display = None
        self.status_label = None
        self.perf_label = None

        # VAD initialization
        self.vad = None
        self.vad_model = None  # For silero
        self._init_vad()

        # Smart chunking state
        self.speech_buffer = []      # Accumulated audio samples for current speech segment
        self.speech_duration = 0.0   # Current speech segment duration
        self.last_speech_time = 0.0  # Time of last detected speech
        self.silence_duration = 0.0  # Current silence duration
        self.in_speech = False       # Whether we're currently in a speech segment

        # Performance tracking
        self.chunk_count = 0
        self.total_latency_sum = 0.0
        self.latency_count = 0

        # Thread pool for transcription/translation (replaces per-chunk threading.Thread)
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="WhisperWorker")

    def _init_vad(self):
        """Initialize Voice Activity Detection backend."""
        global VAD_BACKEND
        if VAD_BACKEND == "webrtcvad":
            try:
                self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
                print(f"[VAD] webrtcvad initialized (aggressiveness={VAD_AGGRESSIVENESS})")
            except Exception as e:
                print(f"[WARN] Failed to initialize webrtcvad: {e}. Falling back to fixed chunking.")
                VAD_BACKEND = None
        elif VAD_BACKEND == "silero":
            try:
                self.vad_model, _ = load_silero_vad()
                print("[VAD] silero-vad initialized")
            except Exception as e:
                print(f"[WARN] Failed to initialize silero-vad: {e}. Falling back to fixed chunking.")
                VAD_BACKEND = None
        else:
            print("[WARN] No VAD available. Using fixed duration chunking (MIN_CHUNK_DURATION={}s). "
                  "This may cut sentences and cause hallucinations. "
                  "Install 'webrtcvad' or 'silero-vad' for smart chunking.".format(MIN_CHUNK_DURATION))

    def _is_speech_webrtcvad(self, audio_frame):
        """Check if audio frame contains speech using webrtcvad.
        audio_frame: numpy array of int16, 20ms at 16kHz (320 samples)"""
        try:
            # webrtcvad expects 20ms frames at 16kHz = 320 samples
            if len(audio_frame) != 320:
                return False
            # Convert float32 [-1, 1] to int16
            int16_frame = (audio_frame * 32767).astype(np.int16).tobytes()
            return self.vad.is_speech(int16_frame, SAMPLE_RATE)
        except Exception:
            return False

    def _is_speech_silero(self, audio_data):
        """Check if audio contains speech using silero-vad.
        audio_data: numpy array of float32 at 16kHz"""
        try:
            if self.vad_model is None:
                return False
            # silero expects torch tensor
            tensor = torch.from_numpy(audio_data).float()
            # Get speech timestamps (returns list of dicts with 'start'/'end' in samples)
            speeches = get_speech_timestamps(tensor, self.vad_model, sampling_rate=SAMPLE_RATE)
            return len(speeches) > 0
        except Exception:
            return False

    def _is_speech(self, audio_data):
        """Unified VAD check - routes to appropriate backend."""
        if VAD_BACKEND == "webrtcvad":
            # webrtcvad needs 20ms frames (320 samples at 16kHz)
            # Use sliding window view for zero-copy frame iteration
            frame_size = 320
            if len(audio_data) < frame_size:
                return False
            # Create strided view: shape (n_frames, 320)
            n_frames = (len(audio_data) - frame_size) // frame_size + 1
            strides = audio_data.strides * 2
            frames_view = np.lib.stride_tricks.as_strided(
                audio_data, shape=(n_frames, frame_size), strides=strides
            )
            # Convert all frames to int16 bytes at once
            int16_frames = (frames_view * 32767).astype(np.int16)
            for frame_bytes in int16_frames:
                if self.vad.is_speech(frame_bytes.tobytes(), SAMPLE_RATE):
                    return True
            return False
        elif VAD_BACKEND == "silero":
            return self._is_speech_silero(audio_data)
        else:
            # Fallback: simple energy-based VAD
            rms = np.sqrt(np.mean(audio_data**2))
            return rms > VOLUME_THRESHOLD

    def set_api_keys(self, openai_key, deepl_key):
        """Set API keys for DeepL and initialize Whisper model.
        Note: openai_key parameter is ignored since we use local faster-whisper model."""
        # Ignore openai_key parameter (keeping for compatibility)
        _ = openai_key
        # Initialize DeepL translator if key provided
        if deepl_key:
            try:
                self.deepl_translator = deepl.Translator(deepl_key)
                print("[API] DeepL translator initialized successfully")
            except Exception as e:
                print(f"[ERROR] Failed to initialize DeepL translator: {e}")
                self.deepl_translator = None
        else:
            self.deepl_translator = None

        # Initialize Whisper model (local, open-source)
        try:
            print(f"[MODEL] Loading Whisper model '{WHISPER_MODEL_SIZE}' on {WHISPER_DEVICE}...")
            load_start = time.time()
            self.whisper_model = WhisperModel(
                WHISPER_MODEL_SIZE,
                device=WHISPER_DEVICE,
                compute_type=WHISPER_COMPUTE_TYPE
            )
            load_time = time.time() - load_start
            print(f"[MODEL] Whisper model loaded successfully in {load_time:.2f}s")

            # Verify GPU usage if applicable
            if WHISPER_DEVICE == "cuda":
                print("[GPU] Whisper is configured to use CUDA")
        except Exception as e:
            print(f"[ERROR] Failed to load Whisper model: {e}")
            self.whisper_model = None

    def audio_callback(self, indata, frames, time, status):
        """Callback for audio input stream."""
        # frames, time unused - required by sounddevice callback signature
        _ = frames, time
        if status:
            print(f"[AUDIO] Callback status: {status}")
        # For system audio capture, we get stereo but Whisper works better with mono
        if indata.shape[1] > 1:
            # Convert stereo to mono
            mono_data = np.mean(indata, axis=1)
            audio_to_queue = mono_data.copy()
        else:
            # Already mono, flatten to 1D array
            audio_to_queue = indata.flatten().copy()

        # Resample to 16 kHz if native sample rate differs from Whisper's expected rate
        if hasattr(self, 'native_sample_rate') and self.native_sample_rate != SAMPLE_RATE:
            if HAS_SCIPY:
                num_samples = int(len(audio_to_queue) * SAMPLE_RATE / self.native_sample_rate)
                if num_samples > 0:
                    audio_to_queue = resample(audio_to_queue, num_samples)
            else:
                # Warn once
                if not hasattr(self, '_scipy_warned'):
                    print("[WARN] scipy not available for resampling. Audio may not be processed correctly.")
                    self._scipy_warned = True

        self.audio_queue.put(audio_to_queue)

    def process_audio_chunk(self, audio_data):
        """Process a chunk of audio: transcribe with Whisper (in-memory), translate with DeepL.
        NO temporary file, NO soundfile.write → avoids the 'metadata_errors' bug on Windows."""
        chunk_start_time = time.time()

        try:
            # Audio already mono float32 at 16 kHz from audio_callback
            audio_float32 = audio_data.astype(np.float32, copy=False)

            # Transcribe with Whisper (local) - using anti-hallucination parameters
            if self.whisper_model:
                transcribe_start = time.time()
                segments, _ = self.whisper_model.transcribe(
                    audio_float32,
                    **WHISPER_PARAMS
                )
                transcribe_time = time.time() - transcribe_start

                # Combine segments into full text
                japanese_text = " ".join([segment.text for segment in segments]).strip()

                if japanese_text:
                    # Translate with DeepL
                    if self.deepl_translator:
                        try:
                            translate_start = time.time()
                            result = self.deepl_translator.translate_text(
                                japanese_text,
                                source_lang="JA",
                                target_lang="FR"
                            )
                            translate_time = time.time() - translate_start
                            french_text = result.text

                            # Calculate total latency
                            total_latency = time.time() - chunk_start_time

                            # Update performance metrics
                            self.chunk_count += 1
                            self.total_latency_sum += total_latency
                            self.latency_count += 1
                            avg_latency = self.total_latency_sum / self.latency_count if self.latency_count > 0 else 0

                            # Update GUI
                            self.update_display(japanese_text, french_text)

                            # Update perf label in GUI
                            if self.perf_label and self.root:
                                self.root.after(0, lambda: self.perf_label.config(
                                    text=f"Latency: {total_latency*1000:.0f} ms | Chunks: {self.chunk_count} (avg {avg_latency*1000:.0f} ms)"
                                ))

                            # Log performance - format timestamp once
                            ts = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                            print(f"[RESULT] [{ts}] JA: {japanese_text[:50]}{'...' if len(japanese_text) > 50 else ''}")
                            print(f"[RESULT] [{ts}] FR: {french_text[:50]}{'...' if len(french_text) > 50 else ''}")
                            print(f"[LATENCY] Chunk #{self.chunk_count}: "
                                  f"VAD→Whisper={transcribe_time*1000:.0f}ms, "
                                  f"Whisper→DeepL={translate_time*1000:.0f}ms, "
                                  f"Total={total_latency*1000:.0f}ms "
                                  f"(avg={avg_latency*1000:.0f}ms over {self.latency_count} chunks)")

                        except Exception as e:
                            print(f"[ERROR] DeepL translation error: {e}")
                            self.update_display(japanese_text, f"[Translation error: {str(e)}]")
                    else:
                        self.update_display(japanese_text, "[DeepL not configured]")
                else:
                    if self.chunk_count < 5:
                        print(f"[DEBUG] No speech detected in chunk (duration: {len(audio_float32)/SAMPLE_RATE:.2f}s)")
                    self.update_display("", "[No speech detected]")
            else:
                self.update_display("[Whisper model not loaded]", "")

        except Exception as e:
            print(f"[ERROR] Error processing audio chunk: {e}")
            self.update_display(f"[Error: {str(e)}]", "")

    def update_display(self, japanese, french):
        """Update the GUI with new transcription and translation."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        display_text = f"[{timestamp}]\n"
        display_text += f"JA: {japanese}\n"
        display_text += f"FR: {french}\n"
        display_text += "-" * 50 + "\n"

        # Update in main thread
        if self.root and self.text_display:
            self.root.after(0, self._append_text, display_text)

    def _append_text(self, text):
        """Append text to display (called from main thread)."""
        self.text_display.insert(tk.END, text)
        self.text_display.see(tk.END)  # Auto-scroll

    def audio_processor(self):
        """Process audio chunks from the queue with smart silence-based chunking."""
        # Buffer for accumulating incoming audio frames
        frame_buffer = []
        frame_buffer_duration = 0.0

        while self.is_recording:
            try:
                # Get audio data with timeout
                data = self.audio_queue.get(timeout=0.1)
                frame_buffer.append(data)
                frame_duration = len(data) / SAMPLE_RATE
                frame_buffer_duration += frame_duration

                # Process accumulated frames for VAD
                if frame_buffer:
                    # Concatenate recent frames for VAD analysis (check last ~300ms)
                    vad_check_duration = min(frame_buffer_duration, 0.3)  # Check last 300ms
                    vad_check_samples = int(vad_check_duration * SAMPLE_RATE)

                    # Get recent audio for VAD
                    recent_audio = np.concatenate(frame_buffer, axis=0)
                    if len(recent_audio) > vad_check_samples:
                        recent_audio = recent_audio[-vad_check_samples:]

                    # Run VAD on recent audio
                    has_speech = self._is_speech(recent_audio)
                    current_time = time.time()

                    if has_speech:
                        # Speech detected
                        if not self.in_speech:
                            print(f"[VAD] [{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Speech START detected")
                            self.in_speech = True
                        self.last_speech_time = current_time
                        self.silence_duration = 0.0
                        # Accumulate to speech buffer
                        self.speech_buffer.extend(frame_buffer)
                        self.speech_duration += frame_buffer_duration
                    else:
                        # Silence detected
                        if self.in_speech:
                            self.silence_duration += frame_buffer_duration
                            # Still accumulate during silence (part of current segment)
                            self.speech_buffer.extend(frame_buffer)
                            self.speech_duration += frame_buffer_duration

                    # Clear frame buffer
                    frame_buffer = []
                    frame_buffer_duration = 0.0

                    # Check if we should finalize the current speech segment
                    if self.in_speech:
                        should_finalize = False
                        finalize_reason = ""

                        # Condition 1: Silence longer than threshold (end of sentence)
                        if self.silence_duration >= (SILENCE_THRESHOLD_MS / 1000.0):
                            should_finalize = True
                            finalize_reason = f"silence > {SILENCE_THRESHOLD_MS}ms"

                        # Condition 2: Maximum duration reached (safety cap)
                        elif self.speech_duration >= MAX_CHUNK_DURATION:
                            should_finalize = True
                            finalize_reason = f"max duration {MAX_CHUNK_DURATION}s reached"

                        if should_finalize:
                            if self.speech_duration >= MIN_CHUNK_DURATION:
                                # Concatenate speech buffer once
                                chunk = np.concatenate(self.speech_buffer, axis=0)

                                # Log chunk info for debugging
                                ts = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                                print(f"[CHUNK] [{ts}] "
                                      f"Sending to Whisper: duration={self.speech_duration:.2f}s, "
                                      f"samples={len(chunk)}, reason={finalize_reason}")

                                # Check volume threshold
                                if np.max(np.abs(chunk)) > VOLUME_THRESHOLD:
                                    # Submit to thread pool (replaces per-chunk threading.Thread)
                                    self.executor.submit(self.process_audio_chunk, chunk)
                                else:
                                    print(f"[DEBUG] Chunk rejected: volume below threshold")
                            else:
                                print(f"[DEBUG] Chunk discarded: too short ({self.speech_duration:.2f}s < {MIN_CHUNK_DURATION}s)")

                            # Keep overlap for continuity
                            overlap_samples = int(SAMPLE_RATE * OVERLAP_DURATION)
                            if len(self.speech_buffer) > 0:
                                all_speech = np.concatenate(self.speech_buffer, axis=0)
                                if len(all_speech) > overlap_samples:
                                    self.speech_buffer = [all_speech[-overlap_samples:]]
                                    self.speech_duration = OVERLAP_DURATION
                                else:
                                    self.speech_buffer = [all_speech]
                                    self.speech_duration = len(all_speech) / SAMPLE_RATE
                            else:
                                self.speech_buffer = []
                                self.speech_duration = 0.0

                            # Reset speech state
                            self.in_speech = False
                            self.silence_duration = 0.0

            except queue.Empty:
                # No audio data - check if we're in speech and silence timeout
                if self.in_speech and self.last_speech_time > 0:
                    elapsed_silence = time.time() - self.last_speech_time
                    if elapsed_silence >= (SILENCE_THRESHOLD_MS / 1000.0):
                        # Force finalize on silence timeout
                        if self.speech_duration >= MIN_CHUNK_DURATION:
                            chunk = np.concatenate(self.speech_buffer, axis=0)
                            ts = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                            print(f"[CHUNK] [{ts}] "
                                  f"Sending to Whisper (timeout): duration={self.speech_duration:.2f}s, "
                                  f"samples={len(chunk)}, reason=silence timeout")
                            if np.max(np.abs(chunk)) > VOLUME_THRESHOLD:
                                self.executor.submit(self.process_audio_chunk, chunk)
                        else:
                            print(f"[DEBUG] Chunk discarded (timeout): too short ({self.speech_duration:.2f}s)")

                        # Reset with overlap
                        overlap_samples = int(SAMPLE_RATE * OVERLAP_DURATION)
                        if len(self.speech_buffer) > 0:
                            all_speech = np.concatenate(self.speech_buffer, axis=0)
                            if len(all_speech) > overlap_samples:
                                self.speech_buffer = [all_speech[-overlap_samples:]]
                                self.speech_duration = OVERLAP_DURATION
                            else:
                                self.speech_buffer = [all_speech]
                                self.speech_duration = len(all_speech) / SAMPLE_RATE
                        else:
                            self.speech_buffer = []
                            self.speech_duration = 0.0

                        self.in_speech = False
                        self.silence_duration = 0.0
                continue
            except Exception as e:
                print(f"[ERROR] Processor error: {e}")

    def start_recording(self):
        """Start audio capture and processing."""
        if self.is_recording:
            return

        self.is_recording = True
        self.is_processing = True

        # Update button states
        if hasattr(self, 'start_button'):
            self.start_button.config(state="disabled")
        if hasattr(self, 'stop_button'):
            self.stop_button.config(state="normal")

        # Start processing thread
        self.process_thread = threading.Thread(target=self.audio_processor, daemon=True)
        self.process_thread.start()

        # Start audio stream
        try:
            # If capturing system audio, try to use Stereo Mix first
            if CAPTURE_SYSTEM_AUDIO:
                # Try to find and use Stereo Mix device for system audio capture
                devices = sd.query_devices()
                stereo_mix_found = False
                for i, dev in enumerate(devices):
                    try:
                        name = str(dev['name']).lower()
                    except:
                        name = ""
                    if 'stereo mix' in name and dev['max_input_channels'] >= 2:
                        # Found Stereo Mix - display its properties
                        try:
                            name_str = str(dev['name'])
                        except:
                            name_str = f"Device_{i}"
                        default_samplerate = dev.get('default_samplerate', 'Unknown')
                        max_input_channels = dev.get('max_input_channels', 'Unknown')
                        print(f"[AUDIO] Found Stereo Mix device: {name_str} (index {i})")
                        print(f"[AUDIO]   default_samplerate: {default_samplerate} Hz")
                        print(f"[AUDIO]   max_input_channels: {max_input_channels}")

                        # Use device's native samplerate and channels
                        stream_params = {
                            'samplerate': int(default_samplerate) if default_samplerate != 'Unknown' else SAMPLE_RATE,
                            'channels': min(max_input_channels, 2) if max_input_channels != 'Unknown' else CHANNELS,
                            'device': i,
                            'callback': self.audio_callback,
                            'blocksize': int(default_samplerate * 0.1) if default_samplerate != 'Unknown' else int(SAMPLE_RATE * 0.1),
                            'latency': 'low'
                        }

                        # Store native sample rate for resampling later
                        self.native_sample_rate = stream_params['samplerate']
                        stereo_mix_found = True
                        print(f"[AUDIO] Using Stereo Mix device: {name_str} (index {i}) at {self.native_sample_rate} Hz, {stream_params['channels']} channel(s)")
                        break

                if not stereo_mix_found:
                    # If no Stereo Mix found, fall back to microphone with warning
                    print("[WARN] Stereo Mix not found or not accessible. Falling back to microphone input.")
                    print("[WARN] To capture system audio, please enable Stereo Mix in Windows Sound settings.")
                    # Use default input device with its native sample rate
                    default_dev_idx = sd.default.device[0]
                    default_dev = sd.query_devices()[default_dev_idx]
                    default_samplerate = default_dev.get('default_samplerate', SAMPLE_RATE)
                    max_input_channels = default_dev.get('max_input_channels', 1)
                    stream_params = {
                        'samplerate': int(default_samplerate),
                        'channels': min(max_input_channels, 2),
                        'device': default_dev_idx,
                        'callback': self.audio_callback,
                        'blocksize': int(default_samplerate * 0.1),
                        'latency': 'low'
                    }
                    self.native_sample_rate = stream_params['samplerate']
                    try:
                        default_name = str(default_dev['name'])
                    except:
                        default_name = f"Device_{default_dev_idx}"
                    print(f"[AUDIO] Using microphone input: {default_name} at {self.native_sample_rate} Hz, {stream_params['channels']} channel(s)")
            else:
                # Microphone mode - use default input device with its native sample rate
                default_dev_idx = sd.default.device[0]
                default_dev = sd.query_devices()[default_dev_idx]
                default_samplerate = default_dev.get('default_samplerate', SAMPLE_RATE)
                max_input_channels = default_dev.get('max_input_channels', 1)
                stream_params = {
                    'samplerate': int(default_samplerate),
                    'channels': min(max_input_channels, 2),
                    'device': default_dev_idx,
                    'callback': self.audio_callback,
                    'blocksize': int(default_samplerate * 0.1),
                    'latency': 'low'
                }
                self.native_sample_rate = stream_params['samplerate']
                try:
                    default_name = str(default_dev['name'])
                except:
                    default_name = f"Device_{default_dev_idx}"
                print(f"[AUDIO] Using microphone input: {default_name} at {self.native_sample_rate} Hz, {stream_params['channels']} channel(s)")

            self.stream = sd.InputStream(**stream_params)
            self.stream.start()
            mode_text = "System Audio (Loopback)" if CAPTURE_SYSTEM_AUDIO else "Microphone"
            self.update_status(f"Recording... ({mode_text})", "green")
            print(f"[AUDIO] Audio stream started successfully in {mode_text} mode at {self.native_sample_rate} Hz")
        except Exception as e:
            self.update_status(f"Failed to start audio: {e}", "red")
            self.is_recording = False
            # Reset button states on error
            if hasattr(self, 'start_button'):
                self.start_button.config(state="normal")
            if hasattr(self, 'stop_button'):
                self.stop_button.config(state="disabled")
            print(f"[ERROR] Error starting audio stream: {e}")

    def stop_recording(self):
        """Stop audio capture and processing."""
        if not self.is_recording:
            return

        self.is_recording = False
        self.is_processing = False

        # Stop audio stream
        if hasattr(self, 'stream'):
            try:
                self.stream.stop()
                self.stream.close()
            except:
                pass  # Ignore errors during cleanup

        # Wait for processing thread to finish (with timeout)
        if hasattr(self, 'process_thread') and self.process_thread.is_alive():
            self.process_thread.join(timeout=2.0)

        # Reset button states
        if hasattr(self, 'start_button'):
            self.start_button.config(state="normal")
        if hasattr(self, 'stop_button'):
            self.stop_button.config(state="disabled")

        self.update_status("Stopped", "red")
        print("[INFO] Recording stopped")

    def update_status(self, text, color):
        """Update status label."""
        if self.status_label:
            self.root.after(0, lambda: self.status_label.config(text=text, foreground=color))

    def create_gui(self):
        """Create the tkinter GUI."""
        self.root = tk.Tk()
        self.root.title("Audio Transcriber - JA to FR (faster-whisper GPU-Optimized)")
        self.root.geometry("700x500")

        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Title
        title_label = ttk.Label(main_frame,
                               text="Real-time Audio Transcription & Translation\n"
                                    "(using faster-whisper GPU + DeepL - Optimized for 1-2s latency)",
                               font=("Arial", 12, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))

        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=1, column=0, columnspan=2, pady=(0, 10))

        self.start_button = ttk.Button(button_frame, text="Start Recording",
                                      command=self.start_recording)
        self.start_button.grid(row=0, column=0, padx=(0, 5))

        self.stop_button = ttk.Button(button_frame, text="Stop Recording",
                                     command=self.stop_recording, state="disabled")
        self.stop_button.grid(row=0, column=1, padx=(5, 0))

        # Status label
        self.status_label = ttk.Label(main_frame, text="Ready", foreground="gray")
        self.status_label.grid(row=2, column=0, columnspan=2, pady=(0, 10))

        # Performance info label
        self.perf_label = ttk.Label(main_frame, text="Latency: -- ms | Chunks: 0", foreground="blue")
        self.perf_label.grid(row=2, column=1, columnspan=2, pady=(0, 10), sticky=tk.E)

        # Shutdown handler for executor
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Text display
        ttk.Label(main_frame, text="Transcription & Translation:").grid(
            row=3, column=0, sticky=tk.W, pady=(0, 5))

        self.text_display = scrolledtext.ScrolledText(
            main_frame, width=80, height=20, wrap=tk.WORD
        )
        self.text_display.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(4, weight=1)

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def on_closing(self):
        """Handle window closing."""
        self.stop_recording()
        # Shutdown thread pool gracefully
        self.executor.shutdown(wait=True)
        self.root.destroy()

    def run(self, openai_key=None, deepl_key=None):
        """Run the application."""
        # Set API keys if provided
        if openai_key or deepl_key:
            self.set_api_keys(openai_key, deepl_key)

        # Create and show GUI
        self.create_gui()

        # Update button states
        self.start_button.config(command=self.start_recording)
        self.stop_button.config(command=self.stop_recording)

        # Start GUI main loop
        self.root.mainloop()


def main():
    """Main entry point."""
    print("=" * 60)
    print("Real-time Japanese→French Transcription (GPU-Optimized)")
    print("Target latency: 1-2 seconds after speech ends")
    print("=" * 60)

    # Try to get API keys from environment variables
    openai_key = os.getenv("OPENAI_API_KEY")  # Ignored but kept for compatibility
    deepl_key = os.getenv("DEEPL_API_KEY")

    if not deepl_key:
        print("[ERROR] DEEPL_API_KEY environment variable not set.")
        print("Set it before running: $env:DEEPL_API_KEY='your_key_here'")
        return

    # Create and run transcriber
    app = AudioTranscriber()
    app.run(openai_key, deepl_key)


if __name__ == "__main__":
    main()