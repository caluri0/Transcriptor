# Real-time Audio Transcription & Translation (JA → FR)

Capture audio from microphone or system output, transcribe Japanese speech using **faster-whisper** (open-source, no API key), translate to French via **DeepL API**, and display results in a live-updating GUI.

**Target latency: 1–2 seconds after speech ends.**

---

## ✨ Features

- **Real-time capture** — Microphone or system audio (Windows WASAPI loopback via Stereo Mix)
- **Local transcription** — faster-whisper (OpenAI Whisper C++ port), runs offline, GPU-accelerated
- **Translation** — DeepL API (free tier available)
- **Responsive GUI** — Tkinter with threaded audio processing
- **Smart Voice Activity Detection** — webrtcvad / silero-vad with 300 ms silence detection (sentence-aware chunking)
- **Anti-hallucination Whisper params** — `beam_size=1`, `condition_on_previous_text=False`, `compression_ratio_threshold`, `no_speech_threshold`
- **In-memory processing** — No temp files, direct numpy → faster-whisper
- **Full GPU utilization** — CUDA float16 on RTX 5060 / RTX 30/40 series
- **Latency logging** — Per-stage timestamps (VAD → Whisper → DeepL → Display)
- **Fallback handling** — Graceful degradation if VAD/scipy/DeepL unavailable

---

## 📋 Requirements

- Python 3.9+
- Windows 10/11 (for system audio capture via Stereo Mix)
- DeepL API key ([get one free](https://www.deepl.com/pro-account))
- Microphone or Stereo Mix enabled
- **Optional but recommended**: `pip install scipy webrtcvad` (resampling + smart VAD)

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Install VAD + resampling for best quality
pip install scipy webrtcvad

# 3. Set your DeepL API key 
# Windows PowerShell
$env:DEEPL_API_KEY="your_key_here"

# 4. Run
python transcriber.py
```

---

## ⚙️ Configuration

Key constants at the top of `transcriber.py`:

| Constant | Value | Description |
|----------|-------|-------------|
| `SAMPLE_RATE` | 16000 | Whisper expects 16 kHz mono |
| `CHANNELS` | 2 | Stereo capture (converted to mono) |
| `MIN_CHUNK_DURATION` | 0.5 s | Minimum speech segment before processing |
| `MAX_CHUNK_DURATION` | 12.0 s | Safety cap for long utterances |
| `SILENCE_THRESHOLD_MS` | 300 ms | Silence duration to finalize segment |
| `OVERLAP_DURATION` | 0.5 s | Overlap kept between segments |
| `VOLUME_THRESHOLD` | 0.005 | RMS gate to ignore near-silence |
| `WHISPER_MODEL_SIZE` | `"medium"` | **Forced for 1–2 s latency** (was auto large-v3) |
| `WHISPER_COMPUTE_TYPE` | `"float16"` | GPU: float16, CPU: int8 |
| `WHISPER_PARAMS` | dict | `beam_size=1`, `vad_filter=False`, `condition_on_previous_text=False`, `temperature=0.0` |
| `CAPTURE_SYSTEM_AUDIO` | `True` | `True` = Stereo Mix, `False` = microphone |
| `VAD_AGGRESSIVENESS` | 2 | webrtcvad 0–3 (higher = more aggressive) |

### Model Latency vs Quality (RTX 5060 laptop)

| Model | Typical Latency | Quality | VRAM |
|-------|----------------|---------|------|
| `large-v3` | 3–5 s | Best | ~8 GB |
| **`medium`** | **1–2 s** | **Excellent** | **~5 GB** |
| `small` | ~0.8 s | Good | ~2 GB |
| `base` | ~0.4 s | Fair | ~1 GB |

> **Current default: `medium`** — best balance for real-time use.

---

## 🎙️ System Audio (Windows)

To capture **what you hear** (speakers/headphones):

1. Enable Stereo Mix:
   - Right-click speaker icon → **Sounds** → **Recording** tab
   - Right-click empty area → **Show Disabled Devices**
   - Right-click **Stereo Mix** → **Enable**

2. Find its device index:
   ```bash
   python find_audio_devices.py
   ```

3. The app auto-detects Stereo Mix by name (`CAPTURE_SYSTEM_AUDIO = True`). If not found, falls back to microphone with warning.

> **Note:** Some apps (DRM-protected, browsers with audio isolation) may block loopback capture.

---

## 🖥️ GUI Overview

| Element | Purpose |
|---------|---------|
| **Start Recording** | Begins capture + transcription |
| **Stop Recording** | Stops cleanly, flushes buffer |
| **Status label** | Shows current mode/state |
| **Latency label** | Rolling avg latency (ms) + chunk count |
| **Text area** | Live JA/FR output with timestamps |

Output format:
```
[14:30:22]
JA: こんにちは、元気ですか？
FR: Bonjour, comment ça va ?
--------------------------------------------------
[14:30:27]
JA: 今日は良い天気ですね。
FR: Il fait beau aujourd'hui.
--------------------------------------------------
```

---

## 🧪 Testing & Utilities

| Script | Purpose |
|--------|---------|
| `find_audio_devices.py` | List all audio devices with indices |
| `test_audio_capture.py` | Verify microphone/system audio works |
| `example_usage.py` | Programmatic usage example |

---

## 📦 Dependencies

```
sounddevice     # Audio I/O (WASAPI loopback)
numpy           # Array math
faster-whisper  # Local Whisper inference (CTranslate2)
torch           # CUDA/CPU backend
deepl           # Translation API
soundfile       # WAV I/O (for testing only)
scipy           # Optional: high-quality resampling (48→16 kHz)
webrtcvad       # Optional: fast VAD backend
silero-vad      # Alternative: ML-based VAD (more accurate)
```

Install all:
```bash
pip install sounddevice numpy faster-whisper torch deepl soundfile scipy webrtcvad
```

Or minimal:
```bash
pip install -r requirements.txt
```

---

## 🐛 Troubleshooting

| Issue | Fix |
|-------|-----|
| No audio / "No input device" | Check Windows Sound → Recording, ensure mic/Stereo Mix is enabled & default |
| DeepL auth error | Verify API key, check free tier limit (500k chars/mo) |
| Whisper model load fails | Ensure RAM/VRAM for model size; try `small` or `base` |
| GUI freezes | Reduce `MAX_CHUNK_DURATION` or use smaller model |
| System audio silent | Enable Stereo Mix, some apps block capture |
| CUDA not detected / RTX 5060 warning | Install PyTorch nightly cu124: `pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu124` |
| `metadata_errors` from soundfile | Fixed — no temp files used anymore |
| Hallucinations ("ご視聴ありがとうございました" repeats) | Fixed — smart VAD + anti-hallucination params |
| High latency (>3 s) | Use `medium` or `small` model, ensure `scipy` + `webrtcvad` installed |

---

## 📄 License

Educational / personal use. DeepL usage subject to their [Terms of Service](https://www.deepl.com/terms).

---

## 🙏 Credits

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — SYSTRAN (CTranslate2 backend)
- [DeepL API](https://www.deepl.com/docs-api)
- [sounddevice](https://python-sounddevice.readthedocs.io/) — WASAPI loopback support
- [webrtcvad](https://github.com/wiseman/py-webrtcvad) — WebRTC VAD
- [silero-vad](https://github.com/snakers4/silero-vad) — Silero VAD
- Built with Python, tkinter, numpy
