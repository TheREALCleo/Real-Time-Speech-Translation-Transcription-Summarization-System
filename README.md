# AI Video Transcriber & Meeting Summarizer

An all-in-one **local** pipeline that turns video recordings into professional subtitles (SRT) and comprehensive meeting minutes (MOM). Built for privacy — everything runs on your machine (GPU recommended), no cloud or API keys required.

---

## Features

* **Fully offline**: No cloud uploads or API keys. Process videos locally.
* **Accurate, fast transcription** using **faster-whisper (large-v3)** with optional auto-translation to English.
* **Movie-style subtitles**: Smart sentence-splitting, line-length limits (42 chars/line), and timing aligned to words for readable SRT files.
* **Structured meeting minutes**: Uses a quantized Qwen 3 model (4-bit) to produce Markdown Minutes of the Meeting (Attendees, Decisions, Action Items, Next Steps).
* **Integrated review**: Launches VLC with custom subtitle rendering to preview video + subtitles immediately.
* **Hardware-friendly**: Uses BitsAndBytes quantization and device-aware loading so large models can run on consumer GPUs (8–12 GB VRAM recommended).

---

## Quick Start

1. Clone the repo:

```bash
git clone https://github.com/yourusername/ai-meeting-assistant.git
cd ai-meeting-assistant
```

2. Create a virtual environment and activate it (recommended):

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

3. Install dependencies. Create a `requirements.txt` with the following (example):

```
torch
transformers
accelerate
bitsandbytes
faster-whisper
python-dotenv
python-multipart
```

Then run:

```bash
pip install -r requirements.txt
```

> **Tip:** Use a compatible PyTorch + CUDA build for your GPU. If you have limited VRAM, make sure to install the `bitsandbytes` and `transformers` versions that work for your CUDA toolkit.

---

## Configuration (.env)

Create a file named `.env` in the project root and set paths and filenames. Example:

```ini
# Input video file to transcribe
in_video=C:/Users/You/Videos/meeting_recording.mp4

# Output subtitle path (SRT)
out_srt=C:/Users/You/Videos/meeting_recording.srt

# Input SRT file (for summarization)
in_srt=C:/Users/You/Videos/meeting_recording.srt

# Output Minutes of the Meeting (Markdown)
out_sum=C:/Users/You/Documents/minutes_of_meeting.md
```

Make sure paths are correct for your OS. On Linux/macOS, use `/home/you/...` style paths.

---

## Scripts & Usage

### 1) `transcriber.py` — Transcribe & preview

Purpose: Detect language, (optionally) translate to English, create professionally formatted `.srt`, and open VLC for review.

Run:

```bash
python transcriber.py
```

What it does:

* Loads `faster-whisper (large-v3)` by default.
* Runs a first-pass language detection with VAD (voice activity detection).
* If `auto_translate=True` and audio is non-English, runs `task="translate"` to generate English subtitles.
* Smartly splits long segments into movie-style subtitle chunks (max 42 chars/line, max 2 lines, max chunk duration default 8s).
* Writes SRT with `UTF-8-sig` encoding for maximum compatibility.
* Launches VLC with custom subtitle settings (configurable path & font).

**Configurable options (function parameters):** `model_size`, `device`, `compute_type`, `batch_size`, `max_subtitle_duration`, `auto_translate`.

**VLC settings** (change `vlc_path` or other flags in `play_with_vlc` if needed):

* `--freetype-font` to specify a font
* `--sub-text-scale` and freetype settings to customize size

### 2) `summarizer.py` — Generate Minutes of the Meeting (MOM)

Purpose: Read the `.srt` transcript, assemble a single transcript string, and generate a polished, structured meeting markdown using a quantized Qwen model.

Run:

```bash
python summarizer.py
```

What it does:

* Loads `Qwen3` (default: `Qwen/Qwen3-4B-Instruct-2507`) with 4-bit quantization via `bitsandbytes`.
* Extracts and concatenates subtitle text from the SRT file (removes indices & timestamps).
* Constructs a rich prompt that asks the LLM to emit a Markdown Minutes of the Meeting (Agenda, Attendees, Decisions, Next Steps, Action Items, Deadlines).
* Generates a Markdown `.md` file with the output.

**Important model options:**

* `BitsAndBytesConfig`: `load_in_4bit=True`, `bnb_4bit_quant_type='nf4'`, `bnb_4bit_use_double_quant=True`, `bnb_4bit_compute_dtype=torch.float16`
* `device_map='cuda'` (or change to run on CPU if GPU unavailable — expect much slower inference)

---

## Example `.env` & run flow

1. Fill `.env` with your paths.
2. Run `python transcriber.py` → `out_srt` is produced.
3. Optionally review the subtitles in VLC.
4. Run `python summarizer.py` → `out_sum` (Markdown MOM) is produced.

---

## File structure (suggested)

```
ai-meeting-assistant/
├─ transcriber.py
├─ summarizer.py
├─ requirements.txt
├─ .env
└─ README.md
```

---

## Customization & Tips

* **Change VLC path**: Update `vlc_path` inside `transcriber.py` to your VLC installation.
* **Language handling**: If your meetings are always in English, set `auto_translate=False` to skip translation and save compute.
* **Lower VRAM**: If you have <8GB VRAM, try a smaller Qwen variant (or CPU), or reduce inference batch sizes and use `low_cpu_mem_usage=True` (already enabled in the script).
* **Subtitle style**: Tweak `max_chars` and `max_lines` in `format_subtitle_text()` to match your preferred reading speed / layout.
* **Segment splitting**: `max_subtitle_duration` controls how long a subtitle chunk can be. Lower values create faster subtitle turnover; higher values reduce subtitle churn.
---

## Security & Privacy

All processing is local. **Do not** share raw video or SRT files if privacy is a concern — keep them on your device. If you ever adapt the pipeline to use a hosted LLM or cloud speech API, be sure to add clear warnings and opt-in controls.

---

## Example Improvements / Roadmap

* Add a CLI with flags for all important parameters (model, device, max duration, auto-translate).
* Support speaker diarization to attribute lines to participants (use `pyannote` or faster local diarization tools).
* Add an optional condensed and an extended summary mode.
* Offer a GUI wrapper (Electron or a simple web UI) for users who prefer not to use the command line.
