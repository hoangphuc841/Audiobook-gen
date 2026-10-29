# PDF to Audiobook Generator

An automated Python pipeline that converts PDF books and light novels into multi-speaker, dramatized MP3 audiobooks using Google Gemini models.

---

## Overview

The pipeline extracts clean prose from a PDF, uses a Gemini diarization model to convert the text into an emotion-annotated multi-speaker script, assigns persistent voices to characters, synthesizes speech with Gemini TTS, and stitches the audio segments into a mastered MP3 audiobook with pause-aware timing and atomic resume checkpoints.

### Key Features

- **Multi-Speaker Diarization**: Detects narrator text versus character dialogue and tags each line with character name, gender, and emotion direction (e.g. *calm*, *commanding*, *whispering fearfully*).
- **Token & Segment Optimization**: Merges consecutive narrative and same-speaker dialogue blocks to optimize API calls and ensure natural speech cadence.
- **Consistent Voice Casting**: Automatically assigns distinct voices from a voice pool and maintains consistent character casting across batches via a persistent `cast_registry.json`.
- **Fault-Tolerant & Resumable**: Atomic checkpoints at the script, line, and batch levels allow interrupted conversions to resume without repeating completed work or wasting API quota.
- **Configurable Models**: Model names are loaded from `.env` or system environment variables rather than hardcoded, with CLI override support.

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Ensure FFmpeg is installed or portable static-ffmpeg is available on your system.)*

### 2. Configure Environment
Copy the example environment file and set your Gemini API key:
```bash
cp .env.example .env
```

Edit `.env`:
```env
GEMINI_API_KEY=your_gemini_api_key_here
SCRIPT_MODEL=gemini-3.8-flash
TTS_MODEL=gemini-3.8-flash-tts
```

### 3. Run Conversion

**Normal (Positional) Syntax:**
```bash
python pdf_to_audiobook.py "novel.pdf"
```

**Flag Syntax:**
```bash
python pdf_to_audiobook.py --pdf "novel.pdf" --output "audiobook.mp3"
```

---

## Documentation

For detailed usage, page-range options, batch sizing, model overrides, and CLI parameter reference, see:
- [CLI Usage Guide](docs/usage.md)
