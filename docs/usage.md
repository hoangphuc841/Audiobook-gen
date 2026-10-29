# PDF to Audiobook CLI Usage Guide

This guide covers usage examples for `pdf_to_audiobook.py`, demonstrating both **normal (positional)** syntax and **flag-based** syntax, along with common options and environment variable configuration.

---

## 1. Prerequisites & Environment Setup

Configure your API key and default models in `.env` (or set them in your system environment):

```env
GEMINI_API_KEY=your_gemini_api_key_here
SCRIPT_MODEL=gemini-3.8-flash
TTS_MODEL=gemini-3.8-flash-tts
```

---

## 2. Normal Usage (Positional Parameter)

In normal usage, provide the input PDF as a positional argument. The output filename is automatically derived from the PDF name (e.g., `novel.pdf` becomes `novel.mp3`).

### Basic Conversion (Entire Book)
```bash
python pdf_to_audiobook.py "Youjo Senki 01 _ Deus lo vult.pdf"
```
*Output: `Youjo Senki 01 _ Deus lo vult.mp3`*

### Converting a Specific Page Range
Convert only a chapter or preview range (1-indexed, inclusive):
```bash
# Convert pages 1 to 20
python pdf_to_audiobook.py "novel.pdf" --start-page 1 --end-page 20

# Convert from page 50 to the end of the book
python pdf_to_audiobook.py "novel.pdf" --start-page 50
```

### Specifying Output Path
```bash
python pdf_to_audiobook.py "novel.pdf" -o "audiobooks/Volume_01.mp3"
```

---

## 3. Flag Usage (`--pdf` / `-f`)

Instead of a positional argument, you can explicitly pass the input file using the `--pdf` (or `-f`) flag.

### Basic Conversion with `--pdf`
```bash
python pdf_to_audiobook.py --pdf "novel.pdf"
```

### Short Flag `-f` and `-o`
```bash
python pdf_to_audiobook.py -f "novel.pdf" -o "novel.mp3"
```

### Overriding Gemini Models via Flags
Override the models specified in your `.env` or system environment on a per-run basis:
```bash
python pdf_to_audiobook.py --pdf "novel.pdf" \
  --script-model "gemini-3.8-flash" \
  --tts-model "gemini-3.8-flash-tts"
```

### Custom Batch Size & Cache Directory
Adjust the number of pages processed per diarization batch and set a custom cache folder for checkpoint files:
```bash
python pdf_to_audiobook.py --pdf "novel.pdf" \
  --batch-size 5 \
  --cache-dir "cache_vol1"
```

### Passing API Key Explicitly
If not using a `.env` file or environment variable:
```bash
python pdf_to_audiobook.py --pdf "novel.pdf" --api-key "AIzaSy..."
```

---

## 4. Full Advanced Example

Combining page selection, model overrides, custom batch size, and custom paths:

```bash
python pdf_to_audiobook.py \
  --pdf "Youjo Senki 01 _ Deus lo vult.pdf" \
  --output "Youjo_Senki_Vol1_Ch1.mp3" \
  --start-page 1 \
  --end-page 35 \
  --batch-size 5 \
  --cache-dir "checkpoints_vol1" \
  --script-model "gemini-3.8-flash" \
  --tts-model "gemini-3.8-flash-tts"
```

---

## 5. Resuming an Interrupted Run

The pipeline saves atomic checkpoint files inside `--cache-dir` (default: `audiobook_cache`):
- `batch_XXXX_pA-B_script.json`: Diarized script cache
- `batch_XXXX_line_YYYY.wav`: Segment audio cache
- `batch_XXXX_pA-B.wav`: Master batch audio cache
- `cast_registry.json`: Persistent character voice mappings

To resume an interrupted conversion, re-run the exact same command. Completed segments and batches will automatically be skipped:
```bash
python pdf_to_audiobook.py "novel.pdf"
```

---

## 6. CLI Parameter Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `PDF_PATH` | Positional | *(Required if `--pdf` omitted)* | Input PDF file path |
| `--pdf`, `-f` | Flag | *(Required if positional omitted)* | Input PDF file path |
| `--output`, `-o` | Flag | `<pdf_name>.mp3` | Output MP3 file destination |
| `--start-page` | Flag | `1` | 1-indexed start page |
| `--end-page` | Flag | `None` (end of book) | 1-indexed end page |
| `--batch-size` | Flag | `10` | Pages processed per diarization batch |
| `--cache-dir` | Flag | `audiobook_cache` | Checkpoint and batch cache directory |
| `--api-key` | Flag | `None` | Gemini API Key (or set `GEMINI_API_KEY` in `.env`) |
| `--script-model` | Flag | `SCRIPT_MODEL` from env | Gemini model for script diarization |
| `--tts-model` | Flag | `TTS_MODEL` from env | Gemini model for speech synthesis |

