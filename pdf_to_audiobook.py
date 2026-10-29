import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Callable, TypeVar

# Initialize portable FFmpeg paths for pydub without lock warnings
import sys

_ffmpeg_bin = os.path.join(sys.prefix, "Lib", "site-packages", "static_ffmpeg", "bin", "win32")
if os.path.exists(_ffmpeg_bin) and _ffmpeg_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _ffmpeg_bin + os.pathsep + os.environ.get("PATH", "")

import pdfplumber
from pydub import AudioSegment

if os.path.exists(_ffmpeg_bin):
    AudioSegment.converter = os.path.join(_ffmpeg_bin, "ffmpeg.exe")
    AudioSegment.ffprobe = os.path.join(_ffmpeg_bin, "ffprobe.exe")
from google import genai
from google.genai import types

def load_local_env() -> None:
    """Loads environment variables from .env in the current directory if present."""
    env_file = Path(".env")
    if env_file.exists():
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip("'\"")
                    if key and val and key not in os.environ:
                        os.environ[key] = val


load_local_env()

T = TypeVar("T")

SCRIPT_MODEL = os.environ.get("SCRIPT_MODEL") or os.environ.get("GEMINI_SCRIPT_MODEL") or os.environ.get("GEMINI_MODEL")
TTS_MODEL = os.environ.get("TTS_MODEL") or os.environ.get("GEMINI_TTS_MODEL")
MAX_RETRIES = 5
BASE_DELAY_SECONDS = 2.0
MAX_MERGED_CHARS = 1200


def get_script_model(override: str | None = None) -> str:
    model = (
        override
        or os.environ.get("SCRIPT_MODEL")
        or os.environ.get("GEMINI_SCRIPT_MODEL")
        or os.environ.get("GEMINI_MODEL")
        or SCRIPT_MODEL
    )
    if not model:
        raise RuntimeError(
            "Script diarization model not found. Set SCRIPT_MODEL in your environment / .env file, "
            "or pass --script-model <MODEL>."
        )
    return model


def get_tts_model(override: str | None = None) -> str:
    model = (
        override
        or os.environ.get("TTS_MODEL")
        or os.environ.get("GEMINI_TTS_MODEL")
        or TTS_MODEL
    )
    if not model:
        raise RuntimeError(
            "TTS model not found. Set TTS_MODEL in your environment / .env file, "
            "or pass --tts-model <MODEL>."
        )
    return model


VOICE_POOL = {
    "Narrator": "Charon",
    "male": ["Puck", "Fenrir", "Orus", "Enceladus", "Iapetus"],
    "female": ["Kore", "Aoede", "Callirhoe", "Leda", "Zephyr"],
    "neutral": ["Zephyr", "Puck"],
}

GENDER_MAP = {
    "m": "male",
    "male": "male",
    "f": "female",
    "female": "female",
    "n": "neutral",
    "neutral": "neutral",
}

SCRIPT_SYSTEM_PROMPT = """
Convert the book text into a chronological multi-speaker audiobook script.

Rules:
1. Preserve every word of the story. Do not summarize or omit narrative text.
2. Assign narrative descriptions and dialogue tags (e.g., "said Mark anxiously") to "Narrator".
3. Merge consecutive "Narrator" paragraphs into a single object (up to ~1000 characters) instead of splitting every paragraph.
4. Assign spoken dialogue to the specific character's name.
5. Provide a concise 1-3 word "e" (emotion) direction (e.g., "calm", "whispering fearfully", "commanding", "sarcastic").
6. Return strict JSON as a list of compact objects:
[
  {"s": "Narrator" | "Character Name", "g": "m" | "f" | "n", "e": "emotion", "t": "Exact words to speak"}
]
"""


def get_client(api_key: str | None = None) -> genai.Client:
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY not found. Pass --api-key <YOUR_KEY>, set the GEMINI_API_KEY "
            "environment variable, or put GEMINI_API_KEY=your_key into a .env file."
        )
    return genai.Client(api_key=key)


def with_retry(fn: Callable[[], T], label: str) -> T:
    """Runs fn() with exponential backoff up to MAX_RETRIES times."""
    last_err: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn()
        except Exception as exc:
            last_err = exc
            if attempt == MAX_RETRIES:
                break
            delay = BASE_DELAY_SECONDS * (2 ** (attempt - 1))
            print(f"   [Retry {attempt}/{MAX_RETRIES}] {label} failed ({exc}). Waiting {delay:.1f}s...")
            time.sleep(delay)
    raise RuntimeError(f"{label} failed after {MAX_RETRIES} attempts: {last_err}") from last_err


def atomic_write_json(path: Path, data: object) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def atomic_export_audio(segment: AudioSegment, path: Path, fmt: str = "wav", **kwargs) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    segment.export(str(tmp_path), format=fmt, **kwargs)
    os.replace(tmp_path, path)


def get_pdf_page_count(pdf_path: str) -> int:
    with pdfplumber.open(pdf_path) as pdf:
        return len(pdf.pages)


def clean_page_text(text: str) -> str:
    """Strips standalone page numbers to avoid wasting tokens or reading page numbers aloud."""
    lines = [line.strip() for line in text.splitlines()]
    cleaned = [line for line in lines if not re.fullmatch(r"\d{1,4}", line)]
    return "\n".join(cleaned).strip()


def extract_pdf_text(pdf_path: str, start_page: int = 1, end_page: int | None = None) -> str:
    """Extracts cleaned text from a 1-indexed inclusive range of PDF pages."""
    pages_text = []
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        last_page = min(end_page, total_pages) if end_page else total_pages

        for i in range(start_page - 1, last_page):
            raw = pdf.pages[i].extract_text()
            if raw:
                cleaned = clean_page_text(raw)
                if cleaned:
                    pages_text.append(cleaned)

    return "\n\n".join(pages_text)


def normalize_segment(item: dict) -> dict:
    """Normalizes compact ('s','g','e','t') or full keys into a standard segment dict."""
    speaker = str(item.get("s") or item.get("speaker") or "Narrator").strip() or "Narrator"
    raw_gender = str(item.get("g") or item.get("gender") or "n").strip().lower()
    gender = GENDER_MAP.get(raw_gender, "neutral")
    emotion = str(item.get("e") or item.get("emotion") or "natural").strip() or "natural"
    text = str(item.get("t") or item.get("text") or "").strip()
    return {"speaker": speaker, "gender": gender, "emotion": emotion, "text": text}


def merge_consecutive_segments(
    segments: list[dict], max_chars: int = MAX_MERGED_CHARS
) -> list[dict]:
    """Merges adjacent lines from the same speaker and emotion up to max_chars."""
    merged: list[dict] = []
    for raw_seg in segments:
        seg = normalize_segment(raw_seg)
        if not seg["text"]:
            continue
        if (
            merged
            and merged[-1]["speaker"] == seg["speaker"]
            and (seg["speaker"] == "Narrator" or merged[-1]["emotion"] == seg["emotion"])
            and len(merged[-1]["text"]) + 1 + len(seg["text"]) <= max_chars
        ):
            merged[-1]["text"] = f"{merged[-1]['text']}\n{seg['text']}"
        else:
            merged.append(seg)
    return merged


def create_annotated_script(
    client: genai.Client, raw_text: str, model: str | None = None
) -> list[dict]:
    """Uses Gemini model from env with retries to convert raw prose into merged speaker segments."""
    script_model = get_script_model(model)

    def _call() -> list[dict]:
        response = client.models.generate_content(
            model=script_model,
            contents=[SCRIPT_SYSTEM_PROMPT, raw_text],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2,
            ),
        )
        if not response.text:
            raise ValueError("Empty response text from diarization model.")
        parsed = json.loads(response.text)
        if not isinstance(parsed, list):
            raise ValueError(f"Expected JSON list from diarization model, got {type(parsed).__name__}.")
        return merge_consecutive_segments(parsed)

    return with_retry(_call, f"Diarization ({script_model})")


def load_cast_registry(cache_dir: Path) -> tuple[dict[str, str], dict[str, int]]:
    cast_file = cache_dir / "cast_registry.json"
    if cast_file.exists():
        with open(cast_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("cast", {}), data.get("counters", {"male": 0, "female": 0, "neutral": 0})
    return {}, {"male": 0, "female": 0, "neutral": 0}


def save_cast_registry(cache_dir: Path, cast: dict[str, str], counters: dict[str, int]) -> None:
    atomic_write_json(cache_dir / "cast_registry.json", {"cast": cast, "counters": counters})


def assign_voice(speaker: str, gender: str, cast_registry: dict, counters: dict) -> str:
    """Assigns a consistent pretrained voice to each unique character."""
    if speaker in cast_registry:
        return cast_registry[speaker]

    if speaker == "Narrator":
        cast_registry[speaker] = VOICE_POOL["Narrator"]
        return cast_registry[speaker]

    gender_key = gender.lower() if gender.lower() in ("male", "female") else "neutral"
    pool = VOICE_POOL[gender_key]

    idx = counters.get(gender_key, 0) % len(pool)
    counters[gender_key] = idx + 1

    cast_registry[speaker] = pool[idx]
    print(f"[Casting] Assigned '{cast_registry[speaker]}' to character '{speaker}' ({gender_key})")
    return cast_registry[speaker]


def pcm_to_segment(audio_bytes: bytes) -> AudioSegment:
    """Wraps 24kHz, 16-bit, mono PCM bytes into a pydub AudioSegment."""
    return AudioSegment(
        data=audio_bytes,
        sample_width=2,
        frame_rate=24000,
        channels=1,
    )


def extract_pcm_bytes(response: types.GenerateContentResponse, model_name: str = "TTS model") -> bytes:
    """Validates candidate parts and returns raw PCM bytes."""
    candidates = getattr(response, "candidates", None)
    if not candidates:
        raise ValueError(f"No candidates returned by {model_name} (feedback: {getattr(response, 'prompt_feedback', None)}).")
    content = getattr(candidates[0], "content", None)
    parts = getattr(content, "parts", None) if content else None
    if not parts or not getattr(parts[0], "inline_data", None) or not parts[0].inline_data.data:
        finish_reason = getattr(candidates[0], "finish_reason", None)
        raise ValueError(f"Missing inline audio data from {model_name} (finish_reason={finish_reason}).")
    return parts[0].inline_data.data


def synthesize_speech_gemini(
    client: genai.Client, text: str, voice_name: str, emotion: str, model: str | None = None
) -> AudioSegment:
    """Calls Gemini TTS model from env with retries and returns an AudioSegment."""
    tts_model = get_tts_model(model)
    directed_prompt = f"Say in a {emotion} tone: {text}"

    def _call() -> AudioSegment:
        response = client.models.generate_content(
            model=tts_model,
            contents=directed_prompt,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=voice_name
                        )
                    )
                ),
            ),
        )
        return pcm_to_segment(extract_pcm_bytes(response, model_name=tts_model))

    return with_retry(_call, f"TTS ({voice_name} via {tts_model})")


def build_audiobook(
    pdf_path: str,
    output_mp3: str,
    start_page: int = 1,
    end_page: int | None = None,
    pages_per_batch: int = 10,
    cache_dir_name: str = "audiobook_cache",
    api_key: str | None = None,
    script_model: str | None = None,
    tts_model: str | None = None,
) -> None:
    cache_dir = Path(cache_dir_name)
    cache_dir.mkdir(parents=True, exist_ok=True)

    active_script_model = get_script_model(script_model)
    active_tts_model = get_tts_model(tts_model)

    total_pages = get_pdf_page_count(pdf_path)
    last_page = min(end_page, total_pages) if end_page else total_pages
    client = get_client(api_key)

    cast_registry, gender_counters = load_cast_registry(cache_dir)
    print(f"Processing '{pdf_path}' pages {start_page} to {last_page} (of {total_pages})...")
    print(f"Using models: script='{active_script_model}', tts='{active_tts_model}'")

    batch_index = 0
    batch_files: list[Path] = []

    for batch_start in range(start_page, last_page + 1, pages_per_batch):
        batch_index += 1
        batch_end = min(batch_start + pages_per_batch - 1, last_page)
        batch_prefix = f"batch_{batch_index:04d}_p{batch_start}-{batch_end}"
        batch_audio_file = cache_dir / f"{batch_prefix}.wav"
        batch_script_file = cache_dir / f"{batch_prefix}_script.json"
        batch_files.append(batch_audio_file)

        if batch_audio_file.exists():
            print(f"[Resume] Skipping batch {batch_index} (pages {batch_start}-{batch_end}), already synthesized.")
            continue

        print(f"\n--- Batch {batch_index} (pages {batch_start}-{batch_end}) ---")
        if batch_script_file.exists():
            print(f"[Resume] Loading cached diarized script: {batch_script_file}")
            with open(batch_script_file, "r", encoding="utf-8") as f:
                script = merge_consecutive_segments(json.load(f))
        else:
            raw_text = extract_pdf_text(pdf_path, batch_start, batch_end)
            if not raw_text:
                print(f"Skipping pages {batch_start}-{batch_end} (no text found).")
                atomic_export_audio(AudioSegment.silent(duration=100), batch_audio_file, fmt="wav")
                continue

            print(f"Diarizing text with {active_script_model}...")
            script = create_annotated_script(client, raw_text, model=active_script_model)
            atomic_write_json(batch_script_file, script)

        batch_audio = AudioSegment.silent(duration=300)
        prev_speaker = None

        print(f"Synthesizing {len(script)} merged segments with {active_tts_model}...")
        for idx, line in enumerate(script):
            speaker = line["speaker"]
            text = line["text"]
            emotion = line["emotion"]
            gender = line["gender"]

            if not text:
                continue

            voice = assign_voice(speaker, gender, cast_registry, gender_counters)
            save_cast_registry(cache_dir, cast_registry, gender_counters)

            line_audio_file = cache_dir / f"{batch_prefix}_line_{idx + 1:04d}.wav"
            if line_audio_file.exists():
                print(f"   [{idx + 1}/{len(script)}] [Resume] Using cached segment {idx + 1} ({speaker})")
                segment = AudioSegment.from_file(str(line_audio_file), format="wav")
            else:
                preview = text.replace("\n", " ")[:45]
                print(f"   [{idx + 1}/{len(script)}] {speaker} ({voice} - {emotion}, {len(text)} chars): {preview}...")
                segment = synthesize_speech_gemini(client, text, voice, emotion, model=active_tts_model)
                atomic_export_audio(segment, line_audio_file, fmt="wav")
                time.sleep(0.3)

            if prev_speaker is not None:
                if prev_speaker == speaker:
                    pause_ms = 350
                elif "Narrator" in (prev_speaker, speaker):
                    pause_ms = 550
                else:
                    pause_ms = 250
                batch_audio += AudioSegment.silent(duration=pause_ms)

            batch_audio += segment
            prev_speaker = speaker

        atomic_export_audio(batch_audio, batch_audio_file, fmt="wav")
        print(f"Saved batch checkpoint: {batch_audio_file}")

    print("\nStitching all batches into master audiobook...")
    master_audio = AudioSegment.silent(duration=500)
    for bf in batch_files:
        if bf.exists():
            master_audio += AudioSegment.from_file(str(bf), format="wav")
            master_audio += AudioSegment.silent(duration=600)

    print(f"Exporting final mastered audiobook to '{output_mp3}'...")
    atomic_export_audio(master_audio, Path(output_mp3), fmt="mp3", bitrate="192k")
    print("Done!")


def main() -> None:
    load_local_env()
    parser = argparse.ArgumentParser(description="Convert a PDF book into a multi-speaker MP3 audiobook.")
    parser.add_argument(
        "pdf",
        nargs="?",
        default=None,
        metavar="PDF_PATH",
        help="Input PDF file path",
    )
    parser.add_argument(
        "--pdf",
        "-f",
        dest="pdf_flag",
        metavar="PATH",
        default=None,
        help="Input PDF file path",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output MP3 file path (default: <pdf_name>.mp3)",
    )
    parser.add_argument("--start-page", type=int, default=1, help="1-indexed start page")
    parser.add_argument("--end-page", type=int, default=None, help="1-indexed end page (omit for full book)")
    parser.add_argument("--batch-size", type=int, default=10, help="Pages per diarization batch")
    parser.add_argument("--cache-dir", default="audiobook_cache", help="Directory for checkpoint batch files")
    parser.add_argument("--api-key", default=None, help="Gemini API Key (or set GEMINI_API_KEY / use .env)")
    parser.add_argument(
        "--script-model",
        default=None,
        help="Gemini model for diarization (env: SCRIPT_MODEL)",
    )
    parser.add_argument(
        "--tts-model",
        default=None,
        help="Gemini model for TTS (env: TTS_MODEL)",
    )
    args = parser.parse_args()

    pdf_path = args.pdf_flag or args.pdf
    if not pdf_path:
        parser.error("Input PDF file path is required. Usage: python pdf_to_audiobook.py <PDF_PATH> or --pdf <PDF_PATH>")

    if not os.path.exists(pdf_path):
        parser.error(f"Input PDF file not found: '{pdf_path}'")

    output_path = args.output
    if not output_path:
        output_path = str(Path(pdf_path).with_suffix(".mp3"))

    build_audiobook(
        pdf_path=pdf_path,
        output_mp3=output_path,
        start_page=args.start_page,
        end_page=args.end_page,
        pages_per_batch=args.batch_size,
        cache_dir_name=args.cache_dir,
        api_key=args.api_key,
        script_model=args.script_model,
        tts_model=args.tts_model,
    )


if __name__ == "__main__":
    main()
