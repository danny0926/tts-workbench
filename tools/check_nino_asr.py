"""Cross-check Japanese dataset clip transcripts with pinned faster-whisper.

ASR output is a separate triage signal. It never edits the source transcript or
changes human-review status. CUDA libraries are reused read-only from .venv-qwen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "Systran/faster-whisper-large-v3"
MODEL_REVISION = "edaa852ec7e145841d8ffdb056a99866b5f0a478"
DEFAULT_MANIFEST = ROOT / "outputs/dataset/manifest.jsonl"
DEFAULT_OUTPUT = ROOT / "outputs/dataset/asr_checks.jsonl"
DEFAULT_CACHE = ROOT / ".cache_hf/asr"
DEFAULT_MODEL_PATH = (
    DEFAULT_CACHE
    / "hub/models--Systran--faster-whisper-large-v3/snapshots"
    / MODEL_REVISION
)
SETTINGS = {
    "language": "ja",
    "task": "transcribe",
    "device": "cuda",
    "compute_type": "int8_float16",
    "beam_size": 5,
    "batch_size": 1,
    "cpu_threads": 4,
    "num_workers": 1,
    "condition_on_previous_text": False,
    "initial_prompt": None,
    "word_timestamps": True,
    "vad_filter": True,
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_no}: expected JSON object")
                rows.append(value)
    return rows


def write_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())


def normalize_for_triage(text: str) -> str:
    value = unicodedata.normalize("NFKC", text)
    return "".join(ch for ch in value if not ch.isspace() and not unicodedata.category(ch).startswith(("P", "S")))


def edit_ratio(left: str, right: str) -> float | None:
    a, b = normalize_for_triage(left), normalize_for_triage(right)
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    # Memory-bounded Levenshtein distance; this is a normalization triage score,
    # not WER or ground-truth accuracy.
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / max(len(a), len(b))


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def prepare_cuda_dlls() -> list[Any]:
    """Add existing Qwen Torch CUDA DLL directory to this process only."""
    dll_dir = ROOT / ".venv-qwen/Lib/site-packages/torch/lib"
    required = ("cublas64_12.dll", "cudnn64_9.dll")
    missing = [name for name in required if not (dll_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"Existing Qwen CUDA DLLs not found in {dll_dir}: {', '.join(missing)}")
    handles: list[Any] = []
    if hasattr(os, "add_dll_directory"):
        handles.append(os.add_dll_directory(str(dll_dir)))
    os.environ["PATH"] = str(dll_dir) + os.pathsep + os.environ.get("PATH", "")
    return handles


def gpu_memory_mib() -> list[dict[str, int]] | None:
    """Read current GPU memory from nvidia-smi without touching the device."""
    executable = shutil.which("nvidia-smi")
    if not executable:
        return None
    try:
        result = subprocess.run(
            [executable, "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=5,
        )
        if result.returncode != 0:
            return None
        values = []
        for line in result.stdout.splitlines():
            used, total = (int(part.strip()) for part in line.split(",", 1))
            values.append({"used_mib": used, "total_mib": total})
        return values or None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def collect_result(segment_iter: Any, info: Any) -> dict[str, Any]:
    texts: list[str] = []
    word_rows: list[dict[str, Any]] = []
    avg_logs: list[float] = []
    no_speech: list[float] = []
    for segment in segment_iter:
        texts.append(segment.text)
        if segment.avg_logprob is not None:
            avg_logs.append(float(segment.avg_logprob))
        if segment.no_speech_prob is not None:
            no_speech.append(float(segment.no_speech_prob))
        for word in segment.words or []:
            word_rows.append({
                "word": word.word,
                "start_seconds": float(word.start),
                "end_seconds": float(word.end),
                "probability": float(word.probability) if word.probability is not None else None,
            })
    return {
        "text": "".join(texts).strip(),
        "language": getattr(info, "language", None),
        "language_probability": getattr(info, "language_probability", None),
        "duration_seconds": getattr(info, "duration", None),
        "duration_after_vad_seconds": getattr(info, "duration_after_vad", None),
        "avg_logprob_mean": sum(avg_logs) / len(avg_logs) if avg_logs else None,
        "no_speech_prob_max": max(no_speech) if no_speech else None,
        "words": word_rows,
    }


def load_completed(path: Path) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    done: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    if not path.exists():
        return done
    for row in read_jsonl(path):
        if row.get("status") == "completed":
            key = (row.get("clip_id", ""), row.get("audio_sha256", ""), row.get("model_revision", ""), row.get("settings_sha256", ""))
            done[key] = row
    return done


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--limit", type=int, help="Process at most this many new eligible clips.")
    parser.add_argument("--start", type=int, default=0, help="Skip this many eligible manifest rows before processing.")
    parser.add_argument("--cpu", action="store_true", help="Use CPU int8 only after a concrete GPU blocker.")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.start < 0:
        parser.error("--start cannot be negative")

    manifest_path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    cache_path = args.cache if args.cache.is_absolute() else ROOT / args.cache
    model_path = args.model_path if args.model_path.is_absolute() else ROOT / args.model_path
    if not manifest_path.is_file():
        print(f"Manifest not found: {manifest_path}", file=sys.stderr)
        return 2
    rows = read_jsonl(manifest_path)
    eligible: list[tuple[dict[str, Any], Path]] = []
    for row in rows:
        if row.get("machine_status") in {"excluded", "decode_failed"}:
            continue
        wav_value = row.get("clip_audio_path")
        if not isinstance(wav_value, str) or not wav_value:
            continue
        wav = resolve_project_path(wav_value)
        if not wav.is_file():
            continue
        eligible.append((row, wav))
    if not eligible:
        print(f"No extracted clip WAVs found in {manifest_path}; waiting for extraction manifest.", file=sys.stderr)
        return 2
    if not model_path.is_dir():
        print(f"Pinned local model snapshot is missing: {model_path}; download it before inference.", file=sys.stderr)
        return 2

    settings = dict(SETTINGS)
    settings["device"] = "cpu" if args.cpu else "cuda"
    settings["compute_type"] = "int8" if args.cpu else "int8_float16"
    settings["settings_sha256"] = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    completed = load_completed(output_path)
    pending: list[tuple[dict[str, Any], Path, str]] = []
    considered = eligible[args.start:]
    for row, wav in considered:
        audio_sha = sha256_file(wav)
        key = (str(row.get("clip_id", "")), audio_sha, MODEL_REVISION, settings["settings_sha256"])
        if key not in completed:
            pending.append((row, wav, audio_sha))
        if args.limit is not None and len(pending) >= args.limit:
            break
    print(json.dumps({"eligible": len(eligible), "new_pending": len(pending), "already_complete_exact_match": len(completed), "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "settings": settings}, ensure_ascii=False))
    if not pending:
        return 0

    handles: list[Any] = []
    if not args.cpu:
        handles = prepare_cuda_dlls()
    gpu_before_model = gpu_memory_mib() if not args.cpu else None
    try:
        from faster_whisper import WhisperModel

        model = WhisperModel(
            str(model_path),
            device=settings["device"],
            compute_type=settings["compute_type"],
            cpu_threads=settings["cpu_threads"],
            num_workers=settings["num_workers"],
            download_root=str(cache_path),
            local_files_only=True,
        )
    except Exception as exc:
        print(f"Could not load pinned ASR model/device: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3

    error_count = 0
    for n, (row, wav, audio_sha) in enumerate(pending, 1):
        base = {
            "clip_id": row.get("clip_id"),
            "audio_path": str(wav.relative_to(ROOT)) if wav.is_relative_to(ROOT) else str(wav),
            "audio_sha256": audio_sha,
            "source_audio": row.get("source_audio"),
            "source_sha256": row.get("source_sha256"),
            "annotation_path": row.get("annotation_path"),
            "annotation_sha256": row.get("annotation_sha256"),
            "source_transcript_path": row.get("source_transcript_path"),
            "source_transcript_sha256": row.get("source_transcript_sha256"),
            "original_asr_text": row.get("text"),
            "machine_review_flags": row.get("machine_review_flags", row.get("flags", [])),
            "human_review_status": row.get("human_review_status", "pending"),
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "model_path": str(model_path),
            "settings": settings,
            "settings_sha256": settings["settings_sha256"],
            "started_at_unix": time.time(),
        }
        try:
            segments, info = model.transcribe(
                str(wav),
                language="ja",
                task="transcribe",
                beam_size=5,
                condition_on_previous_text=False,
                word_timestamps=True,
                vad_filter=True,
            )
            result = collect_result(segments, info)
            duration = float(result.get("duration_seconds") or 0.0)
            ratio = edit_ratio(str(base.get("original_asr_text") or ""), result["text"])
            flags: list[str] = []
            if not result["text"]:
                flags.append("asr_empty_hypothesis")
            if ratio is not None and ratio > 0.35:
                flags.append("normalized_character_disagreement_over_0_35")
            if result["avg_logprob_mean"] is not None and result["avg_logprob_mean"] < -1.0:
                flags.append("asr_avg_logprob_below_minus_1")
            if result["no_speech_prob_max"] is not None and result["no_speech_prob_max"] > 0.6:
                flags.append("asr_no_speech_probability_over_0_6")
            if any(w["start_seconds"] <= 0.10 or (duration > 0 and w["end_seconds"] >= duration - 0.10) for w in result["words"]):
                flags.append("word_timestamp_near_clip_boundary")
            record = {
                **base,
                "status": "completed",
                "result": result,
                "normalized_character_disagreement_ratio": ratio,
                "review_priority_flags": flags,
                "interpretation": "ASR triage only; normalized character difference is not WER or ground-truth accuracy, and no human review status is changed.",
                "completed_at_unix": time.time(),
            }
            if n == 1 and not args.cpu:
                record["gpu_memory_before_model"] = gpu_before_model
                record["gpu_memory_after_first_clip"] = gpu_memory_mib()
            write_jsonl_row(output_path, record)
            memory_note = ""
            if n == 1 and not args.cpu:
                memory_note = f", GPU_MiB before={gpu_before_model}, resident_after={record.get('gpu_memory_after_first_clip')}"
            print(f"[{n}/{len(pending)}] {row.get('clip_id')} completed ({duration:.2f}s), flags={len(flags)}{memory_note}", flush=True)
        except Exception as exc:
            error_count += 1
            message = str(exc)
            fatal = isinstance(exc, (TypeError, ImportError, AttributeError)) or any(
                token in message.lower()
                for token in ("cuda", "cublas", "cudnn", "out of memory", "dll load failed")
            )
            write_jsonl_row(output_path, {
                **base,
                "status": "system_failed" if fatal else "failed",
                "error_type": type(exc).__name__,
                "error": message,
                "completed_at_unix": time.time(),
            })
            print(f"[{n}/{len(pending)}] {row.get('clip_id')} {'aborted' if fatal else 'failed'}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            if fatal:
                print("Stopping after systemic ASR/API/device error; refusing to repeat it across clips.", file=sys.stderr)
                return 3
    print(json.dumps({"processed": len(pending), "failed": error_count, "output": str(output_path)}, ensure_ascii=False))
    return 1 if error_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
