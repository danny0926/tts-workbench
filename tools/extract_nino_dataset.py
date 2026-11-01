#!/usr/bin/env python3
"""Extract planned Nino annotation clips to reviewable PCM16 WAV + UTF-8 text.

This tool only decodes/crops user-provided audio. It never imports a TTS package,
loads a model, uploads audio, denoises, normalizes, trims, or overwrites unknown
outputs. Run with the project-local .venv-qwen Python which has soundfile,
NumPy, and imageio-ffmpeg installed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = ROOT / "outputs/dataset/plan.jsonl"
DEFAULT_EXCLUDED = ROOT / "outputs/dataset/excluded.jsonl"
DEFAULT_SOURCE_INVENTORY = ROOT / "outputs/intake/source_inventory.json"
DEFAULT_DATASET_ROOT = ROOT / "audio/nino-dataset"
DEFAULT_MANIFEST = ROOT / "outputs/dataset/manifest.jsonl"
DEFAULT_QC_SUMMARY = ROOT / "outputs/dataset/audio_qc_summary.json"
SAFE_CLIP_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,180}$")

# These are conservative triage triggers only. A pass is never a human review.
FULL_SCALE_FRACTION_REVIEW = 0.001
FULL_SCALE_RUN_REVIEW_SAMPLES = 48  # 1 ms at 48 kHz
VERY_QUIET_RMS_DBFS = -45.0
VERY_QUIET_PEAK_DBFS = -35.0
TEXT_RATE_HIGH_CHARS_PER_SECOND = 18.0
TEXT_RATE_LOW_CHARS_PER_SECOND = 0.25
TEXT_RATE_LOW_MIN_DURATION_SECONDS = 8.0
SILENCE_ESTIMATE_THRESHOLD_DBFS = -50.0


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path: Path, *, required: bool = True) -> list[dict[str, Any]]:
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"Required input is missing: {path}")
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def write_exact_or_validate(path: Path, data: bytes) -> None:
    """Create a file, or allow reruns only when the existing bytes match exactly."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != data:
            raise FileExistsError(f"Refusing to replace a non-matching existing file: {path}")
        return
    path.write_bytes(data)


def source_id_for(sha256: str) -> str:
    return f"src-{sha256[:16]}"


def inspect_wav(path: Path, *, sf: Any) -> dict[str, Any]:
    info = sf.info(str(path))
    if info.format != "WAV" or info.subtype != "PCM_16":
        raise ValueError(f"Expected WAV PCM_16 at {path}; found {info.format}/{info.subtype}")
    if info.channels != 1:
        raise ValueError(f"Expected mono audio at {path}; found {info.channels} channels")
    if info.frames <= 0 or info.samplerate <= 0:
        raise ValueError(f"Empty or invalid WAV at {path}")
    return {
        "sample_rate_hz": int(info.samplerate),
        "channels": int(info.channels),
        "frames": int(info.frames),
        "duration_seconds": float(info.duration),
        "format": str(info.format),
        "subtype": str(info.subtype),
    }


def source_rate_index(inventory_path: Path) -> dict[str, int]:
    if not inventory_path.is_file():
        return {}
    data = json.loads(inventory_path.read_text(encoding="utf-8-sig"))
    result: dict[str, int] = {}
    for entry in data.get("files", []):
        streams = entry.get("audio_streams") or []
        if entry.get("sha256") and streams and streams[0].get("sample_rate_hz"):
            result[entry["sha256"].lower()] = int(streams[0]["sample_rate_hz"])
    return result


def ensure_source_wav(
    *,
    source_path: Path,
    expected_source_sha: str,
    expected_rate: int | None,
    source_wav_dir: Path,
    ffmpeg: str,
    ffmpeg_version: str,
    sf: Any,
) -> dict[str, Any]:
    if not source_path.is_file():
        raise FileNotFoundError(f"Source MP4 not found: {source_path}")
    actual_source_sha = sha256_file(source_path)
    if actual_source_sha.lower() != expected_source_sha.lower():
        raise ValueError(
            f"Source SHA mismatch for {source_path}: plan={expected_source_sha}, actual={actual_source_sha}"
        )

    source_id = source_id_for(actual_source_sha)
    wav_path = source_wav_dir / f"{source_id}.wav"
    meta_path = source_wav_dir / f"{source_id}.json"
    source_wav_dir.mkdir(parents=True, exist_ok=True)

    if wav_path.exists() or meta_path.exists():
        if not wav_path.is_file() or not meta_path.is_file():
            raise FileExistsError(f"Incomplete source-WAV cache pair; refusing to overwrite: {wav_path}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        wav_info = inspect_wav(wav_path, sf=sf)
        actual_wav_sha = sha256_file(wav_path)
        if (
            meta.get("source_sha256", "").lower() != actual_source_sha.lower()
            or meta.get("decoded_wav_sha256", "").lower() != actual_wav_sha.lower()
            or int(meta.get("sample_rate_hz", -1)) != wav_info["sample_rate_hz"]
            or int(meta.get("frames", -1)) != wav_info["frames"]
            or meta.get("channels") != 1
            or meta.get("subtype") != "PCM_16"
        ):
            raise FileExistsError(f"Source-WAV cache provenance mismatch; refusing to overwrite: {wav_path}")
        if expected_rate is not None and wav_info["sample_rate_hz"] != expected_rate:
            raise ValueError(
                f"Cached decode changed source sample rate: expected {expected_rate}, got {wav_info['sample_rate_hz']}"
            )
        return {
            "source_id": source_id,
            "source_audio": str(source_path.resolve()),
            "source_sha256": actual_source_sha,
            "source_wav_path": wav_path.relative_to(ROOT).as_posix(),
            "source_wav_sha256": actual_wav_sha,
            **wav_info,
            "cache_status": "reused_matching_cache",
            "ffmpeg_version": meta.get("ffmpeg_version"),
        }

    with tempfile.TemporaryDirectory(prefix=f".{source_id}-", dir=source_wav_dir) as tmp_name:
        tmp_dir = Path(tmp_name)
        tmp_wav = tmp_dir / "decoded.wav"
        command = [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source_path),
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(tmp_wav),
        ]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
        if result.returncode != 0 or not tmp_wav.is_file():
            raise RuntimeError(f"ffmpeg decode failed for {source_path}: {result.stderr[-4000:]}")
        wav_info = inspect_wav(tmp_wav, sf=sf)
        if expected_rate is not None and wav_info["sample_rate_hz"] != expected_rate:
            raise ValueError(
                f"Decoded rate differs from source inventory for {source_path}: "
                f"expected native {expected_rate}, got {wav_info['sample_rate_hz']}"
            )
        decoded_sha = sha256_file(tmp_wav)
        meta = {
            "source_id": source_id,
            "source_audio": str(source_path.resolve()),
            "source_sha256": actual_source_sha,
            "decode_method": "ffmpeg audio stream 0:a:0, mono PCM16; no -ar/resampling or audio filters",
            "ffmpeg_version": ffmpeg_version,
            "decoded_wav_sha256": decoded_sha,
            "sample_rate_hz": wav_info["sample_rate_hz"],
            "channels": 1,
            "frames": wav_info["frames"],
            "duration_seconds": wav_info["duration_seconds"],
            "format": "WAV",
            "subtype": "PCM_16",
        }
        tmp_meta = tmp_dir / "decoded.json"
        tmp_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if wav_path.exists() or meta_path.exists():
            raise FileExistsError(f"Source-WAV output appeared during extraction; refusing overwrite: {wav_path}")
        os.replace(tmp_wav, wav_path)
        try:
            os.replace(tmp_meta, meta_path)
        except Exception:
            # Remove only the just-created file after a failed paired write.
            wav_path.unlink(missing_ok=True)
            raise
    return {
        "source_id": source_id,
        "source_audio": str(source_path.resolve()),
        "source_sha256": actual_source_sha,
        "source_wav_path": wav_path.relative_to(ROOT).as_posix(),
        "source_wav_sha256": decoded_sha,
        **wav_info,
        "cache_status": "decoded_once",
        "ffmpeg_version": ffmpeg_version,
    }


def max_true_run(mask: Any, np: Any) -> int:
    if mask.size == 0 or not bool(mask.any()):
        return 0
    starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
    ends = np.flatnonzero(mask & ~np.r_[mask[1:], False]) + 1
    return int((ends - starts).max(initial=0))


def audio_qc(samples: Any, sample_rate: int, np: Any) -> dict[str, Any]:
    if samples.ndim == 2:
        flat = samples.reshape(-1)
    else:
        flat = samples
    if flat.size == 0:
        raise ValueError("Clip contains no PCM samples")
    values = flat.astype(np.int32, copy=False)
    absolute = np.abs(values)
    peak = int(absolute.max(initial=0))
    rms = float(np.sqrt(np.mean(values.astype(np.float64) ** 2)))
    rms_normalized = rms / 32768.0
    peak_normalized = peak / 32768.0
    rms_dbfs = 20.0 * math.log10(rms_normalized) if rms_normalized > 0 else None
    peak_dbfs = 20.0 * math.log10(peak_normalized) if peak_normalized > 0 else None
    positive = values == 32767
    negative = values == -32768
    full_scale = positive | negative
    full_run = max_true_run(full_scale, np)
    same_value_run = max(max_true_run(positive, np), max_true_run(negative, np))
    threshold = 32768.0 * (10.0 ** (SILENCE_ESTIMATE_THRESHOLD_DBFS / 20.0))
    active = np.flatnonzero(absolute >= threshold)
    if active.size:
        leading_silence = float(active[0] / sample_rate)
        trailing_silence = float((values.size - 1 - active[-1]) / sample_rate)
    else:
        leading_silence = trailing_silence = float(values.size / sample_rate)
    count = int(values.size)
    full_count = int(full_scale.sum())
    return {
        "pcm_sample_count_including_channels": count,
        "peak_abs_pcm16": peak,
        "peak_abs_normalized_to_32768": peak_normalized,
        "peak_dbfs": peak_dbfs,
        "rms_normalized_to_32768": rms_normalized,
        "rms_dbfs": rms_dbfs,
        "positive_full_scale_32767_count": int(positive.sum()),
        "negative_full_scale_minus_32768_count": int(negative.sum()),
        "full_scale_count": full_count,
        "full_scale_fraction": full_count / count,
        "longest_contiguous_full_scale_run_samples": full_run,
        "longest_same_exact_full_scale_value_run_samples": same_value_run,
        "leading_silence_estimate_seconds": leading_silence,
        "trailing_silence_estimate_seconds": trailing_silence,
        "silence_estimate_threshold_dbfs": SILENCE_ESTIMATE_THRESHOLD_DBFS,
        "silence_estimate_note": "Amplitude-threshold edge estimate only; not VAD and not a verified speech boundary.",
        "all_samples_finite": bool(np.isfinite(values).all()),
        "all_zero": bool(not np.any(values)),
    }


def machine_status(qc: dict[str, Any], *, text: str, duration: float) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if qc["pcm_sample_count_including_channels"] <= 0:
        return "excluded", ["empty_audio"]
    if not qc["all_samples_finite"]:
        return "excluded", ["non_finite_samples"]
    if qc["all_zero"]:
        return "excluded", ["all_zero_audio"]
    if duration < 0.15:
        return "excluded", ["duration_below_0.15_seconds"]
    if qc["full_scale_fraction"] >= FULL_SCALE_FRACTION_REVIEW or qc["longest_same_exact_full_scale_value_run_samples"] >= FULL_SCALE_RUN_REVIEW_SAMPLES:
        reasons.append("severe_full_scale_saturation_candidate")
    if (
        qc["rms_dbfs"] is not None
        and qc["rms_dbfs"] < VERY_QUIET_RMS_DBFS
        and (qc["peak_dbfs"] is None or qc["peak_dbfs"] < VERY_QUIET_PEAK_DBFS)
    ):
        reasons.append("very_quiet_candidate")
    visible_chars = sum(not ch.isspace() for ch in text)
    char_rate = visible_chars / duration if duration > 0 else math.inf
    if char_rate > TEXT_RATE_HIGH_CHARS_PER_SECOND:
        reasons.append("text_duration_rate_high_machine_flag")
    if duration >= TEXT_RATE_LOW_MIN_DURATION_SECONDS and char_rate < TEXT_RATE_LOW_CHARS_PER_SECOND:
        reasons.append("text_duration_rate_low_machine_flag")
    return ("needs_review", reasons) if reasons else ("candidate_for_review", [])


def temp_write_clip(
    *,
    clip_id: str,
    samples: Any,
    sample_rate: int,
    text_bytes: bytes,
    clip_dir: Path,
    sf: Any,
    np: Any,
    extra_metadata: dict[str, Any] | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    clip_dir.mkdir(parents=True, exist_ok=True)
    wav_path = clip_dir / f"{clip_id}.wav"
    text_path = clip_dir / f"{clip_id}.txt"
    sidecar_path = clip_dir / f"{clip_id}.json"
    existing = [p for p in (wav_path, text_path, sidecar_path) if p.exists()]
    if existing:
        raise FileExistsError(f"Refusing to replace existing clip output(s): {', '.join(map(str, existing))}")
    with tempfile.TemporaryDirectory(prefix=f".{clip_id}-", dir=clip_dir) as tmp_name:
        tmp_dir = Path(tmp_name)
        tmp_wav = tmp_dir / f"{clip_id}.wav"
        tmp_text = tmp_dir / f"{clip_id}.txt"
        tmp_sidecar = tmp_dir / f"{clip_id}.json"
        sf.write(str(tmp_wav), samples, sample_rate, format="WAV", subtype="PCM_16")
        tmp_text.write_bytes(text_bytes)
        info = inspect_wav(tmp_wav, sf=sf)
        qc = audio_qc(samples, sample_rate, np)
        audio_sha = sha256_file(tmp_wav)
        sidecar = {
            "clip_id": clip_id,
            "clip_audio_path": wav_path.relative_to(ROOT).as_posix(),
            "transcript_path": text_path.relative_to(ROOT).as_posix(),
            "clip_sha256": audio_sha,
            "transcript_sha256": hashlib.sha256(text_bytes).hexdigest(),
            "sample_rate_hz": info["sample_rate_hz"],
            "channels": info["channels"],
            "frames": info["frames"],
            "duration_seconds": info["duration_seconds"],
            "format": info["format"],
            "subtype": info["subtype"],
            "qc": qc,
        }
        sidecar.update(extra_metadata or {})
        tmp_sidecar.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        moved: list[Path] = []
        try:
            if any(p.exists() for p in (wav_path, text_path, sidecar_path)):
                raise FileExistsError(f"Output appeared during clip extraction: {clip_id}")
            for source, target in ((tmp_wav, wav_path), (tmp_text, text_path), (tmp_sidecar, sidecar_path)):
                os.replace(source, target)
                moved.append(target)
        except Exception:
            for path in moved:
                path.unlink(missing_ok=True)
            raise
    return wav_path, text_path, sidecar


def validate_existing_clip(
    *, clip_id: str, expected: dict[str, Any], clip_dir: Path
) -> dict[str, Any] | None:
    wav = clip_dir / f"{clip_id}.wav"
    txt = clip_dir / f"{clip_id}.txt"
    meta_path = clip_dir / f"{clip_id}.json"
    present = [p.exists() for p in (wav, txt, meta_path)]
    if not any(present):
        return None
    if not all(present):
        raise FileExistsError(f"Partial existing clip bundle; refusing overwrite: {clip_id}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if (
        meta.get("source_sha256", "").lower() != str(expected["source_sha256"]).lower()
        or meta.get("annotation_sha256", "").lower() != str(expected["annotation_sha256"]).lower()
        or meta.get("transcript_sha256", "").lower() != str(expected["transcript_sha256"]).lower()
        or sha256_file(wav).lower() != str(meta.get("clip_sha256", "")).lower()
        or hashlib.sha256(txt.read_bytes()).hexdigest().lower() != str(expected["transcript_sha256"]).lower()
    ):
        raise FileExistsError(f"Existing clip bundle provenance mismatch; refusing overwrite: {clip_id}")
    return meta


def validate_plan_row(row: dict[str, Any], plan_path: Path) -> None:
    clip_id = row.get("clip_id")
    if not isinstance(clip_id, str) or not SAFE_CLIP_ID.fullmatch(clip_id) or clip_id in {".", ".."}:
        raise ValueError(f"Invalid stable ASCII clip_id in {plan_path}: {clip_id!r}")
    required = (
        "source_audio",
        "source_sha256",
        "annotation_path",
        "annotation_sha256",
        "source_speaker_label",
        "segment_indices",
        "original_lexical_intervals_seconds",
        "original_text_segments",
        "text",
        "padded_start_seconds",
        "padded_end_seconds",
    )
    missing = [key for key in required if key not in row]
    if missing:
        raise ValueError(f"Plan row {clip_id} is missing required field(s): {missing}")
    if not isinstance(row["text"], str) or not row["text"]:
        raise ValueError(f"Plan row {clip_id} has empty text")
    expected_text = "\n".join(row["original_text_segments"])
    if row["text"] != expected_text:
        raise ValueError(f"Plan row {clip_id}: text differs from verbatim newline-joined original_text_segments")
    start, end = float(row["padded_start_seconds"]), float(row["padded_end_seconds"])
    if not (math.isfinite(start) and math.isfinite(end) and start >= 0 and end > start):
        raise ValueError(f"Plan row {clip_id} has invalid padded bounds: {start}, {end}")


def validate_source_annotation(
    row: dict[str, Any],
    *,
    verified_hashes: set[tuple[str, str]],
    annotation_cache: dict[str, list[dict[str, Any]]],
) -> None:
    source = Path(row["source_audio"])
    annotation = Path(row["annotation_path"])
    transcript = Path(row["transcript_path"]) if row.get("transcript_path") else None
    if not source.is_file():
        raise FileNotFoundError(f"Source audio is missing: {source}")
    if not annotation.is_file():
        raise FileNotFoundError(f"Source annotation JSON is missing: {annotation}")
    annotation_key = str(annotation.resolve())
    annotation_hash_key = (annotation_key, str(row["annotation_sha256"]).lower())
    if annotation_hash_key not in verified_hashes:
        if sha256_file(annotation).lower() != str(row["annotation_sha256"]).lower():
            raise ValueError(f"Source annotation SHA mismatch: {annotation}")
        verified_hashes.add(annotation_hash_key)
    if transcript is not None:
        if not transcript.is_file():
            raise FileNotFoundError(f"Source transcript TXT is missing: {transcript}")
        expected = row.get("transcript_sha256")
        transcript_key = (str(transcript.resolve()), str(expected).lower()) if expected else None
        if expected and transcript_key not in verified_hashes:
            if sha256_file(transcript).lower() != str(expected).lower():
                raise ValueError(f"Source transcript SHA mismatch: {transcript}")
            verified_hashes.add(transcript_key)
    if annotation_key not in annotation_cache:
        ann = json.loads(annotation.read_text(encoding="utf-8-sig"))
        segments = ann.get("segments") or ann.get("utterances")
        if not isinstance(segments, list):
            raise ValueError(f"Unrecognized paired annotation JSON schema: {annotation}")
        annotation_cache[annotation_key] = segments
    segments = annotation_cache[annotation_key]
    for index, original_text in zip(row["segment_indices"], row["original_text_segments"]):
        if not isinstance(index, int) or index < 0 or index >= len(segments):
            raise ValueError(f"Plan row {row['clip_id']} references invalid annotation segment index {index!r}")
        source_text = segments[index].get("text")
        if source_text != original_text:
            raise ValueError(
                f"Plan row {row['clip_id']} text mismatch at annotation segment {index}: source JSON differs"
            )


def make_manifest_row(
    *,
    plan_row: dict[str, Any],
    source_meta: dict[str, Any] | None,
    clip_meta: dict[str, Any] | None,
    clip_error: str | None = None,
) -> dict[str, Any]:
    row = dict(plan_row)
    row["plan_machine_status"] = row.get("machine_status")
    row["plan_human_review_status"] = row.get("human_review_status")
    plan_transcript_path = row.pop("transcript_path", None)
    plan_transcript_sha = row.pop("transcript_sha256", None)
    row["source_transcript_path"] = plan_transcript_path
    row["source_transcript_sha256"] = plan_transcript_sha
    row["transcript_provenance"] = "ASR_draft_not_human_verified"
    row["global_identity_verified"] = False
    row["background_audio_removed"] = False
    row["human_review_status"] = "pending"
    row["source_wav"] = source_meta.get("source_wav_path") if source_meta else None
    row["source_wav_sha256"] = source_meta.get("source_wav_sha256") if source_meta else None
    if clip_meta:
        row.update({k: v for k, v in clip_meta.items() if k not in {"qc"}})
        row["qc"] = clip_meta["qc"]
        row["transcript_path"] = clip_meta["transcript_path"]
        row["transcript_sha256"] = clip_meta["transcript_sha256"]
        row["machine_status"] = clip_meta["machine_status"]
        row["machine_review_flags"] = clip_meta["machine_review_flags"]
        row["exclusion_reason"] = clip_meta.get("exclusion_reason")
        row["clip_error"] = None
    else:
        row["clip_audio_path"] = None
        row["clip_sha256"] = None
        row["transcript_path"] = None
        row["transcript_sha256"] = None
        row["machine_status"] = "excluded"
        row["machine_review_flags"] = ["source_decode_or_clip_extraction_failed"]
        row["exclusion_reason"] = clip_error or "audio_not_available"
        row["clip_error"] = clip_error or "audio_not_available"
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--excluded", type=Path, default=DEFAULT_EXCLUDED)
    parser.add_argument("--source-inventory", type=Path, default=DEFAULT_SOURCE_INVENTORY)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--qc-summary", type=Path, default=DEFAULT_QC_SUMMARY)
    args = parser.parse_args()

    # These imports are audio I/O only; this script intentionally never imports torch/qwen.
    import numpy as np
    import soundfile as sf
    import imageio_ffmpeg

    plan_rows = read_jsonl(args.plan)
    excluded_rows = read_jsonl(args.excluded, required=False)
    if not plan_rows:
        raise ValueError(f"No candidate clip rows in plan: {args.plan}")
    clip_ids = [r.get("clip_id") for r in plan_rows]
    if len(set(clip_ids)) != len(clip_ids):
        raise ValueError("Plan contains duplicate clip_id values")
    for row in plan_rows:
        validate_plan_row(row, args.plan)

    # Verify source and annotation provenance for excluded-ledger entries too, and
    # use them to ensure all eight source files receive one cached decode.
    source_rows: dict[str, dict[str, Any]] = {}
    for row in [*plan_rows, *excluded_rows]:
        source_sha = str(row.get("source_sha256", "")).lower()
        if not source_sha:
            continue
        prior = source_rows.get(source_sha)
        if prior and Path(prior["source_audio"]).resolve() != Path(row["source_audio"]).resolve():
            raise ValueError(f"One source SHA is mapped to multiple paths: {source_sha}")
        source_rows[source_sha] = row

    verified_hashes: set[tuple[str, str]] = set()
    annotation_cache: dict[str, list[dict[str, Any]]] = {}
    for row in [*plan_rows, *excluded_rows]:
        # Hash each paired file once while checking every referenced segment verbatim.
        validate_source_annotation(row, verified_hashes=verified_hashes, annotation_cache=annotation_cache)

    # Ensure the plan plus separate exclusion ledger account for each annotation
    # segment exactly once, so no source utterance silently disappears or duplicates.
    accounted: dict[str, list[int]] = defaultdict(list)
    for row in [*plan_rows, *excluded_rows]:
        key = str(Path(row["annotation_path"]).resolve())
        accounted[key].extend(row["segment_indices"])
    for key, indices in accounted.items():
        if len(indices) != len(set(indices)):
            raise ValueError(f"Annotation segments appear more than once across plan/exclusions: {key}")
        expected = set(range(len(annotation_cache[key])))
        if set(indices) != expected:
            missing = sorted(expected - set(indices))
            extra = sorted(set(indices) - expected)
            raise ValueError(f"Annotation segment accounting mismatch for {key}: missing={missing}, extra={extra}")

    rates = source_rate_index(args.source_inventory)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    version_result = subprocess.run([ffmpeg, "-version"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    ffmpeg_version = version_result.stdout.splitlines()[0] if version_result.stdout else "unknown"

    root = ROOT
    dataset_root = args.dataset_root if args.dataset_root.is_absolute() else root / args.dataset_root
    source_wav_dir = dataset_root / "source_wav"
    clip_dir = dataset_root / "clips"
    source_wav_dir.mkdir(parents=True, exist_ok=True)
    clip_dir.mkdir(parents=True, exist_ok=True)

    source_meta_by_sha: dict[str, dict[str, Any]] = {}
    source_failures: dict[str, str] = {}
    source_inventory_rows: list[dict[str, Any]] = []
    for source_sha, source_row in source_rows.items():
        source_path = Path(source_row["source_audio"])
        try:
            source_meta = ensure_source_wav(
                source_path=source_path,
                expected_source_sha=source_sha,
                expected_rate=rates.get(source_sha),
                source_wav_dir=source_wav_dir,
                ffmpeg=ffmpeg,
                ffmpeg_version=ffmpeg_version,
                sf=sf,
            )
            source_meta_by_sha[source_sha] = source_meta
            # Normalize this field so the QC summary stays identical on cache hits.
            source_inventory_rows.append({
                **{k: v for k, v in source_meta.items() if k != "cache_status"},
                "status": "verified_source_wav_cache",
            })
        except Exception as exc:
            source_failures[source_sha] = f"{type(exc).__name__}: {exc}"
            source_inventory_rows.append({
                "source_id": source_id_for(source_sha),
                "source_audio": str(source_path),
                "source_sha256": source_sha,
                "status": "decode_failed",
                "error": source_failures[source_sha],
            })

    manifest_rows: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    flags_counts: Counter[str] = Counter()
    source_duration_seconds = 0.0
    clip_duration_by_status: defaultdict[str, float] = defaultdict(float)
    overall_start = time.perf_counter()

    for index, plan_row in enumerate(plan_rows, 1):
        clip_id = plan_row["clip_id"]
        source_sha = str(plan_row["source_sha256"]).lower()
        source_meta = source_meta_by_sha.get(source_sha)
        if source_meta is None:
            out_row = make_manifest_row(
                plan_row=plan_row,
                source_meta=None,
                clip_meta=None,
                clip_error=source_failures.get(source_sha, "source_decode_failed"),
            )
            manifest_rows.append(out_row)
            status_counts[out_row["machine_status"]] += 1
            flags_counts.update(out_row["machine_review_flags"])
            continue

        wav_path = root / source_meta["source_wav_path"]
        source_rate = int(source_meta["sample_rate_hz"])
        source_frames = int(source_meta["frames"])
        start_seconds = float(plan_row["padded_start_seconds"])
        end_seconds = float(plan_row["padded_end_seconds"])
        start_frame = max(0, int(math.floor(start_seconds * source_rate)))
        end_frame = min(source_frames, int(math.ceil(end_seconds * source_rate)))
        if end_frame <= start_frame:
            out_row = make_manifest_row(plan_row=plan_row, source_meta=source_meta, clip_meta=None, clip_error="invalid_or_empty_crop_bounds")
            manifest_rows.append(out_row)
            status_counts[out_row["machine_status"]] += 1
            flags_counts.update(out_row["machine_review_flags"])
            continue

        try:
            # Reuse an already-complete, provenance-matching bundle on reruns.
            text_bytes = plan_row["text"].encode("utf-8")
            expected_transcript_sha = hashlib.sha256(text_bytes).hexdigest()
            cached_meta = validate_existing_clip(
                clip_id=clip_id,
                expected={
                    "source_sha256": source_sha,
                    "annotation_sha256": plan_row["annotation_sha256"],
                    "transcript_sha256": expected_transcript_sha,
                },
                clip_dir=clip_dir,
            )
            if cached_meta is None:
                with sf.SoundFile(str(wav_path), "r") as source_wav:
                    if source_wav.samplerate != source_rate or source_wav.channels != 1 or source_wav.subtype != "PCM_16":
                        raise ValueError(f"Source cache format changed: {wav_path}")
                    source_wav.seek(start_frame)
                    samples = source_wav.read(frames=end_frame - start_frame, dtype="int16", always_2d=True)
                frames_written = int(samples.shape[0])
                if frames_written <= 0:
                    raise ValueError("empty_audio_crop")
                qc = audio_qc(samples, source_rate, np)
                duration = frames_written / source_rate
                status, machine_flags = machine_status(qc, text=plan_row["text"], duration=duration)
                source_transcript_path = Path(plan_row["transcript_path"]) if plan_row.get("transcript_path") else None
                transcript_hash = hashlib.sha256(text_bytes).hexdigest()
                _, _, clip_sidecar = temp_write_clip(
                    clip_id=clip_id,
                    samples=samples,
                    sample_rate=source_rate,
                    text_bytes=text_bytes,
                    clip_dir=clip_dir,
                    sf=sf,
                    np=np,
                    extra_metadata={
                        "source_sha256": source_sha,
                        "annotation_sha256": plan_row["annotation_sha256"],
                        "source_transcript_path": str(source_transcript_path) if source_transcript_path else None,
                        "source_transcript_sha256": plan_row.get("transcript_sha256"),
                        "source_start_frame": start_frame,
                        "source_end_frame_exclusive": start_frame + frames_written,
                        "actual_start_seconds": start_frame / source_rate,
                        "actual_end_seconds": (start_frame + frames_written) / source_rate,
                        "machine_status": status,
                        "machine_review_flags": machine_flags,
                        "exclusion_reason": machine_flags if status == "excluded" else None,
                        "transcript_provenance": "ASR_draft_not_human_verified",
                        "human_review_status": "pending",
                        "background_audio_removed": False,
                    },
                )
                clip_meta = clip_sidecar
            else:
                clip_meta = cached_meta
                status = str(clip_meta.get("machine_status", "needs_review"))
                machine_flags = list(clip_meta.get("machine_review_flags") or [])

            manifest_row = make_manifest_row(plan_row=plan_row, source_meta=source_meta, clip_meta=clip_meta)
            manifest_rows.append(manifest_row)
            status_counts[manifest_row["machine_status"]] += 1
            flags_counts.update(manifest_row["machine_review_flags"])
            clip_duration_by_status[manifest_row["machine_status"]] += float(manifest_row.get("duration_seconds", 0.0))
            source_duration_seconds += float(manifest_row.get("duration_seconds", 0.0))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            out_row = make_manifest_row(plan_row=plan_row, source_meta=source_meta, clip_meta=None, clip_error=error)
            manifest_rows.append(out_row)
            status_counts[out_row["machine_status"]] += 1
            flags_counts.update(out_row["machine_review_flags"])
        if index % 50 == 0:
            print(f"processed {index}/{len(plan_rows)} planned clips", flush=True)

    manifest_bytes = b"".join(
        (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        for row in manifest_rows
    )
    write_exact_or_validate(args.manifest if args.manifest.is_absolute() else root / args.manifest, manifest_bytes)

    elapsed = time.perf_counter() - overall_start
    summary = {
        "extraction_tool": "tools/extract_nino_dataset.py",
        "execution_python": sys.version.split()[0],
        "audio_io_packages": {"soundfile": sf.__version__, "numpy": np.__version__, "imageio_ffmpeg": imageio_ffmpeg.__version__},
        "ffmpeg_version": ffmpeg_version,
        "source_decode_method": "Each distinct source MP4 once: ffmpeg audio stream 0:a:0 to mono PCM16 WAV, no resampling or audio filters.",
        "crop_method": "Frame interval [floor(padded_start_seconds*source_rate), ceil(padded_end_seconds*source_rate)) clipped to source frame bounds; no trimming or resampling.",
        "source_count_unique": len(source_rows),
        "source_wav_cache_dir": (source_wav_dir.relative_to(root).as_posix()),
        "source_decode_status_counts": dict(Counter(row["status"] for row in source_inventory_rows)),
        "sources": source_inventory_rows,
        "planned_clip_count": len(plan_rows),
        "plan_excluded_clip_count": len(excluded_rows),
        "extracted_clip_count": sum(bool(row.get("clip_audio_path")) for row in manifest_rows),
        "machine_status_counts": dict(status_counts),
        "machine_review_flag_counts": dict(flags_counts),
        "duration_seconds_by_machine_status": dict(clip_duration_by_status),
        "extracted_padded_duration_seconds_including_machine_excluded": source_duration_seconds,
        "manifest_path": (args.manifest if args.manifest.is_absolute() else root / args.manifest).relative_to(root).as_posix(),
        "excluded_plan_path": (args.excluded if args.excluded.is_absolute() else root / args.excluded).relative_to(root).as_posix(),
        "machine_qc_thresholds": {
            "full_scale_fraction_review_at_or_above": FULL_SCALE_FRACTION_REVIEW,
            "same_full_scale_run_review_samples_at_or_above": FULL_SCALE_RUN_REVIEW_SAMPLES,
            "very_quiet_rms_dbfs_below": VERY_QUIET_RMS_DBFS,
            "very_quiet_peak_dbfs_below": VERY_QUIET_PEAK_DBFS,
            "high_visible_text_characters_per_second_above": TEXT_RATE_HIGH_CHARS_PER_SECOND,
            "low_visible_text_characters_per_second_below": TEXT_RATE_LOW_CHARS_PER_SECOND,
            "low_text_rate_min_duration_seconds": TEXT_RATE_LOW_MIN_DURATION_SECONDS,
        },
        "review_policy": {
            "human_review_status": "pending for every extracted clip",
            "candidate_for_review_is_not_trainable_or_verified": True,
            "transcript_provenance": "ASR draft copied verbatim; not human verified",
            "global_speaker_identity_verified": False,
            "background_audio_removed": False,
            "silence_edges_are_threshold_estimates_not_VAD": True,
            "known singing/event exclusions remain in the separate excluded ledger": True,
        },
    }
    summary_bytes = (json.dumps(summary, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    write_exact_or_validate(args.qc_summary if args.qc_summary.is_absolute() else root / args.qc_summary, summary_bytes)
    print(json.dumps({
        "sources": len(source_rows),
        "planned_clips": len(plan_rows),
        "plan_excluded": len(excluded_rows),
        "extracted_clips": summary["extracted_clip_count"],
        "machine_status_counts": summary["machine_status_counts"],
        "manifest": summary["manifest_path"],
        "qc_summary": (args.qc_summary if args.qc_summary.is_absolute() else root / args.qc_summary).relative_to(root).as_posix(),
        "elapsed_seconds": round(elapsed, 2),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
