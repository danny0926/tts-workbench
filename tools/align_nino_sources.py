"""Estimate Japanese word timings against full context audio without editing clips.

This writes alignment sidecars and conservative crop proposals only. It never
rewrites source transcripts, existing manifests, WAV/TXT clips, or review data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen3-ForcedAligner-0.6B-hf"
MODEL_REVISION = "c07281df297b9905d24a508279258cccf987a064"
MODEL_FILE_SHA256 = "00568245ceca5af1991d28562a75fe1ddc9bfeb041c27fda66947ea05c47fb86"
MODEL_FILE_SIZE = 1_835_545_960
DEFAULT_MODEL_PATH = (
    ROOT / ".cache_hf/align/hub/models--Qwen--Qwen3-ForcedAligner-0.6B-hf/snapshots" / MODEL_REVISION
)
DEFAULT_CONTEXT = ROOT / "outputs/dataset/realignment/context-nino-05_summer_memories_nino_study_cd-s0012-s0013.json"
DEFAULT_OUTPUT = ROOT / "outputs/dataset/realignment/forced-alignments.jsonl"
DEFAULT_SUMMARY = ROOT / "outputs/dataset/realignment/forced-alignment-summary.json"
DEFAULT_BASELINE_PLAN = ROOT / "outputs/dataset/plan.jsonl"
DEFAULT_REPAIR_PLAN = ROOT / "outputs/dataset/repair-planning-v2/plan.jsonl"
OUTPUT_ROOT = (ROOT / "outputs/dataset/realignment").resolve()
PRE_MARGIN_SECONDS = 0.15
TAIL_MARGIN_SECONDS = 0.20
TIGHT_GAP_SECONDS = 0.30
LONG_WORD_SECONDS = 2.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8-sig") as f:
        for number, line in enumerate(f, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{number}: expected JSON object")
                out.append(value)
    return out


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_contexts(context_json: Path | None, contexts_jsonl: Path | None) -> list[dict[str, Any]]:
    if context_json and contexts_jsonl:
        raise ValueError("Use either --context-json or --contexts-jsonl, not both")
    if context_json:
        value = json.loads(context_json.read_text(encoding="utf-8-sig"))
        if not isinstance(value, dict):
            raise ValueError("Context JSON must contain an object")
        return [value]
    if contexts_jsonl:
        return read_jsonl(contexts_jsonl)
    return [json.loads(DEFAULT_CONTEXT.read_text(encoding="utf-8-sig"))]


def get_clip_targets(context: dict[str, Any], plan_by_id: dict[str, dict[str, Any]]) -> list[tuple[str, list[int]]]:
    ids = context.get("target_clip_ids")
    clip_segment_rows = context.get("target_clip_to_segment_indices")
    if not isinstance(ids, list) and isinstance(clip_segment_rows, list):
        ids = [row.get("clip_id") for row in clip_segment_rows if isinstance(row, dict) and isinstance(row.get("clip_id"), str)]
    if not isinstance(ids, list):
        single = context.get("target_clip_id")
        ids = [single] if isinstance(single, str) else []
    explicit = context.get("target_clip_segment_indices", {})
    clip_to_segments = {
        row.get("clip_id"): row.get("target_segment_indices")
        for row in clip_segment_rows or []
        if isinstance(row, dict)
    }
    result = []
    for clip_id in ids:
        row = plan_by_id.get(clip_id)
        indices = explicit.get(clip_id) if isinstance(explicit, dict) else None
        if not isinstance(indices, list):
            indices = clip_to_segments.get(clip_id)
        if not isinstance(indices, list) and row:
            indices = row.get("segment_indices")
        if not isinstance(indices, list):
            indices = [s["segment_index"] for s in context.get("segments", []) if s.get("is_target")]
        result.append((clip_id, [int(i) for i in indices]))
    if not result:
        indices = [int(i) for i in context.get("target_segment_indices", [])]
        if indices:
            result.append((str(context.get("context_id", "context")), indices))
    return result


def gpu_memory_mib() -> list[dict[str, int]] | None:
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
        if result.returncode:
            return None
        return [
            {"used_mib": int(a.strip()), "total_mib": int(b.strip())}
            for line in result.stdout.splitlines()
            for a, b in [line.split(",", 1)]
        ] or None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def write_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_mono_16k(path: Path, expected_sample_rate: int):
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio.functional as audio_functional

    samples, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    original_channels = int(samples.shape[1])
    mono = samples.mean(axis=1)
    audio = torch.from_numpy(np.ascontiguousarray(mono))
    if sample_rate != expected_sample_rate:
        audio = audio_functional.resample(audio, orig_freq=sample_rate, new_freq=expected_sample_rate)
    return audio.numpy().astype(np.float32, copy=False), {
        "original_sample_rate_hz": int(sample_rate),
        "original_channels": original_channels,
        "model_sample_rate_hz": int(expected_sample_rate),
        "duration_seconds": len(samples) / sample_rate,
        "downmixed_to_mono": original_channels > 1,
        "resampler": "torchaudio.functional.resample" if sample_rate != expected_sample_rate else "none",
    }


def map_word_ownership(processor: Any, segments: list[dict[str, Any]], word_list: list[str], target_indices: set[int]) -> tuple[list[dict[str, Any]], list[str]]:
    owners: list[dict[str, Any]] = []
    flattened: list[str] = []
    character_spans: list[tuple[int, int, dict[str, Any]]] = []
    character_cursor = 0
    for segment in segments:
        text = segment.get("text")
        if not isinstance(text, str):
            raise ValueError(f"Segment {segment.get('segment_index')} lacks text")
        tokens = list(processor.split_words_for_alignment(text, language="Japanese"))
        flattened.extend(tokens)
        seg_index = int(segment["segment_index"])
        owner = {
            "segment_index": seg_index,
            "is_target": bool(segment.get("is_target", False)) or seg_index in target_indices,
            # These are the original whole-segment spans in the joined source
            # text, not offsets in the processor's punctuation-free characters.
            "char_start": segment.get("char_start"),
            "char_end": segment.get("char_end"),
        }
        owners.extend(dict(owner) for _ in tokens)
        character_end = character_cursor + sum(len(token) for token in tokens)
        if character_end > character_cursor:
            character_spans.append((character_cursor, character_end, owner))
        character_cursor = character_end
    if flattened == word_list:
        return owners, []

    def ambiguous(reason: str | None = None) -> tuple[list[dict[str, Any]], list[str]]:
        flags = ["source_segment_word_mapping_mismatch"]
        if reason:
            flags.append(reason)
        return ([{"segment_index": None, "is_target": False, "char_start": None, "char_end": None} for _ in word_list], flags)

    # Nagisa may split the same Japanese characters differently in a segment
    # and its full context (e.g. "でも" versus "で", "も"). Only boundary changes
    # are recoverable: require exact characters, with no extra normalization.
    if "".join(flattened) != "".join(word_list):
        return ambiguous()
    owners = []
    span_index = 0
    character_cursor = 0
    for word in word_list:
        character_end = character_cursor + len(word)
        while span_index < len(character_spans) and character_cursor >= character_spans[span_index][1]:
            span_index += 1
        if not word or span_index == len(character_spans):
            return ambiguous("source_segment_word_mapping_empty_or_uncovered_word")
        span_start, span_end, owner = character_spans[span_index]
        if character_cursor < span_start or character_end > span_end:
            # A model word spanning source segments has no unique owner. Do not
            # guess one or expose a partially mapped target as a complete crop.
            return ambiguous("source_segment_word_mapping_crosses_segment_boundary")
        owners.append(dict(owner))
        character_cursor = character_end
    return owners, ["source_segment_word_mapping_exact_character_partition"]


def segment_bounds(index: int, segments: list[dict[str, Any]], owners: list[dict[str, Any]], words: list[dict[str, Any]]) -> tuple[float | None, float | None, list[int]]:
    positions = [i for i, owner in enumerate(owners) if owner["segment_index"] == index]
    if not positions or any(words[i].get("decoded_start_seconds") is None for i in positions):
        return None, None, positions
    return min(words[i]["decoded_start_seconds"] for i in positions), max(words[i]["decoded_end_seconds"] for i in positions), positions


def build_clip_proposals(
    context: dict[str, Any],
    segments: list[dict[str, Any]],
    owners: list[dict[str, Any]],
    words: list[dict[str, Any]],
    targets: list[tuple[str, list[int]]],
    baseline_by_id: dict[str, dict[str, Any]],
    repair_by_id: dict[str, dict[str, Any]],
    events: list[dict[str, Any]],
    pre_margin: float,
    tail_margin: float,
) -> list[dict[str, Any]]:
    segment_by_index = {int(s["segment_index"]): s for s in segments}
    proposals = []
    context_start = float(context["context_start_seconds"])
    context_end = float(context["context_end_seconds"])
    for clip_id, target_indices in targets:
        flags: list[str] = []
        present = sorted(i for i in target_indices if i in segment_by_index)
        if len(present) != len(target_indices):
            flags.append("target_segment_missing_from_context")
        positions = [i for i, owner in enumerate(owners) if owner["segment_index"] in present]
        if not positions:
            flags.append("target_has_no_mapped_alignment_words")
            onset = target_end = None
        else:
            onset = min(words[i]["decoded_start_seconds"] for i in positions)
            target_end = max(words[i]["decoded_end_seconds"] for i in positions)
        earlier = [i for i in segment_by_index if present and i < min(present)]
        later = [i for i in segment_by_index if present and i > max(present)]
        previous_index = max(earlier) if earlier else None
        next_index = min(later) if later else None
        previous_end = None
        next_start = None
        if previous_index is not None:
            _, previous_end, previous_positions = segment_bounds(previous_index, segments, owners, words)
            if previous_end is None:
                flags.append("aligned_previous_neighbor_missing")
        if next_index is not None:
            next_start, _, next_positions = segment_bounds(next_index, segments, owners, words)
            if next_start is None:
                flags.append("aligned_next_neighbor_missing")

        start_gap = None if onset is None or previous_end is None else onset - previous_end
        proposed_start = None
        if onset is not None and previous_index is None:
            flags.append("aligned_previous_neighbor_missing")
        elif onset is not None and previous_end is not None:
            if start_gap is not None and start_gap <= 0:
                flags.append("aligned_target_overlaps_previous_neighbor")
            else:
                assert start_gap is not None
                if start_gap < 2 * pre_margin:
                    flags.append("aligned_previous_gap_tight_under_0_30s")
                # All alignment values here are context-relative. Clamp in that
                # coordinate system; convert to source-absolute only when saved.
                proposed_start = max(0.0, onset - min(pre_margin, start_gap / 2))

        end_gap = None if target_end is None or next_start is None else next_start - target_end
        proposed_end = None
        if target_end is not None and next_index is None:
                proposed_end = min(context_end - context_start, target_end + tail_margin)
        elif target_end is not None and next_start is not None:
            if end_gap is not None and end_gap <= 0:
                flags.append("aligned_target_overlaps_next_neighbor")
            else:
                assert end_gap is not None
                if end_gap < 2 * tail_margin:
                    flags.append("aligned_next_gap_tight_under_0_40s")
                proposed_end = min(context_end - context_start, target_end + min(tail_margin, end_gap / 2))

        relevant_events = []
        for event in events:
            try:
                es_source = float(event.get("start_seconds", event.get("start_time", 0)))
                ee_source = float(event.get("end_seconds", event.get("end_time", 0)))
            except (TypeError, ValueError):
                continue
            es = es_source - context_start
            ee = ee_source - context_start
            if proposed_start is not None and onset is not None and es < onset and ee > proposed_start:
                relevant_events.append({"start_seconds_source": es_source, "end_seconds_source": ee_source, "start_seconds_context": es, "end_seconds_context": ee, "text": event.get("text"), "event_index": event.get("event_index"), "speaker_label": event.get("speaker_label"), "time_basis": "source_absolute_from_events_intersecting_context", "annotation_path": context.get("annotation_path"), "annotation_sha256": context.get("annotation_sha256")})
                # Annotation/event timing is unverified and may overlap speech.
                # Preserve it as review evidence; never trim estimated speech.
                flags.append("unverified_annotation_event_overlaps_candidate_head_review")
            if proposed_end is not None and target_end is not None and es < proposed_end and ee > target_end:
                relevant_events.append({"start_seconds_source": es_source, "end_seconds_source": ee_source, "start_seconds_context": es, "end_seconds_context": ee, "text": event.get("text"), "event_index": event.get("event_index"), "speaker_label": event.get("speaker_label"), "time_basis": "source_absolute_from_events_intersecting_context", "annotation_path": context.get("annotation_path"), "annotation_sha256": context.get("annotation_sha256")})
                flags.append("unverified_annotation_event_overlaps_candidate_tail_review")

        baseline = baseline_by_id.get(clip_id, {})
        repaired = repair_by_id.get(clip_id, {})
        target_annotated_start = min(
            (float(segment_by_index[i]["start_seconds"]) for i in present if segment_by_index[i].get("start_seconds") is not None),
            default=None,
        )
        target_annotated_end = max(
            (float(segment_by_index[i]["end_seconds"]) for i in present if segment_by_index[i].get("end_seconds") is not None),
            default=None,
        )
        onset_source = None if onset is None else context_start + onset
        if onset_source is not None and target_annotated_start is not None and onset_source < target_annotated_start - 0.10:
            flags.append("aligned_onset_before_annotation_start")
        if onset_source is not None and target_annotated_start is not None and onset_source > target_annotated_start + 0.20:
            flags.append("aligned_onset_after_annotation_start")
        if start_gap is not None and start_gap < 0:
            flags.append("neighbor_alignment_overlap_requires_listening")
        target_positions = [i for i, owner in enumerate(owners) if owner["segment_index"] in present]
        first_target_word = min(target_positions) if target_positions else None
        last_target_word = max(target_positions) if target_positions else None
        previous_last_word = None
        next_first_word = None
        if previous_index is not None:
            prev_positions = [i for i, owner in enumerate(owners) if owner["segment_index"] == previous_index]
            previous_last_word = max(prev_positions) if prev_positions else None
        if next_index is not None:
            next_positions = [i for i, owner in enumerate(owners) if owner["segment_index"] == next_index]
            next_first_word = min(next_positions) if next_positions else None
        def word_boundary_anomalies(word_index: int | None) -> list[str]:
            if word_index is None:
                return ["boundary_word_missing"]
            result = []
            if "decoder_timestamp_repair_changed_word" in words[word_index].get("review_flags", []):
                result.append("decoder_repaired")
            row = words[word_index]
            if row.get("raw_end_seconds_context", 0) <= row.get("raw_start_seconds_context", 0):
                result.append("raw_nonpositive_duration")
            if row.get("decoded_end_seconds_context", 0) <= row.get("decoded_start_seconds_context", 0):
                result.append("decoded_nonpositive_duration")
            if "decoded_word_outside_context_audio" in row.get("review_flags", []):
                result.append("outside_context_audio")
            return result
        head_uncertainty = {
            "target_first_word": word_boundary_anomalies(first_target_word),
            "previous_neighbor_last_word": word_boundary_anomalies(previous_last_word),
        }
        tail_uncertainty = {
            "target_last_word": word_boundary_anomalies(last_target_word),
            "next_neighbor_first_word": word_boundary_anomalies(next_first_word) if next_index is not None else [],
        }
        head_uncertain = any(head_uncertainty.values())
        tail_uncertain = any(tail_uncertainty.values())
        first_target_repaired = bool(first_target_word is not None and "decoder_timestamp_repair_changed_word" in words[first_target_word].get("review_flags", []))
        last_target_repaired = bool(last_target_word is not None and "decoder_timestamp_repair_changed_word" in words[last_target_word].get("review_flags", []))
        if first_target_repaired:
            flags.append("first_target_word_timestamp_decoder_repaired_review")
        if last_target_repaired:
            flags.append("last_target_word_timestamp_decoder_repaired_review")
        if head_uncertain:
            flags.append("head_boundary_timing_uncertain")
            proposed_start = None
        if tail_uncertain:
            flags.append("tail_boundary_timing_uncertain")
            proposed_end = None

        proposals.append({
            "clip_id": clip_id,
            "target_segment_indices": present,
            "target_annotated_start_seconds": target_annotated_start,
            "target_annotated_end_seconds": target_annotated_end,
            "estimated_target_aligned_onset_source_seconds": None if onset is None else round(context_start + onset, 3),
            "estimated_target_aligned_end_source_seconds": None if target_end is None else round(context_start + target_end, 3),
            "previous_segment_index": previous_index,
            "previous_aligned_end_source_seconds": None if previous_end is None else round(context_start + previous_end, 3),
            "next_segment_index": next_index,
            "next_aligned_start_source_seconds": None if next_start is None else round(context_start + next_start, 3),
            "aligned_gap_before_target_seconds": None if start_gap is None else round(start_gap, 3),
            "aligned_gap_after_target_seconds": None if end_gap is None else round(end_gap, 3),
            "baseline_plan_start_seconds": baseline.get("padded_start_seconds"),
            "baseline_plan_end_seconds": baseline.get("padded_end_seconds"),
            "lead_padding_repair_plan_start_seconds": repaired.get("padded_start_seconds"),
            "lead_padding_repair_plan_end_seconds": repaired.get("padded_end_seconds"),
            "proposed_cut_start_source_seconds": None if proposed_start is None else round(context_start + proposed_start, 3),
            "proposed_cut_end_source_seconds": (
                round(float(baseline.get("padded_end_seconds")), 3)
                if tail_uncertain and baseline.get("padded_end_seconds") is not None
                else None if proposed_end is None else round(context_start + proposed_end, 3)
            ),
            "tail_end_withheld_baseline_retained": bool(tail_uncertain and baseline.get("padded_end_seconds") is not None),
            "requested_pre_margin_seconds": pre_margin,
            "requested_tail_margin_seconds": tail_margin,
            "head_boundary_uncertain": head_uncertain,
            "head_uncertainty_evidence": head_uncertainty,
            "tail_boundary_uncertain": tail_uncertain,
            "tail_uncertainty_evidence": tail_uncertainty,
            "target_boundary_word_timestamps_decoder_repaired": {
                "first_target_word": first_target_repaired,
                "last_target_word": last_target_repaired,
            },
            "target_boundary_word_raw_vs_decoded": {
                "first_target_word": None if first_target_word is None else {
                    "raw_start_seconds_context": words[first_target_word]["raw_start_seconds_context"],
                    "raw_end_seconds_context": words[first_target_word]["raw_end_seconds_context"],
                    "decoded_start_seconds_context": words[first_target_word]["decoded_start_seconds_context"],
                    "decoded_end_seconds_context": words[first_target_word]["decoded_end_seconds_context"],
                },
                "last_target_word": None if last_target_word is None else {
                    "raw_start_seconds_context": words[last_target_word]["raw_start_seconds_context"],
                    "raw_end_seconds_context": words[last_target_word]["raw_end_seconds_context"],
                    "decoded_start_seconds_context": words[last_target_word]["decoded_start_seconds_context"],
                    "decoded_end_seconds_context": words[last_target_word]["decoded_end_seconds_context"],
                },
            },
            "event_constraints_considered": relevant_events,
            "review_flags": sorted(set(flags)),
            "proposal_status": "requires_human_review",
            "warning": "Forced-aligner times are model estimates, not phoneme-level guarantees or human verification; no WAV was created or changed.",
        })
    return proposals


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--context-json", type=Path)
    group.add_argument("--contexts-jsonl", type=Path)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--baseline-plan", type=Path, default=DEFAULT_BASELINE_PLAN)
    parser.add_argument("--repair-plan", type=Path, default=DEFAULT_REPAIR_PLAN)
    parser.add_argument("--pre-margin-seconds", type=float, default=PRE_MARGIN_SECONDS)
    parser.add_argument("--tail-margin-seconds", type=float, default=TAIL_MARGIN_SECONDS)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip context IDs already present in the output JSONL")
    args = parser.parse_args()

    model_path = resolve_project_path(args.model_path)
    output_path = resolve_project_path(args.output)
    summary_path = resolve_project_path(args.summary)
    for path in (output_path, summary_path):
        if not path.resolve().is_relative_to(OUTPUT_ROOT):
            parser.error(f"Output must stay under {OUTPUT_ROOT}: {path}")
    if not model_path.is_dir():
        parser.error(f"Pinned local model snapshot is missing: {model_path}")
    if not (model_path / "model.safetensors").is_file():
        parser.error(f"Pinned model weight file is missing: {model_path / 'model.safetensors'}")
    if not args.overwrite and not args.resume and (output_path.exists() or summary_path.exists()):
        parser.error("Refusing to replace alignment output; choose another path or pass --overwrite explicitly")
    if args.resume and summary_path.exists():
        parser.error("Resume requires a fresh summary path")
    if args.pre_margin_seconds < 0 or args.tail_margin_seconds < 0:
        parser.error("Padding margins cannot be negative")

    contexts = read_contexts(args.context_json, args.contexts_jsonl)
    if not contexts:
        parser.error("No alignment contexts supplied")
    baseline_rows = read_jsonl(resolve_project_path(args.baseline_plan))
    repair_path = resolve_project_path(args.repair_plan)
    repair_rows = read_jsonl(repair_path) if repair_path.is_file() else []
    baseline_by_id = {row["clip_id"]: row for row in baseline_rows}
    repair_by_id = {row["clip_id"]: row for row in repair_rows}

    import numpy as np
    import torch
    import torchaudio
    import soundfile as sf
    from transformers import AutoModelForTokenClassification, AutoProcessor
    from transformers.models.qwen3_asr.processing_qwen3_asr import _fix_timestamps

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable; model initialization was not attempted")
    if not torch.cuda.is_bf16_supported():
        parser.error("CUDA BF16 is unavailable; model initialization was not attempted")

    # Verify downloaded bytes against the pinned official HF LFS digest before loading.
    weights = model_path / "model.safetensors"
    if weights.stat().st_size != MODEL_FILE_SIZE or sha256_file(weights) != MODEL_FILE_SHA256:
        parser.error("Local model.safetensors does not match the pinned official HF size/SHA256")

    gpu_before = gpu_memory_mib()
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForTokenClassification.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": 0},
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model.eval()
    device = torch.device("cuda:0")
    gpu_after_model = gpu_memory_mib()
    print(json.dumps({"model_loaded": True, "gpu_memory_before_model_mib": gpu_before, "gpu_memory_after_model_mib": gpu_after_model}, ensure_ascii=False), flush=True)
    timestamp_token_id = int(model.config.timestamp_token_id)
    timestamp_step_ms = float(processor.timestamp_segment_time)
    input_sample_rate = int(processor.feature_extractor.sampling_rate)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    prior_records = read_jsonl(output_path) if args.resume and output_path.is_file() else []
    existing_context_ids = {str(row.get("context_id")) for row in prior_records}
    completed = list(prior_records)
    try:
        for context_index, context in enumerate(contexts, 1):
            context_id = str(context.get("context_id") or context.get("target_clip_id") or f"context-{context_index:04d}")
            if context_id in existing_context_ids:
                continue
            audio_path_value = context.get("context_wav_path") or context.get("clip_audio_path")
            transcript = context.get("full_matching_asr_draft_text") or context.get("text")
            if not isinstance(audio_path_value, str) or not isinstance(transcript, str) or not transcript.strip():
                raise ValueError(f"{context_id}: missing context WAV or transcript")
            context_status = context.get("context_status")
            if context_status and context_status != "ready_for_review":
                record = {
                    "schema_version": 1,
                    "context_id": context_id,
                    "status": "pending_context_review",
                    "context_status": context_status,
                    "status_reasons": context.get("status_reasons", []),
                    "target_clip_ids": [clip_id for clip_id, _ in get_clip_targets(context, baseline_by_id)],
                    "source_audio": context.get("source_audio"),
                    "source_sha256": context.get("source_sha256"),
                    "annotation_path": context.get("annotation_path"),
                    "annotation_sha256": context.get("annotation_sha256"),
                    "context_start_seconds": context.get("context_start_seconds"),
                    "context_end_seconds": context.get("context_end_seconds"),
                    "human_review_status": context.get("human_review_status", "pending"),
                    "alignment_performed": False,
                    "clip_statuses": [
                        {"clip_id": clip_id, "status": "pending_context_review", "review_flags": context.get("status_reasons", []), "human_review_status": context.get("human_review_status", "pending")}
                        for clip_id, _ in get_clip_targets(context, baseline_by_id)
                    ],
                    "interpretation": "Context was retained for review and not aligned because its upstream context status is not ready_for_review.",
                }
                write_jsonl_row(output_path, record)
                completed.append(record)
                print(json.dumps({"context_id": context_id, "status": record["status"], "context_status": context_status}, ensure_ascii=False), flush=True)
                continue
            audio_path = resolve_project_path(audio_path_value)
            if not audio_path.is_file():
                raise FileNotFoundError(f"{context_id}: context WAV missing: {audio_path}")
            audio_sha = sha256_file(audio_path)
            expected_audio_sha = context.get("context_wav_sha256") or context.get("clip_sha256")
            if expected_audio_sha and audio_sha.lower() != str(expected_audio_sha).lower():
                raise ValueError(f"{context_id}: context WAV SHA256 differs from context metadata")
            context_duration = float(context.get("actual_duration_seconds", context.get("duration_seconds", float(context["context_end_seconds"]) - float(context["context_start_seconds"]))))
            target_pairs = get_clip_targets(context, baseline_by_id)
            target_indices = {i for _, indices in target_pairs for i in indices}
            audio, audio_meta = load_mono_16k(audio_path, input_sample_rate)
            inputs, word_lists = processor.prepare_forced_aligner_inputs(
                audio=audio,
                transcript=transcript,
                language="Japanese",
            )
            if len(word_lists) != 1:
                raise ValueError(f"{context_id}: expected one alignment sequence, got {len(word_lists)}")
            word_list = list(word_lists[0])
            segments = context.get("segments") or []
            owners, mapping_flags = map_word_ownership(processor, segments, word_list, target_indices)
            inputs = inputs.to(device)
            started_at = time.time()
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(**inputs)
            gpu_after_context = gpu_memory_mib()
            print(json.dumps({"context_id": context_id, "forward_pass_complete": True, "gpu_memory_after_context_mib": gpu_after_context}, ensure_ascii=False), flush=True)
            logits = output.logits
            predicted = logits.argmax(dim=-1)
            mask = inputs["input_ids"] == timestamp_token_id
            raw_bins_tensor = predicted[0][mask[0]]
            raw_bins = [int(x) for x in raw_bins_tensor.detach().cpu().tolist()]
            expected_timestamp_count = 2 * len(word_list)
            if len(raw_bins) != expected_timestamp_count:
                raise ValueError(
                    f"{context_id}: timestamp marker count {len(raw_bins)} != 2 * word count {expected_timestamp_count}"
                )
            raw_ms = np.asarray(raw_bins, dtype=np.float64) * timestamp_step_ms
            fixed_ms = np.asarray(_fix_timestamps(raw_ms.copy()), dtype=np.float64)
            if len(fixed_ms) != expected_timestamp_count:
                raise ValueError(f"{context_id}: timestamp repair changed the timestamp count")
            decoded = processor.decode_forced_alignment(
                logits=logits,
                input_ids=inputs["input_ids"],
                word_lists=word_lists,
                timestamp_token_id=timestamp_token_id,
            )[0]
            if len(decoded) != len(word_list):
                raise ValueError(f"{context_id}: decoded word count differs from native word list")

            raw_nonmonotonic = [i for i in range(len(raw_ms) - 1) if raw_ms[i + 1] < raw_ms[i]]
            changed_positions = [i for i, (a, b) in enumerate(zip(raw_ms.tolist(), fixed_ms.tolist())) if a != b]
            word_rows = []
            global_flags = list(mapping_flags)
            for i, (token, timing) in enumerate(zip(word_list, decoded)):
                raw_start = float(raw_ms[2 * i] / 1000.0)
                raw_end = float(raw_ms[2 * i + 1] / 1000.0)
                fixed_start = float(timing["start_time"])
                fixed_end = float(timing["end_time"])
                owner = owners[i] if i < len(owners) else {"segment_index": None, "is_target": False}
                token_flags = []
                if raw_end <= raw_start:
                    token_flags.append("raw_word_duration_nonpositive")
                if fixed_end <= fixed_start:
                    token_flags.append("decoded_word_duration_nonpositive")
                if fixed_end - fixed_start > LONG_WORD_SECONDS:
                    token_flags.append("decoded_word_duration_over_2_seconds_review")
                if fixed_start < 0 or fixed_end > context_duration:
                    token_flags.append("decoded_word_outside_context_audio")
                if 2 * i in changed_positions or 2 * i + 1 in changed_positions:
                    token_flags.append("decoder_timestamp_repair_changed_word")
                word_rows.append({
                    "token_index": i,
                    "segment_index": owner.get("segment_index"),
                    "is_target_segment": bool(owner.get("is_target", False)),
                    "source_segment_char_start": owner.get("char_start"),
                    "source_segment_char_end": owner.get("char_end"),
                    "word": token,
                    "raw_timestamp_start_bin": raw_bins[2 * i],
                    "raw_timestamp_end_bin": raw_bins[2 * i + 1],
                    "raw_start_seconds_context": round(raw_start, 3),
                    "raw_end_seconds_context": round(raw_end, 3),
                    "decoded_start_seconds_context": round(fixed_start, 3),
                    "decoded_end_seconds_context": round(fixed_end, 3),
                    "decoded_start_seconds": round(fixed_start, 3),
                    "decoded_end_seconds": round(fixed_end, 3),
                    "source_absolute_start_seconds": round(float(context["context_start_seconds"]) + fixed_start, 3),
                    "source_absolute_end_seconds": round(float(context["context_start_seconds"]) + fixed_end, 3),
                    "raw_minus_decoded_start_seconds": round(raw_start - fixed_start, 3),
                    "raw_minus_decoded_end_seconds": round(raw_end - fixed_end, 3),
                    "confidence": None,
                    "review_flags": token_flags,
                })
                global_flags.extend(token_flags)
            if raw_nonmonotonic:
                global_flags.append("raw_timestamp_sequence_nonmonotonic")
            if changed_positions:
                global_flags.append("native_decoder_timestamp_repair_applied")
            if len(owners) != len(word_list):
                global_flags.append("transcript_word_segment_coverage_incomplete")
            if len({w["segment_index"] for w in word_rows if w["segment_index"] is not None}) < len(segments):
                global_flags.append("one_or_more_context_segments_have_no_mapped_words")

            segment_rows = []
            for i, segment in enumerate(segments):
                seg_index = int(segment["segment_index"])
                seg_positions = [j for j, owner in enumerate(owners) if owner["segment_index"] == seg_index]
                source_start = segment.get("start_seconds")
                source_end = segment.get("end_seconds")
                segment_rows.append({
                    "segment_index": seg_index,
                    "text": segment.get("text"),
                    "is_target": bool(segment.get("is_target", False)) or seg_index in target_indices,
                    "char_start": segment.get("char_start"),
                    "char_end": segment.get("char_end"),
                    "speaker_label": segment.get("speaker_label"),
                    "source_annotation_start_seconds": source_start,
                    "source_annotation_end_seconds": source_end,
                    "word_start_index": min(seg_positions) if seg_positions else None,
                    "word_end_index_exclusive": max(seg_positions) + 1 if seg_positions else None,
                    "word_count": len(seg_positions),
                    "aligned_start_source_seconds": min((word_rows[j]["source_absolute_start_seconds"] for j in seg_positions), default=None),
                    "aligned_end_source_seconds": max((word_rows[j]["source_absolute_end_seconds"] for j in seg_positions), default=None),
                    "annotation_flags": segment.get("flags", []),
                    "excluded_reason": segment.get("exclusion_reason"),
                    "alignment_status": "mapped" if seg_positions else "no_mapped_words",
                })

            annotation_path = context.get("annotation_path")
            # Context-builder event intervals are context-relative and carry
            # source provenance; never reinterpret them as source-absolute.
            events = context.get("events_intersecting_context") or []
            targets = target_pairs
            proposals = build_clip_proposals(
                context, segments, owners, word_rows, targets,
                baseline_by_id, repair_by_id, events,
                args.pre_margin_seconds, args.tail_margin_seconds,
            )
            record = {
                "schema_version": 1,
                "context_id": context_id,
                "status": "completed",
                "alignment_performed": True,
                "target_clip_ids": [clip_id for clip_id, _ in targets],
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "model_weight_sha256": MODEL_FILE_SHA256,
                "model_weight_size_bytes": MODEL_FILE_SIZE,
                "model_license_card": "apache-2.0",
                "device": "cuda:0",
                "dtype": "bfloat16",
                "attention_implementation": "sdpa",
                "torch_num_threads": 4,
                "torch_num_interop_threads": 1,
                "language": "Japanese",
                "timestamp_token_id": timestamp_token_id,
                "timestamp_segment_time_ms": timestamp_step_ms,
                "reported_time_resolution_seconds": timestamp_step_ms / 1000.0,
                "confidence_available": False,
                "context_audio_path": str(audio_path.relative_to(ROOT)) if audio_path.is_relative_to(ROOT) else str(audio_path),
                "context_audio_sha256": audio_sha,
                "source_audio": context.get("source_audio"),
                "source_sha256": context.get("source_sha256"),
                "annotation_path": annotation_path,
                "annotation_sha256": context.get("annotation_sha256"),
                "source_transcript_path": context.get("source_transcript_path"),
                "source_transcript_sha256": context.get("source_transcript_sha256"),
                "context_start_seconds": float(context["context_start_seconds"]),
                "context_end_seconds": float(context["context_end_seconds"]),
                "context_duration_seconds": context_duration,
                "audio_load": audio_meta,
                "full_matching_asr_draft_text": transcript,
                "word_lists_native": word_list,
                "word_count": len(word_list),
                "timestamp_mask_count": len(raw_bins),
                "expected_timestamp_count": expected_timestamp_count,
                "raw_timestamp_nonmonotonic_position_count": len(raw_nonmonotonic),
                "raw_timestamp_nonmonotonic_positions": raw_nonmonotonic,
                "decoder_repaired_timestamp_position_count": len(changed_positions),
                "decoder_repaired_timestamp_positions": changed_positions,
                "decoder_repaired_first_target_word": any(
                    word_rows[i]["is_target_segment"] and i in {j // 2 for j in changed_positions}
                    and not any(word_rows[k]["is_target_segment"] for k in range(i))
                    for i in range(len(word_rows))
                ),
                "decoder_repaired_last_target_word": any(
                    word_rows[i]["is_target_segment"] and i in {j // 2 for j in changed_positions}
                    and not any(word_rows[k]["is_target_segment"] for k in range(i + 1, len(word_rows)))
                    for i in range(len(word_rows))
                ),
                "source_segment_word_mapping_method": (
                    "unmapped_ambiguous" if "source_segment_word_mapping_mismatch" in mapping_flags
                    else "exact_character_partition" if "source_segment_word_mapping_exact_character_partition" in mapping_flags
                    else "exact_token_sequence"
                ),
                "context_segments": segment_rows,
                "word_alignments": word_rows,
                "clip_proposals": proposals,
                "clip_statuses": [
                    {"clip_id": p["clip_id"], "status": "aligned_proposal_requires_human_review", "review_flags": p["review_flags"], "human_review_status": context.get("human_review_status", "pending")}
                    for p in proposals
                ],
                "source_annotation_review_flags": context.get("source_annotation_review_flags", []),
                "context_speaker_labels_file_local": context.get("context_speaker_labels_file_local", []),
                "global_identity_verified": bool(context.get("global_identity_verified", False)),
                "transcript_provenance": context.get("transcript_provenance", "source_annotation_ASR_draft_unverified"),
                "human_review_status": context.get("human_review_status", "pending"),
                "context_status": context.get("context_status", "ready_for_review"),
                "context_status_reasons": context.get("status_reasons", []),
                "events_intersecting_context": context.get("events_intersecting_context", []),
                "review_flags": sorted(set(global_flags)),
                "interpretation": "Native Qwen forced-alignment timestamps are model estimates with the recorded grid step; decoder repairs are preserved alongside raw timestamp bins. They are not phoneme-level guarantees, confidence scores, or human verification. Crop bounds are proposals only.",
                "started_at_unix": started_at,
                "completed_at_unix": time.time(),
                "gpu_memory_before_model_mib": gpu_before,
                "gpu_memory_after_model_mib": gpu_after_model,
                "gpu_memory_after_context_mib": gpu_after_context,
            }
            write_jsonl_row(output_path, record)
            completed.append(record)
            print(json.dumps({
                "context_id": context_id,
                "words": len(word_rows),
                "changed_timestamp_positions": len(changed_positions),
                "flags": record["review_flags"],
                "clip_proposals": [{
                    "clip_id": p["clip_id"],
                    "estimated_onset": p["estimated_target_aligned_onset_source_seconds"],
                    "previous_aligned_end": p["previous_aligned_end_source_seconds"],
                    "proposed_start": p["proposed_cut_start_source_seconds"],
                    "flags": p["review_flags"],
                } for p in proposals],
                "gpu_memory_after_context_mib": record["gpu_memory_after_context_mib"],
            }, ensure_ascii=False), flush=True)
    finally:
        per_clip_status = {}
        for row in completed:
            for clip_status in row.get("clip_statuses", []):
                per_clip_status[clip_status["clip_id"]] = clip_status
        all_proposals = [p for row in completed for p in row.get("clip_proposals", [])]
        proposal_counts = {
            "proposals_with_earlier_start_than_baseline": sum(
                p.get("proposed_cut_start_source_seconds") is not None
                and p.get("baseline_plan_start_seconds") is not None
                and p["proposed_cut_start_source_seconds"] < p["baseline_plan_start_seconds"]
                for p in all_proposals
            ),
            "model_estimated_onset_before_baseline_start": sum(
                p.get("estimated_target_aligned_onset_source_seconds") is not None
                and p.get("baseline_plan_start_seconds") is not None
                and p["estimated_target_aligned_onset_source_seconds"] < p["baseline_plan_start_seconds"]
                for p in all_proposals
            ),
            "head_boundary_uncertain": sum(bool(p.get("head_boundary_uncertain")) for p in all_proposals),
            "tail_boundary_uncertain": sum(bool(p.get("tail_boundary_uncertain")) for p in all_proposals),
            "head_boundary_decoder_repair_anomalies": sum(
                any("decoder_repaired" in reasons for reasons in (p.get("head_uncertainty_evidence") or {}).values())
                for p in all_proposals
            ),
            "tail_boundary_decoder_repair_anomalies": sum(
                any("decoder_repaired" in reasons for reasons in (p.get("tail_uncertainty_evidence") or {}).values())
                for p in all_proposals
            ),
            "head_boundary_nonpositive_duration_anomalies": sum(
                any("nonpositive_duration" in reasons for reasons in (p.get("head_uncertainty_evidence") or {}).values())
                for p in all_proposals
            ),
            "tail_boundary_nonpositive_duration_anomalies": sum(
                any("nonpositive_duration" in reasons for reasons in (p.get("tail_uncertainty_evidence") or {}).values())
                for p in all_proposals
            ),
            "heads_withheld": sum(p.get("proposed_cut_start_source_seconds") is None for p in all_proposals),
            "ends_withheld_model_proposal": sum(p.get("proposed_cut_end_source_seconds") is None for p in all_proposals),
            "ends_retained_from_baseline_due_tail_uncertainty": sum(bool(p.get("tail_end_withheld_baseline_retained")) for p in all_proposals),
            "deferred_contexts": sum(row.get("status") == "pending_context_review" for row in completed),
        }
        summary = {
            "contexts_completed": len(completed),
            "contexts_requested": len(contexts),
            "contexts_aligned": sum(row.get("status") == "completed" for row in completed),
            "contexts_deferred": sum(row.get("status") == "pending_context_review" for row in completed),
            "clips_accounted": len(per_clip_status),
            "proposal_counts_are_estimates_not_confirmed_word_cuts": proposal_counts,
            "per_clip_status": per_clip_status,
            "output_jsonl": str(output_path),
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "model_weight_sha256": MODEL_FILE_SHA256,
            "review_only": True,
            "audio_files_modified": False,
            "source_text_modified": False,
            "human_review_status_changed": False,
            "note": "No existing WAV/TXT clip, baseline plan/manifest, review page, or review storage is modified.",
        }
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
