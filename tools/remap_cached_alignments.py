"""Rebuild source ownership/proposals from cached alignments without model inference.

Only processor text segmentation is used. Audio, model weights, native words,
raw bins, decoded timings and their anomaly evidence are never regenerated.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from align_nino_sources import (
    DEFAULT_BASELINE_PLAN,
    DEFAULT_MODEL_PATH,
    DEFAULT_REPAIR_PLAN,
    MODEL_REVISION,
    OUTPUT_ROOT,
    PRE_MARGIN_SECONDS,
    ROOT,
    TAIL_MARGIN_SECONDS,
    build_clip_proposals,
    get_clip_targets,
    map_word_ownership,
    read_jsonl,
    resolve_project_path,
    sha256_file,
)


DEFAULT_INPUT = OUTPUT_ROOT / "remaining-v3b/forced-alignments.jsonl"
DEFAULT_CONTEXTS = ROOT / "outputs/dataset/remaining-v3b/contexts.jsonl"
DEFAULT_OUTPUT = OUTPUT_ROOT / "remaining-v3b/forced-alignments-remapped.jsonl"
DEFAULT_SUMMARY = OUTPUT_ROOT / "remaining-v3b/remap-summary.json"
MAPPING_FLAGS = {
    "source_segment_word_mapping_mismatch",
    "source_segment_word_mapping_exact_character_partition",
    "source_segment_word_mapping_crosses_segment_boundary",
    "source_segment_word_mapping_empty_or_uncovered_word",
    "transcript_word_segment_coverage_incomplete",
    "one_or_more_context_segments_have_no_mapped_words",
}
OWNER_FIELDS = {
    "segment_index", "is_target_segment", "source_segment_char_start", "source_segment_char_end",
}
DERIVED_FIELDS = {
    "source_segment_word_mapping_method", "context_segments", "word_alignments",
    "clip_proposals", "clip_statuses", "review_flags", "cached_alignment_remap",
    "decoder_repaired_first_target_word", "decoder_repaired_last_target_word",
    "annotation_path", "annotation_sha256", "source_transcript_path", "source_transcript_sha256",
    "source_annotation_review_flags", "context_speaker_labels_file_local", "global_identity_verified",
    "transcript_provenance", "human_review_status", "context_status", "context_status_reasons",
    "events_intersecting_context",
}
def unique_by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed = {row["context_id"]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("Duplicate context IDs are not safe to remap")
    return indexed


def verify_preserved(before: dict[str, Any], after: dict[str, Any]) -> None:
    """Compare every field, allowing only explicitly derived mapping metadata."""
    if {k: v for k, v in before.items() if k not in DERIVED_FIELDS} != {
        k: v for k, v in after.items() if k not in DERIVED_FIELDS
    }:
        raise ValueError(f"{before['context_id']}: non-derived record content changed")
    strip_owner = lambda row: {k: v for k, v in row.items() if k not in OWNER_FIELDS}
    if [strip_owner(w) for w in before["word_alignments"]] != [
        strip_owner(w) for w in after["word_alignments"]
    ]:
        raise ValueError(f"{before['context_id']}: native word/timestamp/anomaly evidence changed")


def remap_record(
    original: dict[str, Any], context: dict[str, Any], processor: Any,
    baseline: dict[str, dict[str, Any]], repair: dict[str, dict[str, Any]], provenance: dict[str, Any],
) -> dict[str, Any]:
    cid = original["context_id"]
    transcript = context.get("full_matching_asr_draft_text") or context.get("text")
    if original.get("full_matching_asr_draft_text") != transcript:
        raise ValueError(f"{cid}: cached transcript differs from current context")
    for key in ("context_start_seconds", "context_end_seconds", "source_audio", "source_sha256"):
        if original.get(key) != context.get(key):
            raise ValueError(f"{cid}: {key} differs from current context")
    if original.get("context_audio_sha256") != context.get("context_wav_sha256"):
        raise ValueError(f"{cid}: audio hash differs from current context")
    if original.get("model_revision") != MODEL_REVISION:
        raise ValueError(f"{cid}: unexpected model revision")
    words_native = original["word_lists_native"]
    if words_native != [w["word"] for w in original["word_alignments"]]:
        raise ValueError(f"{cid}: saved native word sequence is inconsistent")
    targets = get_clip_targets(context, baseline)
    if original["target_clip_ids"] != [clip_id for clip_id, _ in targets]:
        raise ValueError(f"{cid}: target clip IDs differ from current context")
    target_indices = {i for _, indices in targets for i in indices}
    segments = context["segments"]
    if [(s["segment_index"], s["text"]) for s in segments] != [
        (s["segment_index"], s["text"]) for s in original["context_segments"]
    ]:
        raise ValueError(f"{cid}: source segment text differs from cached alignment")
    owners, mapping_flags = map_word_ownership(processor, segments, words_native, target_indices)
    record = copy.deepcopy(original)
    words = record["word_alignments"]
    for word, owner in zip(words, owners):
        word.update({
            "segment_index": owner["segment_index"],
            "is_target_segment": owner["is_target"],
            "source_segment_char_start": owner["char_start"],
            "source_segment_char_end": owner["char_end"],
        })
    mapping_method = (
        "unmapped_ambiguous" if "source_segment_word_mapping_mismatch" in mapping_flags
        else "exact_character_partition" if "source_segment_word_mapping_exact_character_partition" in mapping_flags
        else "exact_token_sequence"
    )
    record["source_segment_word_mapping_method"] = mapping_method
    record["context_segments"] = []
    for segment in segments:
        index = int(segment["segment_index"])
        positions = [i for i, owner in enumerate(owners) if owner["segment_index"] == index]
        record["context_segments"].append({
            "segment_index": index, "text": segment["text"],
            "is_target": bool(segment.get("is_target", False)) or index in target_indices,
            "char_start": segment.get("char_start"), "char_end": segment.get("char_end"),
            "speaker_label": segment.get("speaker_label"),
            "source_annotation_start_seconds": segment.get("start_seconds"),
            "source_annotation_end_seconds": segment.get("end_seconds"),
            "word_start_index": min(positions) if positions else None,
            "word_end_index_exclusive": max(positions) + 1 if positions else None,
            "word_count": len(positions),
            "aligned_start_source_seconds": min((words[i]["source_absolute_start_seconds"] for i in positions), default=None),
            "aligned_end_source_seconds": max((words[i]["source_absolute_end_seconds"] for i in positions), default=None),
            "annotation_flags": segment.get("flags", []), "excluded_reason": segment.get("exclusion_reason"),
            "alignment_status": "mapped" if positions else "no_mapped_words",
        })
    global_flags = (set(original.get("review_flags", [])) - MAPPING_FLAGS) | set(mapping_flags)
    if len({o["segment_index"] for o in owners if o["segment_index"] is not None}) < len(segments):
        global_flags.add("one_or_more_context_segments_have_no_mapped_words")
    record["review_flags"] = sorted(global_flags)
    target_words = [word for word in words if word["is_target_segment"]]
    for name, position in (("first", 0), ("last", -1)):
        record[f"decoder_repaired_{name}_target_word"] = bool(
            target_words and "decoder_timestamp_repair_changed_word" in target_words[position]["review_flags"]
        )
    for key in (
        "annotation_path", "annotation_sha256", "source_transcript_path", "source_transcript_sha256",
        "source_annotation_review_flags", "context_speaker_labels_file_local", "global_identity_verified",
        "transcript_provenance", "human_review_status", "context_status", "events_intersecting_context",
    ):
        record[key] = copy.deepcopy(context.get(key, original.get(key)))
    record["context_status_reasons"] = copy.deepcopy(context.get("status_reasons", []))
    old_proposals = original.get("clip_proposals", [])
    margins = {(p.get("requested_pre_margin_seconds", PRE_MARGIN_SECONDS), p.get("requested_tail_margin_seconds", TAIL_MARGIN_SECONDS)) for p in old_proposals}
    if len(margins) > 1:
        raise ValueError(f"{cid}: inconsistent cached proposal margins")
    pre_margin, tail_margin = next(iter(margins), (PRE_MARGIN_SECONDS, TAIL_MARGIN_SECONDS))
    record["clip_proposals"] = build_clip_proposals(
        context, segments, owners, words, targets, baseline, repair,
        context.get("events_intersecting_context", []), pre_margin, tail_margin,
    )
    record["clip_statuses"] = [
        {"clip_id": p["clip_id"], "status": "aligned_proposal_requires_human_review",
         "review_flags": p["review_flags"], "human_review_status": context.get("human_review_status", "pending")}
        for p in record["clip_proposals"]
    ]
    record["cached_alignment_remap"] = dict(provenance)
    verify_preserved(original, record)
    return record


def usable_head(proposal: dict[str, Any]) -> bool:
    return proposal.get("proposed_cut_start_source_seconds") is not None and not proposal.get("head_boundary_uncertain", True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--contexts", type=Path, default=DEFAULT_CONTEXTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--baseline-plan", type=Path, default=DEFAULT_BASELINE_PLAN)
    parser.add_argument("--repair-plan", type=Path, default=DEFAULT_REPAIR_PLAN)
    args = parser.parse_args()
    input_path, contexts_path, output_path, summary_path = map(
        resolve_project_path, (args.input, args.contexts, args.output, args.summary)
    )
    for path in (output_path, summary_path):
        if not path.resolve().is_relative_to(OUTPUT_ROOT) or path.exists() or path.resolve() == input_path.resolve():
            parser.error(f"Choose a fresh sidecar path under {OUTPUT_ROOT}: {path}")
    originals = read_jsonl(input_path)
    contexts = unique_by_id(read_jsonl(contexts_path))
    if set(unique_by_id(originals)) != set(contexts):
        parser.error("Cached alignments and current contexts must have identical unique context IDs")
    baseline = {r["clip_id"]: r for r in read_jsonl(resolve_project_path(args.baseline_plan))}
    repair = {r["clip_id"]: r for r in read_jsonl(resolve_project_path(args.repair_plan))}
    input_sha = sha256_file(input_path)
    provenance = {
        "input_jsonl": str(input_path.relative_to(ROOT)), "input_sha256": input_sha,
        "contexts_jsonl": str(contexts_path.relative_to(ROOT)), "contexts_sha256": sha256_file(contexts_path),
        "processor_revision": MODEL_REVISION, "model_weights_loaded": False,
        "model_forward_performed": False, "audio_loaded": False, "gpu_used": False,
        "event_time_basis": "source_absolute_from_current_context_metadata",
    }
    # Block GPU visibility before importing processor dependencies. No model
    # class is imported or constructed, and no audio/tensor inputs are prepared.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(resolve_project_path(args.model_path), local_files_only=True)
    remapped = [remap_record(r, contexts[r["context_id"]], processor, baseline, repair, provenance) for r in originals]
    import torch

    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA unexpectedly initialized during text-only remapping")
    if sha256_file(input_path) != input_sha:
        raise ValueError("Raw alignment input changed during remapping")
    before_failures = {r["context_id"] for r in originals if "source_segment_word_mapping_mismatch" in r.get("review_flags", [])}
    after_failures = {r["context_id"] for r in remapped if "source_segment_word_mapping_mismatch" in r.get("review_flags", [])}
    all_clip_ids = sorted({clip for r in remapped for clip in r["target_clip_ids"]})
    # Derive actual E IDs from saved records rather than guessing source slugs.
    e_ids = [clip for clip in all_clip_ids if any(
        ("nino-05_" in clip and f"-s{i:04d}-s{i:04d}" in clip) for i in (19, 70, 94, 151, 177, 271)
    ) or ("nino-08_" in clip and "-s0004-s0004" in clip)]
    e_summary = []
    for clip in e_ids:
        before = [(r["context_id"], p) for r in originals for p in r["clip_proposals"] if p["clip_id"] == clip]
        after = [(r["context_id"], p) for r in remapped for p in r["clip_proposals"] if p["clip_id"] == clip]
        was_usable = any(usable_head(p) for _, p in before)
        now_usable = any(usable_head(p) for _, p in after)
        e_summary.append({
            "clip_id": clip, "usable_head_before": was_usable, "usable_head_after": now_usable,
            "recovered_usable_head": not was_usable and now_usable,
            "contexts": [{"context_id": cid, "proposed_cut_start_source_seconds": p.get("proposed_cut_start_source_seconds"),
                          "head_boundary_uncertain": p.get("head_boundary_uncertain"), "review_flags": p["review_flags"]} for cid, p in after],
        })
    summary = {
        "contexts_preserved": len(remapped), "unique_contexts_preserved": len({r["context_id"] for r in remapped}),
        "unique_target_clips_preserved": len(all_clip_ids), "mapping_failures_before": len(before_failures),
        "mapping_failures_after": len(after_failures), "recovered_context_ids": sorted(before_failures - after_failures),
        "remaining_mapping_failure_context_ids": sorted(after_failures),
        "mapping_methods": dict(Counter(r["source_segment_word_mapping_method"] for r in remapped)),
        "seven_unresolved_E_clip_heads": e_summary,
        "E_clips_recovered_usable_head": [r["clip_id"] for r in e_summary if r["recovered_usable_head"]],
        "heads_usable_before": sum(usable_head(p) for r in originals for p in r["clip_proposals"]),
        "heads_usable_after": sum(usable_head(p) for r in remapped for p in r["clip_proposals"]),
        "all_native_words_bins_timings_and_word_anomalies_preserved": True,
        "all_other_non_derived_fields_preserved": True, "input_sha256_after": sha256_file(input_path),
        "cuda_initialized": torch.cuda.is_initialized(), "human_review_status": "pending",
        "wav_cuts_created": 0, "label_changes": 0, "provenance": provenance,
        "output_jsonl": str(output_path.relative_to(ROOT)),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="\n") as f:
        for record in remapped:
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    with summary_path.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps(summary, ensure_ascii=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
