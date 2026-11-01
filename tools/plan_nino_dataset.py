#!/usr/bin/env python3
"""Build a review-first clip plan from existing Nino JSON annotations."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = Path(os.environ.get("TTS_SOURCE_ROOT", str(ROOT / "audio/source")))
OUTPUT = ROOT / "outputs/dataset"
LEAD_PAD = 0.15
TAIL_PAD = 0.20
MERGE_GAP = 0.70
MAX_MERGED_SPAN = 12.0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def union_seconds(intervals):
    merged = []
    for start, end in sorted((float(a), float(b)) for a, b in intervals if b > a):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return sum(end - start for start, end in merged)


def union_by_source(rows, get_intervals):
    by_source = defaultdict(list)
    for row in rows:
        by_source[row["source_audio"]].extend(get_intervals(row))
    return sum(union_seconds(intervals) for intervals in by_source.values())


def event_crosses(events, start, end):
    return any(float(e.get("start_time", 0)) < end and float(e.get("end_time", 0)) > start for e in events)


def event_text_is_music(text):
    return "\u97f3\u697d" in text or "bgm" in text.lower() or "music" in text.lower()


def stable_slug(annotation_stem):
    # Annotation basenames are already ASCII stable IDs; sanitize defensively.
    return re.sub(r"[^A-Za-z0-9_-]+", "-", annotation_stem).strip("-_").lower()


def make_exclusion(source, annotation, transcript, media_seconds, seg, index, reason, extra_flags=()):
    start, end = seg.get("start_time"), seg.get("end_time")
    return {
        "clip_id": f"nino-{source['source_id']}-s{index:04d}",
        "source_audio": source["source_audio"],
        "source_sha256": source["source_sha256"],
        "annotation_path": annotation["path"],
        "annotation_sha256": annotation["sha256"],
        "annotation_source_sha256_inherited": annotation["inherited_source_sha256"],
        "transcript_path": transcript["path"],
        "transcript_sha256": transcript["sha256"],
        "source_container_duration_seconds": media_seconds,
        "source_speaker_label": (seg.get("speaker") or {}).get("id", "unknown"),
        "identity_verified": False,
        "segment_indices": [index],
        "original_lexical_intervals_seconds": [[start, end]] if isinstance(start, (int, float)) and isinstance(end, (int, float)) else [],
        "original_text_segments": [seg.get("text", "")],
        "text": seg.get("text", ""),
        "exclusion_reason": reason,
        "flags": list(extra_flags),
        "machine_status": "excluded_pending_human_review",
        "human_review_status": "pending",
    }


def load_inventory(path):
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    return {Path(item["absolute_path"]).name: item for item in data.get("files", [])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--source-inventory", type=Path, default=ROOT / "outputs/intake/source_inventory.json")
    parser.add_argument("--readiness", type=Path, default=ROOT / "outputs/intake/dataset_readiness.json")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--overwrite", action="store_true", help="Replace only plan.jsonl, excluded.jsonl, and plan_summary.json in the output directory.")
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    allowed_output_root = (ROOT / "outputs").resolve()
    if not args.output_dir.is_relative_to(allowed_output_root):
        parser.error(f"--output-dir must stay inside the project outputs directory: {allowed_output_root}")
    output_targets = [args.output_dir / name for name in ("plan.jsonl", "excluded.jsonl", "plan_summary.json")]
    existing_outputs = [path for path in output_targets if path.exists()]
    if existing_outputs and not args.overwrite:
        parser.error(
            "Refusing to replace existing planning artifact(s): "
            + ", ".join(str(path) for path in existing_outputs)
            + ". Choose a new --output-dir or pass --overwrite explicitly."
        )
    source_root = args.source_root.resolve(strict=True)
    annotation_dir = source_root / "Nino_TTS_Merged"
    inventory = load_inventory(args.source_inventory)
    files = []
    excluded = []
    all_segment_ids = set()

    for annotation_path in sorted(annotation_dir.glob("*.json")):
        annotation_raw = annotation_path.read_bytes()
        annotation = json.loads(annotation_raw.decode("utf-8-sig"))
        source_name = Path(annotation["source_file"]).name.removesuffix(".json")
        source_audio = source_root / source_name
        if not source_audio.is_file():
            raise FileNotFoundError(f"Source MP4 missing: {source_audio}")
        media_meta = inventory.get(source_name)
        if not media_meta or not media_meta.get("sha256"):
            raise ValueError(f"Actual MP4 SHA256 missing from source inventory: {source_name}")
        media_seconds = float(media_meta["container_duration_seconds"])
        source = {"source_id": stable_slug(annotation_path.stem), "source_audio": str(source_audio.resolve()),
                  "source_sha256": media_meta["sha256"]}
        annotation_info = {"path": str(annotation_path.resolve()), "sha256": hashlib.sha256(annotation_raw).hexdigest(),
                           "inherited_source_sha256": annotation.get("source_sha256")}
        transcript_path = annotation_path.with_suffix(".txt")
        if not transcript_path.is_file():
            raise FileNotFoundError(f"Paired TXT missing: {transcript_path}")
        transcript_info = {"path": str(transcript_path.resolve()), "sha256": sha256(transcript_path)}
        segments = annotation.get("segments", [])
        events = annotation.get("events", [])
        speaker_labels = {((s.get("speaker") or {}).get("id", "unknown")) for s in segments}
        source_candidates = []
        source_excluded = []

        for index, seg in enumerate(segments):
            ident = (annotation_path.name, index)
            if ident in all_segment_ids:
                raise ValueError(f"Duplicate annotation segment ID: {ident}")
            all_segment_ids.add(ident)
            start, end = seg.get("start_time"), seg.get("end_time")
            text = seg.get("text") if isinstance(seg.get("text"), str) else ""
            speaker = (seg.get("speaker") or {}).get("id", "unknown")
            duration = (float(end) - float(start)) if isinstance(start, (int, float)) and isinstance(end, (int, float)) else None
            flags = [str(f) for f in seg.get("flags", [])]
            reason = None
            if not isinstance(start, (int, float)) or not isinstance(end, (int, float)) or end <= start or start < 0 or end > media_seconds:
                reason = "invalid_or_out_of_media_interval"
            elif not text.strip():
                reason = "empty_or_whitespace_text"
            elif text.strip().startswith("[") and text.strip().endswith("]"):
                reason = "non_speech_annotation"
            elif duration < 1.0:
                reason = "standalone_under_1_second"
            elif duration > 15.0:
                reason = "over_15_seconds_preserve_sentence_boundary"
            elif any("singing" in f.lower() for f in flags):
                reason = "possible_singing_flag_requires_audio_review"
            if reason:
                source_excluded.append(make_exclusion(source, annotation_info, transcript_info, media_seconds, seg, index, reason, flags))
                continue
            other_voice_overlap = any(
                (other.get("speaker") or {}).get("id", "unknown") != speaker
                and float(other.get("start_time", 0)) < float(end)
                and float(other.get("end_time", 0)) > float(start)
                for j, other in enumerate(segments) if j != index
            )
            if other_voice_overlap:
                source_excluded.append(make_exclusion(source, annotation_info, transcript_info, media_seconds, seg, index,
                                                      "overlaps_another_annotated_speaker", flags))
                continue
            overlaps = [e for e in events if float(e.get("start_time", 0)) < end and float(e.get("end_time", 0)) > start]
            source_candidates.append({"index": index, "segment": seg, "speaker": speaker, "start": float(start), "end": float(end),
                                      "duration": duration, "flags": flags, "overlaps": overlaps})

        by_speaker = defaultdict(list)
        for candidate in source_candidates:
            by_speaker[candidate["speaker"]].append(candidate)
        groups = []
        for speaker, candidates in by_speaker.items():
            candidates.sort(key=lambda c: (c["start"], c["end"], c["index"]))
            pending = []

            def flush():
                nonlocal pending
                if pending:
                    groups.append({"speaker": speaker, "members": pending})
                    pending = []

            for candidate in candidates:
                if candidate["duration"] >= 3.0:
                    flush()
                    groups.append({"speaker": speaker, "members": [candidate]})
                    continue
                if pending:
                    gap = candidate["start"] - pending[-1]["end"]
                    envelope = candidate["end"] - pending[0]["start"]
                    crossing_event = event_crosses(events, pending[-1]["end"], candidate["start"])
                    crossing_voice = any(
                        (other.get("speaker") or {}).get("id", "unknown") != speaker
                        and float(other.get("start_time", 0)) < candidate["start"]
                        and float(other.get("end_time", 0)) > pending[-1]["end"]
                        for other in segments
                    )
                    if gap <= MERGE_GAP and gap >= 0 and envelope <= MAX_MERGED_SPAN and not crossing_event and not crossing_voice:
                        pending.append(candidate)
                    else:
                        flush()
                        pending = [candidate]
                else:
                    pending = [candidate]
            flush()

        groups.sort(key=lambda g: (g["members"][0]["start"], g["members"][0]["index"]))
        group_rows = []
        for group in groups:
            members = group["members"]
            lexical_start = min(c["start"] for c in members)
            lexical_end = max(c["end"] for c in members)
            indices = [c["index"] for c in members]
            original_texts = [c["segment"].get("text", "") for c in members]
            flags = ["transcript_asr_unverified", "alignment_unverified", "audio_not_reviewed", "background_music_and_effects_not_removed",
                     "speaker_identity_file_local_unverified"]
            if len(speaker_labels) > 1:
                flags.append("multi_speaker_source_requires_review")
            if any(c["overlaps"] for c in members):
                flags.append("annotated_event_overlaps_lexical_interval")
            if any(event_text_is_music(e.get("text", "")) for c in members for e in c["overlaps"]):
                flags.append("annotated_music_event_overlap")
            if len(members) == 1 and members[0]["duration"] < 3.0:
                flags.append("review_short_1_to_3_seconds")
                clip_class = "review_short"
            elif lexical_end - lexical_start > 12.0:
                flags.append("allowed_up_to_15_seconds_review_length")
                clip_class = "review_long"
            else:
                clip_class = "target_length"
            flags.extend(sorted({f for c in members for f in c["flags"]}))
            lexical_intervals = [[c["start"], c["end"]] for c in members]
            group_rows.append({"speaker": group["speaker"], "members": members, "indices": indices, "texts": original_texts,
                               "lexical_start": lexical_start, "lexical_end": lexical_end, "lexical_intervals": lexical_intervals,
                               "flags": sorted(set(flags)), "clip_class": clip_class})

        # Pads stay inside media bounds, do not enter neighboring speech/events, and never overlap other planned clips.
        for row in group_rows:
            start = max(0.0, row["lexical_start"] - LEAD_PAD)
            end = min(media_seconds, row["lexical_end"] + TAIL_PAD)
            before = [float(s["end_time"]) for s in segments if isinstance(s.get("end_time"), (int, float)) and float(s["end_time"]) <= row["lexical_start"]]
            after = [float(s["start_time"]) for s in segments if isinstance(s.get("start_time"), (int, float)) and float(s["start_time"]) >= row["lexical_end"]]
            if before:
                start = max(start, max(before))
            if after:
                end = min(end, min(after))
            for event in events:
                es, ee = float(event.get("start_time", 0)), float(event.get("end_time", 0))
                if ee <= row["lexical_start"] and ee > start:
                    start = ee
                if es >= row["lexical_end"] and es < end:
                    end = es
                if es < row["lexical_start"] < ee:
                    start = row["lexical_start"]
                if es < row["lexical_end"] < ee:
                    end = row["lexical_end"]
            row["padded_start"] = start
            row["padded_end"] = max(row["lexical_end"], end)

        for previous, current in zip(group_rows, group_rows[1:]):
            if previous["lexical_end"] > current["lexical_start"]:
                # This shouldn't occur with current annotations; keep both spans accounted but prevent duplicate extraction.
                raise ValueError(f"Planned lexical groups overlap: {source['source_id']} {previous['indices']} {current['indices']}")
            boundary = (previous["lexical_end"] + current["lexical_start"]) / 2
            previous["padded_end"] = min(previous["padded_end"], boundary)
            current["padded_start"] = max(current["padded_start"], boundary)

        for row in group_rows:
            if not (0 <= row["padded_start"] <= row["lexical_start"] <= row["lexical_end"] <= row["padded_end"] <= media_seconds):
                raise ValueError(f"Invalid padded bounds for {source['source_id']} {row['indices']}")
            for event in events:
                es, ee = float(event.get("start_time", 0)), float(event.get("end_time", 0))
                if row["padded_start"] < row["lexical_start"] and es < row["lexical_start"] and ee > row["padded_start"]:
                    raise ValueError(f"Lead pad intersects annotated event for {source['source_id']} {row['indices']}")
                if row["lexical_end"] < row["padded_end"] and es < row["padded_end"] and ee > row["lexical_end"]:
                    raise ValueError(f"Tail pad intersects annotated event for {source['source_id']} {row['indices']}")
        for previous, current in zip(group_rows, group_rows[1:]):
            if previous["padded_end"] > current["padded_start"]:
                raise ValueError(f"Padded clip intervals overlap: {source['source_id']} {previous['indices']} {current['indices']}")

        for row in group_rows:
            first, last = min(row["indices"]), max(row["indices"])
            clip = {
                "clip_id": f"nino-{source['source_id']}-s{first:04d}-s{last:04d}",
                "source_audio": source["source_audio"],
                "source_sha256": source["source_sha256"],
                "annotation_path": annotation_info["path"],
                "annotation_sha256": annotation_info["sha256"],
                "annotation_source_sha256_inherited": annotation_info["inherited_source_sha256"],
                "transcript_path": transcript_info["path"],
                "transcript_sha256": transcript_info["sha256"],
                "source_container_duration_seconds": media_seconds,
                "source_speaker_label": row["speaker"],
                "identity_verified": False,
                "segment_indices": row["indices"],
                "original_lexical_intervals_seconds": row["lexical_intervals"],
                "source_lexical_start_seconds": row["lexical_start"],
                "source_lexical_end_seconds": row["lexical_end"],
                "padded_start_seconds": round(row["padded_start"], 6),
                "padded_end_seconds": round(row["padded_end"], 6),
                "original_text_segments": row["texts"],
                "text": "\n".join(row["texts"]),
                "flags": row["flags"],
                "clip_class": row["clip_class"],
                "machine_status": "planned_for_review",
                "human_review_status": "pending",
            }
            files.append(clip)
        excluded.extend(source_excluded)

    # Accounting invariant: every source annotation segment is in one and only one ledger row.
    planned_ids = [(Path(c["annotation_path"]).name, i) for c in files for i in c["segment_indices"]]
    excluded_ids = [(Path(c["annotation_path"]).name, i) for c in excluded for i in c["segment_indices"]]
    accounted = planned_ids + excluded_ids
    if len(accounted) != len(set(accounted)) or set(accounted) != all_segment_ids:
        raise ValueError("Segment accounting failed: duplicate or missing annotation segments")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "plan.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in files), encoding="utf-8")
    (args.output_dir / "excluded.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in excluded), encoding="utf-8")
    readiness = json.loads(args.readiness.read_text(encoding="utf-8-sig"))
    plan_span = union_by_source(files, lambda c: c["original_lexical_intervals_seconds"])
    excluded_span = union_by_source(excluded, lambda c: c["original_lexical_intervals_seconds"])
    reasons = Counter(x["exclusion_reason"] for x in excluded)
    source_paths = sorted({row["source_audio"] for row in files + excluded})
    per_source = []
    for source_path in source_paths:
        source_plan = [row for row in files if row["source_audio"] == source_path]
        source_excluded = [row for row in excluded if row["source_audio"] == source_path]
        meta = (source_plan or source_excluded)[0]
        source_plan_union = union_seconds([span for row in source_plan for span in row["original_lexical_intervals_seconds"]])
        source_ex_union = union_seconds([span for row in source_excluded for span in row["original_lexical_intervals_seconds"]])
        source_reasons = Counter(row["exclusion_reason"] for row in source_excluded)
        per_source.append({
            "source_audio": source_path,
            "source_sha256_actual_mp4": meta["source_sha256"],
            "annotation_source_sha256_inherited": meta["annotation_source_sha256_inherited"],
            "container_duration_seconds": meta["source_container_duration_seconds"],
            "planned_clip_count": len(source_plan),
            "planned_segment_count": sum(len(row["segment_indices"]) for row in source_plan),
            "planned_annotated_speech_union_seconds": round(source_plan_union, 3),
            "excluded_segment_count": sum(len(row["segment_indices"]) for row in source_excluded),
            "excluded_annotated_span_union_seconds": round(source_ex_union, 3),
            "exclusion_reason_counts": dict(sorted(source_reasons.items())),
        })
    summary = {
        "source_audio_files": readiness["source_audio_file_count"],
        "source_media_duration_seconds": readiness["source_audio_total_duration_seconds"],
        "source_media_duration_minutes": readiness["source_audio_total_duration_minutes"],
        "annotated_segment_count": readiness["annotation_total_segment_count"],
        "annotated_candidate_union_seconds": readiness["union_of_annotated_candidate_speech_seconds"],
        "annotated_candidate_union_minutes": readiness["union_candidate_minutes"],
        "planned_clip_count": len(files),
        "planned_segment_count": len(planned_ids),
        "planned_annotated_speech_union_seconds": round(plan_span, 3),
        "planned_annotated_speech_union_minutes": round(plan_span / 60, 3),
        "planned_share_of_raw_media_percent": round(plan_span / readiness["source_audio_total_duration_seconds"] * 100, 2),
        "planned_share_of_annotated_candidate_percent": round(plan_span / readiness["union_of_annotated_candidate_speech_seconds"] * 100, 2),
        "planned_source_clip_envelope_seconds": round(sum(c["padded_end_seconds"] - c["padded_start_seconds"] for c in files), 3),
        "excluded_segment_count": len(excluded_ids),
        "excluded_annotated_span_union_seconds": round(excluded_span, 3),
        "exclusion_reason_counts": dict(sorted(reasons.items())),
        "per_source": per_source,
        "review_short_clip_count": sum(c["clip_class"] == "review_short" for c in files),
        "long_review_clip_count": sum(c["clip_class"] == "review_long" for c in files),
        "segment_accounting_complete": True,
        "actual_source_sha_verified_from": "outputs/intake/source_inventory.json SHA256 values computed from source MP4 bytes; annotation.source_sha256 is stored separately as inherited metadata",
        "human_review_complete": False,
        "clean_trainable_duration_minutes": None,
        "note": "This is a cut/review plan only. Candidate timestamps/text are unreviewed ASR/AI annotations; music/SFX remain, source-local speaker labels are not cross-file identities, and no clip is asserted clean or trainable.",
    }
    (args.output_dir / "plan_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True))


if __name__ == "__main__":
    main()
