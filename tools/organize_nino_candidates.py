#!/usr/bin/env python3
"""Partition the frozen Nino candidate manifest into pending review queues.

Metadata-only: does not read audio, rewrite transcripts, or change review status.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "outputs/dataset/manifest_asr.jsonl"
DEFAULT_READINESS = ROOT / "outputs/intake/dataset_readiness.json"
OUT = ROOT / "outputs/dataset"
TIERS = ("priority_review", "needs_attention", "short_review")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def recursive_ratio(obj: Any) -> float | None:
    """Find a normalized-character disagreement ratio in ASR metadata."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_norm = key.lower().replace("-", "_")
            if "ratio" in key_norm and ("disagreement" in key_norm or "normalized_char" in key_norm):
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    pass
                else:
                    if math.isfinite(value):
                        return value
        for value in obj.values():
            found = recursive_ratio(value)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = recursive_ratio(value)
            if found is not None:
                return found
    return None


def asr_status(row: dict[str, Any]) -> str:
    check = row.get("asr_check")
    if isinstance(check, dict):
        return str(check.get("status", "missing"))
    return "missing"


def lexical_seconds(row: dict[str, Any]) -> float:
    return sum(max(0.0, float(end) - float(start)) for start, end in row.get("original_lexical_intervals_seconds", []))


def padded_seconds(row: dict[str, Any]) -> float:
    return max(0.0, float(row.get("padded_end_seconds", 0)) - float(row.get("padded_start_seconds", 0)))


def write_new_or_identical(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise FileExistsError(f"Refusing to replace non-matching existing output: {path}")
        return
    path.write_bytes(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--readiness", type=Path, default=DEFAULT_READINESS)
    args = parser.parse_args()
    rows = read_jsonl(args.manifest)
    if not rows:
        raise ValueError(f"Empty manifest: {args.manifest}")
    ids = [row.get("clip_id") for row in rows]
    if None in ids or len(set(ids)) != len(ids):
        raise ValueError("Manifest must contain unique, nonempty clip_id values")

    readiness = json.loads(args.readiness.read_text(encoding="utf-8-sig"))
    multiple_speaker_source_hashes = set()
    for source in readiness.get("files", []):
        if len(source.get("speaker_labels", {})) > 1:
            actual_hash = source.get("source_sha256_actual_mp4")
            if actual_hash:
                multiple_speaker_source_hashes.add(str(actual_hash).lower())

    partitions: dict[str, list[dict[str, Any]]] = {tier: [] for tier in TIERS}
    reasons_by_clip: dict[str, list[str]] = {}
    for row in rows:
        clip_id = str(row["clip_id"])
        if row.get("human_review_status") != "pending":
            raise ValueError(f"Expected human_review_status=pending for {clip_id}")
        cls = row.get("clip_class")
        if cls == "review_short":
            tier = "short_review"
            reasons = ["clip_class_review_short"]
        elif cls != "target_length":
            tier = "needs_attention"
            reasons = [f"unsupported_clip_class:{cls!r}"]
        else:
            reasons = []
            if row.get("machine_status") != "candidate_for_review":
                reasons.append(f"machine_status:{row.get('machine_status')}")
            if asr_status(row) != "completed":
                reasons.append(f"asr_status:{asr_status(row)}")
            asr_flags = row.get("asr_review_priority_flags") or []
            reasons.extend(f"asr_review_priority_flag:{flag}" for flag in asr_flags)
            ratio = recursive_ratio(row.get("asr_check", {}))
            if ratio is None:
                reasons.append("asr_disagreement_ratio_missing")
            elif ratio > 0.20:
                reasons.append(f"asr_disagreement_ratio_above_0.20:{ratio:.6f}")
            source_flags = row.get("flags") or []
            for flag in source_flags:
                low = str(flag).lower()
                if any(token in low for token in ("uncertain", "incomplete", "interrupted")):
                    reasons.append(f"source_wording_caution:{flag}")
            source_sha = str(row.get("source_sha256", "")).lower()
            if source_sha in multiple_speaker_source_hashes:
                reasons.append("source_has_multiple_file_local_speaker_labels")
            if reasons:
                tier = "needs_attention"
            else:
                tier = "priority_review"
        # Keep the complete merged source row and add only queue metadata.
        output_row = dict(row)
        output_row["organization_tier"] = tier
        output_row["organization_reasons"] = reasons
        partitions[tier].append(output_row)
        reasons_by_clip[clip_id] = reasons

    total_partitioned = sum(map(len, partitions.values()))
    if total_partitioned != len(rows) or sum(len(v) for v in partitions.values()) != len(set(ids)):
        raise RuntimeError("Partitions are not a disjoint complete cover of the input manifest")

    for tier in TIERS:
        payload = b"".join((json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8") for row in partitions[tier])
        write_new_or_identical(OUT / f"{tier}.jsonl", payload)

    summary: dict[str, Any] = {
        "input_manifest": args.manifest.relative_to(ROOT).as_posix() if args.manifest.is_absolute() and args.manifest.is_relative_to(ROOT) else str(args.manifest),
        "metadata_only": True,
        "clip_count": len(rows),
        "tiers": {},
        "clip_id_to_tier": {},
        "selection_rules": {
            "short_review": "All review_short clips; preserve their ASR, QC, source flags and pending human status.",
            "priority_review": "target_length + machine candidate_for_review + ASR completed + no ASR priority flags + normalized-character disagreement ratio <= 0.20 + no source wording uncertain/incomplete/interrupted flags + source has one file-local speaker label.",
            "needs_attention": "Every other target_length clip, with concrete reasons recorded per row.",
            "global_caveats": "ASR draft, unreviewed audio/alignment, retained background audio, and unverified global identity remain caveats; they do not by themselves remove clips from priority_review.",
            "human_review": "All tiers remain pending; queue placement is not approval, verification, or training readiness.",
        },
    }
    for tier, group in partitions.items():
        summary["tiers"][tier] = {
            "clip_count": len(group),
            "padded_seconds": round(sum(padded_seconds(r) for r in group), 6),
            "lexical_seconds": round(sum(lexical_seconds(r) for r in group), 6),
            "jsonl_path": f"outputs/dataset/{tier}.jsonl",
        }
        for row in group:
            summary["clip_id_to_tier"][row["clip_id"]] = tier
    summary["needs_attention_reason_counts"] = dict(Counter(reason for row in partitions["needs_attention"] for reason in row["organization_reasons"]))
    payload = (json.dumps(summary, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    write_new_or_identical(OUT / "organization_summary.json", payload)
    print(json.dumps({"clip_count": len(rows), "tiers": {k: len(v) for k, v in partitions.items()}, "summary": "outputs/dataset/organization_summary.json"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
