"""Merge matching ASR triage records into a new dataset manifest."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


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


def key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("clip_id", "")),
        str(row.get("clip_sha256") or row.get("audio_sha256") or ""),
        str(row.get("model_revision", "")),
        str(row.get("settings_sha256", "")),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "outputs/dataset/manifest.jsonl")
    parser.add_argument("--checks", type=Path, default=ROOT / "outputs/dataset/asr_checks.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/dataset/manifest_asr.jsonl")
    parser.add_argument("--summary", type=Path, default=ROOT / "outputs/dataset/asr_summary.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    paths = [args.manifest, args.checks, args.output, args.summary]
    paths = [p if p.is_absolute() else ROOT / p for p in paths]
    manifest_path, checks_path, output_path, summary_path = paths
    if output_path.exists() and not args.overwrite:
        parser.error(f"Refusing to overwrite {output_path}; pass --overwrite to replace it")
    manifest = read_jsonl(manifest_path)
    checks = read_jsonl(checks_path)

    by_key: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    check_status_counts: Counter[str] = Counter(str(row.get("status", "unknown")) for row in checks)
    for check in checks:
        if check.get("status") == "completed":
            by_key[key(check)] = check

    counts: Counter[str] = Counter()
    priority_flag_counts: Counter[str] = Counter()
    output_rows: list[dict[str, Any]] = []
    for source_row in manifest:
        row = dict(source_row)
        audio_sha = str(row.get("clip_sha256") or "")
        matched = None
        # Model/settings come from available checks. Audio hash and clip id are
        # always required, so stale checks cannot attach to a changed WAV.
        if audio_sha:
            candidates = [
                check for check in by_key.values()
                if check.get("clip_id") == row.get("clip_id")
                and check.get("audio_sha256") == audio_sha
            ]
            if candidates:
                matched = candidates[-1]
        if matched:
            row["asr_check"] = {
                "status": "completed",
                "audio_sha256": matched.get("audio_sha256"),
                "model_id": matched.get("model_id"),
                "model_revision": matched.get("model_revision"),
                "settings": matched.get("settings"),
                "result": matched.get("result"),
                "normalized_character_disagreement_ratio": matched.get("normalized_character_disagreement_ratio"),
                "interpretation": matched.get("interpretation"),
            }
            row["asr_review_priority_flags"] = matched.get("review_priority_flags", [])
            priority_flag_counts.update(row["asr_review_priority_flags"])
            counts["completed"] += 1
        else:
            row["asr_check"] = {"status": "pending_or_no_matching_audio_hash"}
            row["asr_review_priority_flags"] = []
            counts["pending"] += 1
        # Deliberately leave the original `text`, machine status and human
        # review status untouched. ASR never approves, rewrites, or excludes.
        output_rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as f:
        for row in output_rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    summary = {
        "manifest_input": str(manifest_path),
        "asr_checks_input": str(checks_path),
        "manifest_output": str(output_path),
        "clip_rows": len(output_rows),
        "check_record_status_counts": dict(check_status_counts),
        "asr_status_counts": dict(counts),
        "asr_review_priority_flag_counts": dict(priority_flag_counts),
        "interpretation": "ASR is a separate triage signal; original text and human review statuses are preserved. Character disagreement is not WER or accuracy.",
        "human_review_status_changed": False,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
