#!/usr/bin/env python3
"""Validate a Nino style review export and create an immutable caption rollup."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "outputs/dataset/style-review-v1/decisions.json"
DEFAULT_OUT = ROOT / "outputs/dataset/style-review-v1/rollup-v1"
SERVER = ROOT / "runtime/nino_style_review_server.py"
ALL_ROWS = ROOT / "outputs/dataset/training-v4/all.jsonl"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_new(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        f.write(data)


def json_bytes(obj) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--reference", type=Path,
                        help="Optional older downloaded export to preserve and compare as a reference snapshot")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    input_path = args.input.resolve()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty output directory: {out}")

    spec = importlib.util.spec_from_file_location("nino_style_review_server", SERVER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    app = module.ReviewApp(ROOT / "outputs/dataset/style-review-v1")
    incoming_bytes = input_path.read_bytes()
    incoming = json.loads(incoming_bytes.decode("utf-8"))
    app.validate_store(incoming)
    if len(incoming["decisions"]) != 30 or len({d["clip_id"] for d in incoming["decisions"]}) != 30:
        raise ValueError("Expected exactly 30 unique reviewed clip decisions")
    if incoming.get("collection_fingerprint") != app.collection_fp:
        raise ValueError("Input collection fingerprint is not the current collection fingerprint")

    # Direct file read only: ReviewApp.read_store() can migrate and rewrite a compatible store.
    live_path = app.store_path
    live = load_json(live_path) if live_path.exists() else app.empty_store()
    app.validate_store(live)
    live_by_id = {d["clip_id"]: d for d in live["decisions"]}
    in_by_id = {d["clip_id"]: d for d in incoming["decisions"]}
    conflicts = []
    for cid in sorted(set(live_by_id) | set(in_by_id)):
        left, right = in_by_id.get(cid), live_by_id.get(cid)
        if left != right:
            conflicts.append({"clip_id": cid, "input_status": left.get("status") if left else None,
                              "live_status": right.get("status") if right else None,
                              "input_decision": left, "live_decision": right})
    reference_bytes = args.reference.resolve().read_bytes() if args.reference else None
    reference = json.loads(reference_bytes.decode("utf-8")) if reference_bytes else None
    if reference:
        app.validate_store(reference)

    manifest_by_id = app.clips
    all_rows = {}
    with ALL_ROWS.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            all_rows[row["clip_id"]] = row
    if len(all_rows) != 380:
        raise ValueError(f"Expected 380 original v4 rows, got {len(all_rows)}")

    confirmed_rows = []
    statuses = Counter(d["status"] for d in incoming["decisions"])
    for d in incoming["decisions"]:
        if d["status"] != "confirmed":
            continue
        c = manifest_by_id[d["clip_id"]]
        original = all_rows.get(d["clip_id"])
        if original is None:
            raise ValueError(f"Missing original training-v4 row: {d['clip_id']}")
        checked = app.validate_decision(d, preserve_timestamp=True)
        caption = " ".join(app.caption_parts[k][checked[k]] for k in ("attitude", "effort", "delivery")
                            if app.caption_parts[k].get(checked[k])).strip()
        if d.get("caption_ja") != caption or d.get("label_basis") != "human_listening":
            raise ValueError(f"Caption proof mismatch for {d['clip_id']}")
        if original.get("audio_sha256") != c["clip_audio_sha256"] or original.get("text") != c["text_ja"]:
            raise ValueError(f"training-v4 text/audio proof mismatch for {d['clip_id']}")
        confirmed_rows.append({
            "clip_id": d["clip_id"], "clip_audio_sha256": c["clip_audio_sha256"],
            "text_ja": c["text_ja"], "speaker_id": "nino", "caption": caption,
            "attitude": d["attitude"], "effort": d["effort"], "delivery": d["delivery"],
            "notes": d.get("notes", ""), "label_basis": "human_listening",
            "confirmed_at": d["confirmed_at"], "source_id": c["source_id"],
            "source_family": c["source_family"], "source_sha256": c["source_sha256"],
            "original_split": original["split"], "clip_duration_seconds": c["clip_duration_seconds"],
            "parent_dataset_fingerprint": app.parent_fp,
            "style_collection_fingerprint": app.collection_fp,
            "review_input_sha256": digest(incoming_bytes),
        })

    sources = {s["source_id"]: s for s in app.manifest["sources"]}
    groups = {key: defaultdict(lambda: {"count": 0, "seconds": 0.0}) for key in ("attitude", "effort", "delivery")}
    cross = defaultdict(lambda: {"count": 0, "seconds": 0.0})
    source_dist = defaultdict(lambda: {"count": 0, "seconds": 0.0})
    split_dist = defaultdict(lambda: {"count": 0, "seconds": 0.0})
    confirmed_seconds = 0.0
    for row in confirmed_rows:
        sec = row["clip_duration_seconds"]
        confirmed_seconds += sec
        for key in groups:
            g = groups[key][row[key]]
            g["count"] += 1; g["seconds"] += sec
        cross[f"{row['attitude']}|{row['effort']}|{row['delivery']}"]["count"] += 1
        cross[f"{row['attitude']}|{row['effort']}|{row['delivery']}"]["seconds"] += sec
        source_dist[row["source_id"]]["count"] += 1; source_dist[row["source_id"]]["seconds"] += sec
        split_dist[row["original_split"]]["count"] += 1; split_dist[row["original_split"]]["seconds"] += sec

    total_seconds = sum(c["clip_duration_seconds"] for c in app.clips.values())
    draft_seconds = sum(manifest_by_id[d["clip_id"]]["clip_duration_seconds"]
                        for d in incoming["decisions"] if d["status"] == "draft")
    all_attitudes = {x["id"] for x in app.manifest["label_schema"]["attitude"]}
    missing = sorted(all_attitudes - set(groups["attitude"]))
    summary = {
        "schema": "nino-style-review-rollup-v1", "input_path": str(input_path),
        "review_input_sha256": digest(incoming_bytes), "collection_fingerprint": app.collection_fp,
        "parent_dataset_fingerprint": app.parent_fp,
        "input_matches_live_semantically": not conflicts,
        "live_newer_or_conflicts": conflicts,
        "reference_export": ({"path": str(args.reference.resolve()),
                              "sha256": digest(reference_bytes),
                              "confirmed": sum(d["status"] == "confirmed" for d in reference["decisions"]),
                              "draft": sum(d["status"] == "draft" for d in reference["decisions"]),
                              "stale_against_input": reference != incoming,
                              "user_resolved_stale_draft_id": "nino-06_03_walk_to_school-s0003-s0003"}
                             if reference else None),
        "decision_counts": {"total": len(incoming["decisions"]), "confirmed": statuses["confirmed"],
                            "draft": statuses["draft"], "needs_review": statuses["needs_review"]},
        "durations_seconds": {"all_30": total_seconds, "confirmed": confirmed_seconds,
                              "draft": draft_seconds},
        "confirmed_label_counts_and_seconds": {k: dict(v) for k, v in groups.items()},
        "confirmed_cross_tabs": dict(cross),
        "highlight_cross_tabs": {key: dict(cross.get(key, {"count": 0, "seconds": 0.0}))
                                 for key in ("daily|light|conversation", "daily|medium|conversation",
                                             "affectionate|strong|conversation", "affectionate|strong|raised_voice")},
        "confirmed_source_distribution": {k: dict(v) for k, v in sorted(source_dist.items())},
        "confirmed_original_split_distribution": {k: dict(v) for k, v in sorted(split_dist.items())},
        "source_quota_distribution": {sid: {"count": sum(1 for c in app.clips.values() if c["source_id"] == sid),
                                             "source_family": src["source_family"]}
                                      for sid, src in sorted(sources.items())},
        "original_v4_rows": 380, "confirmed_caption_rows": len(confirmed_rows),
        "original_v4_remainder_after_reviewed_ids": 380 - len(in_by_id),
        "missing_attitude_categories": missing,
        "sample_scope_note": "這 30 筆是按 8 個 source 配額抽樣，不代表完整 380 筆的情緒分布；只有本次 confirmed 決策匯出為 caption，整批標註尚未完成。",
        "unmodified_notes_observation": [r["clip_id"] for r in confirmed_rows if r["notes"] == "ㄓ"],
    }
    jsonl = b"".join((json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
                    for row in confirmed_rows)
    # Prepare every artifact in memory, then create only fresh files.
    outputs = {
        "summary.json": json_bytes(summary),
        "confirmed-captions.jsonl": jsonl,
        "source-review-snapshot.json": incoming_bytes,
        "conflicts.json": json_bytes(conflicts),
    }
    if reference_bytes is not None:
        outputs["reference-export-snapshot.json"] = reference_bytes
    out.mkdir(parents=True, exist_ok=True)
    try:
        for name, content in outputs.items():
            write_new(out / name, content)
    except FileExistsError as e:
        raise SystemExit(f"Refusing to overwrite existing artifact: {e.filename}")
    print(json.dumps({"output": str(out), "confirmed": len(confirmed_rows), "draft": statuses["draft"],
                      "total_seconds": total_seconds, "confirmed_seconds": confirmed_seconds,
                      "conflicts": len(conflicts), "missing_attitudes": missing}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
