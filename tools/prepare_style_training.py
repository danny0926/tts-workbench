#!/usr/bin/env python3
"""Create a caption-only manifest from explicitly confirmed listening decisions."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def prepare(manifest, review, rollup):
    originals = read_rows(manifest)
    store = json.loads(review.read_text(encoding="utf-8-sig"))
    captions = read_rows(rollup)
    by_id = {row["clip_id"]: row for row in originals}
    decisions = {row["clip_id"]: row for row in store["decisions"]}
    if len(by_id) != len(originals) or len(decisions) != len(store["decisions"]):
        raise ValueError("Duplicate clip IDs in manifest or human decisions")
    if len({row["clip_id"] for row in captions}) != len(captions):
        raise ValueError("Duplicate caption rollup IDs")
    confirmed = {key: value for key, value in decisions.items() if value.get("status") == "confirmed"}
    if set(confirmed) != {row["clip_id"] for row in captions} or not confirmed:
        raise ValueError("Rollup must contain exactly the nonempty confirmed decision set")
    if {row.get("parent_fingerprint") for row in originals} != {store.get("parent_dataset_fingerprint")} or not store.get("parent_dataset_fingerprint"):
        raise ValueError("Human style decisions and training manifest have different parent lineage")
    if any(row.get("caption") for row in originals):
        raise ValueError("This controlled experiment requires a baseline with blank captions")
    replacements = {}
    for row in captions:
        clip = row["clip_id"]
        original = by_id.get(clip)
        decision = confirmed[clip]
        if original is None or not decision.get("heard_audio"):
            raise ValueError(f"Unknown or unlistened confirmed clip: {clip}")
        if row.get("clip_audio_sha256") != original["audio_sha256"] or decision.get("clip_audio_sha256") != original["audio_sha256"]:
            raise ValueError(f"Audio identity differs: {clip}")
        if row.get("source_sha256") != original["source_sha256"] or row.get("text_ja") != original["text"] or row.get("original_split") != original["split"]:
            raise ValueError(f"Text, source or split differs: {clip}")
        caption = row.get("caption", "")
        if not caption.strip() or caption != decision.get("caption_ja"):
            raise ValueError(f"Caption differs from the human decision: {clip}")
        replacements[clip] = caption
    train_sources = {row["source_sha256"] for row in originals if row["split"] == "train"}
    validation_sources = {row["source_sha256"] for row in originals if row["split"] == "validation"}
    if not train_sources or not validation_sources or train_sources & validation_sources:
        raise ValueError("Train and validation source groups must be nonempty and disjoint")
    if any(row["split"] not in {"train", "validation"} for row in originals):
        raise ValueError("Unexpected dataset split")
    result = [{**row, "caption": replacements.get(row["clip_id"], "")} for row in originals]
    summary = {"records": len(result), "splits": dict(Counter(row["split"] for row in result)),
               "confirmed_caption_records": len(replacements), "blank_caption_records": len(result) - len(replacements),
               "source_disjoint": True, "all_noncaption_fields_unchanged": True, "audio_quality_status": "unreviewed",
               "input_sha256": {str(path.name): hashlib.sha256(path.read_bytes()).hexdigest() for path in (manifest, review, rollup)}}
    return result, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--rollup", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="New JSONL file beside the baseline manifest so relative latent paths still resolve")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    rows, summary = prepare(args.manifest, args.review, args.rollup)
    if args.out.resolve().parent != args.manifest.resolve().parent:
        raise ValueError("Output must be beside the baseline manifest to preserve relative latent paths")
    proof = args.out.with_suffix(".summary.json")
    if args.out.exists() or proof.exists():
        raise FileExistsError("Refusing to overwrite an existing manifest or summary")
    if not args.dry_run:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        with proof.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"written": not args.dry_run, **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
