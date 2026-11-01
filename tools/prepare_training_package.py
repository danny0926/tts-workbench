#!/usr/bin/env python3
"""Prepare immutable human-reviewed audio/text packages (v2/v3/v4)."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs/dataset"
OLD_FP = "4012b11aefc89f5a458d18c33a49b8ba31b20865225619c5f3f778e7966f1cae"
TARGET_VALIDATION_FRACTION = 0.10
LATENT_SETTINGS = {
    "format": "DACVAE deterministic encoder mean", "device": "cpu", "dtype": "float32",
    "deterministic_encode": True, "normalize_db": -16.0,
    "codec_sample_rate_resampling": "DACVAECodec.encode_waveform internal resampling",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def dumps(rows) -> str:
    return "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows)


def write_immutable(path: Path, content: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"Refusing to overwrite different artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def approved_rows(export: Path, reviews_path: Path, fingerprint: str):
    review = read_json(reviews_path)
    if review.get("dataset_fingerprint") != fingerprint:
        raise ValueError("Review fingerprint does not match the requested parent fingerprint")
    decisions = {item["clip_id"]: item for item in review.get("decisions", [])}
    rows = read_jsonl(export)
    approved = {key: value for key, value in decisions.items() if value.get("decision") == "approved"}
    if len(rows) != len(approved) or len({r.get("clip_id") for r in rows}) != len(rows):
        raise ValueError("Export rows do not conserve the approved human review decisions")
    for row in rows:
        decision = approved.get(row.get("clip_id"))
        if not decision or row.get("review_status") != "approved":
            raise ValueError(f"Export contains a non-approved row: {row.get('clip_id')}")
        if row.get("reviewed_text") != decision.get("reviewed_text") or row.get("clip_audio_sha256") != decision.get("clip_audio_sha256"):
            raise ValueError(f"Review/export text or audio proof mismatch: {row.get('clip_id')}")
        decision_caption = decision.get("style_caption")
        export_caption = row.get("reviewer_style_caption")
        if decision_caption is not None and not isinstance(decision_caption, str):
            raise ValueError(f"Review style caption must be a string: {row.get('clip_id')}")
        if export_caption is not None and not isinstance(export_caption, str):
            raise ValueError(f"Export style caption must be a string: {row.get('clip_id')}")
        decision_caption = (decision_caption or "").strip()
        export_caption = (export_caption or "").strip()
        if decision_caption != export_caption:
            raise ValueError(f"Review/export style caption proof mismatch: {row.get('clip_id')}")
        row["reviewer_style_caption"] = decision_caption
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v2")
    parser.add_argument("--export", type=Path)
    parser.add_argument("--reviews", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    version = args.version
    export = args.export or DATA / f"reviewed-{version}/reviewed_audio_manifest.jsonl"
    reviews_path = args.reviews or DATA / f"reviews/{version}.json"
    out = args.out or DATA / f"training-{version}"
    fingerprint = read_json(reviews_path).get("dataset_fingerprint")
    if not fingerprint:
        raise ValueError("Missing review fingerprint")
    rows = approved_rows(export, reviews_path, fingerprint)

    prior_all_path = DATA / "training-v2/all.jsonl"
    prior = read_jsonl(prior_all_path) if prior_all_path.is_file() else []
    prior_split_by_source = {}
    for item in prior:
        source = item["source_sha256"]
        split = item["split"]
        if source in prior_split_by_source and prior_split_by_source[source] != split:
            raise ValueError(f"v2 source group crossed split: {source}")
        prior_split_by_source[source] = split

    prepared, by_source = [], defaultdict(list)
    for row in rows:
        if row.get("source_manifest_fingerprint") != fingerprint:
            raise ValueError(f"Parent lineage mismatch for {row.get('clip_id')}")
        base = export.parent
        wav, transcript = base / row["clip_audio_path"], base / row["transcript_path"]
        if not wav.is_file() or not transcript.is_file():
            raise FileNotFoundError(f"Missing reviewed audio/text file for {row['clip_id']}")
        text = transcript.read_text(encoding="utf-8").rstrip("\r\n")
        if text != row["reviewed_text"] or sha256(wav) != row["clip_audio_sha256"]:
            raise ValueError(f"Reviewed transcript/audio proof mismatch for {row['clip_id']}")
        duration = row.get("clip_duration_seconds")
        if not isinstance(duration, (int, float)) or duration <= 0:
            raise ValueError(f"Missing verified duration for {row['clip_id']}")
        item = {"audio": wav.relative_to(ROOT).as_posix(), "text": text, "speaker_id": "nino", "language": "ja", "caption": row["reviewer_style_caption"],
                "clip_id": row["clip_id"], "audio_sha256": row["clip_audio_sha256"], "source_sha256": row["source_sha256"],
                "source_path": row["source_audio"], "source_bounds_seconds": {"start": row["source_timing"]["actual_start_seconds"], "end": row["source_timing"]["actual_end_seconds"]},
                "clip_duration_seconds": duration, "parent_fingerprint": fingerprint, "text_provenance": row["text_provenance"],
                "machine_flags_preserved": row["machine_flags_preserved"]}
        prepared.append(item)
        by_source[item["source_sha256"]].append(item)

    # Keep every established v2 source assignment. Assign only previously unseen sources
    # deterministically, balancing duration toward the original 10% validation target.
    new_sources = sorted(set(by_source) - set(prior_split_by_source))
    source_split = {s: prior_split_by_source[s] for s in by_source if s in prior_split_by_source}
    train_dur = sum(x["clip_duration_seconds"] for s, group in by_source.items() if source_split.get(s) == "train" for x in group)
    val_dur = sum(x["clip_duration_seconds"] for s, group in by_source.items() if source_split.get(s) == "validation" for x in group)
    for source in sorted(new_sources, key=lambda s: (-sum(x["clip_duration_seconds"] for x in by_source[s]), s)):
        duration = sum(x["clip_duration_seconds"] for x in by_source[source])
        if abs((val_dur + duration) / max(train_dur + val_dur + duration, 1) - TARGET_VALIDATION_FRACTION) < abs(val_dur / max(train_dur + val_dur + duration, 1) - TARGET_VALIDATION_FRACTION):
            source_split[source] = "validation"; val_dur += duration
        else:
            source_split[source] = "train"; train_dur += duration
    if set(source_split) != set(by_source) or "train" not in set(source_split.values()) or "validation" not in set(source_split.values()):
        raise ValueError("Could not produce nonempty inherited whole-source splits")
    split_map = {x["clip_id"]: source_split[x["source_sha256"]] for x in prepared}
    train = [x for x in prepared if split_map[x["clip_id"]] == "train"]
    validation = [x for x in prepared if split_map[x["clip_id"]] == "validation"]
    if {x["source_sha256"] for x in train} & {x["source_sha256"] for x in validation}:
        raise AssertionError("Source SHA groups overlap between splits")
    all_with_split = [{**x, "split": split_map[x["clip_id"]]} for x in prepared]
    total_duration = sum(x["clip_duration_seconds"] for x in prepared)
    summary = {"human_approved_clips": len(prepared), "parent_review_fingerprint": fingerprint, "audio_text_prepared": True,
               "model_specific_preprocessing_completed": False, "training_started": False, "total_duration_seconds": total_duration,
               "target_validation_fraction": TARGET_VALIDATION_FRACTION, "actual_validation_fraction": sum(x["clip_duration_seconds"] for x in validation) / total_duration,
               "split_policy": "whole original source SHA-256 groups; preserve all v2 assignments; assign new groups deterministically toward 10% validation duration",
               "training_source_sha256s": sorted({x["source_sha256"] for x in train}), "validation_source_sha256s": sorted({x["source_sha256"] for x in validation}),
               "training_clips": len(train), "training_duration_seconds": sum(x["clip_duration_seconds"] for x in train),
               "validation_clips": len(validation), "validation_duration_seconds": sum(x["clip_duration_seconds"] for x in validation),
               "metadata_format": "metadata.jsonl and validation_metadata.jsonl use audio_path/text/speaker_id fields; paths are project-relative."}
    write_immutable(out / "all.jsonl", dumps(all_with_split))
    write_immutable(out / "train.jsonl", dumps(train)); write_immutable(out / "validation.jsonl", dumps(validation))
    def metadata_rows(group):
        records = []
        for item in group:
            record = {"audio_path": item["audio"], "text": item["text"], "speaker_id": item["speaker_id"]}
            if item["caption"]:
                record["caption"] = item["caption"]
            records.append(record)
        return records

    write_immutable(out / "metadata.jsonl", dumps(metadata_rows(train)))
    write_immutable(out / "validation_metadata.jsonl", dumps(metadata_rows(validation)))
    write_immutable(out / "summary.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    if version in {"v3", "v4"}:
        build_cache_index(prepared, out, fingerprint)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_cache_index(prepared, out: Path, fingerprint: str):
    old_dir = DATA / "training-v2/irodori/latents"
    status = read_json(DATA / "training-v2/irodori/preparation-status.json")
    codec = status["codec_sha256"]
    reused = []
    for item in prepared:
        stem = item["clip_id"]
        meta_path, latent_path = old_dir / f"{stem}.meta.json", old_dir / f"{stem}.pt"
        if not meta_path.is_file() or not latent_path.is_file():
            continue
        meta = read_json(meta_path)
        encoding_parent = meta.get("parent_fingerprint")
        if (meta.get("clip_id") != stem or meta.get("audio_sha256") != item["audio_sha256"] or meta.get("codec_sha256") != codec
                or meta.get("settings") != LATENT_SETTINGS or not encoding_parent):
            continue
        reused.append({"clip_id": stem, "audio_sha256": item["audio_sha256"], "codec_sha256": codec,
                       "settings": LATENT_SETTINGS, "latent_path": latent_path.relative_to(ROOT).as_posix(),
                       "metadata_path": meta_path.relative_to(ROOT).as_posix(), "latent_sha256": sha256(latent_path),
                       "metadata_sha256": sha256(meta_path), "encoding_parent_fingerprint": encoding_parent,
                       "package_parent_fingerprint": fingerprint})
    write_immutable(out / "irodori/cache-index.jsonl", dumps(reused))
    prepared_ids = {x["clip_id"] for x in prepared}
    status_v3 = {"status": "resource_limited", "resource_limited": True, "prepared_audio_clips": len(prepared),
                 "reused_latents": len(reused), "missing_latents": len(prepared) - len(reused), "training_started": False,
                 "model_specific_preprocessing_completed": False, "latent_manifest_assembled": False,
                 "parent_fingerprint": fingerprint, "encoding_parent_fingerprint": OLD_FP, "codec_sha256": codec,
                 "cache_index": "irodori/cache-index.jsonl", "reused_clip_ids": [x["clip_id"] for x in reused],
                 "missing_clip_ids": sorted(prepared_ids - {x["clip_id"] for x in reused}),
                 "resume_note": "Existing paired v2 caches are referenced by proof; no encoding/model loading was performed."}
    write_immutable(out / "irodori/preparation-status.json", json.dumps(status_v3, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
