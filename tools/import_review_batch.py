#!/usr/bin/env python3
"""Merge one partial human review batch into a new immutable dataset version.

The importer validates both review fingerprints, changes only IDs explicitly
present in the candidate review, and delegates approved-media export to the
shared dataset exporter. Dry runs do not write, copy, or alias any files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import export_reviewed_dataset as exporter  # noqa: E402

VERSION_RE = re.compile(r"^v[0-9]+(?:\.[0-9]+)?$")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def resolve_input(value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def load_review(path: Path) -> tuple[dict[str, Any], bytes, str]:
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Review root must be a JSON object: {path}")
    return value, raw, sha256_bytes(raw)


def audio_sha(row: dict[str, Any]) -> str:
    value = exporter.first_value(row, "clip_sha256", "clip_audio_sha256", "wav_sha256")
    if not value:
        raise ValueError(f"{row.get('clip_id')}: manifest row is missing a clip audio SHA256")
    return str(value).lower()


def decision_identity(row: dict[str, Any]) -> tuple[Any, ...]:
    """Fields that make a repeated review decision meaningfully identical."""
    return tuple(row.get(key) for key in (
        "decision", "reviewed_text", "approval_checks", "problem_issues", "problem_notes",
    ))


def rows_as_jsonl(rows: dict[str, dict[str, Any]]) -> bytes:
    return b"".join((json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8") for row in rows.values())


def review_json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def batch_import(
    *,
    base_manifest_path: Path,
    base_review_path: Path,
    candidate_manifest_path: Path,
    candidate_review_path: Path,
    version: str,
    dry_run: bool = False,
) -> dict[str, Any]:
    if not VERSION_RE.fullmatch(version):
        raise ValueError(f"Unsafe version {version!r}; expected vN or vN.N")

    base_manifest_path = resolve_input(base_manifest_path)
    base_review_path = resolve_input(base_review_path)
    candidate_manifest_path = resolve_input(candidate_manifest_path)
    candidate_review_path = resolve_input(candidate_review_path)
    for path in (base_manifest_path, base_review_path, candidate_manifest_path, candidate_review_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    base_raw, base_fingerprint, base_rows = exporter.read_manifest(base_manifest_path)
    candidate_raw, candidate_fingerprint, candidate_rows = exporter.read_manifest(candidate_manifest_path)
    base_review, base_review_raw, base_review_sha = load_review(base_review_path)
    candidate_review, candidate_review_raw, candidate_review_sha = load_review(candidate_review_path)
    base_decisions = exporter.validate_review(base_review, base_fingerprint, base_rows)
    round_decisions = exporter.validate_review(candidate_review, candidate_fingerprint, candidate_rows)
    base_by_id = {d["clip_id"]: d for d in base_decisions}
    round_by_id = {d["clip_id"]: d for d in round_decisions}

    if set(base_by_id) != set(base_rows):
        missing = sorted(set(base_rows) - set(base_by_id))
        extra = sorted(set(base_by_id) - set(base_rows))
        raise ValueError(f"Base review must cover the complete base manifest; missing={missing[:5]}, extra={extra[:5]}")
    if set(candidate_rows) - set(base_rows):
        raise ValueError(f"Candidate IDs are absent from the base manifest: {sorted(set(candidate_rows)-set(base_rows))[:5]}")
    if set(round_by_id) - set(candidate_rows):
        raise ValueError(f"Round decision IDs are absent from the candidate manifest: {sorted(set(round_by_id)-set(candidate_rows))[:5]}")

    base_reviewed_at = base_review.get("exported_at")
    output_root = ROOT / "outputs/dataset" / version
    manifest_out = output_root / "manifest.jsonl"
    review_out = ROOT / "outputs/dataset/reviews" / f"{version}.json"
    reviewed_out = ROOT / "outputs/dataset" / f"reviewed-{version}"
    summary_out = output_root / "summary.json"
    remaining_out = output_root / "remaining_review.jsonl"

    merged_rows = dict(base_rows)
    changed_ids: list[str] = []
    already_applied_ids: list[str] = []
    candidate_sources: dict[str, tuple[Path, str]] = {}
    planned_aliases: dict[str, Path] = {}

    for clip_id, round_decision in round_by_id.items():
        candidate = candidate_rows[clip_id]
        base_row = base_rows[clip_id]
        base_decision = base_by_id[clip_id]
        candidate_path = exporter.resolve_workspace_path(candidate.get("clip_audio_path"), field="clip_audio_path", clip_id=clip_id)
        candidate_hash = exporter.sha256_file(candidate_path)
        if candidate_hash != audio_sha(candidate):
            raise ValueError(f"{clip_id}: candidate WAV bytes do not match candidate manifest SHA256")
        if round_decision.get("clip_audio_sha256") and str(round_decision["clip_audio_sha256"]).lower() != candidate_hash:
            raise ValueError(f"{clip_id}: review WAV hash differs from candidate audio")
        candidate_sources[clip_id] = (candidate_path, candidate_hash)

        base_hash = audio_sha(base_row)
        already_applied = (
            base_hash == candidate_hash
            and str(base_row.get("text", base_row.get("transcript", ""))) == str(candidate.get("text", candidate.get("transcript", "")))
            and decision_identity(base_decision) == decision_identity(round_decision)
        )
        if already_applied:
            already_applied_ids.append(clip_id)
            continue
        if base_decision.get("decision") != "needs_review":
            raise ValueError(
                f"{clip_id}: round requires a base needs_review decision, got {base_decision.get('decision')!r}; "
                "approved/rejected rows may only be repeated as exact idempotent matches"
            )

        replacement = dict(candidate)
        source_rel = str(candidate.get("clip_audio_path"))
        if candidate_path.name != f"{clip_id}.wav":
            alias_rel = Path("outputs/dataset") / version / "source_audio" / f"{clip_id}.wav"
            planned_aliases[clip_id] = ROOT / alias_rel
            replacement["clip_audio_path"] = alias_rel.as_posix()
        replacement["review_batch_import"] = {
            "version": version,
            "base_manifest_path": base_manifest_path.relative_to(ROOT).as_posix() if base_manifest_path.is_relative_to(ROOT) else str(base_manifest_path),
            "base_manifest_fingerprint": base_fingerprint,
            "base_review_path": base_review_path.relative_to(ROOT).as_posix() if base_review_path.is_relative_to(ROOT) else str(base_review_path),
            "base_review_sha256": base_review_sha,
            "candidate_manifest_path": candidate_manifest_path.relative_to(ROOT).as_posix() if candidate_manifest_path.is_relative_to(ROOT) else str(candidate_manifest_path),
            "candidate_manifest_fingerprint": candidate_fingerprint,
            "candidate_source_audio_path": source_rel,
            "candidate_review_path": candidate_review_path.relative_to(ROOT).as_posix() if candidate_review_path.is_relative_to(ROOT) else str(candidate_review_path),
            "candidate_review_sha256": candidate_review_sha,
            "candidate_audio_sha256": candidate_hash,
            "human_decision": round_decision.get("decision"),
            "human_recorded_at": round_decision.get("recorded_at"),
        }
        merged_rows[clip_id] = replacement
        changed_ids.append(clip_id)

    merged_fingerprint = exporter.canonical_fingerprint(merged_rows)
    merged_decisions = [dict(round_by_id.get(cid, old)) for cid, old in base_by_id.items()]
    source_reviews = list(base_review.get("source_reviews", []))
    known_sources = {(item.get("path"), item.get("sha256"), item.get("dataset_fingerprint")) for item in source_reviews if isinstance(item, dict)}
    for item in (
        {"path": base_review_path.relative_to(ROOT).as_posix() if base_review_path.is_relative_to(ROOT) else str(base_review_path), "sha256": base_review_sha, "dataset_fingerprint": base_fingerprint},
        {"path": candidate_review_path.relative_to(ROOT).as_posix() if candidate_review_path.is_relative_to(ROOT) else str(candidate_review_path), "sha256": candidate_review_sha, "dataset_fingerprint": candidate_fingerprint},
    ):
        key = (item["path"], item["sha256"], item["dataset_fingerprint"])
        if key not in known_sources:
            source_reviews.append(item)
            known_sources.add(key)
    merged_review = {
        "schema": "nino-human-review-v1",
        "dataset_fingerprint": merged_fingerprint,
        "manifest_path": manifest_out.relative_to(ROOT).as_posix(),
        "exported_at": candidate_review.get("exported_at", base_reviewed_at),
        "decisions": merged_decisions,
        "source_reviews": source_reviews,
    }
    # Verify the final review against the selected manifest before any output is written.
    merged_validated_decisions = exporter.validate_review(merged_review, merged_fingerprint, merged_rows)
    status_counts = Counter(d["decision"] for d in merged_validated_decisions)
    approved_count = sum(d["decision"] == "approved" for d in merged_validated_decisions)
    remaining_count = sum(d["decision"] == "needs_review" for d in merged_validated_decisions)
    rejected_count = sum(d["decision"] == "rejected" for d in merged_validated_decisions)
    unreviewed_candidate_ids = sorted(set(candidate_rows) - set(round_by_id))
    no_op = not changed_ids and len(already_applied_ids) == len(round_by_id)

    # Check all media that the shared exporter will need without materializing aliases.
    for decision in merged_validated_decisions:
        if decision["decision"] != "approved":
            continue
        cid = decision["clip_id"]
        if cid in candidate_sources and cid in changed_ids:
            source_path, source_hash = candidate_sources[cid]
            if source_hash != audio_sha(merged_rows[cid]):
                raise ValueError(f"{cid}: selected candidate media hash mismatch")
        else:
            source_path = exporter.resolve_workspace_path(merged_rows[cid].get("clip_audio_path"), field="clip_audio_path", clip_id=cid)
            if exporter.sha256_file(source_path) != audio_sha(merged_rows[cid]):
                raise ValueError(f"{cid}: selected base media hash mismatch")

    remaining_rows = [
        {"clip_id": d["clip_id"], "decision": d, "manifest_row": merged_rows[d["clip_id"]]}
        for d in merged_validated_decisions if d["decision"] == "needs_review"
    ]
    remaining_bytes = b"".join((json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8") for r in remaining_rows)
    merged_manifest_bytes = rows_as_jsonl(merged_rows)
    merged_review_bytes = review_json_bytes(merged_review)

    result: dict[str, Any] = {
        "version": version,
        "dry_run": dry_run,
        "no_op_already_applied": no_op,
        "no_audio_text_or_review_status_changes": no_op,
        "base_count": len(base_rows),
        "candidate_count": len(candidate_rows),
        "round_decision_count": len(round_by_id),
        "applied_count": len(changed_ids),
        "already_applied_count": len(already_applied_ids),
        "changed_ids": changed_ids,
        "already_applied_ids": already_applied_ids,
        "unreviewed_candidate_ids": unreviewed_candidate_ids,
        "status_counts": {"approved": approved_count, "needs_review": remaining_count, "rejected": rejected_count},
        "exported_approved_count": approved_count,
        "remaining_review_count": remaining_count,
        "base_fingerprint": base_fingerprint,
        "candidate_fingerprint": candidate_fingerprint,
        "master_fingerprint": merged_fingerprint,
        "master_manifest_sha256": sha256_bytes(merged_manifest_bytes),
        "review_sha256": sha256_bytes(merged_review_bytes),
        "output_paths": {
            "manifest": manifest_out.relative_to(ROOT).as_posix(),
            "review": review_out.relative_to(ROOT).as_posix(),
            "reviewed_export": reviewed_out.relative_to(ROOT).as_posix(),
            "summary": summary_out.relative_to(ROOT).as_posix(),
            "remaining_review": remaining_out.relative_to(ROOT).as_posix(),
        },
    }
    if dry_run or no_op:
        result["writes_performed"] = False
        return result

    # Actual imports only write into new version/review/export targets.
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Version directory is not empty; refusing overwrite: {output_root}")
    if review_out.exists():
        raise FileExistsError(f"Review output already exists; refusing overwrite: {review_out}")
    if reviewed_out.exists():
        raise FileExistsError(f"Reviewed export already exists; refusing overwrite: {reviewed_out}")
    for alias in planned_aliases.values():
        if alias.exists():
            raise FileExistsError(f"Alias path already exists; refusing overwrite: {alias}")

    # Preflight approved candidate and base media with the same strict basename/hash rules as exporter.
    for decision in merged_validated_decisions:
        if decision["decision"] != "approved":
            continue
        cid = decision["clip_id"]
        row = merged_rows[cid]
        if cid in planned_aliases:
            candidate_path, candidate_hash = candidate_sources[cid]
            if candidate_path.name == f"{cid}.wav" or candidate_hash != audio_sha(row):
                raise ValueError(f"{cid}: invalid planned alias source")
        else:
            source_path = exporter.resolve_workspace_path(row.get("clip_audio_path"), field="clip_audio_path", clip_id=cid)
            if source_path.name != f"{cid}.wav":
                raise ValueError(f"{cid}: selected WAV basename must be {cid}.wav")

    # Materialize only byte-identical aliases needed by the shared exporter.
    for clip_id, alias in planned_aliases.items():
        source_path, source_hash = candidate_sources[clip_id]
        payload = source_path.read_bytes()
        if sha256_bytes(payload) != source_hash:
            raise ValueError(f"{clip_id}: candidate changed before alias copy")
        alias.parent.mkdir(parents=True, exist_ok=True)
        alias.write_bytes(payload)

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_out.write_bytes(merged_manifest_bytes)
    review_out.parent.mkdir(parents=True, exist_ok=True)
    review_out.write_bytes(merged_review_bytes)
    remaining_out.write_bytes(remaining_bytes)
    exported = exporter.export_reviewed(manifest_out, review_out, reviewed_out)
    if exported != approved_count:
        raise RuntimeError(f"Shared exporter emitted {exported}, expected {approved_count}")
    result["exported_approved_count"] = exported
    result["writes_performed"] = True
    result["reviewed_export_manifest"] = (reviewed_out / "reviewed_audio_manifest.jsonl").relative_to(ROOT).as_posix()
    summary_out.write_bytes(review_json_bytes(result))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--base-review", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--candidate-review", type=Path, required=True)
    parser.add_argument("--version", required=True, help="New output version: vN or vN.N")
    parser.add_argument("--dry-run", action="store_true", help="Validate and summarize; write no files")
    args = parser.parse_args()
    result = batch_import(
        base_manifest_path=args.base_manifest,
        base_review_path=args.base_review,
        candidate_manifest_path=args.candidate_manifest,
        candidate_review_path=args.candidate_review,
        version=args.version,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
