#!/usr/bin/env python3
"""Export only explicitly approved audio/text pairs from local review JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "outputs" / "dataset" / "manifest.jsonl"
DEFAULT_OUTPUT = ROOT / "outputs" / "dataset" / "reviewed"
VALID_DECISIONS = {"approved", "needs_review", "rejected"}
VALID_PROBLEM_ISSUES = {"clipped_start", "clipped_end"}
CLIP_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_fingerprint(rows: dict[str, dict[str, Any]]) -> str:
    identity = []
    for clip_id, row in rows.items():
        audio_sha = row.get("clip_sha256") or row.get("clip_audio_sha256") or row.get("wav_sha256")
        if not audio_sha:
            raise ValueError(f"{clip_id}: manifest is missing the clip WAV SHA256")
        identity.append({
            "clip_id": clip_id,
            "clip_audio_sha256": str(audio_sha).lower(),
            "original_lexical_intervals_seconds": row.get("original_lexical_intervals_seconds", []),
            "original_text_segments": row.get("original_text_segments", [row.get("text", "")]),
        })
    identity.sort(key=lambda item: item["clip_id"])
    canonical = json.dumps(
        {"schema": "nino-review-dataset-identity-v1", "clips": identity},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return sha256_bytes(canonical)


def read_manifest(path: Path) -> tuple[bytes, str, dict[str, dict[str, Any]]]:
    raw = path.read_bytes()
    rows: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"Manifest line {line_number} is not an object")
        clip_id = str(row.get("clip_id", "")).strip()
        if not CLIP_ID_RE.fullmatch(clip_id) or clip_id in rows:
            raise ValueError(f"Invalid or duplicate clip_id on manifest line {line_number}: {clip_id!r}")
        rows[clip_id] = row
    return raw, canonical_fingerprint(rows), rows


def resolve_workspace_path(value: Any, *, field: str, clip_id: str) -> Path:
    if not value:
        raise ValueError(f"{clip_id}: missing {field}")
    candidate = Path(str(value))
    resolved = candidate.resolve() if candidate.is_absolute() else (ROOT / candidate).resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"{clip_id}: {field} escapes the workspace: {value}") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"{clip_id}: {field} does not exist: {resolved}")
    return resolved


def first_value(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if value:
            return value
    return None


def validate_review(review: dict[str, Any], fingerprint: str, rows: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    if review.get("schema") != "nino-human-review-v1":
        raise ValueError("Review JSON schema must be nino-human-review-v1")
    if review.get("dataset_fingerprint") != fingerprint:
        raise ValueError("Review JSON fingerprint does not match the current input manifest")
    decisions = review.get("decisions")
    if not isinstance(decisions, list):
        raise ValueError("Review JSON decisions must be a list")
    seen: set[str] = set()
    for decision in decisions:
        if not isinstance(decision, dict):
            raise ValueError("Each review decision must be an object")
        clip_id = str(decision.get("clip_id", ""))
        if clip_id not in rows:
            raise ValueError(f"Review JSON contains unknown clip_id: {clip_id!r}")
        if clip_id in seen:
            raise ValueError(f"Review JSON contains duplicate decision for clip_id: {clip_id}")
        seen.add(clip_id)
        if decision.get("decision") not in VALID_DECISIONS:
            raise ValueError(f"{clip_id}: invalid human decision {decision.get('decision')!r}")
        expected_draft = str(rows[clip_id].get("text", rows[clip_id].get("transcript", "")))
        recorded_draft = decision.get("original_draft_text")
        if recorded_draft != expected_draft:
            raise ValueError(f"{clip_id}: review draft text does not match the manifest text")
        expected_audio_sha = rows[clip_id].get("clip_sha256", rows[clip_id].get("clip_audio_sha256", rows[clip_id].get("wav_sha256")))
        recorded_audio_sha = decision.get("clip_audio_sha256")
        if recorded_audio_sha is not None and str(recorded_audio_sha).lower() != str(expected_audio_sha).lower():
            raise ValueError(f"{clip_id}: review decision WAV hash does not match the manifest")
        problem_issues = decision.get("problem_issues", [])
        if not isinstance(problem_issues, list) or any(not isinstance(issue, str) for issue in problem_issues):
            raise ValueError(f"{clip_id}: problem_issues must be a list of strings")
        if len(problem_issues) != len(set(problem_issues)) or any(issue not in VALID_PROBLEM_ISSUES for issue in problem_issues):
            raise ValueError(f"{clip_id}: problem_issues contains duplicate or unsupported issue codes")
        problem_notes = decision.get("problem_notes", "")
        if not isinstance(problem_notes, str):
            raise ValueError(f"{clip_id}: problem_notes must be a string")
        if decision["decision"] == "approved":
            checks = decision.get("approval_checks")
            if not isinstance(checks, dict) or not all(
                checks.get(key) is True
                for key in ("correct_speaker", "complete_audio", "corrected_text")
            ):
                raise ValueError(f"{clip_id}: approval is missing explicit speaker/audio/text confirmations")
            if not isinstance(decision.get("reviewed_text"), str) or not decision["reviewed_text"].strip():
                raise ValueError(f"{clip_id}: approved review has empty reviewed_text")
            if problem_issues:
                raise ValueError(f"{clip_id}: approved review still has unresolved audio problems")
        elif decision.get("reviewed_text") is not None and not isinstance(decision.get("reviewed_text"), str):
            raise ValueError(f"{clip_id}: reviewed_text must be a string")
    return decisions


def approved_sources(
    decisions: list[dict[str, Any]], rows: dict[str, dict[str, Any]]
) -> list[tuple[dict[str, Any], dict[str, Any], Path, str]]:
    prepared = []
    for decision in decisions:
        if decision["decision"] != "approved":
            continue
        clip_id = decision["clip_id"]
        row = rows[clip_id]
        source_path = resolve_workspace_path(row.get("clip_audio_path"), field="clip_audio_path", clip_id=clip_id)
        if source_path.name != f"{clip_id}.wav":
            raise ValueError(f"{clip_id}: clip_audio_path basename must be {clip_id}.wav")
        source_hash = sha256_file(source_path)
        expected_hash = first_value(row, "clip_audio_sha256", "clip_sha256", "audio_sha256", "wav_sha256")
        if expected_hash and source_hash.lower() != str(expected_hash).lower():
            raise ValueError(f"{clip_id}: WAV hash mismatch against the source manifest")
        prepared.append((decision, row, source_path, source_hash))
    return prepared


def export_reviewed(
    manifest_path: Path,
    review_path: Path,
    output_dir: Path,
    *,
    dry_run: bool = False,
) -> int:
    _, fingerprint, rows = read_manifest(manifest_path)
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if not isinstance(review, dict):
        raise ValueError("Review JSON root must be an object")
    decisions = validate_review(review, fingerprint, rows)
    prepared = approved_sources(decisions, rows)
    print(f"Input clips: {len(rows)}; human decisions: {len(decisions)}; approved: {len(prepared)}")
    if dry_run:
        print("Dry run: no WAV/TXT files or reviewed manifest written.")
        return len(prepared)

    reviewed_root = (ROOT / "outputs" / "dataset").resolve()
    output_dir = output_dir.resolve()
    try:
        relative_output = output_dir.relative_to(reviewed_root)
    except ValueError as exc:
        raise ValueError(f"Reviewed exports must stay under {reviewed_root}") from exc
    if not relative_output.parts:
        raise ValueError("Choose a new subdirectory under outputs/dataset for reviewed exports")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty; choose a new path: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = output_dir / "clips"
    text_dir = output_dir / "text"
    if prepared:
        clips_dir.mkdir(parents=True, exist_ok=True)
        text_dir.mkdir(parents=True, exist_ok=True)

    output_rows: list[dict[str, Any]] = []
    for decision, row, source_path, source_hash in prepared:
        clip_id = decision["clip_id"]
        out_wav = clips_dir / f"{clip_id}.wav"
        out_text = text_dir / f"{clip_id}.txt"
        shutil.copy2(source_path, out_wav)
        reviewed_text = decision["reviewed_text"]
        text_bytes = (reviewed_text.rstrip("\r\n") + "\n").encode("utf-8")
        out_text.write_bytes(text_bytes)
        checks = decision["approval_checks"]
        output_rows.append({
            "clip_id": clip_id,
            "review_status": "approved",
            "human_reviewed_at": decision.get("recorded_at"),
            "approval_checks": checks,
            "variant_label": row.get("variant_label"),
            "reviewer_style_caption": decision.get("style_caption", ""),
            "problem_issues": decision.get("problem_issues", []),
            "problem_notes": decision.get("problem_notes", ""),
            "reviewed_text": reviewed_text,
            "original_draft_text": row.get("text", row.get("transcript", "")),
            "source_audio": row.get("source_audio"),
            "source_sha256": row.get("source_sha256"),
            "annotation_path": row.get("annotation_path"),
            "annotation_sha256": row.get("annotation_sha256"),
            "source_speaker_label": row.get("source_speaker_label"),
            "human_verified_speaker": True,
            "identity_verified_in_source": row.get("identity_verified", False),
            "source_segment_indices": row.get("segment_indices"),
            "original_start_seconds": row.get("original_start_seconds", row.get("start_sec")),
            "original_end_seconds": row.get("original_end_seconds", row.get("end_sec")),
            "padded_start_seconds": row.get("padded_start_seconds"),
            "padded_end_seconds": row.get("padded_end_seconds"),
            "source_timing": {
                key: value for key, value in row.items()
                if any(part in key.lower() for part in ("start", "end", "time", "interval", "frame", "sample"))
                and isinstance(value, (int, float, str, list, tuple))
                and not isinstance(value, bool)
            },
            "source_manifest_fingerprint": fingerprint,
            "clip_audio_path": (Path("clips") / out_wav.name).as_posix(),
            "clip_audio_sha256": source_hash,
            "transcript_path": (Path("text") / out_text.name).as_posix(),
            "reviewed_text_sha256": sha256_bytes(text_bytes),
            "native_sample_rate": row.get("native_sample_rate", row.get("sample_rate")),
            "clip_duration_seconds": row.get("clip_duration_seconds", row.get("duration_seconds")),
            "machine_flags_preserved": row.get("machine_flags", {
                key: value for key, value in row.items()
                if any(part in key.lower() for part in ("flag", "unverified", "background", "identity", "qc"))
            }),
            "source_text_provenance": row.get("text_provenance", row.get("transcript_provenance", "machine draft; unverified")),
            "text_provenance": "human_reviewed_corrected",
            "original_text_segments": row.get("original_text_segments"),
            "latents_created": False,
            "training_ready": False,
            "processing_note": "Reviewed audio/text export only; no latent encoding or training preparation was performed.",
        })

    manifest_out = output_dir / "reviewed_audio_manifest.jsonl"
    manifest_out.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output_rows),
        encoding="utf-8",
    )
    if output_rows:
        print(f"Exported {len(output_rows)} explicitly approved WAV/TXT pair(s) to {output_dir}")
    else:
        print(f"No clips are human-approved; wrote an empty reviewed manifest at {manifest_out} and copied no media.")
    return len(output_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--review", type=Path, required=True, help="JSON downloaded by the local review page")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    manifest = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    review = args.review if args.review.is_absolute() else ROOT / args.review
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    export_reviewed(manifest, review, output_dir, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
