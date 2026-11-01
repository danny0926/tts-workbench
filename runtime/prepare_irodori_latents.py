#!/usr/bin/env python3
"""CPU-only, resumable DACVAE preparation for a reviewed package manifest."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "outputs/dataset/training-v2/all.jsonl"
DEFAULT_CODEC = ROOT / ".cache_hf/hub/models--Aratako--Semantic-DACVAE-Japanese-32dim/snapshots/47376ee24834d7a05a48ebabfe3cde29b3c5e214/weights.pth"
DEFAULT_OUT = ROOT / "outputs/dataset/training-v2/irodori"
PARENT_FINGERPRINT = "4012b11aefc89f5a458d18c33a49b8ba31b20865225619c5f3f778e7966f1cae"
CODEC_REPO = "Aratako/Semantic-DACVAE-Japanese-32dim"
CODEC_REVISION = "47376ee24834d7a05a48ebabfe3cde29b3c5e214"
SETTINGS = {
    "format": "DACVAE deterministic encoder mean",
    "device": "cpu",
    "dtype": "float32",
    "deterministic_encode": True,
    "normalize_db": -16.0,
    "codec_sample_rate_resampling": "DACVAECodec.encode_waveform internal resampling",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != data:
            raise FileExistsError(f"Refusing to overwrite different metadata: {path}")
        return
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(data, encoding="utf-8", newline="\n")
    os.replace(temp, path)


def load_reusable_cache_index(index_path: Path, manifest_rows: list[dict], codec_sha: str, parent_fingerprint: str) -> dict[str, tuple[Path, Path]]:
    """Validate cache proofs against current audio rows without opening latent tensors."""
    manifest_by_id = {row["clip_id"]: row for row in manifest_rows}
    reuse = {}
    for entry in read_jsonl(index_path):
        clip_id = entry.get("clip_id")
        row = manifest_by_id.get(clip_id)
        if row is None:
            raise ValueError(f"Cache index references a clip outside the current manifest: {clip_id}")
        if entry.get("audio_sha256") != row.get("audio_sha256"):
            raise ValueError(f"Cache-index audio hash differs from current manifest: {clip_id}")
        if (entry.get("codec_sha256") != codec_sha or entry.get("settings") != SETTINGS
                or entry.get("package_parent_fingerprint") != parent_fingerprint):
            raise ValueError(f"Reusable latent has incompatible codec/settings: {clip_id}")
        latent = (ROOT / entry["latent_path"]).resolve()
        metadata = (ROOT / entry["metadata_path"]).resolve()
        if not latent.is_file() or not metadata.is_file() or sha256(latent) != entry.get("latent_sha256") or sha256(metadata) != entry.get("metadata_sha256"):
            raise ValueError(f"Reusable latent proof/file mismatch: {clip_id}")
        meta = json.loads(metadata.read_text(encoding="utf-8"))
        if (meta.get("clip_id") != clip_id or meta.get("audio_sha256") != row.get("audio_sha256")
                or meta.get("codec_sha256") != codec_sha or meta.get("settings") != SETTINGS
                or meta.get("parent_fingerprint") != entry.get("encoding_parent_fingerprint")):
            raise ValueError(f"Reusable metadata does not match cache-index lineage: {clip_id}")
        if clip_id in reuse:
            raise ValueError(f"Duplicate cache-index entry: {clip_id}")
        reuse[clip_id] = (latent, metadata)
    return reuse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--codec-weights", type=Path, default=DEFAULT_CODEC)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cache-index", type=Path, help="Optional JSONL index of verified reusable latent files.")
    parser.add_argument("--limit", type=int, help="Encode only the first N rows (CPU smoke run).")
    parser.add_argument("--compare-existing", action="store_true", help="Re-encode one cached row and compare without writing.")
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    weights = args.codec_weights.resolve()
    out = args.output_dir.resolve()
    if not manifest.is_file() or not weights.is_file():
        raise FileNotFoundError(f"Missing manifest or local codec weights: {manifest}; {weights}")
    rows = read_jsonl(manifest)
    parent_fingerprints = {row.get("parent_fingerprint") for row in rows}
    if not rows or len(parent_fingerprints) != 1 or not next(iter(parent_fingerprints)):
        raise ValueError("Input manifest must contain rows from exactly one reviewed parent fingerprint")
    parent_fingerprint = next(iter(parent_fingerprints))
    if len({row["clip_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate clip_id in input manifest")
    manifest_rows = rows
    if args.compare_existing:
        if args.limit is not None:
            raise ValueError("Use either --compare-existing or --limit, not both")
        rows = rows[:1]
    elif args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        rows = rows[: args.limit]

    # Hash the large codec checkpoint once. The digest is reused for every
    # cache-index entry and all later provenance checks.
    codec_sha = sha256(weights)
    reuse = load_reusable_cache_index(args.cache_index.resolve(), manifest_rows, codec_sha, parent_fingerprint) if args.cache_index else {}

    # Verify all input WAVs before importing/loading the large codec model.
    for row in rows:
        wav = (ROOT / Path(row["audio"])).resolve()
        if not wav.is_file() or sha256(wav) != row["audio_sha256"]:
            raise ValueError(f"Missing or changed audio for {row['clip_id']}: {wav}")
        if not row.get("text") or row.get("speaker_id") != "nino" or row.get("language") != "ja":
            raise ValueError(f"Invalid reviewed text/speaker/language for {row['clip_id']}")
        cached = reuse.get(row["clip_id"])
        if cached:
            metadata = json.loads(cached[1].read_text(encoding="utf-8"))
            if metadata.get("audio_sha256") != row["audio_sha256"] or metadata.get("clip_id") != row["clip_id"] or metadata.get("codec_sha256") != codec_sha or metadata.get("settings") != SETTINGS:
                raise ValueError(f"Reusable cache metadata does not match manifest row: {row['clip_id']}")

    cache_key_inputs = {
        "codec_repo": CODEC_REPO,
        "codec_revision": CODEC_REVISION,
        "codec_sha256": codec_sha,
        "settings": SETTINGS,
    }
    cache_key = hashlib.sha256(json.dumps(cache_key_inputs, sort_keys=True).encode("utf-8")).hexdigest()
    out.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(ROOT / ".cache_hf")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    os.environ["OPENBLAS_NUM_THREADS"] = "2"
    sys.path.insert(0, str(ROOT / "vendor/irodori-tts"))
    import torch
    import soundfile as sf
    from irodori_tts.codec import DACVAECodec

    torch.set_num_threads(2)
    torch.manual_seed(0)
    codec = DACVAECodec.load(
        repo_id=str(weights), device="cpu", dtype=torch.float32,
        deterministic_encode=True, deterministic_decode=False, normalize_db=-16.0,
    )
    if codec.latent_dim != 32:
        raise ValueError(f"Expected v4.1 Small 32-dimensional codec, got {codec.latent_dim}")
    # DACVAECodec's encode path uses only encoder, _pad, and quantizer.in_proj.
    # Release the unused reconstruction decoder after load to reduce resident RAM.
    if getattr(codec.model, "decoder", None) is None:
        raise ValueError("Expected the local DACVAE model to expose a decoder before pruning")
    codec.model.decoder = None
    gc.collect()

    output_rows = []
    completed = 0
    for row in rows:
        cached_pair = reuse.get(row["clip_id"])
        if cached_pair:
            latent_path, metadata_path = cached_pair
            cached_meta = json.loads(metadata_path.read_text(encoding="utf-8"))
            latent_path_text = Path(os.path.relpath(latent_path, out)).as_posix()
            num_frames = int(cached_meta["num_frames"])
            cache_key = cached_meta["cache_key"]
        else:
            latent_path = out / "latents" / f"{row['clip_id']}.pt"
            metadata_path = latent_path.with_suffix(".meta.json")
            latent_path.parent.mkdir(parents=True, exist_ok=True)
            cache = {
                "cache_key": cache_key,
                "audio_sha256": row["audio_sha256"],
                "codec_sha256": codec_sha,
                "settings": SETTINGS,
                "clip_id": row["clip_id"],
                "parent_fingerprint": parent_fingerprint,
            }
            if args.compare_existing and not (latent_path.is_file() and metadata_path.is_file()):
                raise FileNotFoundError(f"Cannot compare without a complete existing cache pair: {latent_path}")
            if latent_path.exists() or metadata_path.exists():
                if not latent_path.is_file() or not metadata_path.is_file():
                    raise FileExistsError(f"Incomplete latent cache entry: {latent_path}")
                cached_meta = json.loads(metadata_path.read_text(encoding="utf-8"))
                if any(cached_meta.get(key) != value for key, value in cache.items()):
                    raise FileExistsError(f"Existing latent has a different provenance key: {latent_path}")
                latent = torch.load(latent_path, map_location="cpu", weights_only=True)
                if not isinstance(latent, torch.Tensor) or latent.ndim != 2 or latent.shape[1] != 32:
                    raise ValueError(f"Invalid latent tensor: {latent_path}")
                if args.compare_existing:
                    wav_path = ROOT / Path(row["audio"])
                    audio, sample_rate = sf.read(str(wav_path), dtype="float32", always_2d=True)
                    waveform = torch.from_numpy(audio.T.copy()).unsqueeze(0)
                    recomputed = codec.encode_waveform(waveform, int(sample_rate)).squeeze(0).cpu().contiguous()
                    if recomputed.shape != latent.shape:
                        raise ValueError(f"Decoder-pruned latent shape mismatch: {tuple(recomputed.shape)} != {tuple(latent.shape)}")
                    max_error = float((recomputed - latent).abs().max().item())
                    exact = bool(torch.equal(recomputed, latent))
                    print(json.dumps({"clip_id": row["clip_id"], "shape": list(latent.shape), "bitwise_equal": exact, "max_abs_error": max_error, "cached_latent_overwritten": False}, ensure_ascii=False))
                    if not exact:
                        raise ValueError(f"Decoder-pruned re-encode differs from cached latent (max_abs_error={max_error})")
                    return
            else:
                wav_path = ROOT / Path(row["audio"])
                audio, sample_rate = sf.read(str(wav_path), dtype="float32", always_2d=True)
                waveform = torch.from_numpy(audio.T.copy()).unsqueeze(0)
                latent = codec.encode_waveform(waveform, int(sample_rate)).squeeze(0).cpu().contiguous()
                if latent.ndim != 2 or latent.shape[1] != 32 or not torch.isfinite(latent).all():
                    raise ValueError(f"Encoder returned invalid latent for {row['clip_id']}: {tuple(latent.shape)}")
                temp = latent_path.with_suffix(".pt.tmp")
                torch.save(latent, temp)
                os.replace(temp, latent_path)
                cache["num_frames"] = int(latent.shape[0])
                atomic_json(metadata_path, cache)
                latent_path_text = latent_path.relative_to(out).as_posix()
                num_frames = int(latent.shape[0])
        item = {
            "text": row["text"],
            "latent_path": latent_path_text,
            "num_frames": num_frames,
            "speaker_id": row["speaker_id"],
            "caption": row.get("caption", ""),
            "clip_id": row["clip_id"],
            "split": row["split"],
            "audio_sha256": row["audio_sha256"],
            "source_sha256": row["source_sha256"],
            "source_path": row["source_path"],
            "source_bounds_seconds": row["source_bounds_seconds"],
            "parent_fingerprint": parent_fingerprint,
            "latent_cache_key": cache_key,
            "machine_flags_preserved": row["machine_flags_preserved"],
        }
        output_rows.append(item)
        completed += 1
        print(f"[{completed}/{len(rows)}] {row['clip_id']} frames={num_frames}" + (" (reused)" if cached_pair else ""), flush=True)

    if args.compare_existing:
        raise ValueError("--compare-existing requires an existing latent and metadata pair")
    if args.limit is not None:
        print(f"CPU smoke run complete: {completed} latent(s); no full manifest written.")
        return
    manifest_path = out / "manifest.jsonl"
    manifest_text = "".join(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n" for item in output_rows)
    if manifest_path.exists() and manifest_path.read_text(encoding="utf-8") != manifest_text:
        raise FileExistsError(f"Refusing to overwrite different training manifest: {manifest_path}")
    if not manifest_path.exists():
        manifest_path.write_text(manifest_text, encoding="utf-8", newline="\n")
    summary = {
        "parent_fingerprint": parent_fingerprint,
        "codec_repo": CODEC_REPO,
        "codec_revision": CODEC_REVISION,
        "codec_sha256": codec_sha,
        "cache_key": cache_key,
        "preprocessing": SETTINGS,
        "latent_count": completed,
        "latent_dim": codec.latent_dim,
        "train_count": sum(item["split"] == "train" for item in output_rows),
        "validation_count": sum(item["split"] == "validation" for item in output_rows),
        "manifest_path": str(manifest_path.relative_to(ROOT).as_posix()),
    }
    atomic_json(out / "preparation.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
