#!/usr/bin/env python3
"""Validate the prepared Irodori manifest and optionally launch training.

Training is intentionally opt-in via --launch-training. The default path only
checks dataset/config contracts and never imports the upstream training module.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "outputs/dataset/training-v2/irodori/manifest.jsonl"
DEFAULT_CONFIG = ROOT / "config/nino-irodori-lora-v2.yaml"
DEFAULT_WEIGHTS = ROOT / ".cache_hf/hub/models--Aratako--Irodori-TTS-v4.1-Small/snapshots/2b28324dc263ed5e6638b3cf3dd94c82ead07b4b/model.safetensors"
UPSTREAM = ROOT / "vendor/irodori-tts/train.py"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate(manifest: Path, config: Path) -> tuple[list[dict], dict]:
    if not manifest.is_file() or not config.is_file():
        raise FileNotFoundError(f"Missing manifest or config: {manifest}; {config}")
    import yaml

    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or set(cfg) != {"model", "train"}:
        raise ValueError("Training YAML must contain exactly model and train sections")
    if cfg["model"].get("latent_dim") != 32 or not cfg["train"].get("lora_enabled"):
        raise ValueError("Config does not target v4.1-Small 32-dim LoRA")
    rows = load_jsonl(manifest)
    if not rows:
        raise ValueError("Prepared manifest is empty")
    parent_fingerprints = {row.get("parent_fingerprint") for row in rows}
    if len(parent_fingerprints) != 1 or not next(iter(parent_fingerprints)):
        raise ValueError("Manifest rows must share one reviewed parent fingerprint")
    clips = [row.get("clip_id") for row in rows]
    if len(set(clips)) != len(clips) or any(not clip for clip in clips):
        raise ValueError("Manifest clip IDs must be present and unique")
    train = [row for row in rows if row.get("split") == "train"]
    valid = [row for row in rows if row.get("split") == "validation"]
    if len(train) + len(valid) != len(rows):
        raise ValueError(f"Unexpected split counts: train={len(train)} validation={len(valid)} total={len(rows)}")
    train_sources = {row.get("source_sha256") for row in train}
    valid_sources = {row.get("source_sha256") for row in valid}
    if not train_sources or not valid_sources or train_sources & valid_sources:
        raise ValueError("Original source hashes must be disjoint and nonempty across splits")
    if {row["clip_id"] for row in train} & {row["clip_id"] for row in valid}:
        raise ValueError("Clip IDs overlap across train and validation")
    for row in rows:
        for key in ("text", "latent_path", "num_frames", "speaker_id", "caption"):
            if key not in row:
                raise ValueError(f"Missing Irodori field {key!r} for {row.get('clip_id')}")
        if not row["text"] or row["speaker_id"] != "nino" or int(row["num_frames"]) <= 0:
            raise ValueError(f"Invalid text/speaker/frame count for {row['clip_id']}")
        if row.get("parent_fingerprint") != next(iter(parent_fingerprints)):
            raise ValueError(f"Unexpected parent lineage for {row['clip_id']}")
        latent = (manifest.parent / row["latent_path"]).resolve()
        if not latent.is_file():
            raise FileNotFoundError(f"Missing latent for {row['clip_id']}: {latent}")
    return rows, cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--init-checkpoint", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output-dir", type=Path, help="Overrides config output_dir.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--launch-training", action="store_true", help="Explicitly run upstream training.")
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    config = args.config.resolve()
    rows, cfg = validate(manifest, config)
    print(json.dumps({
        "validated": True,
        "training_started": False,
        "records": len(rows),
        "train": sum(row["split"] == "train" for row in rows),
        "validation": sum(row["split"] == "validation" for row in rows),
        "source_disjoint": True,
        "config": str(config),
        "manifest": str(manifest),
        "output_dir": str(args.output_dir or (ROOT / cfg["train"]["output_dir"])),
    }, ensure_ascii=False, indent=2))
    if not args.launch_training:
        return
    if not args.init_checkpoint.is_file():
        raise FileNotFoundError(f"Missing cached model checkpoint: {args.init_checkpoint}")

    vendor = ROOT / "vendor/irodori-tts"
    sys.path.insert(0, str(vendor))
    import torch
    import importlib.util

    spec = importlib.util.spec_from_file_location("irodori_upstream_train", UPSTREAM)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load pinned upstream trainer: {UPSTREAM}")
    upstream = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = upstream
    spec.loader.exec_module(upstream)
    train_ids = [index for index, row in enumerate(rows) if row["split"] == "train"]
    valid_ids = [index for index, row in enumerate(rows) if row["split"] == "validation"]

    def explicit_split(*, num_samples: int, valid_ratio: float, seed: int):
        del seed
        if num_samples != len(rows) or valid_ratio <= 0:
            raise ValueError("Pinned-split wrapper requires the complete manifest and validation enabled")
        return torch.tensor(train_ids, dtype=torch.int64), torch.tensor(valid_ids, dtype=torch.int64)

    upstream.split_train_valid_indices = explicit_split
    output_dir = args.output_dir or (ROOT / cfg["train"]["output_dir"])
    sys.argv = [
        str(UPSTREAM), "--config", str(config), "--manifest", str(manifest),
        "--output-dir", str(output_dir), "--init-checkpoint", str(args.init_checkpoint.resolve()),
        "--device", args.device,
    ]
    upstream.main()


if __name__ == "__main__":
    main()
