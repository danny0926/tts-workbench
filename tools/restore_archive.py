#!/usr/bin/env python3
"""Preview or restore curated metadata; never restore media or overwrite files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data/archive"


def remap(value, mappings):
    if isinstance(value, dict):
        return {key: remap(item, mappings) for key, item in value.items()}
    if isinstance(value, list):
        return [remap(item, mappings) for item in value]
    if not isinstance(value, str):
        return value
    was_path = value.startswith("${TTS_")
    for token, destination in mappings.items():
        value = value.replace("${" + token + "}", destination.as_posix())
    if "${TTS_" in value:
        raise ValueError("Unmapped archive path placeholder: " + value)
    return value.replace("\\", "/") if was_path else value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Write missing metadata files after a full conflict check")
    parser.add_argument("--source-root", type=Path, default=ROOT / "audio/source")
    parser.add_argument("--lab-root", type=Path, default=ROOT / "lab-gpu-v4")
    parser.add_argument("--prep-root", type=Path, default=ROOT / "lab-cpu-v4")
    args = parser.parse_args()
    mappings = {"TTS_REPO_ROOT": ROOT, "TTS_SOURCE_ROOT": args.source_root.resolve(),
                "TTS_LAB_ROOT": args.lab_root.resolve(), "TTS_PREP_ROOT": args.prep_root.resolve(),
                "TTS_PRIVATE_INPUT_ROOT": ROOT / "inputs/reviews", "TTS_EXTERNAL_ROOT": ROOT / "inputs/external"}
    planned = []
    for source in sorted(ARCHIVE.rglob("*")):
        if not source.is_file():
            continue
        relative = source.relative_to(ARCHIVE)
        if relative.parts[0] not in {"outputs", "lab-gpu-v4"}:
            raise ValueError(f"Unexpected archive destination: {relative}")
        target = (ROOT / relative).resolve()
        if not target.is_relative_to(ROOT):
            raise ValueError(f"Unsafe archive destination: {target}")
        raw = source.read_text(encoding="utf-8-sig")
        if source.suffix == ".json":
            content = json.dumps(remap(json.loads(raw), mappings), ensure_ascii=False, indent=2) + "\n"
        elif source.suffix == ".jsonl":
            content = "".join(json.dumps(remap(json.loads(line), mappings), ensure_ascii=False, separators=(",", ":")) + "\n"
                              for line in raw.splitlines() if line.strip())
        else:
            content = remap(raw, mappings)
        if target.exists() and target.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"Refusing to overwrite different local metadata: {target}")
        planned.append((target, content))
    if args.apply:
        for target, content in planned:
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("x", encoding="utf-8", newline="\n") as stream:
                    stream.write(content)
    print(json.dumps({"applied": args.apply, "metadata_files": len(planned), "media_files_restored": 0,
                      "source_root": str(args.source_root.resolve()),
                      "note": "Original byte hashes are provenance only. Media, weights and latent caches must be supplied separately."},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
