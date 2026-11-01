#!/usr/bin/env python3
"""Capture a SHA-256 inventory for a pinned upstream checkout without its .git."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

EXPECTED = "59be938c99d48b5653ca5aa3fdd0cddaa56940e5"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if commit != EXPECTED:
        raise ValueError(f"Expected pinned upstream {EXPECTED}, found {commit}")
    tracked = subprocess.check_output(["git", "-C", str(source), "ls-files", "-z"]).decode().split("\0")
    files = []
    for relative in sorted(item for item in tracked if item):
        path = source / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        files.append({"path": relative.replace("\\", "/"), "size_bytes": path.stat().st_size,
                      "sha256": sha256(path)})
    payload = {"upstream_commit": commit, "source_root": source.name, "file_count": len(files), "files": files}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"commit": commit, "file_count": len(files), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
