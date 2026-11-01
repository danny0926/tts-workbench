"""Pinned, CPU-only local launcher for the official Irodori-TTS inference CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "vendor" / "irodori-tts"
HF_REPO = "Aratako/Irodori-TTS-v4.1-Small"
HF_REVISION = "2b28324dc263ed5e6638b3cf3dd94c82ead07b4b"
CODEC_REPO = "Aratako/Semantic-DACVAE-Japanese-32dim"
CODEC_REVISION = "47376ee24834d7a05a48ebabfe3cde29b3c5e214"
WATERMARK_REPO = "sony/silentcipher"
HF_HOME = ROOT / ".cache_hf"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    text_source = parser.add_mutually_exclusive_group(required=True)
    text_source.add_argument("--text")
    text_source.add_argument("--text-file", type=Path,
                             help="Read UTF-8 synthesis text from a file (safe for Windows shells).")
    parser.add_argument("--ref-wav", action="append", required=True,
                        help="Short same-speaker reference WAV; repeat to concatenate clips.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Replace the WAV and metadata JSON when they already exist.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-steps", type=int, default=40)
    args = parser.parse_args()
    text = args.text if args.text is not None else args.text_file.expanduser().read_text(encoding="utf-8").strip()
    if not text:
        parser.error("Text must not be empty.")

    refs = [Path(p).expanduser().resolve() for p in args.ref_wav]
    missing = [str(p) for p in refs if not p.is_file()]
    if missing:
        parser.error("Missing reference WAV(s): " + ", ".join(missing))
    if not (ROOT / ".venv-irodori-cpu" / "Scripts" / "python.exe").is_file():
        parser.error("Create .venv-irodori-cpu first (see runtime/irodori-cpu.txt).")

    os.environ["HF_HOME"] = str(HF_HOME)
    os.environ["HF_HUB_CACHE"] = str(HF_HOME / "hub")
    os.environ["TORCH_NUM_THREADS"] = "4"
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = output.with_suffix(".json")
    if not args.overwrite and (output.exists() or metadata_path.exists()):
        parser.error(f"Refusing to overwrite existing output/metadata: {output} / {metadata_path}")
    if args.overwrite:
        output.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)

    downloader = (
        "import json; from huggingface_hub import hf_hub_download, snapshot_download; "
        f"model=snapshot_download(repo_id={HF_REPO!r}, revision={HF_REVISION!r}, "
        "allow_patterns=['model.safetensors', 'tokenizer/*']); "
        f"codec=hf_hub_download(repo_id={CODEC_REPO!r}, filename='weights.pth', "
        f"revision={CODEC_REVISION!r}); "
        f"watermark=snapshot_download(repo_id={WATERMARK_REPO!r}, revision='main'); "
        "print(json.dumps({'model_snapshot': model, 'codec_checkpoint': codec, "
        "'watermark_snapshot': watermark}))"
    )
    python = ROOT / ".venv-irodori-cpu" / "Scripts" / "python.exe"
    download_paths = json.loads(subprocess.check_output(
        [str(python), "-c", downloader], cwd=ROOT, env=os.environ.copy(), text=True
    ).strip().splitlines()[-1])
    snapshot = Path(download_paths["model_snapshot"])
    checkpoint = snapshot / "model.safetensors"
    codec_checkpoint = Path(download_paths["codec_checkpoint"])
    watermark_snapshot = Path(download_paths["watermark_snapshot"])
    if not checkpoint.is_file() or not (snapshot / "tokenizer" / "tokenizer.json").is_file():
        raise RuntimeError(f"Pinned model snapshot is missing checkpoint/tokenizer assets: {snapshot}")
    if not codec_checkpoint.is_file():
        raise RuntimeError(f"Pinned codec checkpoint is missing: {codec_checkpoint}")
    if not (watermark_snapshot / "44_1_khz" / "73999_iteration" / "hparams.yaml").is_file():
        raise RuntimeError(f"SilentCipher 44.1 kHz config is missing: {watermark_snapshot}")

    command = [
        str(python), str(REPO / "infer.py"), "--checkpoint", str(checkpoint),
        "--text", text, "--output-wav", str(output),
        "--model-device", "cpu", "--codec-device", "cpu",
        "--codec-repo", str(codec_checkpoint),
        "--no-compile-model", "--num-steps", str(args.num_steps),
        "--seed", str(args.seed), "--ref-wavs", *map(str, refs),
    ]
    started = time.perf_counter()
    print("Running official Irodori inference on CPU; model revision:", HF_REVISION, flush=True)
    process = subprocess.Popen(
        command, cwd=REPO, env={**os.environ.copy(), "HF_HUB_OFFLINE": "1"},
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
        errors="replace", bufsize=1,
    )
    run_lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        run_lines.append(line)
        print(line, end="", flush=True)
    completed_code = process.wait()
    elapsed = time.perf_counter() - started
    if completed_code:
        return completed_code
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"Inference reported success but output is missing/empty: {output}")

    import soundfile as sf
    import numpy as np

    info = sf.info(str(output))
    audio, sample_rate = sf.read(str(output), dtype="float32", always_2d=True)
    finite = bool(np.isfinite(audio).all())
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) if audio.size else 0.0
    if not finite or sample_rate <= 0 or len(audio) == 0 or peak <= 1e-7 or rms <= 1e-8:
        raise RuntimeError("Generated WAV failed finite, nonzero audio/sample-rate verification.")
    timing_match = re.search(r"\[timing\] total_to_decode: ([0-9.]+) s", "".join(run_lines))
    decode_seconds = float(timing_match.group(1)) if timing_match else None
    stage_timings_ms = {
        name: float(seconds)
        for name, seconds in re.findall(r"\[timing\] ([^:\r\n]+): ([0-9.]+) ms", "".join(run_lines))
    }
    run_log = "".join(run_lines)
    if "SilentCipher package is unavailable" in run_log:
        watermark_status = "package_unavailable; official unwatermarked fallback"
    elif "SilentCipher model could not be loaded" in run_log:
        watermark_status = "model_unavailable; official unwatermarked fallback"
    elif "silentcipher_watermark" in stage_timings_ms:
        watermark_status = "loaded_and_applied"
    else:
        watermark_status = "no load failure warning observed"
    metadata = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "text": text,
        "language": "ja",
        "mode": "voice_clone",
        "model_id": HF_REPO,
        "output_wav": str(output),
        "audio_duration_seconds": float(info.duration),
        "generation_seconds": decode_seconds if decode_seconds is not None else elapsed,
        "generation_seconds_synchronized": decode_seconds,
        "timing_scope": (
            "Official infer.py total_to_decode: after model load and request planning, including "
            "tokenization, reference encoding, synthesis, codec decoding, and SilentCipher "
            "watermarking when available; excludes model/codec loading, HF downloads, and WAV saving."
        ),
        "model_repo": HF_REPO,
        "model_revision": HF_REVISION,
        "model_checkpoint_sha256": sha256_file(checkpoint),
        "codec_repo": CODEC_REPO,
        "codec_revision": CODEC_REVISION,
        "codec_checkpoint_sha256": sha256_file(codec_checkpoint),
        "watermark_repo": WATERMARK_REPO,
        "watermark_revision": watermark_snapshot.name,
        "watermark_status": watermark_status,
        "official_code_commit": subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip(),
        "device": "cpu",
        "reference_wavs": [
            {"path": str(p), "sha256": sha256_file(p), "seconds": sf.info(str(p)).duration}
            for p in refs
        ],
        "seed": args.seed,
        "num_steps": args.num_steps,
        "elapsed_seconds_including_model_load": elapsed,
        "stage_timings_ms": stage_timings_ms,
        "output": {
            "path": str(output), "sha256": sha256_file(output),
            "sample_rate": int(info.samplerate), "channels": int(info.channels),
            "frames": int(info.frames), "seconds": float(info.duration),
            "peak": peak, "rms": rms, "all_samples_finite": finite,
        },
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Verified finite WAV: {output} ({info.duration:.2f}s @ {info.samplerate} Hz)")
    print(f"Metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
