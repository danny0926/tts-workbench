#!/usr/bin/env python3
"""Memory-guarded, one-shot GPU baseline launcher for pinned Irodori-TTS v4 Small."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "vendor" / "irodori-tts"
VENV = ROOT / ".venv-irodori-gpu"
PYTHON = VENV / "Scripts" / "python.exe"
MODEL_REPO = "Aratako/Irodori-TTS-v4.1-Small"
MODEL_REVISION = "2b28324dc263ed5e6638b3cf3dd94c82ead07b4b"
MODEL_PATH = ROOT / ".cache_hf" / "hub" / "models--Aratako--Irodori-TTS-v4.1-Small" / "snapshots" / MODEL_REVISION / "model.safetensors"
MODEL_SHA256 = "c85de88c01700cb53538e706f128ebcb1b8513ad21d7d0e75f58bc82cdbf89f6"
CODEC_REPO = "Aratako/Semantic-DACVAE-Japanese-32dim"
CODEC_REVISION = "47376ee24834d7a05a48ebabfe3cde29b3c5e214"
CODEC_PATH = ROOT / ".cache_hf" / "hub" / "models--Aratako--Semantic-DACVAE-Japanese-32dim" / "snapshots" / CODEC_REVISION / "weights.pth"
CODEC_SHA256 = "db120339c5ee7eca1912cdf29bc612b947a0808e69c3cebfb4936b45a762c1d5"
TOKENIZER_JSON = MODEL_PATH.parent / "tokenizer" / "tokenizer.json"
TOKENIZER_CONFIG = MODEL_PATH.parent / "tokenizer" / "tokenizer_config.json"
WATERMARK_HPARAMS = ROOT / ".cache_hf" / "hub" / "models--sony--silentcipher" / "snapshots" / "a1c4d021905e0dc5b24be5f68db5fc4dba410ee1" / "44_1_khz" / "73999_iteration" / "hparams.yaml"
REFERENCE_WAV = ROOT / "audio" / "nino" / "irodori_33s_concat.wav"
REFERENCE_SHA256 = "e3f0a658c3bde4ce29d47090f86d86705f1d003eef0a6d1686f41c1285373d1d"
TEXT_FILE = ROOT / "runtime" / "irodori_gpu_neutral_prompt.txt"
OUTPUT_DIR = ROOT / "outputs" / "nino-irodori-gpu"
CODE_COMMIT = "59be938c99d48b5653ca5aa3fdd0cddaa56940e5"

# Cold-start guard only. The official loader may hold the checkpoint in host
# memory before conversion; these are not steady-state hardware requirements.
MIN_FREE_PHYSICAL = 8 * 1024**3
MIN_FREE_COMMIT = 12 * 1024**3
MIN_FREE_GPU = int(3.2 * 1024**3)
TIMING_SCOPE = (
    "Upstream infer.py total_to_decode, after model/codec loading and request planning; "
    "includes text tokenization, reference encoding when a WAV is used, synthesis, codec "
    "decoding, and SilentCipher watermarking when available; excludes model/codec load and WAV save."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_lora_adapter(path: Path | None) -> dict | None:
    """Validate adapter files without opening or hashing its weight payload."""
    if path is None:
        return None
    adapter_dir = path.expanduser().resolve()
    if not adapter_dir.is_dir():
        raise ValueError(f"LoRA adapter directory does not exist: {adapter_dir}")
    config_path = adapter_dir / "adapter_config.json"
    if not config_path.is_file():
        raise ValueError(f"LoRA adapter is missing adapter_config.json: {adapter_dir}")
    if list(adapter_dir.glob("adapter_model*part*")):
        raise ValueError(f"LoRA adapter contains an incomplete .part weight file: {adapter_dir}")
    weights = [adapter_dir / name for name in ("adapter_model.safetensors", "adapter_model.bin")]
    present = [candidate for candidate in weights if candidate.is_file() and candidate.stat().st_size > 0]
    if len(present) != 1:
        raise ValueError(
            "LoRA adapter must contain exactly one non-empty adapter_model.safetensors "
            f"or adapter_model.bin: {adapter_dir}"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    provenance_keys = ("base_model_name_or_path", "revision", "peft_type", "task_type", "r", "lora_alpha", "target_modules")
    provenance = {key: config[key] for key in provenance_keys if key in config}
    return {
        "directory": str(adapter_dir),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "weights_path": str(present[0]),
        "weights_size_bytes": present[0].stat().st_size,
        "provenance": provenance,
    }


def windows_memory_status() -> dict[str, int]:
    if os.name != "nt":
        raise RuntimeError("This launcher memory guard currently supports Windows only.")

    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ctypes.WinError(ctypes.get_last_error())
    return {
        "physical_total_bytes": int(status.ullTotalPhys),
        "physical_available_bytes": int(status.ullAvailPhys),
        "commit_limit_bytes": int(status.ullTotalPageFile),
        "commit_available_bytes": int(status.ullAvailPageFile),
    }


def preflight(args: argparse.Namespace) -> dict:
    report: dict = {
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "model_loaded": False,
        "cuda_tensor_allocated": False,
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "model_sha256_expected": MODEL_SHA256,
        "codec_repo": CODEC_REPO,
        "codec_revision": CODEC_REVISION,
        "codec_sha256_expected": CODEC_SHA256,
        "official_code_commit_expected": CODE_COMMIT,
        "venv_python_expected": str(PYTHON),
        "planned_request": {
            "model_device": "cuda",
            "model_precision": args.model_precision,
            "codec_device": args.codec_device,
            "codec_precision": args.codec_precision,
            "sampling_steps": 40,
            "seed": 1234,
            "context_kv_cache": True,
            "sdpa_cudnn_priority_enabled": args.sdpa_cudnn == "1",
            "watermark": "normal upstream behavior enabled",
            "reference": (
                {"kind": "latent_cache", "path": str(args.reference_latent.expanduser().resolve())}
                if args.reference_latent is not None
                else {"kind": "wav", "path": str(REFERENCE_WAV), "duration_seconds": 33.05, "sha256_expected": REFERENCE_SHA256}
            ),
            "text_file": str(TEXT_FILE),
        },
        "minimum_headroom": {
            "physical_available_bytes": MIN_FREE_PHYSICAL,
            "commit_available_bytes": MIN_FREE_COMMIT,
            "gpu_free_bytes": MIN_FREE_GPU,
            "scope": "Conservative cold-start loader budget; not a product or steady-state minimum.",
        },
    }
    reasons: list[str] = []
    try:
        report["lora_adapter"] = inspect_lora_adapter(args.lora_adapter)
    except Exception as exc:
        reasons.append(f"Invalid LoRA adapter: {type(exc).__name__}: {exc}")

    try:
        report["memory"] = windows_memory_status()
    except Exception as exc:
        reasons.append(f"Windows memory query failed: {type(exc).__name__}: {exc}")

    try:
        import torch
        import torchaudio

        report["torch"] = torch.__version__
        report["torchaudio"] = torchaudio.__version__
        report["torch_cuda_build"] = torch.version.cuda
        if not torch.__version__.startswith("2.10.0+cu128") or torch.version.cuda != "12.8":
            reasons.append("The private venv does not contain the expected PyTorch 2.10 cu128 build.")
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            free_bytes, total_bytes = torch.cuda.mem_get_info(0)
            capability = (int(props.major), int(props.minor))
            report["gpu"] = {
                "name": props.name,
                "compute_capability": list(capability),
                "native_bf16_capability": capability[0] >= 8,
                "free_bytes": int(free_bytes),
                "total_bytes": int(total_bytes),
            }
            if "3050 Ti" not in props.name:
                reasons.append(f"Expected RTX 3050 Ti; found {props.name}.")
        else:
            report["gpu"] = {"available": False}
            reasons.append("CUDA is not available in the private venv.")
    except Exception as exc:
        reasons.append(f"CUDA environment query failed: {type(exc).__name__}: {exc}")

    try:
        prefix = Path(sys.prefix).resolve()
        expected_prefix = VENV.resolve()
        report["python"] = {
            "executable": sys.executable,
            "version": sys.version.split()[0],
            "prefix": str(prefix),
            "base_prefix": sys.base_prefix,
            "is_private_venv": prefix == expected_prefix and sys.prefix != sys.base_prefix,
        }
        if prefix != expected_prefix or sys.prefix == sys.base_prefix:
            reasons.append("Run this script with .venv-irodori-gpu\\Scripts\\python.exe.")
        if sys.version_info[:2] != (3, 11):
            reasons.append("The private environment must use Python 3.11.")
    except Exception as exc:
        reasons.append(f"Venv verification failed: {type(exc).__name__}: {exc}")

    expected_files = {
        "model_checkpoint": MODEL_PATH,
        "tokenizer": TOKENIZER_JSON,
        "tokenizer_config": TOKENIZER_CONFIG,
        "codec_checkpoint": CODEC_PATH,
        "watermark_config": WATERMARK_HPARAMS,
        "reference_wav": REFERENCE_WAV,
        "request_text": TEXT_FILE,
        "upstream_infer": REPO / "infer.py",
    }
    report["assets"] = {name: {"path": str(path), "exists": path.is_file()} for name, path in expected_files.items()}
    reasons.extend(f"Required local asset is missing: {path}" for path in expected_files.values() if not path.is_file())

    try:
        commit = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
        report["official_code_commit"] = commit
        if commit != CODE_COMMIT:
            reasons.append(f"Upstream code pin mismatch: {commit} != {CODE_COMMIT}.")
    except Exception as exc:
        reasons.append(f"Cannot verify upstream source commit: {type(exc).__name__}: {exc}")

    memory = report.get("memory", {})
    if memory.get("physical_available_bytes", 0) < MIN_FREE_PHYSICAL:
        reasons.append("Insufficient free physical RAM for the conservative cold-start budget (8 GiB required).")
    if memory.get("commit_available_bytes", 0) < MIN_FREE_COMMIT:
        reasons.append("Insufficient free commit for the conservative cold-start budget (12 GiB required).")
    if report.get("gpu", {}).get("free_bytes", 0) < MIN_FREE_GPU:
        reasons.append("Insufficient free GPU memory for the baseline guard (3.2 GiB required).")

    report["reasons"] = reasons
    report["status"] = "ready" if not reasons else "blocked"
    return report


def write_unique_json(directory: Path, prefix: str, value: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    path = directory / f"{prefix}-{stamp}.json"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight", action="store_true", help="Query headroom and write a report; never loads the model.")
    action.add_argument("--run", action="store_true", help="Run one normal one-shot baseline only when the guard is clear.")
    parser.add_argument("--model-precision", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--codec-device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--codec-precision", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--reference-latent", type=Path, help="Optional existing Irodori latent cache; defaults to the pinned 33.05s WAV.")
    parser.add_argument("--lora-adapter", type=Path, help="Optional complete PEFT LoRA adapter directory for this run.")
    parser.add_argument("--sdpa-cudnn", choices=["0", "1"], default="0", help="Upstream SDPA cuDNN priority; desktop baseline defaults to 0.")
    args = parser.parse_args()

    if not PYTHON.is_file():
        parser.error(f"Private GPU venv is missing: {PYTHON}")
    if Path(sys.prefix).resolve() != VENV.resolve() or sys.prefix == sys.base_prefix:
        parser.error(f"Invoke with the private interpreter: {PYTHON}")

    report = preflight(args)
    report_path = write_unique_json(OUTPUT_DIR, "preflight", report)
    print(json.dumps({"status": report["status"], "report": str(report_path), "reasons": report["reasons"]}, ensure_ascii=False))
    if report["status"] != "ready":
        return 2
    if args.preflight:
        return 0

    text = TEXT_FILE.read_text(encoding="utf-8").strip()
    if not text:
        raise RuntimeError(f"Request text is empty: {TEXT_FILE}")
    if args.reference_latent is not None:
        reference = args.reference_latent.expanduser().resolve()
        if not reference.is_file():
            raise FileNotFoundError(reference)
    else:
        reference = REFERENCE_WAV
        if sha256_file(reference) != REFERENCE_SHA256:
            raise RuntimeError("Pinned reference WAV hash mismatch.")
    if MODEL_PATH.stat().st_size != 3_064_295_596 or sha256_file(MODEL_PATH) != MODEL_SHA256:
        raise RuntimeError("Pinned model checkpoint size/hash mismatch; refusing inference.")
    if sha256_file(CODEC_PATH) != CODEC_SHA256:
        raise RuntimeError("Pinned codec checkpoint hash mismatch; refusing inference.")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = OUTPUT_DIR / f"gpu-bf16-baseline-{stamp}.wav"
    metadata_path = output.with_suffix(".json")
    if output.exists() or metadata_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output} or {metadata_path}")

    os.environ["HF_HOME"] = str(ROOT / ".cache_hf")
    os.environ["HF_HUB_CACHE"] = str(ROOT / ".cache_hf" / "hub")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["IRODORI_SDPA_CUDNN"] = args.sdpa_cudnn

    command = [
        str(PYTHON), str(REPO / "infer.py"), "--checkpoint", str(MODEL_PATH),
        "--text", text, "--output-wav", str(output),
        "--model-device", "cuda", "--model-precision", args.model_precision,
        "--codec-device", args.codec_device, "--codec-precision", args.codec_precision,
        "--codec-repo", str(CODEC_PATH), "--no-compile-model", "--num-steps", "40", "--seed", "1234",
    ]
    if args.lora_adapter is not None:
        adapter = report["lora_adapter"]
        command += ["--lora-adapter", adapter["directory"]]
    command += ["--ref-latent", str(reference)] if args.reference_latent is not None else ["--ref-wavs", str(reference)]

    started = time.perf_counter()
    process = subprocess.Popen(
        command, cwd=REPO, env=os.environ.copy(), stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        lines.append(line)
        print(line, end="", flush=True)
    return_code = process.wait()
    elapsed = time.perf_counter() - started
    if return_code:
        return return_code
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"Inference exited successfully but output is missing: {output}")

    import numpy as np
    import soundfile as sf

    info = sf.info(str(output))
    audio, sample_rate = sf.read(str(output), dtype="float32", always_2d=True)
    finite = bool(np.isfinite(audio).all())
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) if audio.size else 0.0
    if not finite or not len(audio) or peak <= 1e-7 or rms <= 1e-8:
        raise RuntimeError("Generated WAV failed the finite/nonzero audio check.")

    log = "".join(lines)
    total_match = re.search(r"\[timing\] total_to_decode: ([0-9.]+) s", log)
    decode_seconds = float(total_match.group(1)) if total_match else None
    stage_timings_ms = {name: float(value) for name, value in re.findall(r"\[timing\] ([^:\r\n]+): ([0-9.]+) ms", log)}
    if "SilentCipher package is unavailable" in log:
        watermark_status = "package_unavailable; official fallback"
    elif "SilentCipher model could not be loaded" in log:
        watermark_status = "model_unavailable; official fallback"
    elif "silentcipher_watermark" in stage_timings_ms:
        watermark_status = "loaded_and_applied"
    else:
        watermark_status = "no load failure warning observed"

    metadata = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "text": text,
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "model_checkpoint_sha256": MODEL_SHA256,
        "lora_adapter": report.get("lora_adapter"),
        "official_code_commit": CODE_COMMIT,
        "codec_repo": CODEC_REPO,
        "codec_revision": CODEC_REVISION,
        "codec_checkpoint_sha256": CODEC_SHA256,
        "device": "cuda",
        "model_precision": args.model_precision,
        "codec_device": args.codec_device,
        "codec_precision": args.codec_precision,
        "sdpa_cudnn_priority_enabled": args.sdpa_cudnn == "1",
        "context_kv_cache": True,
        "watermark_status": watermark_status,
        "reference": {"kind": "latent_cache" if args.reference_latent else "wav", "path": str(reference), "sha256": sha256_file(reference)},
        "seed": 1234,
        "num_steps": 40,
        "elapsed_seconds_including_inference_process": elapsed,
        "generation_seconds_synchronized": decode_seconds,
        "audio_duration_seconds": float(info.duration),
        "rtf": decode_seconds / float(info.duration) if decode_seconds is not None and info.duration > 0 else None,
        "timing_scope": TIMING_SCOPE,
        "stage_timings_ms": stage_timings_ms,
        "ttfa_measured": False,
        "output": {
            "path": str(output), "sha256": sha256_file(output), "sample_rate": int(sample_rate),
            "channels": int(info.channels), "frames": int(info.frames), "peak": peak,
            "rms": rms, "all_samples_finite": finite,
        },
    }
    with metadata_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(f"Verified finite WAV: {output} ({info.duration:.2f}s @ {info.samplerate} Hz)")
    print(f"Metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
