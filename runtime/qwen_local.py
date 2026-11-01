"""Run one local Qwen3-TTS CustomVoice utterance and save WAV + metadata."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path


DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
DEFAULT_REVISION = "85e237c12c027371202489a0ec509ded67b5e4b5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True)
    parser.add_argument("--language", required=True, choices=("Chinese", "Japanese"))
    parser.add_argument("--speaker", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true", help="allow replacing an existing output WAV/JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    # Keep all Hugging Face artifacts in the workspace, including when invoked directly.
    workspace = Path(__file__).resolve().parents[1]
    os.environ["HF_HOME"] = str(workspace / ".cache_hf")

    if args.max_new_tokens < 1 or args.max_new_tokens > 2048:
        raise ValueError("--max-new-tokens must be between 1 and 2048")
    metadata_path = args.output.with_suffix(".json")
    if not args.overwrite and (args.output.exists() or metadata_path.exists()):
        raise FileExistsError(f"Output already exists; pass --overwrite to replace it: {args.output}")

    import numpy as np
    import soundfile as sf
    import torch
    import qwen_tts
    from qwen_tts import Qwen3TTSModel

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")
    if args.dtype == "auto":
        dtype = torch.bfloat16 if args.device.startswith("cuda") and torch.cuda.is_bf16_supported() else (
            torch.float16 if args.device.startswith("cuda") else torch.float32
        )
    else:
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    load_start = time.perf_counter()
    model = Qwen3TTSModel.from_pretrained(
        args.model,
        revision=args.revision,
        device_map=args.device,
        dtype=dtype,
        attn_implementation="sdpa",
    )
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_start

    loaded_config = getattr(getattr(model, "model", None), "config", None)
    if loaded_config is None:
        loaded_config = getattr(model, "config", None)
    resolved_revision = getattr(loaded_config, "_commit_hash", None)

    generation_start = time.perf_counter()
    result = model.generate_custom_voice(
        text=args.text,
        language=args.language,
        speaker=args.speaker,
        max_new_tokens=args.max_new_tokens,
    )
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    generation_seconds = time.perf_counter() - generation_start

    audio, sample_rate = result[0], int(result[1])
    audio = np.asarray(audio).squeeze()
    if audio.ndim != 1 or audio.size == 0:
        raise RuntimeError(f"Expected one non-empty mono waveform; got shape {audio.shape}")
    if not np.isfinite(audio).all() or not np.any(audio):
        raise RuntimeError("Generated waveform is non-finite or silent")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(args.output), audio, sample_rate, subtype="PCM_16")
    audio_duration = float(audio.size / sample_rate)

    try:
        package_version = qwen_tts.__version__
    except AttributeError:
        from importlib.metadata import version
        package_version = version("qwen-tts")
    metadata = {
        "text": args.text,
        "language": args.language,
        "speaker": args.speaker,
        "mode": "preset_custom_voice",
        "model_id": args.model,
        "requested_revision": args.revision,
        "resolved_revision": resolved_revision,
        "device": args.device,
        "dtype": str(dtype),
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "sample_rate_hz": sample_rate,
        "audio_duration_seconds": audio_duration,
        "audio_peak_abs": float(np.max(np.abs(audio))),
        "audio_rms": float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))),
        "load_seconds": load_seconds,
        "load_seconds_includes_model_download_if_cache_was_empty": True,
        "generation_seconds_synchronized": generation_seconds,
        "warmup_runs": 0,
        "rtf_generation_seconds_per_audio_second": generation_seconds / audio_duration,
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else None,
        "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved() if args.device.startswith("cuda") else None,
        "gpu_name": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchaudio": __import__("torchaudio").__version__,
            "qwen-tts": package_version,
        },
        "output_wav": str(args.output.resolve()),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
