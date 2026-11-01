"""Local Qwen3-TTS Base voice cloning from one private reference recording."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import sys
import time
from importlib.metadata import version
from pathlib import Path


DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
DEFAULT_REVISION = "5d83992436eae1d760afd27aff78a71d676296fc"
LANGUAGES = {
    "Chinese": "Chinese",
    "Chinese (Traditional)": "Chinese",
    "zh": "Chinese",
    "zh-TW": "Chinese",
    "Japanese": "Japanese",
    "ja": "Japanese",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="one target utterance")
    source.add_argument("--prompts", type=Path, help="UTF-8 JSONL with id/text/language fields")
    p.add_argument("--sample-id", default="sample-001", help="output name when --text is used")
    p.add_argument("--ref-audio", required=True, type=Path, help="local WAV reference recording")
    p.add_argument("--ref-text-file", type=Path, help="exact UTF-8 reference transcript; required for ICL cloning")
    p.add_argument("--ref-transcript-provenance", choices=("asr_draft", "audio_reviewed"), default="asr_draft")
    p.add_argument("--x-vector-only", action="store_true", help="use speaker embedding only if no exact transcript exists")
    p.add_argument("--language", choices=tuple(LANGUAGES), help="required for --text; JSONL may set language per row")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/nino-qwen"))
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--revision", default=DEFAULT_REVISION)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def normalize_language(value: str | None) -> str:
    if value not in LANGUAGES:
        raise ValueError(f"Unsupported or missing language: {value!r}; use Chinese or Japanese")
    return LANGUAGES[value]


def load_samples(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.text is not None:
        if not args.language:
            raise ValueError("--language is required with --text")
        samples = [{"id": args.sample_id, "language": normalize_language(args.language), "text": args.text}]
    else:
        samples = []
        with args.prompts.open("r", encoding="utf-8-sig") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {args.prompts}:{line_no}: {exc}") from exc
                if not isinstance(item.get("id"), str) or not item["id"].strip():
                    raise ValueError(f"Missing non-empty id at {args.prompts}:{line_no}")
                if not isinstance(item.get("text"), str) or not item["text"].strip():
                    raise ValueError(f"Missing non-empty text at {args.prompts}:{line_no}")
                language = normalize_language(item.get("language", args.language))
                samples.append({"id": item["id"], "language": language, "text": item["text"]})
    if not samples:
        raise ValueError("No target utterances were found")
    safe_ids = [re.sub(r"[^A-Za-z0-9_.-]+", "_", s["id"]).strip("._") for s in samples]
    if any(not x for x in safe_ids) or len(set(safe_ids)) != len(safe_ids):
        raise ValueError("Prompt ids must produce unique, non-empty safe filenames")
    for sample, safe_id in zip(samples, safe_ids):
        sample["filename_id"] = safe_id
    return samples


def main() -> None:
    args = parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    if not (1 <= args.max_new_tokens <= 2048):
        raise ValueError("--max-new-tokens must be between 1 and 2048")
    workspace = Path(__file__).resolve().parents[1]
    os.environ["HF_HOME"] = str(workspace / ".cache_hf")
    ref_audio = args.ref_audio.resolve(strict=True)
    if ref_audio.suffix.lower() not in {".wav", ".wave"}:
        raise ValueError("Use a staged WAV reference on Windows; the installed loader does not guarantee MP4 decoding")
    samples = load_samples(args)

    ref_text: str | None = None
    if args.ref_text_file:
        ref_text = args.ref_text_file.read_text(encoding="utf-8-sig").strip()
    if not args.x_vector_only and not ref_text:
        raise ValueError("Exact --ref-text-file transcript is required for ICL cloning; use --x-vector-only only when unavailable")
    if args.x_vector_only and ref_text:
        raise ValueError("Do not pass a transcript with --x-vector-only; choose ICL mode with the exact transcript")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        for suffix in (".wav", ".json"):
            target = args.output_dir / f"{sample['filename_id']}{suffix}"
            if target.exists() and not args.overwrite:
                raise FileExistsError(f"Output already exists; pass --overwrite to replace: {target}")

    ref_audio_sha256 = hashlib.sha256(ref_audio.read_bytes()).hexdigest()
    ref_text_sha256 = hashlib.sha256(ref_text.encode("utf-8")).hexdigest() if ref_text is not None else None

    import numpy as np
    import soundfile as sf
    import torch
    import qwen_tts
    from qwen_tts import Qwen3TTSModel

    torch.set_num_threads(4)
    ref_info = sf.info(str(ref_audio))

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
    load_start = time.perf_counter()
    model_path = Path(args.model)
    if model_path.is_dir():
        model_source = str(model_path.resolve())
        snapshot_revision = model_path.name if re.fullmatch(r"[0-9a-f]{40}", model_path.name) else None
    else:
        from huggingface_hub import snapshot_download

        offline = os.environ.get("HF_HUB_OFFLINE", "").lower() in {"1", "true", "yes"}
        model_source = snapshot_download(
            repo_id=args.model,
            revision=args.revision,
            local_files_only=offline,
        )
        snapshot_revision = Path(model_source).name if re.fullmatch(r"[0-9a-f]{40}", Path(model_source).name) else None
    try:
        model = Qwen3TTSModel.from_pretrained(
            model_source,
            device_map=args.device,
            dtype=dtype,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
    except torch.cuda.OutOfMemoryError as exc:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise RuntimeError(
            "CUDA ran out of memory while loading the model. This process released cached CUDA allocations; "
            "retry separately with --device cpu --dtype float32 if a slower CPU run is acceptable."
        ) from exc
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - load_start
    loaded_config = getattr(getattr(model, "model", None), "config", None)
    if loaded_config is None:
        loaded_config = getattr(model, "config", None)
    resolved_revision = getattr(loaded_config, "_commit_hash", None) or snapshot_revision

    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    prompt_start = time.perf_counter()
    clone_prompt = model.create_voice_clone_prompt(
        ref_audio=str(ref_audio),
        ref_text=ref_text,
        x_vector_only_mode=args.x_vector_only,
    )
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
        prompt_peak_allocated = torch.cuda.max_memory_allocated()
        prompt_peak_reserved = torch.cuda.max_memory_reserved()
    else:
        prompt_peak_allocated = prompt_peak_reserved = None
    prompt_prep_seconds = time.perf_counter() - prompt_start

    try:
        qwen_version = qwen_tts.__version__
    except AttributeError:
        qwen_version = version("qwen-tts")
    base_metadata = {
        "mode": "voice_clone_x_vector_only" if args.x_vector_only else "voice_clone_icl",
        "model_id": args.model,
        "requested_revision": args.revision,
        "resolved_revision": resolved_revision,
        "ref_audio_path": str(ref_audio),
        "ref_audio_sha256": ref_audio_sha256,
        "ref_audio_duration_seconds": float(ref_info.duration),
        "ref_audio_sample_rate_hz": int(ref_info.samplerate),
        "ref_audio_channels": int(ref_info.channels),
        "ref_transcript": ref_text,
        "ref_transcript_sha256": ref_text_sha256,
        "ref_transcript_provenance": args.ref_transcript_provenance if ref_text is not None else None,
        "ref_transcript_audio_reviewed": (args.ref_transcript_provenance == "audio_reviewed") if ref_text is not None else None,
        "device": args.device,
        "dtype": str(dtype),
        "seed_base": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "model_load_seconds_cold_or_cached_no_warmup": model_load_seconds,
        "prompt_preparation_seconds": prompt_prep_seconds,
        "prompt_peak_gpu_allocated_bytes": prompt_peak_allocated,
        "prompt_peak_gpu_reserved_bytes": prompt_peak_reserved,
        "warmup_runs": 0,
        "load_time_includes_download_if_cache_was_empty": True,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchaudio": __import__("torchaudio").__version__,
            "qwen-tts": qwen_version,
        },
        "gpu_name": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None,
    }

    for index, sample in enumerate(samples):
        sample_seed = args.seed + index
        torch.manual_seed(sample_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(sample_seed)
            torch.cuda.reset_peak_memory_stats()
        generation_start = time.perf_counter()
        try:
            wavs, sample_rate = model.generate_voice_clone(
                text=sample["text"],
                language=sample["language"],
                voice_clone_prompt=clone_prompt,
                max_new_tokens=args.max_new_tokens,
            )
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError as exc:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise RuntimeError(
                "CUDA ran out of memory during voice cloning. This process released cached CUDA allocations; "
                "retry separately with --device cpu --dtype float32 if a slower CPU run is acceptable."
            ) from exc
        generation_seconds = time.perf_counter() - generation_start
        audio = np.asarray(wavs[0]).squeeze()
        if audio.ndim != 1 or audio.size == 0 or not np.isfinite(audio).all() or not np.any(audio):
            raise RuntimeError(f"Generated waveform is empty, non-finite, silent or non-mono: {audio.shape}")
        sample_rate = int(sample_rate)
        duration = float(audio.size / sample_rate)
        output_wav = args.output_dir / f"{sample['filename_id']}.wav"
        sf.write(str(output_wav), audio, sample_rate, subtype="PCM_16")
        generation_peak_allocated = torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else None
        generation_peak_reserved = torch.cuda.max_memory_reserved() if args.device.startswith("cuda") else None
        metadata = {
            **base_metadata,
            "sample_id": sample["id"],
            "text": sample["text"],
            "language": sample["language"],
            "sample_seed": sample_seed,
            "sample_rate_hz": sample_rate,
            "audio_duration_seconds": duration,
            "audio_peak_abs": float(np.max(np.abs(audio))),
            "audio_rms": float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))),
            "generation_seconds_synchronized": generation_seconds,
            "rtf_generation_seconds_per_audio_second": generation_seconds / duration,
            "generation_peak_gpu_allocated_bytes": generation_peak_allocated,
            "generation_peak_gpu_reserved_bytes": generation_peak_reserved,
            "output_wav": str(output_wav.resolve()),
        }
        output_json = output_wav.with_suffix(".json")
        output_json.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
