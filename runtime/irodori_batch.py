"""Synthesize selected UTF-8 benchmark prompts sequentially with one Irodori runtime."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "vendor" / "irodori-tts"
MODEL_ID = "Aratako/Irodori-TTS-v4.1-Small"
MODEL_REVISION = "2b28324dc263ed5e6638b3cf3dd94c82ead07b4b"
CODEC_ID = "Aratako/Semantic-DACVAE-Japanese-32dim"
CODEC_REVISION = "47376ee24834d7a05a48ebabfe3cde29b3c5e214"
WATERMARK_ID = "sony/silentcipher"
WATERMARK_REVISION = "a1c4d021905e0dc5b24be5f68db5fc4dba410ee1"
TARGET_IDS = ("ja-mixed-01", "ja-context-01", "ja-dates-01")
TIMING_SCOPE = (
    "Official Irodori total_to_decode after model load and request planning; includes "
    "tokenization, reference encoding, synthesis, codec decoding, and SilentCipher watermarking "
    "when available; excludes model/codec load, downloads, and WAV saving."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "benchmarks" / "prompts.jsonl")
    parser.add_argument("--reference", type=Path, default=ROOT / "audio" / "nino" / "irodori_33s_concat.wav")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "nino-irodori")
    parser.add_argument("--id", dest="sample_ids", action="append", help="Benchmark id; repeat to select IDs.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-steps", type=int, default=40)
    args = parser.parse_args()

    manifest = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    reference = args.reference if args.reference.is_absolute() else ROOT / args.reference
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    sample_ids = args.sample_ids or list(TARGET_IDS)
    prompts: dict[str, dict] = {}
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("id") in sample_ids:
            prompts[row["id"]] = row
    missing_ids = [sample_id for sample_id in sample_ids if sample_id not in prompts]
    if missing_ids:
        parser.error(f"Prompt ids missing from {manifest}: {missing_ids}")
    for sample_id in sample_ids:
        if prompts[sample_id].get("language") != "ja" or not str(prompts[sample_id].get("text", "")).strip():
            parser.error(f"Expected a non-empty Japanese prompt for {sample_id} in {manifest}")
    if not reference.is_file():
        parser.error(f"Reference WAV is missing: {reference}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {sample_id: output_dir / f"{sample_id}.wav" for sample_id in sample_ids}
    collisions = [str(p) for p in output_paths.values() if p.exists() or p.with_suffix(".json").exists()]
    if collisions:
        parser.error("Refusing to overwrite existing sample output(s): " + ", ".join(collisions))

    hf_home = ROOT / ".cache_hf"
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_HUB_CACHE"] = str(hf_home / "hub")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["TORCH_NUM_THREADS"] = "4"
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"

    model_snapshot = hf_home / "hub" / "models--Aratako--Irodori-TTS-v4.1-Small" / "snapshots" / MODEL_REVISION
    checkpoint = model_snapshot / "model.safetensors"
    codec_checkpoint = (
        hf_home / "hub" / "models--Aratako--Semantic-DACVAE-Japanese-32dim" / "snapshots"
        / CODEC_REVISION / "weights.pth"
    )
    watermark_snapshot = hf_home / "hub" / "models--sony--silentcipher" / "snapshots" / WATERMARK_REVISION
    cache_files = (
        checkpoint,
        model_snapshot / "tokenizer" / "tokenizer.json",
        model_snapshot / "tokenizer" / "tokenizer_config.json",
        codec_checkpoint,
        watermark_snapshot / "44_1_khz" / "73999_iteration" / "hparams.yaml",
        watermark_snapshot / "44_1_khz" / "73999_iteration" / "enc_c.ckpt",
        watermark_snapshot / "44_1_khz" / "73999_iteration" / "dec_c.ckpt",
        watermark_snapshot / "44_1_khz" / "73999_iteration" / "dec_m_0.ckpt",
    )
    missing_cache = [str(path) for path in cache_files if not path.is_file()]
    if missing_cache:
        parser.error("Pinned local model cache is incomplete; use irodori_local.py once to prefetch: " + ", ".join(missing_cache))

    sys.path.insert(0, str(REPO))
    import numpy as np
    import soundfile as sf
    import torch
    from irodori_tts.inference_runtime import (
        InferenceRuntime,
        RuntimeKey,
        SamplingRequest,
        resolve_cfg_scales,
        save_wav,
    )

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    print(f"Pinned assets ready; CPU threads={torch.get_num_threads()}; model={MODEL_ID}@{MODEL_REVISION}", flush=True)

    # Hash large shared assets once for the batch, not once per sample.
    model_sha256 = sha256_file(checkpoint)
    codec_sha256 = sha256_file(codec_checkpoint)
    reference_info = sf.info(str(reference))
    reference_sha256 = sha256_file(reference)
    shared_reference = [{
        "path": str(reference.resolve()),
        "sha256": reference_sha256,
        "seconds": float(reference_info.duration),
    }]
    code_commit = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()

    runtime_key = RuntimeKey(
        checkpoint=str(checkpoint),
        model_device="cpu",
        codec_repo=str(codec_checkpoint),
        model_precision="fp32",
        codec_device="cpu",
        codec_precision="fp32",
        codec_deterministic_encode=True,
        codec_deterministic_decode=True,
        compile_model=False,
        compile_dynamic=False,
        split_rare_tokens=True,
    )
    load_started = time.perf_counter()
    runtime = InferenceRuntime.from_key(runtime_key)
    model_load_seconds = time.perf_counter() - load_started
    print(f"Irodori model/codec/watermark ready; shared_load_seconds={model_load_seconds:.3f}", flush=True)

    for sample_id in sample_ids:
        row = prompts[sample_id]
        text = str(row["text"])
        caption_raw = row.get("caption")
        caption = None if caption_raw is None or not str(caption_raw).strip() else str(caption_raw)
        requested_scales = {
            "text": float(row.get("cfg_scale_text", 3.0)),
            "caption": float(row.get("cfg_scale_caption", 3.0)),
            "speaker": float(row.get("cfg_scale_speaker", 5.0)),
        }
        use_caption_condition = bool(runtime.model_cfg.use_caption_condition and caption is not None)
        use_speaker_condition = bool(runtime.model_cfg.use_speaker_condition_resolved)
        cfg_text, cfg_caption, cfg_speaker, cfg_messages = resolve_cfg_scales(
            cfg_guidance_mode="independent",
            cfg_scale_text=requested_scales["text"],
            cfg_scale_caption=requested_scales["caption"],
            cfg_scale_speaker=requested_scales["speaker"],
            cfg_scale=None,
            use_caption_condition=use_caption_condition,
            use_speaker_condition=use_speaker_condition,
        )
        for message in cfg_messages:
            print(message, flush=True)
        request = SamplingRequest(
            text=text,
            caption=caption,
            ref_wavs=[str(reference.resolve())],
            num_steps=args.num_steps,
            seed=args.seed,
            cfg_guidance_mode="independent",
            cfg_scale_text=cfg_text,
            cfg_scale_caption=cfg_caption,
            cfg_scale_speaker=cfg_speaker,
        )
        started = time.perf_counter()
        result = runtime.synthesize(request, log_fn=None)
        synthesis_call_seconds = time.perf_counter() - started
        out_path = output_paths[sample_id]
        save_wav(out_path, result.audio, result.sample_rate)
        info = sf.info(str(out_path))
        audio, sample_rate = sf.read(str(out_path), dtype="float32", always_2d=True)
        finite = bool(np.isfinite(audio).all())
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) if audio.size else 0.0
        if not finite or sample_rate <= 0 or not info.frames or peak <= 1e-7 or rms <= 1e-8:
            raise RuntimeError(f"Generated WAV is empty, silent, or non-finite: {out_path}")

        stage_timings_ms = {name: float(seconds * 1000.0) for name, seconds in result.stage_timings}
        if "silentcipher_watermark" in stage_timings_ms:
            watermark_status = "loaded_and_applied"
        elif any("SilentCipher watermark is unavailable" in str(message) for message in result.messages):
            watermark_status = "unavailable; official unwatermarked fallback"
        else:
            watermark_status = "no load failure warning observed"
        metadata = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "sample_id": sample_id,
            "category": row.get("category"),
            "source_prompt_id": row.get("source_prompt_id"),
            "variant_label": row.get("variant_label"),
            "text": text,
            "original_text": row.get("original_text"),
            "comparison_text": row.get("comparison_text"),
            "language": "ja",
            "mode": "voice_clone",
            "conditioning": {
                "caption": caption,
                "speaker_reference_paths": [str(reference.resolve())],
                "use_caption_condition": use_caption_condition,
                "use_speaker_condition": use_speaker_condition,
            },
            "cfg_guidance_mode": "independent",
            "cfg_scales_requested": requested_scales,
            "cfg_scales_resolved": {
                "text": cfg_text,
                "caption": cfg_caption,
                "speaker": cfg_speaker,
            },
            "model_id": MODEL_ID,
            "model_repo": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "model_checkpoint_sha256": model_sha256,
            "official_code_commit": code_commit,
            "device": "cpu",
            "dtype": "fp32",
            "model_precision": "fp32",
            "codec_repo": CODEC_ID,
            "codec_revision": CODEC_REVISION,
            "codec_checkpoint_sha256": codec_sha256,
            "codec_device": "cpu",
            "codec_precision": "fp32",
            "watermark_repo": WATERMARK_ID,
            "watermark_revision": WATERMARK_REVISION,
            "watermark_status": watermark_status,
            "reference_wavs": shared_reference,
            "seed": int(result.used_seed),
            "num_steps": int(args.num_steps),
            "compile_model": False,
            "model_load_seconds_shared": model_load_seconds,
            "generation_seconds": float(result.total_to_decode),
            "generation_seconds_synchronized": float(result.total_to_decode),
            "generation_call_wall_seconds": synthesis_call_seconds,
            "timing_scope": TIMING_SCOPE,
            "stage_timings_ms": stage_timings_ms,
            "audio_duration_seconds": float(info.duration),
            "output_wav": str(out_path.resolve()),
            "output": {
                "path": str(out_path.resolve()),
                "sha256": sha256_file(out_path),
                "sample_rate": int(info.samplerate),
                "channels": int(info.channels),
                "frames": int(info.frames),
                "seconds": float(info.duration),
                "peak": peak,
                "rms": rms,
                "all_samples_finite": finite,
            },
        }
        sidecar = out_path.with_suffix(".json")
        sidecar.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(
            f"Completed {sample_id}: {info.duration:.2f}s, total_to_decode={result.total_to_decode:.3f}s, "
            f"output={out_path}",
            flush=True,
        )
        del result, audio
        gc.collect()

    print("Completed selected Irodori samples; shared model remained loaded across requests.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
