#!/usr/bin/env python3
"""Standard-library audio intake and TTS benchmark planning utilities."""
import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import wave
from pathlib import Path


def emit(value):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    print(json.dumps(value, ensure_ascii=False, indent=2))


def error(message):
    emit({"ok": False, "error": message})
    return 2


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def doctor(_args):
    info = {"ok": True, "python": sys.version.split()[0], "python_implementation": platform.python_implementation(),
            "os": platform.platform(), "disk": shutil.disk_usage(Path.cwd())._asdict()}
    try:
        proc = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=8, check=False)
        info["nvidia_smi"] = {"available": True, "returncode": proc.returncode,
                              "output": (proc.stdout or proc.stderr).strip()[:3000]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        info["nvidia_smi"] = {"available": False, "message": str(exc)}
    emit(info)
    return 0


def check_audio(args):
    wav_path, txt_path = Path(args.wav), Path(args.transcript)
    for path, label in ((wav_path, "WAV"), (txt_path, "transcript")):
        if not path.is_file():
            return error(f"{label} file does not exist: {path}")
    try:
        transcript_bytes = txt_path.read_bytes()
        transcript = transcript_bytes.decode("utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        return error(f"Transcript must be readable UTF-8 text: {exc}")
    if not transcript.strip():
        return error("Transcript is empty")
    try:
        with wave.open(str(wav_path), "rb") as wav:
            channels, width, rate, frames, compression = (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(),
                                                            wav.getnframes(), wav.getcomptype())
            if rate <= 0:
                return error("Invalid WAV sample rate: must be greater than zero")
            if frames <= 0:
                return error("WAV contains zero frames")
            if compression != "NONE":
                return error(f"Unsupported WAV compression: {compression}; provide uncompressed PCM WAV")
            if width not in (1, 2, 3, 4):
                return error(f"Unsupported PCM sample width: {width * 8} bits")
            raw = wav.readframes(frames)
            expected_bytes = frames * channels * width
            if len(raw) != expected_bytes:
                return error(f"Truncated WAV frame data: expected {expected_bytes} bytes, found {len(raw)}")
    except (wave.Error, EOFError, OSError) as exc:
        return error(f"Cannot read PCM WAV: {exc}")
    warnings = []
    if rate not in (44100, 48000):
        warnings.append("Sample rate is not preferred 44.1/48 kHz; preserve the original unless conversion is required.")
    if channels not in (1, 2):
        warnings.append("Unusual channel count; confirm this is intentional.")
    if width == 1:
        # PCM WAV 8-bit samples are unsigned, centered at 128.
        vals = (b - 128 for b in raw)
        peak_limit = 128
    else:
        import array
        if sys.byteorder != "little":
            return error("Big-endian host decoding is unsupported for PCM peak analysis")
        vals = array.array({2: "h", 4: "i"}.get(width, "b")) if width != 3 else None
        if width == 3:
            vals = (int.from_bytes(raw[i:i+3], "little", signed=True) for i in range(0, len(raw), 3))
        else:
            vals.frombytes(raw)
            vals = iter(vals)
        peak_limit = (1 << (width * 8 - 1))
    peak = 0
    clipped = 0
    count = 0
    for sample in vals:
        count += 1
        magnitude = abs(sample)
        peak = max(peak, magnitude)
        if magnitude >= peak_limit - 1:
            clipped += 1
    if peak == 0:
        warnings.append("Audio is all-zero digital silence.")
    if clipped:
        warnings.append(f"Nonzero clipping detected in {clipped}/{count} PCM samples.")
    result = {"ok": True, "file": str(wav_path.resolve()), "language": args.language, "speaker": args.speaker,
              "duration_seconds": round(frames / rate, 6), "sample_rate_hz": rate, "channels": channels,
              "bit_depth": width * 8, "encoding": "PCM", "pcm_peak": peak,
              "pcm_peak_normalized": round(peak / peak_limit, 8), "clipping_fraction": round(clipped / count, 8) if count else 0,
              "audio_sha256": sha256(wav_path), "transcript_sha256": hashlib.sha256(transcript_bytes).hexdigest(),
              "transcript_characters": len(transcript.strip()), "warnings": warnings}
    emit(result)
    return 0


def load_manifest(manifest_path):
    if not manifest_path.is_file():
        return [], [f"Manifest does not exist: {manifest_path}"]
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return [], [f"Cannot read JSON manifest: {exc}"]
    if not isinstance(data, dict) or not isinstance(data.get("clips"), list):
        return [], ["Manifest schema: expected an object containing a 'clips' array"]
    jobs = []
    errors = []
    allowed = {"zh-TW": "Chinese", "ja": "Japanese"}
    for index, clip in enumerate(data["clips"]):
        if not isinstance(clip, dict):
            errors.append(f"clips[{index}] must be an object")
            continue
        fields = ("wav", "transcript", "language", "speaker")
        missing = [key for key in fields if key not in clip or clip[key] is None or clip[key] == ""]
        if missing:
            errors.append(f"clips[{index}] missing required fields: {', '.join(missing)}")
            continue
        invalid = [key for key in ("wav", "transcript", "language", "speaker") if not isinstance(clip[key], str)]
        if invalid:
            errors.append(f"clips[{index}] fields must be strings: {', '.join(invalid)}")
            continue
        if clip["language"] not in allowed:
            errors.append(f"clips[{index}].language must be zh-TW or ja")
            continue
        base = manifest_path.parent
        wav_path = Path(clip["wav"])
        txt_path = Path(clip["transcript"])
        if not wav_path.is_absolute(): wav_path = base / wav_path
        if not txt_path.is_absolute(): txt_path = base / txt_path
        if not wav_path.is_file() or not txt_path.is_file():
            errors.append(f"clips[{index}] missing file(s): " + ", ".join(str(p) for p in (wav_path, txt_path) if not p.is_file()))
            continue
        jobs.append({"job_id": f"{clip['language']}-{clip['speaker']}-{index + 1:03d}", "language": clip["language"],
                     "language_name": allowed[clip["language"]], "speaker": clip["speaker"],
                     "wav": str(wav_path.resolve()), "transcript": str(txt_path.resolve()), "status": "planned"})
    return jobs, errors


def plan(args):
    jobs, errors = load_manifest(Path(args.input_manifest))
    if errors:
        emit({"ok": False, "errors": errors, "planned_jobs": jobs})
        return 2
    if not jobs:
        return error("Manifest contains no clips")
    emit({"ok": True, "inference_performed": False, "planned_jobs": jobs})
    return 0


def matrix(args):
    clips, errors = load_manifest(Path(args.input_manifest))
    if errors:
        emit({"ok": False, "errors": errors, "planned_jobs": []})
        return 2
    models_path, prompts_path = Path(args.models), Path(args.prompts)
    try:
        models_data = json.loads(models_path.read_text(encoding="utf-8-sig"))
        prompt_rows = [json.loads(line) for line in prompts_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return error(f"Cannot read models or prompts JSON: {exc}")
    models = models_data.get("models") if isinstance(models_data, dict) else None
    if not isinstance(models, list):
        return error("Models schema: expected an object containing a 'models' array")
    if not isinstance(prompt_rows, list) or any(not isinstance(p, dict) or not all(isinstance(p.get(k), str) and p[k] for k in ("id", "language", "text")) for p in prompt_rows):
        return error("Prompts schema: each JSONL row must have nonempty string id, language, and text")
    lang_names = {"zh-TW": "Chinese", "ja": "Japanese"}
    grouped = {}
    for clip in clips:
        key = (clip["language"], clip["speaker"])
        grouped.setdefault(key, []).append({"wav": clip["wav"], "transcript": clip["transcript"]})
    jobs = []
    model_errors = []
    job_num = 0
    for model in models:
        if not isinstance(model, dict) or not all(isinstance(model.get(k), str) and model[k] for k in ("id", "hf_id", "mode")):
            model_errors.append("Each model must contain string id, hf_id, and mode")
            continue
        if model["mode"] == "local_smoke":
            continue
        languages = model.get("languages")
        if not isinstance(languages, list) or any(not isinstance(value, str) for value in languages):
            model_errors.append(f"Model {model['id']} must have a string-array languages field")
            continue
        for (language, speaker), reference_pool in grouped.items():
            language_name = lang_names[language]
            if language_name not in languages:
                continue
            for prompt in prompt_rows:
                if prompt["language"] != language:
                    continue
                job_num += 1
                jobs.append({"job_id": f"matrix-{job_num:05d}", "model_id": model["id"], "hf_id": model["hf_id"],
                             "checkpoint_sha": model.get("checkpoint_sha"), "prompt_id": prompt["id"], "prompt_text": prompt["text"],
                             "speaker": speaker, "language": language, "reference_pool": reference_pool, "status": "planned"})
    if model_errors:
        emit({"ok": False, "errors": model_errors, "planned_jobs": jobs})
        return 2
    emit({"ok": True, "inference_performed": False, "mode": "planned_matrix", "seed": args.seed,
          "job_count": len(jobs), "planned_jobs": jobs})
    return 0


def main():
    parser = argparse.ArgumentParser(description="Prepare local audio and plan TTS benchmark jobs; performs no inference.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Show runtime, disk, and read-only nvidia-smi metadata")
    check = sub.add_parser("check-audio", help="Inspect PCM WAV and UTF-8 transcript")
    check.add_argument("wav"); check.add_argument("--transcript", required=True); check.add_argument("--language", choices=("zh-TW", "ja"), required=True); check.add_argument("--speaker", required=True)
    planning = sub.add_parser("plan", help="Validate input manifest and list planned benchmark jobs")
    planning.add_argument("--input-manifest", required=True)
    matrix_parser = sub.add_parser("matrix", help="Plan compatible reference-audio model and prompt combinations")
    matrix_parser.add_argument("--input-manifest", required=True)
    matrix_parser.add_argument("--models", default="config/models.json")
    matrix_parser.add_argument("--prompts", default="benchmarks/prompts.jsonl")
    matrix_parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    return {"doctor": doctor, "check-audio": check_audio, "plan": plan, "matrix": matrix}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
