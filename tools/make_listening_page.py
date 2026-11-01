#!/usr/bin/env python3
"""Build a self-contained offline listening page for local Nino reference tests."""
from __future__ import annotations

import base64
import html
import json
import mimetypes
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFS = ROOT / "outputs/intake/nino_reference.json"
OUT_DIRS = (ROOT / "outputs/nino-qwen", ROOT / "outputs/nino-irodori")
PROMPTS = ROOT / "benchmarks/prompts.jsonl"
FEEDBACK = ROOT / "outputs/listener-feedback.json"
OUTPUT = ROOT / "outputs/nino-listen.html"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def audio_uri(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "audio/wav"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def audio_duration(path: Path):
    try:
        with wave.open(str(path), "rb") as wav:
            return wav.getnframes() / wav.getframerate()
    except (OSError, wave.Error, ZeroDivisionError):
        return None


def reference_details(meta: dict, base: Path) -> tuple[list[str], float]:
    values = []
    single = meta.get("ref_audio_path") or meta.get("reference_wav") or meta.get("reference_audio_path")
    many = meta.get("reference_wavs") or meta.get("reference_audio_paths")
    if single:
        values.append(single)
    if isinstance(many, list):
        values.extend(many)
    labels = []
    durations = []
    for value in values:
        if isinstance(value, dict):
            raw_path = value.get("path") or value.get("wav_path") or value.get("ref_audio_path")
            fallback_duration = value.get("seconds") or value.get("duration_seconds")
        else:
            raw_path, fallback_duration = value, None
        if not isinstance(raw_path, str) or not raw_path:
            continue
        path = Path(raw_path)
        if not path.is_absolute():
            local_path = base / path
            path = local_path if local_path.is_file() else ROOT / path
        duration = audio_duration(path) if path.is_file() else None
        if duration is None and isinstance(fallback_duration, (int, float)):
            duration = float(fallback_duration)
        name = path.name or raw_path
        labels.append(f"{name} · {duration:.2f} s" if isinstance(duration, (int, float)) else name)
        if isinstance(duration, (int, float)):
            durations.append(float(duration))
    return labels, sum(durations)


def collect_samples():
    samples = []
    mode_labels = {
        "voice_clone_x_vector_only": "聲音特徵（未使用逐字稿）",
        "voice_clone_icl": "聲音＋逐字稿（ASR 草稿）",
        "voice_clone": "參考錄音聲音複製",
    }
    for directory in OUT_DIRS:
        if not directory.exists():
            continue
        for meta_path in sorted(directory.rglob("*.json")):
            try:
                meta = read_json(meta_path)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            if not isinstance(meta, dict) or not isinstance(meta.get("text"), str):
                continue
            wav_value = meta.get("output_wav") or meta.get("output_audio")
            wav_path = Path(wav_value) if isinstance(wav_value, str) and wav_value else meta_path.with_suffix(".wav")
            if not wav_path.is_absolute():
                wav_path = ROOT / wav_path
            if not wav_path.is_file():
                continue
            try:
                rating_key = wav_path.resolve().relative_to(ROOT.resolve()).as_posix()
            except ValueError:
                rating_key = str(wav_path.resolve())
            model_name = meta.get("model_id") or meta.get("model") or directory.name
            timing = meta.get("generation_seconds_synchronized") or meta.get("generation_seconds") or meta.get("total_to_decode")
            timing_scope = meta.get("timing_scope")
            if not timing_scope:
                family = str(model_name) + " " + directory.name
                if "qwen" in family.lower():
                    timing_scope = "Qwen generation time excludes one-time clone-prompt preparation."
                elif "irodori" in family.lower() or "total_to_decode" in meta:
                    timing_scope = "Irodori total includes reference encoding, conditioning preparation, and watermarking."
            device_dtype = [str(meta[key]) for key in ("device", "dtype") if meta.get(key) not in (None, "")]
            ref_labels, ref_seconds = reference_details(meta, meta_path.parent)
            actual_text = meta.get("actual_spoken_text") or meta.get("spoken_text") or meta.get("text")
            samples.append({
                "text": meta["text"],
                "comparison_text": meta.get("comparison_text") or meta.get("group_text") or meta.get("text"),
                "actual_text": actual_text,
                "variant_label": meta.get("variant_label") or meta.get("variant"),
                "caption": meta.get("caption") or meta.get("caption_text") or (meta.get("conditioning") or {}).get("caption"),
                "language": meta.get("language", "Japanese"),
                "model": model_name,
                "mode": mode_labels.get(meta.get("mode") or meta.get("clone_mode"), meta.get("mode") or meta.get("clone_mode") or "模式未提供"),
                "duration": audio_duration(wav_path) or meta.get("audio_duration_seconds") or meta.get("duration_seconds"),
                "generation": timing,
                "steps": meta.get("num_steps") or meta.get("steps") or meta.get("num_inference_steps"),
                "timing_scope": str(timing_scope) if timing_scope else None,
                "device_dtype": " · ".join(device_dtype),
                "reference_labels": ref_labels,
                "reference_seconds": ref_seconds,
                "rating_key": rating_key,
                "wav": wav_path,
                "meta_path": meta_path,
            })
    return samples


def main():
    try:
        feedback = read_json(FEEDBACK)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        feedback = {}
    feedback_target = feedback.get("target_sentence")
    preferred_model = feedback.get("preferred_model_id")
    refs_data = read_json(REFS)
    refs = []
    for ref in refs_data.get("references", []):
        if ref.get("include_in_listening_page") is False:
            continue
        wav_path = Path(ref["wav_path"]) if ref.get("wav_path") else None
        if not wav_path or not wav_path.is_file():
            continue
        refs.append({**ref, "uri": audio_uri(wav_path)})

    planned_prompts = []
    if PROMPTS.is_file():
        for line in PROMPTS.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict) and item.get("language") == "ja" and isinstance(item.get("text"), str):
                planned_prompts.append(item["text"])
    samples = collect_samples()
    if samples:
        prompts = list(dict.fromkeys(sample["comparison_text"] for sample in samples))
    else:
        prompts = list(dict.fromkeys(planned_prompts))
    by_text = {text: sorted((sample for sample in samples if sample["comparison_text"] == text),
                            key=lambda sample: (0 if sample["model"] == preferred_model and text == feedback_target else 1,
                                                -sample["reference_seconds"])) for text in prompts}

    sections = []
    for index, text in enumerate(prompts, 1):
        cards = []
        entries = by_text[text]
        if entries:
            for item in entries:
                duration = f" · {item['duration']:.2f} s" if isinstance(item["duration"], (int, float)) else ""
                generation = f" · 計時 {item['generation']:.2f} s" if isinstance(item["generation"], (int, float)) else ""
                steps = f" · {item['steps']} steps" if isinstance(item["steps"], (int, float)) else ""
                device_dtype = " · " + item["device_dtype"] if item["device_dtype"] else ""
                timing_scope = '<small class="timing-scope">計時範圍：' + html.escape(item["timing_scope"]) + '</small>' if item["timing_scope"] else ""
                ref_info = " · reference: " + ", ".join(item["reference_labels"]) if item["reference_labels"] else " · reference metadata unavailable"
                variant_info = " · " + str(item["variant_label"]) if item["variant_label"] else ""
                actual_details = ""
                if item["actual_text"] != item["comparison_text"] or item["caption"] or item["variant_label"]:
                    actual_details = '<details class="variant"><summary>實際口白／變體資訊</summary>'
                    if item["variant_label"]:
                        actual_details += '<p>變體：' + html.escape(str(item["variant_label"])) + '</p>'
                    if item["actual_text"]:
                        actual_details += '<p>實際口白：' + html.escape(str(item["actual_text"])) + '</p>'
                    if item["caption"]:
                        actual_details += '<p>字幕：' + html.escape(str(item["caption"])) + '</p>'
                    actual_details += '</details>'
                rating_key = html.escape(item["rating_key"], quote=True)
                cards.append(
                    '<article class="sample"><h4>' + html.escape(str(item["model"])) + '</h4><p>' +
                    html.escape(str(item["mode"])) + html.escape(variant_info + duration + generation + steps + device_dtype + ref_info) + timing_scope + '</p><audio controls preload="metadata" src="' +
                    html.escape(audio_uri(item["wav"]), quote=True) + '"></audio>' + actual_details + '<div class="ratings">' +
                    ''.join(f'<label>{label}<select data-rating="{key}" data-key="{rating_key}"><option value="">—</option>' +
                            ''.join(f'<option value="{score}">{score}</option>' for score in range(1, 6)) + '</select></label>'
                            for key, label in (("naturalness", "自然度"), ("reading", "讀音"), ("similarity", "聲線相似度"))) +
                    f'<label>備註<textarea data-note data-key="{rating_key}"></textarea></label></div></article>')
        else:
            cards.append('<p class="pending">樣本尚未產生，等待模型執行。</p>')
        feedback_html = ""
        if text == feedback_target and feedback.get("feedback_text"):
            scope = feedback.get("note", "")
            feedback_html = '<p class="feedback">單一聆聽者對此配對的回饋：「' + html.escape(str(feedback["feedback_text"])) + '」。' + html.escape(str(scope)) + '</p>'
        sections.append('<section class="prompt"><h3>目標句 ' + str(index) + '</h3><blockquote>' +
                        html.escape(text) + '</blockquote>' + feedback_html + '<div class="grid">' + ''.join(cards) + '</div></section>')

    ref_cards = []
    for ref in refs:
        title = html.escape(str(ref.get("reference_id", "Japanese reference")))
        wav_path = Path(ref["wav_path"])
        transcript_path = Path(ref["transcript_path"]) if ref.get("transcript_path") else None
        transcript = transcript_path.read_text(encoding="utf-8-sig") if transcript_path and transcript_path.is_file() else ""
        provenance = html.escape(str(ref.get("transcript_provenance", "")))
        seconds = audio_duration(wav_path)
        ref_cards.append('<article class="reference"><h3>' + title + ' · ' + html.escape(wav_path.name) + (f' · {seconds:.2f} s' if isinstance(seconds, (int, float)) else '') +
                         '</h3><audio controls preload="metadata" src="' + html.escape(ref["uri"], quote=True) + '"></audio><p>' + provenance +
                         '</p><details><summary>參考逐字稿草稿</summary><pre>' + html.escape(transcript) + '</pre></details></article>')

    document = '''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Nino 本機聲音試作</title><style>
body{font:16px/1.55 system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#17212b;background:#f6f8fa}h1,h2{line-height:1.2}.note{padding:1rem;background:#fff5d8;border-radius:8px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:1rem}.reference,.sample,.prompt{background:white;border:1px solid #d8dee4;border-radius:10px;padding:1rem;margin:1rem 0}audio{width:100%}.pending{color:#665500}.ratings{display:grid;grid-template-columns:repeat(3,1fr);gap:.5rem;margin-top:.8rem}.ratings label{font-size:.86rem}.ratings select,.ratings textarea{display:block;width:100%;box-sizing:border-box}.ratings textarea{grid-column:1/-1;min-height:3rem}.ratings label:last-child{grid-column:1/-1}blockquote{margin:0 0 .7rem;padding:.5rem 1rem;border-left:3px solid #789;background:#f8fafc}button{padding:.55rem .8rem;margin-right:.5rem}pre{white-space:pre-wrap}
</style><body><h1>Nino 本機聲音試作</h1><p class="note">此離線頁播放目前已完成的本機聲音試作樣本；尚未產生的目標句不列入比較。本頁不代表已完成完整模型調查。參考逐字稿仍是未經人工聽校的 ASR 草稿；不提供自動品質分數或排名。各輸出音量未經正規化，請調至相近播放音量後再比較。</p>
<h2>參考錄音</h2>''' + ''.join(ref_cards) + '<h2>依目標句比較生成結果</h2>' + ''.join(sections) + '''
<p><button id="save">儲存評分到此瀏覽器</button><button id="export">匯出評分 JSON</button><span id="status"></span></p>
<script>(()=>{const key='nino-local-ratings-v1';function read(){try{return JSON.parse(localStorage.getItem(key)||'{}')}catch{return {}}}function save(){const d={};document.querySelectorAll('[data-rating],[data-note]').forEach(e=>{const value=e.value.trim();if(!value)return;const k=e.dataset.key;(d[k]??={})[e.dataset.rating||'notes']=value});localStorage.setItem(key,JSON.stringify(d));document.getElementById('status').textContent='已儲存於此瀏覽器。'}const d=read();document.querySelectorAll('[data-rating],[data-note]').forEach(e=>{const v=d[e.dataset.key]?.[e.dataset.rating||'notes'];if(v!==undefined)e.value=v});document.getElementById('save').onclick=save;document.getElementById('export').onclick=()=>{save();const blob=new Blob([JSON.stringify(read(),null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='nino-ratings.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}})();</script></body></html>'''
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(document, encoding="utf-8")
    print(f"Wrote {OUTPUT.relative_to(ROOT)} with {len(refs)} embedded reference players, {len(prompts)} target sentences, {len(samples)} generated samples.")


if __name__ == "__main__":
    main()
