#!/usr/bin/env python3
"""Build an offline, one-clip timing comparison from realignment pilot artifacts."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = "nino-05_summer_memories_nino_study_cd-s0012-s0013"
REALIGN = ROOT / "outputs" / "dataset" / "realignment"
DEFAULT_CONTEXT = REALIGN / f"context-{DEFAULT_TARGET}.json"
DEFAULT_ASR = REALIGN / f"context-{DEFAULT_TARGET}-asr-summary.json"
DEFAULT_ALIGNMENT = REALIGN / "forced-alignments-pilot-corrected.jsonl"
DEFAULT_REPAIR = ROOT / "outputs" / "dataset" / "repairs" / f"{DEFAULT_TARGET}-head-restored-candidate.json"
DEFAULT_ALIGNED_HEAD = REALIGN / "candidates" / f"{DEFAULT_TARGET}-aligned-head-candidate.json"
DEFAULT_OUTPUT = REALIGN / "review_alignment.html"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def read_alignment(path: Path, target_clip_id: str) -> dict[str, Any]:
    matches = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if target_clip_id in row.get("target_clip_ids", []):
            matches.append(row)
    if len(matches) != 1:
        raise ValueError(f"Expected one forced-alignment row for {target_clip_id}; found {len(matches)}")
    return matches[0]


def resolve_audio(value: Any, clip_id: str) -> Path:
    if not value:
        raise ValueError(f"{clip_id}: missing audio path")
    candidate = Path(str(value))
    resolved = candidate.resolve() if candidate.is_absolute() else (ROOT / candidate).resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"{clip_id}: audio path escapes workspace: {value}") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"{clip_id}: audio file not found: {resolved}")
    return resolved


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audio_link(path: Path, output_path: Path) -> str:
    return Path(os.path.relpath(path, output_path.parent)).as_posix()


def fmt_seconds(value: Any) -> str:
    if value is None:
        return "未提供"
    return f"{float(value):.2f} 秒"


def fmt_range(start: Any, end: Any) -> str:
    if start is None or end is None:
        return "未提出安全裁切範圍"
    return f"{float(start):.2f}–{float(end):.2f} 秒"


def checked_audio(path: Path, expected_sha: Any, clip_id: str) -> dict[str, str]:
    actual_sha = sha256_file(path)
    if expected_sha and actual_sha.lower() != str(expected_sha).lower():
        raise ValueError(f"{clip_id}: SHA256 mismatch for {path}")
    return {"path": str(path), "sha256": actual_sha}


def prepare_case(
    context_path: Path,
    asr_path: Path,
    alignment_path: Path,
    repair_path: Path,
    aligned_head_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    context = read_json(context_path)
    asr = read_json(asr_path)
    repair = read_json(repair_path)
    aligned_head = read_json(aligned_head_path)
    clip_id = str(repair.get("clip_id", ""))
    if not clip_id or context.get("target_clip_id") != clip_id or asr.get("target_clip_id") != clip_id or aligned_head.get("clip_id") != clip_id:
        raise ValueError("Context, ASR, and repair artifacts must identify the same target clip")
    alignment = read_alignment(alignment_path, clip_id)
    if alignment.get("context_id") != context.get("context_id"):
        raise ValueError("Forced alignment does not match the selected context")
    if alignment.get("context_audio_sha256") != context.get("context_wav_sha256"):
        raise ValueError("Forced alignment context WAV hash does not match the context metadata")
    if asr.get("input_audio_sha256") != context.get("context_wav_sha256"):
        raise ValueError("Whisper context WAV hash does not match the context metadata")

    target_indices = set(context.get("target_segment_indices") or repair.get("target_segment_indices", []))
    target_segments = [
        row for row in alignment.get("context_segments", [])
        if row.get("segment_index") in target_indices and row.get("is_target")
    ]
    if not target_segments:
        raise ValueError(f"{clip_id}: forced alignment contains no target segments")
    target_words = [
        row for row in alignment.get("word_alignments", [])
        if row.get("segment_index") in target_indices
    ]
    if not target_words:
        raise ValueError(f"{clip_id}: forced alignment contains no target word timings")
    original_path = resolve_audio(repair.get("original_clip_audio_path"), clip_id)
    shifted_path = resolve_audio(repair.get("candidate_wav_path"), clip_id)
    aligned_head_path = resolve_audio(aligned_head.get("candidate_wav_path"), clip_id)
    context_audio_path = resolve_audio(alignment.get("context_audio_path"), clip_id)
    original_audio = checked_audio(original_path, repair.get("original_clip_sha256"), clip_id)
    shifted_audio = checked_audio(shifted_path, repair.get("candidate_sha256"), clip_id)
    aligned_head_audio = checked_audio(aligned_head_path, aligned_head.get("candidate_sha256"), clip_id)
    context_audio = checked_audio(context_audio_path, alignment.get("context_audio_sha256"), clip_id)
    if aligned_head.get("baseline_clip_sha256") != original_audio["sha256"]:
        raise ValueError(f"{clip_id}: aligned-head candidate does not name the current baseline clip hash")
    if "Qwen forced aligner" not in str(aligned_head.get("onset_reference", "")):
        raise ValueError(f"{clip_id}: aligned-head onset is not documented as a Qwen forced-aligner estimate")

    previous_segments = [row for row in alignment.get("context_segments", []) if row.get("segment_index", -1) < min(target_indices)]
    next_segments = [row for row in alignment.get("context_segments", []) if row.get("segment_index", -1) > max(target_indices)]
    previous_segment = previous_segments[-1] if previous_segments else {}
    next_segment = next_segments[0] if next_segments else {}

    asr_words = asr.get("word_timestamps_source_absolute", [])
    asr_onset = next((
        word for word in asr_words
        if word.get("word") == "私"
        and 80.0 <= float(word.get("source_absolute_start_seconds", -1)) <= 85.0
    ), None)
    first_aligned = target_words[0]
    last_aligned = target_words[-1]
    old = repair["old_bounds"]
    new = repair["new_bounds"]
    context_start = float(alignment["context_start_seconds"])
    context_end = float(alignment["context_end_seconds"])

    return {
        "clip_id": clip_id,
        "target_text": repair.get("transcript_text", ""),
        "transcript_provenance": repair.get("transcript_provenance", ""),
        "source_filename": Path(str(repair.get("source_audio", ""))).name,
        "source_sha256": repair.get("source_sha256"),
        "old_bounds": old,
        "new_bounds": new,
        "aligned_head_bounds": aligned_head.get("candidate_bounds", {}),
        "aligned_head_shift_seconds": (aligned_head.get("change_vs_baseline") or {}).get("start_shift_earlier_seconds"),
        "aligned_head_status": aligned_head.get("status", "pending_listening_confirmation"),
        "aligned_head_onset_reference": aligned_head.get("onset_reference", ""),
        "aligned_head_end_policy": aligned_head.get("end_bound_policy", ""),
        "added_context_seconds": repair.get("added_leading_context_seconds"),
        "previous_clip_padded_end_seconds": repair.get("previous_clip_padded_end_seconds"),
        "source_context_choice": "起點接續前一片段的 padded end；依時間標記未包含前一段的詞語區間。",
        "original_audio_url": audio_link(original_path, output_path),
        "shifted_audio_url": audio_link(shifted_path, output_path),
        "aligned_head_audio_url": audio_link(aligned_head_path, output_path),
        "context_audio_url": audio_link(context_audio_path, output_path),
        "audio_hashes": {
            "original": original_audio["sha256"], "shifted": shifted_audio["sha256"],
            "aligned_head": aligned_head_audio["sha256"], "context": context_audio["sha256"],
        },
        "context_start": context_start,
        "context_end": context_end,
        "context_duration": alignment.get("context_duration_seconds"),
        "context_sha256": alignment.get("context_audio_sha256"),
        "context_segments": context.get("segments", []),
        "target_segment_indices": sorted(target_indices),
        "target_aligned_segments": target_segments,
        "first_aligned_word": first_aligned,
        "last_aligned_word": last_aligned,
        "aligned_target_start": target_segments[0].get("aligned_start_source_seconds"),
        "aligned_target_end": target_segments[-1].get("aligned_end_source_seconds"),
        "previous_aligned_end": previous_segment.get("aligned_end_source_seconds"),
        "next_aligned_start": next_segment.get("aligned_start_source_seconds"),
        "previous_aligned_end": previous_segment.get("aligned_end_source_seconds"),
        "next_aligned_start": next_segment.get("aligned_start_source_seconds"),
        "asr_model_id": asr.get("model_id"),
        "asr_model_revision": asr.get("model_revision"),
        "asr_language": asr.get("language", "ja"),
        "asr_onset_word": asr_onset,
        "asr_review_flags": asr.get("review_priority_flags", []),
        "asr_probability_available": asr_onset.get("probability") if asr_onset else None,
        "aligner_model_id": alignment.get("model_id"),
        "aligner_revision": alignment.get("model_revision"),
        "aligner_device": alignment.get("device"),
        "aligner_dtype": alignment.get("dtype"),
        "aligner_resolution": alignment.get("reported_time_resolution_seconds"),
        "aligner_confidence_available": alignment.get("confidence_available"),
        "raw_nonmonotonic_count": alignment.get("raw_timestamp_nonmonotonic_position_count"),
        "decoder_repaired_count": alignment.get("decoder_repaired_timestamp_position_count"),
        "decoder_repaired_first_target_word": alignment.get("decoder_repaired_first_target_word"),
        "decoder_repaired_last_target_word": alignment.get("decoder_repaired_last_target_word"),
        "alignment_status": alignment.get("status"),
        "alignment_review_flags": alignment.get("review_flags", []),
        "source_context_warning": context.get("warnings", []),
        "alignment_interpretation": alignment.get("interpretation", ""),
    }


def render_html(case: dict[str, Any], context_path: Path, alignment_path: Path) -> str:
    j = lambda value: json.dumps(value, ensure_ascii=False)
    target_indices = ", ".join(str(i) for i in case["target_segment_indices"])
    first = case["first_aligned_word"]
    last = case["last_aligned_word"]
    asr_word = case.get("asr_onset_word") or {}
    source_start = float(case["context_start"])
    source_end = float(case["context_end"])
    context_play_start = max(source_start, float(first.get("source_absolute_start_seconds", case["old_bounds"]["start_seconds"])))
    context_play_end = min(source_end, float(case["old_bounds"]["end_seconds"]))
    return f'''<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>逐句時間對齊比較</title>
<style>
:root{{color-scheme:light dark;--bg:#10151c;--panel:#19222d;--panel2:#202c39;--ink:#edf3f8;--muted:#a9b8c7;--line:#354557;--accent:#8fd6ff;--warn:#ffd27b;--red:#ff9696}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}}main{{max-width:1120px;margin:auto;padding:20px}}h1{{font-size:23px;margin:0 0 6px}}h2{{font-size:18px;margin:0 0 10px}}h3{{font-size:16px;margin:0 0 8px}}p{{margin:6px 0}}.muted{{color:var(--muted);font-size:13px}}.banner{{border-left:4px solid var(--warn);background:#49391f33;padding:12px 14px;margin:14px 0;border-radius:6px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px}}.card{{background:var(--panel);border:1px solid var(--line);border-radius:11px;padding:14px;margin:12px 0}}.grid .card{{margin:0}}audio{{width:100%;margin:8px 0}}button{{font:inherit;color:var(--ink);background:var(--panel2);border:1px solid var(--line);border-radius:8px;padding:8px 12px;cursor:pointer}}button:hover{{border-color:var(--accent)}}.metric{{display:grid;grid-template-columns:180px 1fr;gap:5px 12px;font-size:14px}}.metric strong{{font-weight:650}}.pill{{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:2px 8px;margin:2px 4px 2px 0;font-size:12px;color:var(--warn)}}.quote{{font-size:17px;white-space:pre-wrap;margin:7px 0}}.danger{{color:var(--red)}}.bar{{height:10px;background:var(--panel2);border-radius:8px;position:relative;margin:10px 0}}.mark{{position:absolute;top:-5px;width:2px;height:20px;background:var(--accent)}}.mark.warn{{background:var(--warn)}}.time-labels{{display:flex;justify-content:space-between;color:var(--muted);font-size:11px}}details{{margin-top:10px}}summary{{cursor:pointer;color:var(--muted)}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:var(--panel2);padding:10px;border-radius:8px;max-height:300px;overflow:auto}}table{{width:100%;border-collapse:collapse;font-size:13px}}th,td{{padding:7px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}@media(max-width:650px){{main{{padding:12px}}.metric{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>逐句時間對齊比較</h1>
<div class="muted">片段 {html.escape(case['clip_id'])} · 來源 {html.escape(case['source_filename'])} · 原始音訊與文字保持不變</div>
<div class="banner"><strong>所有音訊版本與時間都仍待人工聽校。</strong>「字頭對齊候選」只依強制對齊估計的「私」起點往前留 0.15 秒，並保留目前結尾 88.22 秒；對齊器不提供可信的句尾時間（最後詞時間曾經解碼修補）。這是比較用候選，不是核准結果。</div>
<section class="card"><h2>要比較的原始文字</h2><div class="quote">{html.escape(case['target_text'])}</div><div class="muted">逐字稿來源：{html.escape(case['transcript_provenance'])}</div></section>
<section class="grid">
  <div class="card"><h3>目前片段</h3><div class="metric"><strong>來源範圍</strong><span>{fmt_range(case['old_bounds']['start_seconds'],case['old_bounds']['end_seconds'])}</span><strong>長度</strong><span>{fmt_seconds(case['old_bounds']['duration_seconds'])}</span></div><audio controls preload="none" src="{html.escape(case['original_audio_url'],quote=True)}"></audio></div>
  <div class="card"><h3>較早的固定前導探索版（+0.78 秒）</h3><div class="metric"><strong>來源範圍</strong><span>{fmt_range(case['new_bounds']['start_seconds'],case['new_bounds']['end_seconds'])}</span><strong>新增前導</strong><span>{fmt_seconds(case['added_context_seconds'])}</span><strong>與前一段的關係</strong><span>起點等於前一片段的 padded end {fmt_seconds(case['previous_clip_padded_end_seconds'])}</span></div><audio controls preload="none" src="{html.escape(case['shifted_audio_url'],quote=True)}"></audio><div class="muted">這是待聽校的固定前導探索版，不是逐詞對齊結果。{html.escape(case['source_context_choice'])}</div></div>
  <div class="card"><h3>強制對齊字頭候選（待聽校）</h3><div class="metric"><strong>來源範圍</strong><span>{fmt_range(case['aligned_head_bounds'].get('start_seconds'),case['aligned_head_bounds'].get('end_seconds'))}</span><strong>比原片提前</strong><span>{fmt_seconds(case['aligned_head_shift_seconds'])}</span><strong>字頭留白</strong><span>「私」模型估計起點前 0.15 秒</span><strong>結尾</strong><span>保留原片 88.22 秒，不採用有解碼修補的句尾時間</span></div><audio id="alignedHeadPlayer" controls preload="none" src="{html.escape(case['aligned_head_audio_url'],quote=True)}"></audio><div class="muted">候選狀態：{html.escape(str(case['aligned_head_status']))}。逐字稿未修改。</div></div>
</section>
<section class="card"><h2>完整上下文音訊</h2><div class="metric"><strong>來源範圍</strong><span>{fmt_range(source_start,source_end)}</span><strong>上下文長度</strong><span>{fmt_seconds(case['context_duration'])}</span></div><audio id="contextPlayer" controls preload="none" src="{html.escape(case['context_audio_url'],quote=True)}"></audio><button id="playEstimatedRange" type="button">播放對齊估計的字頭至原片結尾</button><span class="muted"> {fmt_range(context_play_start,context_play_end)} · 在完整上下文中定位播放</span></section>
<section class="card"><h2>逐句邊界比較</h2><table><thead><tr><th>資料來源</th><th>起點</th><th>終點</th><th>說明</th></tr></thead><tbody>
<tr><td>原標註，目標段 {target_indices}</td><td>{fmt_seconds(case['old_bounds']['start_seconds'])}</td><td>{fmt_seconds(case['old_bounds']['end_seconds'])}</td><td>目前片段時間範圍，包含 padded end。</td></tr>
<tr><td>Whisper「私」</td><td>{fmt_seconds(asr_word.get('source_absolute_start_seconds'))}</td><td>{fmt_seconds(asr_word.get('source_absolute_end_seconds'))}</td><td>模型詞語時間戳；token probability {asr_word.get('probability','未提供')}，不是人工信心值。此估計與前一句邊界重疊，不能單獨當作裁切真值。</td></tr>
<tr><td>強制對齊首詞「{html.escape(str(first.get('word','')))}」</td><td>{fmt_seconds(first.get('source_absolute_start_seconds'))}</td><td>{fmt_seconds(first.get('source_absolute_end_seconds'))}</td><td>模型估計，無詞級 confidence。</td></tr>
<tr><td>強制對齊末詞「{html.escape(str(last.get('word','')))}」</td><td>{fmt_seconds(last.get('source_absolute_start_seconds'))}</td><td>{fmt_seconds(last.get('source_absolute_end_seconds'))}</td><td>{'此詞時間由解碼器修補，需特別聽校。' if 'decoder_timestamp_repair_changed_word' in (last.get('review_flags') or []) else '模型估計，需聽校。'} {html.escape(', '.join(last.get('review_flags') or []))}</td></tr>
<tr><td>前一句強制對齊</td><td>—</td><td>{fmt_seconds(case['previous_aligned_end'])}</td><td>強制對齊模型估計的前一句語音終點。</td></tr>
<tr><td>字頭候選</td><td>{fmt_seconds(case['aligned_head_bounds'].get('start_seconds'))}</td><td>{fmt_seconds(case['aligned_head_bounds'].get('end_seconds'))}</td><td>強制對齊估計「私」起點 {fmt_seconds(first.get('source_absolute_start_seconds'))}，候選提前 {fmt_seconds(case['aligned_head_shift_seconds'])}，句尾維持原片。</td></tr>
</tbody></table></section>
<section class="grid">
  <div class="card"><h3>Whisper 上下文估計</h3><div class="metric"><strong>模型</strong><span>{html.escape(str(case['asr_model_id']))}</span><strong>revision</strong><span>{html.escape(str(case['asr_model_revision']))}</span><strong>語言 / 狀態</strong><span>{html.escape(str(case['asr_language']))} · 尚未人工確認</span></div><div class="quote">{html.escape(str(asr_word.get('word','')))}：{fmt_range(asr_word.get('source_absolute_start_seconds'),asr_word.get('source_absolute_end_seconds'))}</div><div class="muted">此詞語時間來自完整上下文 Whisper 結果；不代表來源逐字稿或實際音節邊界已確認。</div></div>
  <div class="card"><h3>Qwen 強制對齊估計</h3><div class="metric"><strong>模型</strong><span>{html.escape(str(case['aligner_model_id']))}</span><strong>revision</strong><span>{html.escape(str(case['aligner_revision']))}</span><strong>裝置 / dtype</strong><span>{html.escape(str(case['aligner_device']))} / {html.escape(str(case['aligner_dtype']))}</span><strong>時間解析度</strong><span>{fmt_seconds(case['aligner_resolution'])} · 詞語信心值：{'有提供' if case['aligner_confidence_available'] else '未提供'}</span><strong>原始非單調位置</strong><span>{case['raw_nonmonotonic_count']}</span><strong>解碼修補位置</strong><span>{case['decoder_repaired_count']}</span></div><div class="muted">目標第一詞修補：{case['decoder_repaired_first_target_word']}；最後一詞修補：{case['decoder_repaired_last_target_word']}。強制對齊器本身不產生 WAV；字頭候選是另外依來源音訊裁切，原始片段與逐字稿未修改。</div></div>
</section>
<details><summary>試點來源、上下文與模型資訊</summary><pre>{html.escape(json.dumps({"context_id": context_path.stem, "alignment_status": case['alignment_status'], "alignment_interpretation": case['alignment_interpretation'], "aligned_head_onset_reference": case['aligned_head_onset_reference'], "aligned_head_end_policy": case['aligned_head_end_policy'], "context_audio_sha256": case['context_sha256'], "audio_hashes": case['audio_hashes'], "source_sha256": case['source_sha256'], "asr_review_flags": case['asr_review_flags'], "source_context_warning": case['source_context_warning']},ensure_ascii=False,indent=2))}</pre></details>
</main>
<script>
const player=document.getElementById("contextPlayer");
const startLocal={context_play_start - source_start:.6f};
const endLocal={context_play_end - source_start:.6f};
let stopAtEnd=null;
document.getElementById("playEstimatedRange").addEventListener("click",async()=>{{
  player.pause(); if(stopAtEnd)clearInterval(stopAtEnd); player.currentTime=startLocal;
  try{{await player.play();}}catch(error){{return;}}
  stopAtEnd=setInterval(()=>{{if(player.currentTime>=endLocal){{player.pause();clearInterval(stopAtEnd);stopAtEnd=null;}}}},80);
}});
</script></body></html>'''


def build_preview(
    context_path: Path,
    asr_path: Path,
    alignment_path: Path,
    repair_path: Path,
    aligned_head_path: Path,
    output_path: Path,
) -> None:
    case = prepare_case(context_path, asr_path, alignment_path, repair_path, aligned_head_path, output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_html(case, context_path, alignment_path), encoding="utf-8")
    print(f"Built offline alignment comparison for {case['clip_id']}: {output_path}")
    print(f"Original source bounds: {case['old_bounds']['start_seconds']:.2f}–{case['old_bounds']['end_seconds']:.2f} s")
    print(f"Aligned-head candidate: {case['aligned_head_bounds']['start_seconds']:.2f}–{case['aligned_head_bounds']['end_seconds']:.2f} s; listening confirmation is pending.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--asr-summary", type=Path, default=DEFAULT_ASR)
    parser.add_argument("--alignment-jsonl", type=Path, default=DEFAULT_ALIGNMENT)
    parser.add_argument("--repair-record", type=Path, default=DEFAULT_REPAIR)
    parser.add_argument("--aligned-head-record", type=Path, default=DEFAULT_ALIGNED_HEAD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    paths = [args.context, args.asr_summary, args.alignment_jsonl, args.repair_record, args.aligned_head_record, args.output]
    resolved = [path if path.is_absolute() else ROOT / path for path in paths]
    build_preview(*resolved)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
