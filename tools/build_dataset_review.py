#!/usr/bin/env python3
"""Build a small, offline browser review page for a clip JSONL manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "outputs" / "dataset" / "manifest.jsonl"
DEFAULT_OUTPUT = ROOT / "outputs" / "dataset" / "review.html"
DEFAULT_ORGANIZATION = ROOT / "outputs" / "dataset" / "organization_summary.json"


def _source_name(value: Any) -> str:
    raw = str(value or "")
    return raw.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1] if raw else "(unknown source)"


def _source_timing(row: dict[str, Any]) -> dict[str, Any]:
    words = ("start", "end", "time", "interval", "frame", "sample")
    return {
        key: value
        for key, value in row.items()
        if any(word in key.lower() for word in words)
        and isinstance(value, (int, float, str, list, tuple))
        and not isinstance(value, bool)
    }


def _machine_flags(row: dict[str, Any]) -> dict[str, Any]:
    markers = (
        "flag", "unverified", "background", "identity", "overlap", "event",
        "music", "noise", "qc", "machine_status", "review_status", "asr",
        "priority", "mismatch", "difference", "disagreement",
    )
    flags: dict[str, Any] = {}
    for nested_key in ("machine_flags", "flags", "qc_stats"):
        nested = row.get(nested_key)
        if isinstance(nested, dict):
            flags.update(nested)
        elif isinstance(nested, list):
            flags.update({str(item): True for item in nested})
    for key, value in row.items():
        normalized = key.lower()
        if key in {"asr_check", "asr_check_text", "normalized_character_disagreement_ratio", "asr_review_priority_flags", "asr_status"}:
            continue
        if key not in {"machine_flags", "flags", "qc_stats"} and any(marker in normalized for marker in markers):
            flags.setdefault(key, value)
    return flags


def _asr_summary(row: dict[str, Any]) -> dict[str, Any]:
    check = row.get("asr_check") if isinstance(row.get("asr_check"), dict) else {}
    result = check.get("result") if isinstance(check.get("result"), dict) else {}
    return {
        "text": result.get("text", row.get("asr_check_text")),
        "normalized_character_disagreement_ratio": row.get(
            "normalized_character_disagreement_ratio", check.get("normalized_character_disagreement_ratio")
        ),
        "review_priority_flags": row.get("asr_review_priority_flags", check.get("asr_review_priority_flags", [])),
    }


def _raw_review_metadata(row: dict[str, Any]) -> dict[str, Any]:
    keep = {
        "asr_check", "asr_status", "asr_review_priority_flags", "normalized_character_disagreement_ratio",
        "qc", "qc_stats", "qc_flags", "machine_review_flags", "source_wav", "source_wav_sha256",
        "source_start_frame", "source_end_frame_exclusive", "actual_start_seconds", "actual_end_seconds",
        "sample_rate_hz", "channels", "frames", "subtype", "clip_sha256", "transcript_sha256",
    }
    return {key: row[key] for key in keep if key in row}


def _has_entries(value: Any) -> bool:
    if isinstance(value, (list, dict, tuple, set)):
        return bool(value)
    return isinstance(value, str) and bool(value.strip())


def _asr_review_state(row: dict[str, Any]) -> tuple[str, bool]:
    check = row.get("asr_check")
    check = check if isinstance(check, dict) else {}
    result = check.get("result")
    result = result if isinstance(result, dict) else {}
    text = result.get("text", row.get("asr_check_text"))
    ratio = row.get("normalized_character_disagreement_ratio", check.get("normalized_character_disagreement_ratio"))
    priorities = row.get("asr_review_priority_flags", check.get("asr_review_priority_flags"))
    explicit_status = row.get("asr_status", check.get("status"))
    has_asr_metadata = bool(check) or any(key in row for key in (
        "asr_check_text", "asr_review_priority_flags", "normalized_character_disagreement_ratio", "asr_status",
    ))
    if not has_asr_metadata:
        return "pending", False
    if explicit_status is not None:
        status = str(explicit_status).strip().lower()
        if status not in {"ok", "success", "complete", "completed", "ready"}:
            return str(explicit_status), True
    elif not result and text is None:
        return "missing_result", True
    if _has_entries(priorities):
        return "priority_review", True
    if text is None or not str(text).strip():
        return "empty_hypothesis", True
    try:
        if ratio is not None and float(ratio) > 0.35:
            return "high_disagreement", True
    except (TypeError, ValueError):
        return "invalid_disagreement_ratio", True
    return "checked", False


def _review_priority_flags(row: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("machine_review_flags", "qc_flags"):
        value = row.get(key)
        if isinstance(value, list):
            values.extend(str(item) for item in value if str(item).strip())
        elif isinstance(value, dict):
            values.extend(str(name) for name, enabled in value.items() if enabled)
    state, needs_asr_review = _asr_review_state(row)
    if needs_asr_review:
        values.append(f"ASR:{state}")
    return list(dict.fromkeys(values))


def _canonical_fingerprint(rows: list[dict[str, Any]]) -> str:
    identity = []
    for row in rows:
        audio_sha = row.get("clip_sha256") or row.get("clip_audio_sha256") or row.get("wav_sha256")
        if not audio_sha:
            raise ValueError(f"{row.get('clip_id', '(unknown clip)')}: manifest is missing the clip WAV SHA256")
        identity.append({
            "clip_id": row["clip_id"],
            "clip_audio_sha256": str(audio_sha).lower(),
            "original_lexical_intervals_seconds": row.get("original_lexical_intervals_seconds", []),
            "original_text_segments": row.get("original_text_segments", [row.get("text", "")]),
        })
    identity.sort(key=lambda item: item["clip_id"])
    canonical = json.dumps(
        {"schema": "nino-review-dataset-identity-v1", "clips": identity},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _resolve_clip_path(relative_path: Any, clip_id: str) -> Path:
    if not relative_path:
        raise ValueError(f"{clip_id}: manifest row is missing clip_audio_path")
    candidate = Path(str(relative_path))
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (ROOT / candidate).resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"{clip_id}: clip_audio_path escapes the workspace: {relative_path}") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"{clip_id}: clip WAV not found: {resolved}")
    return resolved


def prepare_records(
    manifest_path: Path, output_path: Path, organization_path: Path | None = None
) -> tuple[list[dict[str, Any]], str]:
    manifest_bytes = manifest_path.read_bytes()
    organization: dict[str, str] = {}
    if organization_path is not None and organization_path.is_file():
        summary = json.loads(organization_path.read_text(encoding="utf-8"))
        clip_tiers = summary.get("clip_id_to_tier") if isinstance(summary, dict) else None
        if not isinstance(clip_tiers, dict):
            raise ValueError(f"{organization_path}: expected an object field named clip_id_to_tier")
        organization = {str(clip_id): str(tier) for clip_id, tier in clip_tiers.items()}
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(manifest_bytes.decode("utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Manifest line {line_number} must be a JSON object")
        clip_id = str(row.get("clip_id", "")).strip()
        if not clip_id or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for ch in clip_id):
            raise ValueError(f"Manifest line {line_number} has an invalid ASCII clip_id: {clip_id!r}")
        if clip_id in seen:
            raise ValueError(f"Duplicate clip_id in manifest: {clip_id}")
        seen.add(clip_id)

        clip_path = _resolve_clip_path(row.get("clip_audio_path"), clip_id)
        audio_url = Path(os.path.relpath(clip_path, output_path.parent)).as_posix()
        source_audio = row.get("source_audio") or row.get("source_filename") or row.get("source_file")
        text_segments = row.get("original_text_segments")
        if row.get("text") is not None:
            draft_text = str(row["text"])
        elif isinstance(text_segments, list):
            draft_text = "\n".join(str(part) for part in text_segments)
        else:
            draft_text = str(row.get("transcript", ""))
        duration = row.get("clip_duration_seconds", row.get("duration_seconds", row.get("duration_sec")))
        if duration is None and isinstance(row.get("clip_duration"), (int, float)):
            duration = row["clip_duration"]

        rows.append({
            "clip_id": clip_id,
            "organization_tier": organization.get(clip_id, row.get("organization_tier", "")),
            "organization_reasons": row.get("organization_reasons", []),
            "clip_audio_sha256": row.get("clip_sha256", row.get("clip_audio_sha256", row.get("wav_sha256", ""))),
            "audio_url": audio_url,
            "audio_exists": clip_path.is_file(),
            "draft_text": draft_text,
            "duration_seconds": duration,
            "source_audio": str(source_audio or ""),
            "source_filename": _source_name(source_audio),
            "source_speaker_label": row.get("source_speaker_label", row.get("speaker_label", "")),
            "source_timing": _source_timing(row),
            "machine_flags": _machine_flags(row),
            "asr_summary": _asr_summary(row),
            "raw_review_metadata": _raw_review_metadata(row),
            "asr_status": _asr_review_state(row)[0],
            "asr_needs_review": _asr_review_state(row)[1],
            "review_priority_flags": _review_priority_flags(row),
            "machine_status": row.get("machine_status", "machine candidate"),
            "transcript_path": row.get("transcript_path", ""),
            "source_transcript_path": row.get("source_transcript_path", ""),
            "text_provenance": row.get("text_provenance", row.get("transcript_provenance", "ASR draft; unverified")),
        })
    if not rows:
        raise ValueError(f"No clip rows found in {manifest_path}")
    unknown_tiers = set(organization) - seen
    if unknown_tiers:
        raise ValueError(f"Organization summary contains unknown clip IDs: {sorted(unknown_tiers)[:5]}")
    source_rows = [json.loads(line) for line in manifest_bytes.decode("utf-8").splitlines() if line.strip()]
    return rows, _canonical_fingerprint(source_rows)


HTML_TEMPLATE = r'''<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>本機語音資料集審核</title>
<style>
:root{color-scheme:light dark;--bg:#10151c;--panel:#19222d;--panel2:#202c39;--ink:#edf3f8;--muted:#a9b8c7;--line:#354557;--accent:#8fd6ff;--good:#8cdb9c;--warn:#ffd27b;--bad:#ff9696}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
header{position:sticky;top:0;z-index:5;background:#10151cf2;border-bottom:1px solid var(--line);padding:12px 18px;display:flex;gap:14px;align-items:center;flex-wrap:wrap}
h1{font-size:18px;margin:0}.count{color:var(--muted);font-size:13px}.layout{display:grid;grid-template-columns:minmax(230px,310px) minmax(0,1fr);gap:14px;padding:14px;max-width:1500px;margin:auto}
aside,.card{background:var(--panel);border:1px solid var(--line);border-radius:12px}.filters{padding:12px;position:sticky;top:74px;align-self:start;max-height:calc(100vh - 90px);overflow:auto}.filters h2,.card h2{font-size:15px;margin:0 0 10px}
label{display:block;font-size:12px;color:var(--muted);margin:9px 0 4px}select,textarea,input[type=text]{width:100%;background:var(--panel2);color:var(--ink);border:1px solid var(--line);border-radius:7px;padding:8px;font:inherit}
button{font:inherit;background:var(--panel2);color:var(--ink);border:1px solid var(--line);padding:8px 11px;border-radius:8px;cursor:pointer}button:hover:not(:disabled){border-color:var(--accent)}button:disabled{opacity:.45;cursor:not-allowed}
.toolbar{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0}.list{display:grid;gap:6px;margin-top:12px}.clip-button{text-align:left;font-size:12px;padding:8px}.clip-button[aria-current=true]{border-color:var(--accent);box-shadow:inset 3px 0 var(--accent)}
main{min-width:0}.card{padding:14px;margin-bottom:12px}.toprow{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap}.clip-title{font-size:18px;font-weight:700}.status{font-size:13px;font-weight:700;padding:3px 8px;border-radius:999px;background:#344252}.status.approved{color:var(--good)}.status.needs_review{color:var(--warn)}.status.rejected{color:var(--bad)}
.meta{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:6px;color:var(--muted);font-size:13px;margin-top:10px}.meta strong{color:var(--ink);font-weight:600}.flags{display:flex;gap:6px;flex-wrap:wrap;margin-top:10px}.flag{border:1px solid var(--line);background:#253241;border-radius:999px;padding:3px 8px;font-size:12px;color:#ffe0a6}.flag.neutral{color:var(--muted)}
audio{width:100%;margin:8px 0}.warning{border-left:3px solid var(--warn);padding:7px 10px;background:#4a3b201e;color:var(--warn);font-size:13px}.draft{min-height:120px;resize:vertical}.caption{min-height:54px;resize:vertical}.checks{display:grid;gap:8px;margin:12px 0}.checks label{display:flex;gap:9px;align-items:flex-start;margin:0;color:var(--ink);font-size:14px}.checks input{margin-top:4px}
.actions{display:flex;gap:8px;flex-wrap:wrap}.approve{border-color:#4b9563}.reject{border-color:#a65050}.message{min-height:20px;margin-top:8px;color:var(--muted);font-size:13px}.empty{padding:18px;color:var(--muted)}.small{font-size:12px;color:var(--muted)}
kbd{display:inline-block;font-family:inherit;font-size:12px;line-height:1.4;border:1px solid var(--line);border-radius:4px;padding:0 5px;background:var(--bg);color:var(--ink)}.shortcuts{display:flex;flex-wrap:wrap;gap:8px 15px;padding:11px 14px;margin-bottom:12px;border:1px solid var(--line);border-radius:10px;color:var(--muted);font-size:12px}.shortcuts strong,.shortcuts .shortcut-note{flex-basis:100%}.shortcut-item{display:inline-flex;align-items:center;gap:4px;flex-wrap:wrap}.checks kbd,.problem kbd{margin-left:auto;flex-shrink:0}
.asr-summary{border:1px solid var(--line);border-radius:8px;padding:9px 11px;margin:8px 0;color:var(--muted);font-size:13px}.asr-summary strong{color:var(--ink)}.asr-text{margin-top:5px;white-space:pre-wrap;overflow-wrap:anywhere;color:var(--ink)}details.raw{margin-top:10px;color:var(--muted);font-size:12px}details.raw summary{cursor:pointer}details.raw pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:300px;overflow:auto;background:var(--panel2);padding:9px;border-radius:8px;color:var(--ink)}
.problem{border:1px solid var(--line);border-radius:9px;padding:8px 12px 12px;margin:14px 0}.problem legend{font-size:14px;font-weight:700;color:var(--ink);padding:0 5px}.problem label{display:flex;gap:9px;align-items:flex-start;color:var(--ink);font-size:14px;margin:8px 0}.problem input{margin-top:4px}.problem textarea{min-height:68px}
@media(max-width:760px){.layout{grid-template-columns:1fr}.filters{position:static;max-height:none}header{position:static}}
</style>
</head>
<body>
<header>
  <h1>本機語音資料集審核</h1>
  <span class="count" id="summary"></span>
  <span class="small">離線頁面 · 音訊保留在工作區 · 不會預設任何審核結果</span>
  <button id="exportButton" type="button">匯出人工審核 JSON</button>
</header>
<div class="layout">
  <aside class="filters">
    <h2>音訊清單與篩選</h2>
    <label for="sourceFilter">來源檔案</label><select id="sourceFilter"><option value="">所有來源</option></select>
    <label for="groupFilter">資料分組</label><select id="groupFilter"><option value="">所有資料</option></select>
    <label for="flagFilter">機器標記</label><select id="flagFilter">
      <option value="">所有片段</option><option value="flagged">有任何機器標記</option>
      <option value="asr">ASR 待確認</option><option value="background">含背景音訊</option>
      <option value="identity">說話者身分未確認</option>
    </select>
    <label for="statusFilter">人工審核狀態</label><select id="statusFilter">
      <option value="">所有狀態</option><option value="pending">待審核</option>
      <option value="approved">已核准</option><option value="needs_review">需要再確認</option><option value="rejected">已排除</option>
    </select>
    <label for="sortBy">排序</label><select id="sortBy">
      <option value="source">來源檔案與時間</option><option value="flags">有機器標記的優先</option><option value="status">審核狀態</option>
    </select>
    <div class="list" id="clipList"></div>
  </aside>
  <main>
    <div class="shortcuts" aria-label="鍵盤快捷鍵">
      <strong>快捷鍵（主鍵盤／數字鍵盤）</strong>
      <span class="shortcut-item"><kbd>1</kbd>／<kbd>Space</kbd> 播放／暫停</span>
      <span class="shortcut-item"><kbd>2</kbd> 三項確認全選／清除</span>
      <span class="shortcut-item"><kbd>3</kbd> 字頭被切</span>
      <span class="shortcut-item"><kbd>4</kbd> 字尾被切</span>
      <span class="shortcut-item"><kbd>5</kbd>／<kbd>←</kbd> 上一筆</span>
      <span class="shortcut-item"><kbd>6</kbd>／<kbd>→</kbd> 下一筆</span>
      <span class="shortcut-item"><kbd>7</kbd> 核准</span>
      <span class="shortcut-item"><kbd>8</kbd> 需要再確認</span>
      <span class="shortcut-note">2 只切換確認勾選，不會核准。編輯文字、操作選單或輸入法組字時不啟用快捷鍵。</span>
    </div>
    <section class="card" id="clipCard">
      <div class="toprow"><div class="clip-title" id="clipTitle">載入中…</div><div class="status" id="statusBadge">待審核</div></div>
      <div class="meta" id="clipMeta"></div>
      <div class="flags" id="machineFlags"></div>
      <div class="warning">這些是機器產生的候選資料。審核者明確確認下方三項之前，不會核准任何片段。</div>
      <div class="actions"><button id="playPauseButton" type="button" aria-keyshortcuts="1 Space">播放／暫停 <kbd>1</kbd>／<kbd>Space</kbd></button></div>
      <audio id="player" controls preload="none"></audio>
      <div class="small" id="playerMessage"></div>
      <div class="asr-summary">
        <strong>ASR 核對資訊</strong><span id="asrStatus"></span>
        <div class="asr-text" id="asrText"></div>
        <div id="asrRatio"></div><div id="asrPriorities"></div>
        <div class="small">字詞時間戳接近片段邊界是機器提醒，不代表已確認切掉字。兩份辨識文字的差異僅供分流，不代表正確率。</div>
      </div>
      <label for="transcript">辨識草稿 · 請聆聽後修正</label>
      <textarea id="transcript" class="draft" spellcheck="false"></textarea>
      <div class="small" id="provenance"></div>
      <details class="raw"><summary>原始 ASR、QC 與來源技術欄位</summary><pre id="rawMetadata"></pre></details>
    </section>
    <section class="card">
      <h2>人工審核</h2>
      <div class="actions"><button id="toggleChecksButton" type="button" aria-keyshortcuts="2">三項確認全選／清除 <kbd>2</kbd></button></div>
      <div class="checks">
        <label><input type="checkbox" id="checkSpeaker" aria-keyshortcuts="2"> 我已確認片段中是目標說話者。<kbd>2</kbd></label>
        <label><input type="checkbox" id="checkComplete" aria-keyshortcuts="2"> 我已完整聆聽，並確認音訊完整且可用。<kbd>2</kbd></label>
        <label><input type="checkbox" id="checkText" aria-keyshortcuts="2"> 我已校對文字，確認內容符合實際聽到的語句。<kbd>2</kbd></label>
      </div>
      <label for="styleCaption">選填：審核者撰寫的說話風格描述</label>
      <textarea id="styleCaption" class="caption" placeholder="例如：平靜、自然、戲劇化。請由審核者自行判斷，不會自動推測。"></textarea>
      <fieldset class="problem">
        <legend>音訊問題</legend>
        <label><input type="checkbox" id="issueClippedStart" aria-keyshortcuts="3"> 字頭被切（開頭語音不完整）<kbd>3</kbd></label>
        <label><input type="checkbox" id="issueClippedEnd" aria-keyshortcuts="4"> 字尾被切（結尾語音不完整）<kbd>4</kbd></label>
        <label for="problemNotes">問題備註（與說話風格描述分開）</label>
        <textarea id="problemNotes" class="caption" placeholder="例如：開頭的「私」只剩下後半音節。"></textarea>
        <div class="small">若音節被切，請選「需要再確認」並標記問題；保留完整逐字稿。未解問題請勾選，備註可保留核對說明。修剪後必須重新聽校。</div>
      </fieldset>
      <div class="actions">
        <button id="previousButton" type="button" aria-keyshortcuts="5 ArrowLeft">上一筆 <kbd>5</kbd>／<kbd>←</kbd></button><button id="nextButton" type="button" aria-keyshortcuts="6 ArrowRight">下一筆 <kbd>6</kbd>／<kbd>→</kbd></button>
        <button id="approveButton" class="approve" type="button" aria-keyshortcuts="7">核准 <kbd>7</kbd></button>
        <button id="needsReviewButton" type="button" aria-keyshortcuts="8">需要再確認 <kbd>8</kbd></button><button id="rejectButton" class="reject" type="button">排除</button>
        <button id="clearButton" type="button">清除決定</button>
      </div>
      <div class="message" id="message" role="status" aria-live="polite"></div>
      <div class="small">核准需要勾選全部確認項目、提供非空白校對文字，並清除所有未解問題標記與備註。修改已儲存決定時，該決定會撤銷並回到待審核。</div>
    </section>
  </main>
</div>
<script>
const DATASET_FINGERPRINT = __FINGERPRINT__;
const CLIPS = __CLIPS__;
const STORAGE_PREFIX = `nino-dataset-review:v2:${DATASET_FINGERPRINT}:`;
const STORAGE_UNAVAILABLE_MESSAGE = "此檔案網址無法使用瀏覽器本機儲存。請改用允許本機檔案儲存的瀏覽器，才能記錄審核結果。";
const byId = new Map(CLIPS.map(clip => [clip.clip_id, clip]));
const $ = id => document.getElementById(id);
let selectedId = CLIPS[0]?.clip_id ?? null;
let visibleIds = [];
let storageReady = false;
try {
  const probe = STORAGE_PREFIX + "__probe__";
  localStorage.setItem(probe, "1"); localStorage.removeItem(probe); storageReady = true;
} catch (error) {
  $("message").textContent = STORAGE_UNAVAILABLE_MESSAGE;
}
$("exportButton").disabled = !storageReady;

function storageKey(id) { return STORAGE_PREFIX + id; }
function draftStorageKey(id) { return STORAGE_PREFIX + "draft:" + id; }
function getDecision(id) {
  if (!storageReady) return null;
  try {
    const raw = localStorage.getItem(storageKey(id));
    if (!raw) return null;
    const value = JSON.parse(raw);
    return ["approved", "needs_review", "rejected"].includes(value.decision) ? value : null;
  } catch (_) { return null; }
}
function getDraft(id) {
  if (!storageReady) return null;
  try {
    const raw=localStorage.getItem(draftStorageKey(id));
    if (!raw) return null;
    const value=JSON.parse(raw);
    return value && typeof value === "object" ? value : null;
  } catch (_) { return null; }
}
function savePendingDraft() {
  if (!storageReady || !selectedId || getDecision(selectedId)) return;
  const value={clip_id:selectedId,reviewed_text:$("transcript").value,style_caption:$("styleCaption").value,problem_issues:selectedProblemIssues(),problem_notes:$("problemNotes").value,approval_checks:{correct_speaker:$("checkSpeaker").checked,complete_audio:$("checkComplete").checked,corrected_text:$("checkText").checked},updated_at:new Date().toISOString()};
  try { localStorage.setItem(draftStorageKey(selectedId),JSON.stringify(value)); }
  catch (_) { $("message").textContent="無法將未完成草稿儲存至瀏覽器本機空間。"; }
}
function currentStatus(id) { return getDecision(id)?.decision ?? "pending"; }
function visibleStatus(status) { return ({pending:"待審核",approved:"已核准",needs_review:"需要再確認",rejected:"已排除"})[status] || status; }
function visibleAsrStatus(status) { return ({pending:"待核對",checked:"已核對",priority_review:"已核對，需聽校",missing_result:"缺少辨識結果，需聽校",empty_hypothesis:"辨識文字為空，需聽校",high_disagreement:"文字差異較高，需聽校",invalid_disagreement_ratio:"差異資料異常，需聽校"})[status] || (status ? `${status}，需聽校` : "待核對"); }
function selectedProblemIssues() {
  const issues=[];
  if ($("issueClippedStart").checked) issues.push("clipped_start");
  if ($("issueClippedEnd").checked) issues.push("clipped_end");
  return issues;
}
function asText(value) { return value == null ? "" : String(value); }
function pretty(value) {
  if (value === true) return "yes";
  if (value === false) return "no";
  if (value && typeof value === "object") return JSON.stringify(value);
  return String(value);
}
function flagKind(key, value) {
  const k = key.toLowerCase();
  if (k.includes("background") && value) return "background";
  if (k.includes("identity") && value === false) return "identity";
  return "other";
}
function hasFlag(clip, wanted) {
  if (wanted === "flagged") return (clip.review_priority_flags || []).length > 0;
  if (wanted === "asr") return clip.asr_needs_review === true;
  const flags = clip.machine_flags || {};
  return Object.entries(flags).some(([key, value]) => {
    return flagKind(key, value) === wanted;
  });
}
function sortStart(clip) {
  const timing = clip.source_timing || {};
  for (const [key, value] of Object.entries(timing)) if (key.toLowerCase().includes("start") && Number.isFinite(Number(value))) return Number(value);
  return 0;
}
function refreshVisible() {
  const source = $("sourceFilter").value;
  const group = $("groupFilter").value;
  const flag = $("flagFilter").value;
  const status = $("statusFilter").value;
  const sort = $("sortBy").value;
  let clips = CLIPS.filter(clip => {
    if (source && clip.source_filename !== source) return false;
    if (group && clip.organization_tier !== group) return false;
    if (flag && !hasFlag(clip, flag)) return false;
    if (status && currentStatus(clip.clip_id) !== status) return false;
    return true;
  });
  clips.sort((a,b) => {
    if (sort === "flags") return (b.review_priority_flags||[]).length-(a.review_priority_flags||[]).length || a.source_filename.localeCompare(b.source_filename) || sortStart(a)-sortStart(b);
    if (sort === "status") return currentStatus(a.clip_id).localeCompare(currentStatus(b.clip_id)) || a.source_filename.localeCompare(b.source_filename) || sortStart(a)-sortStart(b);
    return a.source_filename.localeCompare(b.source_filename) || sortStart(a)-sortStart(b) || a.clip_id.localeCompare(b.clip_id);
  });
  visibleIds = clips.map(clip => clip.clip_id);
  if (!visibleIds.includes(selectedId)) selectedId = visibleIds[0] ?? null;
  renderList(); renderCurrent(); updateSummary();
}
function renderList() {
  const list = $("clipList"); list.replaceChildren();
  if (!visibleIds.length) { const empty=document.createElement("div"); empty.className="empty"; empty.textContent="沒有符合篩選條件的片段。"; list.append(empty); return; }
  for (const id of visibleIds) {
    const clip=byId.get(id), button=document.createElement("button");
    button.type="button"; button.className="clip-button"; button.setAttribute("aria-current", String(id===selectedId));
    button.textContent=`${id} · ${visibleStatus(currentStatus(id))} · ${clip.source_filename}`;
    button.addEventListener("click",()=>{selectedId=id;renderList();renderCurrent();}); list.append(button);
  }
}
function renderCurrent() {
  const clip=byId.get(selectedId);
  const disabled=!clip;
  for (const id of ["previousButton","nextButton","approveButton","needsReviewButton","rejectButton","clearButton","toggleChecksButton","transcript","styleCaption","checkSpeaker","checkComplete","checkText","issueClippedStart","issueClippedEnd","problemNotes"]) $(id).disabled=disabled || !storageReady;
  $("playPauseButton").disabled=disabled;
  if (!clip) { $("clipTitle").textContent="未選取片段"; $("clipMeta").replaceChildren(); $("machineFlags").replaceChildren(); $("player").removeAttribute("src"); return; }
  const saved=getDecision(clip.clip_id);
  const draft=getDraft(clip.clip_id);
  $("clipTitle").textContent=clip.clip_id;
  const status=$("statusBadge"); status.textContent=visibleStatus(currentStatus(clip.clip_id)); status.className="status "+currentStatus(clip.clip_id);
  const meta=$("clipMeta"); meta.replaceChildren();
  const fields=[
    ["來源檔案",clip.source_filename], ["來源路徑",clip.source_audio],
    ["來源說話者標記",clip.source_speaker_label], ["片段長度",clip.duration_seconds == null ? "未知" : `${clip.duration_seconds} 秒`],
    ["來源時間資訊",Object.entries(clip.source_timing||{}).map(([k,v])=>`${k}: ${pretty(v)}`).join(" · ") || "未提供"],
    ["資料分組",clip.organization_tier || "未分組"], ["機器處理狀態",clip.machine_status], ["ASR 檢查狀態",clip.asr_status === "pending" ? "待核對" : clip.asr_status], ["片段 ID",clip.clip_id]
  ];
  for (const [name,value] of fields) { const item=document.createElement("div"), strong=document.createElement("strong"); strong.textContent=name+": "; item.append(strong,document.createTextNode(asText(value)||"未提供")); meta.append(item); }
  const flagBox=$("machineFlags"); flagBox.replaceChildren();
  const flags=Object.entries(clip.machine_flags||{});
  if (!flags.length) { const tag=document.createElement("span"); tag.className="flag neutral"; tag.textContent="沒有提供機器標記"; flagBox.append(tag); }
  for (const [key,value] of flags) { const tag=document.createElement("span"); tag.className="flag"; tag.textContent=`${key}: ${pretty(value)}`; flagBox.append(tag); }
  const player=$("player"); player.src=clip.audio_url; player.load(); $("playerMessage").textContent=clip.audio_exists?"播放器使用工作區中的相對 WAV 路徑；頁面未嵌入音訊資料。":"找不到音訊檔案。";
  const asr=clip.asr_summary||{};
  $("asrStatus").textContent=` · ${visibleAsrStatus(clip.asr_status)}`;
  $("asrText").textContent=asr.text == null ? "尚無 ASR 核對文字。" : `辨識結果：${asr.text}`;
  const ratio=asr.normalized_character_disagreement_ratio;
  $("asrRatio").textContent=ratio == null ? "正規化字元差異：未提供（此數值僅供分流，不代表辨識準確率）。" : `正規化字元差異：${ratio}（僅供分流，不代表辨識準確率）。`;
  const priorities=asr.review_priority_flags;
  $("asrPriorities").textContent=priorities && (Array.isArray(priorities)?priorities.length:Object.keys(priorities).length) ? `ASR 優先確認標記：${JSON.stringify(priorities)}` : "沒有明確 ASR 優先確認標記。";
  $("rawMetadata").textContent=JSON.stringify(clip.raw_review_metadata||{},null,2);
  $("transcript").value=saved?.reviewed_text ?? draft?.reviewed_text ?? clip.draft_text;
  $("styleCaption").value=saved?.style_caption ?? draft?.style_caption ?? "";
  const problemIssues=saved?.problem_issues ?? draft?.problem_issues ?? [];
  $("issueClippedStart").checked=problemIssues.includes("clipped_start");
  $("issueClippedEnd").checked=problemIssues.includes("clipped_end");
  $("problemNotes").value=saved?.problem_notes ?? draft?.problem_notes ?? "";
  const checks=saved?.approval_checks||draft?.approval_checks||{};
  $("checkSpeaker").checked=!!checks.correct_speaker; $("checkComplete").checked=!!checks.complete_audio; $("checkText").checked=!!checks.corrected_text;
  const sourceTranscript=clip.source_transcript_path?` · 來源逐字稿：${clip.source_transcript_path}`:"";
  $("provenance").textContent=`文字來源：${clip.text_provenance||"未提供"} · 片段逐字稿：${clip.transcript_path||"未提供"}${sourceTranscript}`;
  const at=visibleIds.indexOf(selectedId); $("previousButton").disabled=!storageReady||at<=0; $("nextButton").disabled=!storageReady||at<0||at>=visibleIds.length-1;
  $("message").textContent=!storageReady?STORAGE_UNAVAILABLE_MESSAGE:saved?`已儲存人工決定：${visibleStatus(saved.decision)}。`:"等待人工審核。機器標記不代表已核准。";
  updateApproveButton();
}
function updateApproveButton() {
  const unresolved=selectedProblemIssues().length>0;
  $("approveButton").disabled=!storageReady||!selectedId||unresolved||!$("checkSpeaker").checked||!$("checkComplete").checked||!$("checkText").checked||!$("transcript").value.trim();
}
function invalidateChangedSavedDecision() {
  const saved=getDecision(selectedId); if (!saved) return;
  const previousIssues=saved.problem_issues||[];
  const currentIssues=selectedProblemIssues();
  if ($("transcript").value!==saved.reviewed_text || $("styleCaption").value!==(saved.style_caption||"") || $("problemNotes").value!==(saved.problem_notes||"") || JSON.stringify(currentIssues)!==JSON.stringify(previousIssues)) {
    try { localStorage.removeItem(storageKey(selectedId)); } catch (_) {}
    $("checkSpeaker").checked=false; $("checkComplete").checked=false; $("checkText").checked=false;
    savePendingDraft();
    updateSummary(); renderList(); renderCurrent();
    $("message").textContent="文字、風格或問題標記已修改；先前決定已撤銷，草稿保留並回到待審核。";
  }
}
function invalidateChangedSavedChecks(saveDraft=true) {
  const saved=getDecision(selectedId); if (!saved) return;
  const checks={correct_speaker:$("checkSpeaker").checked,complete_audio:$("checkComplete").checked,corrected_text:$("checkText").checked};
  const prior=saved.approval_checks||{};
  if (checks.correct_speaker===!!prior.correct_speaker && checks.complete_audio===!!prior.complete_audio && checks.corrected_text===!!prior.corrected_text) return;
  try { localStorage.removeItem(storageKey(selectedId)); } catch (_) {}
  $("checkSpeaker").checked=false; $("checkComplete").checked=false; $("checkText").checked=false;
  if (saveDraft) savePendingDraft();
  const status=$("statusBadge"); status.textContent=visibleStatus("pending"); status.className="status pending";
  updateSummary(); renderList(); updateApproveButton();
  $("message").textContent="確認項目已變更；先前決定已撤銷。草稿已保留，請重新完成所有確認。";
}
function recordDecision(decision) {
  if (!storageReady) { $("message").textContent=STORAGE_UNAVAILABLE_MESSAGE; return; }
  if (!selectedId) { $("message").textContent="請先選取片段再記錄決定。"; return; }
  const checks={correct_speaker:$("checkSpeaker").checked,complete_audio:$("checkComplete").checked,corrected_text:$("checkText").checked};
  const reviewedText=$("transcript").value;
  const issues=selectedProblemIssues();
  const problemNotes=$("problemNotes").value.trim();
  if (decision==="approved") {
    const missing=[];
    if (issues.length) missing.push("清除字頭／字尾被切的問題標記");
    if (!checks.correct_speaker||!checks.complete_audio||!checks.corrected_text) missing.push("完成三項確認");
    if (!reviewedText.trim()) missing.push("提供非空白的校對文字");
    if (missing.length) { $("message").textContent="尚未核准：請"+missing.join("、")+"。"; return; }
  }
  const clip=byId.get(selectedId);
  const value={clip_id:selectedId,clip_audio_sha256:clip.clip_audio_sha256,decision,reviewed_text:reviewedText,original_draft_text:clip.draft_text,style_caption:$("styleCaption").value.trim(),problem_issues:issues,problem_notes:problemNotes,approval_checks:checks,recorded_at:new Date().toISOString()};
  try { localStorage.setItem(storageKey(selectedId),JSON.stringify(value)); }
  catch (_) { $("message").textContent="無法將此決定儲存至瀏覽器本機空間。"; return; }
  try { localStorage.removeItem(draftStorageKey(selectedId)); } catch (_) {}
  $("message").textContent=`已儲存人工決定：${visibleStatus(decision)}。`;
  renderList(); renderCurrent(); updateSummary();
}
function updateSummary() {
  const counts={pending:0,approved:0,needs_review:0,rejected:0};
  for (const clip of CLIPS) counts[currentStatus(clip.clip_id)]++;
  $("summary").textContent=`${CLIPS.length} 個片段 · ${counts.approved} 已核准 · ${counts.needs_review} 需再確認 · ${counts.rejected} 已排除 · ${counts.pending} 待審核 · 資料指紋 ${DATASET_FINGERPRINT.slice(0,12)}`;
}
function move(delta) { const at=visibleIds.indexOf(selectedId), next=at+delta; if (next>=0&&next<visibleIds.length){selectedId=visibleIds[next];renderList();renderCurrent();} }
function toggleAllChecks() {
  if (!selectedId || !storageReady) return;
  const checks=[$("checkSpeaker"),$("checkComplete"),$("checkText")];
  const target=!checks.every(box=>box.checked);
  for (const box of checks) box.checked=target;
  invalidateChangedSavedChecks(false);
  // A changed saved decision clears checks during invalidation; restore the explicit bulk target.
  for (const box of checks) box.checked=target;
  savePendingDraft(); updateApproveButton();
  $("message").textContent=target?"已勾選三項確認；尚未核准。":"已清除三項確認；尚未核准。";
}
function toggleIssue(id) {
  if (!selectedId || !storageReady) return;
  const hadDecision=!!getDecision(selectedId), box=$(id), name=id==="issueClippedStart"?"字頭被切":"字尾被切";
  box.checked=!box.checked; box.dispatchEvent(new Event("change",{bubbles:true}));
  $("message").textContent=name+(box.checked?"：已勾選；請選擇「需要再確認」。":"：已取消勾選。")+(hadDecision?" 先前決定已撤銷，請重新確認。":"");
}
function playPause() {
  if (!byId.has(selectedId)) { $("message").textContent="請先選取片段再播放。"; return; }
  const player=$("player"), failed=()=>{$("message").textContent="無法播放音訊，請確認 WAV 檔案仍在原路徑，或使用播放器重試。";};
  try {
    if (player.paused) { const result=player.play(); if (result?.catch) result.catch(failed); }
    else player.pause();
  } catch (_) { failed(); }
}
function isTypingTarget(target) {
  return target instanceof Element && (target.isContentEditable || !!target.closest('textarea,select,input:not([type=checkbox]):not([type=radio]):not([type=button]):not([type=submit]),[role="textbox"]'));
}
function handleShortcut(event) {
  if (event.defaultPrevented || event.isComposing || event.keyCode===229 || event.ctrlKey || event.altKey || event.metaKey || event.shiftKey) return;
  // Use the produced key so a NumLock-off numpad arrow is never treated as a digit.
  const key=event.key===" "?"Space":event.key;
  if (isTypingTarget(event.target)) return;
  const recognized=["1","Space","2","3","4","5","ArrowLeft","6","ArrowRight","7","8"].includes(key);
  if (!recognized) return;
  event.preventDefault();
  if (event.repeat) return;
  if (key==="1"||key==="Space") { playPause(); return; }
  if (key==="2") { toggleAllChecks(); return; }
  if (key==="3") { toggleIssue("issueClippedStart"); return; }
  if (key==="4") { toggleIssue("issueClippedEnd"); return; }
  if (key==="5"||key==="ArrowLeft") { move(-1); return; }
  if (key==="6"||key==="ArrowRight") { move(1); return; }
  if (key==="7") { recordDecision("approved"); return; }
  recordDecision("needs_review");
}
function exportReview() {
  const decisions=[];
  for (const clip of CLIPS) { const d=getDecision(clip.clip_id); if (d) decisions.push(d); }
  const result={schema:"nino-human-review-v1",dataset_fingerprint:DATASET_FINGERPRINT,manifest_path:DATASET_MANIFEST_PATH,exported_at:new Date().toISOString(),decisions};
  const blob=new Blob([JSON.stringify(result,null,2)+"\n"],{type:"application/json;charset=utf-8"});
  const url=URL.createObjectURL(blob), anchor=document.createElement("a"); anchor.href=url; anchor.download=`dataset-review-${DATASET_FINGERPRINT.slice(0,12)}.json`; anchor.click(); setTimeout(()=>URL.revokeObjectURL(url),1000);
  $("message").textContent=`已匯出 ${decisions.length} 筆實際人工決定；待審核片段不會匯出。`;
}
const DATASET_MANIFEST_PATH=__MANIFEST_PATH__;
for (const id of ["sourceFilter"]) {
  const select=$(id), names=[...new Set(CLIPS.map(c=>c.source_filename).filter(Boolean))].sort();
  for (const name of names) { const option=document.createElement("option"); option.value=name; option.textContent=name; select.append(option); }
}
const groupSelect=$("groupFilter");
const groupLabels={priority_review:"優先聽校候選",needs_attention:"需再確認",short_review:"短片段"};
const groupValues=[...new Set(CLIPS.map(c=>c.organization_tier).filter(Boolean))].sort();
for (const value of groupValues) { const option=document.createElement("option"); option.value=value; option.textContent=groupLabels[value]||value; groupSelect.append(option); }
for (const id of ["sourceFilter","groupFilter","flagFilter","statusFilter","sortBy"]) $(id).addEventListener("change",refreshVisible);
$("previousButton").addEventListener("click",()=>move(-1)); $("nextButton").addEventListener("click",()=>move(1));
$("playPauseButton").addEventListener("click",playPause); $("toggleChecksButton").addEventListener("click",toggleAllChecks);
window.addEventListener("keydown",handleShortcut);
$("approveButton").addEventListener("click",()=>recordDecision("approved")); $("needsReviewButton").addEventListener("click",()=>recordDecision("needs_review")); $("rejectButton").addEventListener("click",()=>recordDecision("rejected"));
$("exportButton").addEventListener("click",exportReview);
$("clearButton").addEventListener("click",()=>{if(!selectedId||!storageReady)return;localStorage.removeItem(storageKey(selectedId));$("checkSpeaker").checked=false;$("checkComplete").checked=false;$("checkText").checked=false;savePendingDraft();$("message").textContent="已清除決定；片段回到待審核。";refreshVisible();});
for (const id of ["transcript","styleCaption","problemNotes"]) $(id).addEventListener("input",()=>{invalidateChangedSavedDecision();savePendingDraft();updateApproveButton();});
for (const id of ["issueClippedStart","issueClippedEnd"]) $(id).addEventListener("change",()=>{invalidateChangedSavedDecision();savePendingDraft();updateApproveButton();});
for (const id of ["checkSpeaker","checkComplete","checkText"]) $(id).addEventListener("change",()=>{invalidateChangedSavedChecks();savePendingDraft();updateApproveButton();});
refreshVisible();
</script>
</body>
</html>
'''


def build_html(manifest_path: Path, output_path: Path, organization_path: Path | None = None) -> int:
    rows, fingerprint = prepare_records(manifest_path, output_path, organization_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(rows, ensure_ascii=True, separators=(",", ":"))
    data = data.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    manifest_label = str(manifest_path.resolve()).replace("\\", "/")
    page = HTML_TEMPLATE.replace("__FINGERPRINT__", json.dumps(fingerprint))
    page = page.replace("__MANIFEST_PATH__", json.dumps(manifest_label))
    page = page.replace("__CLIPS__", data)
    output_path.write_text(page, encoding="utf-8")
    print(f"Built offline review page for {len(rows)} clips: {output_path}")
    print(f"Manifest fingerprint: {fingerprint}")
    print("WAV data is not embedded; playback uses relative local audio URLs.")
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--organization-summary", type=Path, default=DEFAULT_ORGANIZATION)
    args = parser.parse_args()
    manifest = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    output = args.output if args.output.is_absolute() else ROOT / args.output
    organization = args.organization_summary if args.organization_summary.is_absolute() else ROOT / args.organization_summary
    build_html(manifest, output, organization)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
