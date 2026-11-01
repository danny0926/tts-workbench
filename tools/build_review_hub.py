"""Build a read-only status hub from an explicit reviewed dataset version."""
from __future__ import annotations

import argparse
import datetime as dt
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs/dataset"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v3", help="review/export version to show (for example v3)")
    args = parser.parse_args()
    version = args.version
    reviews = read_json(DATA / f"reviews/{version}.json")
    decisions = reviews["decisions"]
    approved = {r["clip_id"] for r in decisions if r.get("decision") == "approved"}
    pending = {r["clip_id"] for r in decisions if r.get("decision") == "needs_review"}
    rejected = {r["clip_id"] for r in decisions if r.get("decision") == "rejected"}
    if len(approved | pending | rejected) != len(decisions) or (approved & pending) or (approved & rejected) or (pending & rejected):
        raise ValueError("Review decisions are not a unique partition")

    prior_summary = read_json(DATA / "v3/summary.json") if version == "v3" else None
    if version == "v3" and (len(approved), len(pending), len(rejected)) != (377, 3, 4):
        raise ValueError("v3 review counts do not match the imported source decisions")
    if version == "v4" and (len(approved), len(pending), len(rejected), len(decisions)) != (380, 0, 4, 384):
        raise ValueError("v4 review counts do not match the completed import")
    reviewed_href = f"reviewed-{version}/reviewed_audio_manifest.jsonl"
    if not (DATA / reviewed_href).is_file():
        raise FileNotFoundError(DATA / reviewed_href)
    training_dir = DATA / f"training-{version}"
    training = read_json(training_dir / "summary.json")
    prep = read_json(training_dir / "irodori/preparation-status.json")
    cache_rows = (training_dir / "irodori/cache-index.jsonl").read_text(encoding="utf-8").splitlines() if (training_dir / "irodori/cache-index.jsonl").is_file() else []
    reused = len([line for line in cache_rows if line.strip()])
    prepared_count = int(prep.get("prepared_audio_clips", training["human_approved_clips"]))
    completed = int(prep.get("completed_latents", reused))
    missing = max(0, prepared_count - completed)
    newly_encoded = max(0, completed - reused)

    candidate = DATA / "remaining-v5"
    candidate_rows = []
    candidate_manifest = candidate / "candidate_manifest.jsonl"
    if pending and candidate_manifest.is_file():
        candidate_rows = [json.loads(line) for line in candidate_manifest.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        candidate_ids = {row["clip_id"] for row in candidate_rows}
        if pending and candidate_ids != pending:
            raise ValueError(f"v5 candidate set differs from current pending decisions: {len(candidate_ids)} vs {len(pending)}")
    has_review = bool(pending and (candidate / "review.html").is_file())
    has_compare = bool(pending and (candidate / "compare.html").is_file())
    pending_detail = ("資料審核完成，380核准、4排除、0待審" if version == "v4" and not pending else
                      (f'<a href="remaining-v5/review.html">開啟 {len(candidate_rows)} 個待審候選</a>' if has_review and len(candidate_rows) == len(pending)
                       else f"尚有 {len(pending)} 個待審候選；審核頁尚未就緒。"))
    total_duration = float(training["total_duration_seconds"] if version == "v4" else
                           prior_summary.get("export_duration_seconds", training["total_duration_seconds"]))
    minutes, seconds = divmod(round(total_duration), 60)
    date = dt.date.today().isoformat()
    train_count, val_count = training["training_clips"], training["validation_clips"]
    training_started = bool(prep.get("training_started", False))
    if completed == prepared_count and completed > 0:
        latent_status = (f"Latent 編碼完成 {completed}/{prepared_count} 筆（重用 {reused} 筆、此次新編碼 {newly_encoded} 筆）。"
                        f"訓練 {train_count} 筆、驗證 {val_count} 筆。")
    else:
        latent_status = f"Latent 快取可重用 {reused}/{prepared_count} 筆；尚缺 {prep.get('missing_latents', missing)} 筆。"
    training_notice = "訓練已啟動；進度以 server 紀錄為準。" if training_started else "尚未開始訓練。"
    run_links = ""
    if training_started:
        status_href = html.escape("../../" + prep["training_status_snapshot"])
        log_href = html.escape("../../" + prep["training_log_snapshot"])
        synced_at = html.escape(prep.get("last_status_sync_at_utc", ""))
        run_links = (f' <a href="{status_href}">本機狀態快照</a> · '
                     f'<a href="{log_href}">本機執行記錄</a><br>最近同步 {synced_at} UTC')
    training_completed = (prep.get("training_run_status") == "completed"
                          and int(prep.get("completed_optimizer_steps", 0)) >= int(prep.get("training_max_steps", 500)))
    if training_completed:
        best_adapter = html.escape("../../" + prep["best_adapter_snapshot"])
        final_adapter = html.escape("../../" + prep["final_adapter_snapshot"])
        training_notice = f"訓練已完成：{prep['completed_optimizer_steps']}/{prep.get('training_max_steps', 500)} optimizer steps。"
        run_links += (f'<br><a href="{best_adapter}">最佳驗證 LoRA（step 450）</a> · '
                      f'<a href="{final_adapter}">最終 LoRA（step 500）</a>')
    elif prep.get("training_run_status") == "failed":
        training_notice = "訓練失敗；請查看 server status 與 log。"
    elif training_started:
        training_notice = "訓練已啟動；進度以 server 紀錄為準。"
    pending_section = '<p class="status">資料審核完成，380核准、4排除、0待審</p>' if version == "v4" and not pending else ""
    if pending:
        compare_link = '<p><a href="remaining-v5/compare.html">開啟音訊比較頁</a></p>' if has_compare else ""
        pending_section = f'<section class="card section"><h2>待審候選</h2><p>{pending_detail}</p>{compare_link}</section>'
    page = f'''<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>音訊資料審核與訓練準備</title>
<style>
:root{{color-scheme:light;--ink:#17232d;--muted:#566675;--line:#dce4e9;--paper:#f4f7f8;--card:#fff;--blue:#126b82;--pale:#e8f3f6}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif}}
main{{max-width:960px;margin:auto;padding:clamp(20px,5vw,52px)}}h1{{font-size:clamp(1.8rem,4vw,2.6rem);line-height:1.2;margin:8px 0}}h2{{font-size:1.2rem;margin:0 0 12px}}
.muted{{color:var(--muted)}}.stats{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:26px 0}}.card{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px}}
.stats strong{{font-size:2rem;display:block}}.pending{{background:#fffdf7;border-color:#d4bd84}}.section{{margin-top:20px}}a{{color:var(--blue);font-weight:650}}.status{{padding:14px 18px;border-left:4px solid #c29227;background:#fffdf7}}
@media(max-width:650px){{.stats{{grid-template-columns:1fr}}}}
</style></head><body><main>
<p class="muted">資料集版本 {html.escape(version)} · 更新日期 {date}</p><h1>人工審核與訓練準備</h1>
<p class="muted">以下狀態取自目前版本的審核決策與實際產物。</p>
<section class="stats" aria-label="審核統計"><div class="card"><strong>{len(approved)}</strong><span>人工核准</span></div>
<div class="card pending"><strong>{len(pending)}</strong><span>待審</span></div><div class="card"><strong>{len(rejected)}</strong><span>拒絕</span></div></section>
{pending_section}
<section class="card section"><h2>訓練資料封裝</h2>
<p>音訊與文字已準備 {training['human_approved_clips']} 筆；來源分組切分為訓練 {train_count} 筆、驗證 {val_count} 筆。</p>
<p>總音訊時長約 {minutes} 分 {seconds:02d} 秒。文字保留人工審核內容，風格描述維持空白。</p>
<p class="status">{latent_status}<br>{training_notice}{run_links}</p>
</section>
<section class="section"><a href="{html.escape(reviewed_href)}">核准音訊匯出清單</a> · <a href="training-{html.escape(version)}/summary.json">訓練封裝摘要</a></section>
</main></body></html>'''
    output = DATA / "review-hub.html"
    output.write_text(page, encoding="utf-8", newline="\n")
    print(json.dumps({"output": output.relative_to(ROOT).as_posix(), "version": version, "approved": len(approved), "needs_review": len(pending),
                      "rejected": len(rejected), "candidate_page_ready": has_review, "reused_latents": reused,
                      "completed_latents": completed, "newly_encoded_latents": newly_encoded,
                      "missing_latents": missing, "resource_limited": prep.get("resource_limited"),
                      "training_started": prep.get("training_started", False),
                      "training_run_status": prep.get("training_run_status"),
                      "training_completed": training_completed}, ensure_ascii=False))


if __name__ == "__main__":
    main()
