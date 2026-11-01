# 工具地圖與環境

| 工作 | 入口 | 依賴與限制 |
| --- | --- | --- |
| 收件與比較規劃 | `ttsbench.py` | Python 標準函式庫 |
| 規劃、抽取 | `tools/plan_nino_dataset.py`、`extract_nino_dataset.py` | 抽取需 FFmpeg；自己的來源 inventory 與 readiness |
| ASR／alignment | `check_nino_asr.py`、`align_nino_sources.py` | 選用 faster-whisper／alignment 依賴 |
| 離線審核／匯入 | `build_dataset_review.py`、`export_reviewed_dataset.py`、`import_review_batch.py` | fingerprint 與人工確認 gate |
| 語氣 UI／本機保存 | `build_nino_style_review.py`、`runtime/nino_style_review_server.py` | 標準函式庫；保存服務只綁 loopback |
| caption-only manifest | `prepare_style_training.py` | 明確輸入，先 `--dry-run`，不覆寫基準 |
| 訓練包／latent／訓練 | `prepare_training_package.py`、`runtime/prepare_irodori_latents.py`、`runtime/train_irodori_v2.py` | 真實資料、固定 codec／模型及選用 ML 環境 |
| Irodori CPU／GPU 推論 | `runtime/irodori_local.py`、`irodori_gpu.py`、`irodori_batch.py` | 選擇相符環境；本機資產自行提供 |
| Qwen 推論 | `runtime/qwen_local.py`、`qwen_clone.py` | clone 另需有權使用的 reference |
| 資源 gate／上游清單 | `lab-gpu-v4/runner/vram_gate.py`、`capture_upstream_inventory.py` | Linux／GPU 工具；不會補回私人實驗結果 |
| 本機 archive 復原 | `tools/restore_archive.py` | archive 未隨公開版提供；需自行從私人備份放回 |

其餘保留工具先讀 docstring 與 `--help`。一些工具仍有原版本名稱、固定資料 identity 或特定檔案 schema，不保證適用任意 manifest。

基本 metadata 與審核工具建議 Python 3.11+。先建立專案 venv；抽取、ASR、alignment、推論與訓練分別準備選用依賴，不混裝所有模型。

```powershell
python -m venv .venv
powershell -ExecutionPolicy Bypass -File tools/bootstrap_irodori.ps1
```

bootstrap 只在明確執行時取得固定上游，不啟動模型。依釘選上游的 `pyproject.toml`／`uv.lock` 選 CPU 或 CUDA extra。`runtime/*.txt` 是各歷史環境的依賴紀錄，並非適用所有平台的單一 requirements。公開版未驗證新機器的完整 ML 安裝或效能。
