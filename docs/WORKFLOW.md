# 操作流程

## 來源與資料

先為自己有權使用的素材建立來源 ledger，記錄 URL、發布者、取得日期、可取得內容與權利範圍。真實 ledger、音訊、逐字稿與人工備註保留本機，不加入 Git。來源長度及候選數量不代表可訓練時長。

`ttsbench.py check-audio` 檢查 PCM WAV、UTF-8 逐字稿與基本訊號；`plan`／`matrix` 只規劃，不推論。

```powershell
python ttsbench.py check-audio audio/reference/ja/sample.wav --transcript audio/reference/ja/sample.txt --language ja --speaker speaker-01
python ttsbench.py plan --input-manifest config/intake.example.json
```

## 逐字稿審核

候選規劃、抽取與機器線索依序可使用 `plan_nino_dataset.py`、`extract_nino_dataset.py`、`check_nino_asr.py`、`merge_dataset_asr.py`、`organize_nino_candidates.py`。各工具接受自己的輸入／輸出契約，先查看 `--help`。ASR 是未校對線索，不能代替人工確認。

```powershell
python tools/build_dataset_review.py --manifest outputs/dataset/working/manifest.jsonl --output outputs/dataset/working/review.html
```

逐字稿頁：`1`／Space 播放，`2` 確認欄位全選／清除，`3` 字頭問題，`4` 字尾問題，`5`／左鍵前一筆，`6`／右鍵下一筆，`7` 核准，`8` 需要確認。輸入文字或 IME 組字時停用快捷鍵。需實際聽音並確認 speaker、完整性與逐字稿；不能由機器代核准。

離線頁的內容需匯出人工審核 JSON 才有可攜備份。`export_reviewed_dataset.py` 只匯出明確核准且 identity 相符的項目；`import_review_batch.py` 將相符的部分批次匯入新版本，先用 `--dry-run`。保留排除決定及父 fingerprint。

## 語氣、caption 與訓練

`build_nino_style_review.py` 依自己的 manifest 與標籤 config 建頁；`nino_style_review_server.py` 提供只綁 127.0.0.1 的本機保存服務。標註須分別選 attitude、effort、delivery，只有實際聽音並 confirmed 的決定可進 caption。混合或不確定的片段保留 needs_review 與備註。

`prepare_style_training.py` 驗證 confirmed／heard_audio、文字、音訊 hash、split 與父 lineage，只產生新的 caption-only manifest，保留其餘欄位。輸出須與基準 manifest 同目錄，避免破壞 latent 相對路徑。`summarize_nino_style_review.py` 仍有原版本的固定集合契約，不能直接套用任意資料集。

`prepare_training_package.py` 檢查實際核准的 WAV/TXT；`prepare_irodori_latents.py` 需要另備 codec 與 ML 環境。`train_irodori_v2.py` 預設驗證，只有明確加 `--launch-training` 才啟動訓練。不同原始來源 SHA 的 train／validation 分組必須隔離。

比較時固定模型、checkpoint step、reference、seed、sampling、backend 與 dtype。數值 loss、訓練完成及人工聽感是不同證據，不能互相取代。
