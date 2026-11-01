# TTS Workbench

語音資料整理、ASR 交叉核對、人工逐字稿與語氣審核、Irodori 訓練及本機推論工具。桌寵應用另見 [nino-desktop-pet](https://github.com/danny0926/nino-desktop-pet)。

這個公開版包含可重用程式、設定範例與原創測試句。真實逐字稿、人工備註、試聽回饋、研究索引、訓練清單、媒體、權重及含研究資料的舊 Git 歷史均不在此版本。資料綁定的一次性修補、特定實驗 runner 與私人 Colab 產物亦未公開。

## 開始使用

Python 3.11+；基本收件與規劃工具只用標準函式庫，不需安裝模型。

```powershell
python ttsbench.py --help
python ttsbench.py plan --input-manifest config/intake.example.json
python tools/build_dataset_review.py --help
```

`config/intake.example.json` 的音訊與逐字稿路徑是範例，需自行準備有權使用的檔案；`benchmarks/prompts.jsonl` 是原創測試句，並非真實對話或研究逐字稿。此公開版不附帶可直接訓練的資料集。

## 工具與流程

- [操作流程](docs/WORKFLOW.md)：來源、人工審核、匯入、caption、訓練與比較的界線。
- [工具地圖與環境](docs/TOOLS.md)：依工作選擇 CLI 與選用依賴。
- [本機資料](data/README.md)：研究資料與私人資產的保存位置。
- [上游與權利範圍](docs/PROVENANCE.md)：固定版本、第三方依賴及素材要求。
- [代理規則](AGENTS.md)：修改與資料處理契約。

`config/nino-irodori-lora-v2.yaml` 是起始設定，未保證適用任意 GPU。`config/nino-style-review-v*.json` 是單筆標註設計範例，使用時須填入自己的集合 fingerprint 與實際筆數。部分保留工具沿用早期版本名稱及資料契約；先看 `--help`，不可把缺失的私人資料當成已準備完成。

公開版的基本檢查不代表已在新機器完成模型安裝、推論、訓練或人工聽感評估。
