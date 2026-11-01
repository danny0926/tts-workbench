# 代理工作規則

先讀 README、docs/WORKFLOW.md 與 docs/TOOLS.md，再讀本次涉及的程式。桌寵在獨立的 nino-desktop-pet 儲存庫。

- 優先完成設計與功能；相關修改完成後做最小必要驗證，不重跑仍適用的檢查。
- 派工遵循使用者全域規則，列出 Target、Change、Constraints、Ownership、Observable acceptance；不同 worker 不同時修改同一檔案。
- 這是程式與範例的公開儲存庫。不得提交真實研究逐字稿、人工備註、試聽回饋、來源索引、資料集、對話、記憶、螢幕畫面、憑證、模型、音訊或衍生 tensor。
- 原始資料與人工決定只讀。裁切、修補、訓練、匯入與分析寫入新版本，保留 ID、來源區間、SHA-256、父 fingerprint 與排除紀錄。
- ASR、字幕線索、機器 flags、上下文提案與人工聽校分開。只有實際聽音並明確 confirmed 的標註可進 caption；不得代勾已聽或代核准。
- 新素材記錄來源、發布者、取得日期、授權與存取狀態；原文或來源內容未取得時如實註明。
- 不把命令句、角色印象、音量或機器推測直接當成確認情緒；attitude、effort、delivery 分別判斷。
- 本機與伺服器只用專案隔離的環境、cache 與輸出目錄。不得終止其他工作；沿用 VRAM gate 與互斥鎖。
- 訓練必須另有具體授權；保留 `--launch-training` 的明確啟動 gate。訓練與 validation 按來源 SHA 分組隔離。
- 不宣稱即時性、聲線或情緒改善，除非有相符量測與人工聽感。
- 驗證以 Python AST、修改的 JSON／JSONL、CLI 入口及實際改動契約為主；不為驗證安裝大型模型或執行 ML。文件與程式以 UTF-8 保存。
