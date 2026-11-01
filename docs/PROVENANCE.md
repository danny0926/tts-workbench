# 上游與權利範圍

這裡包含自行編寫的資料處理、審核 UI、推論整合與訓練 wrapper。第三方 vendor、模型、媒體與真實研究資料未打包。

| 依賴 | 固定來源或用途 |
| --- | --- |
| [Aratako/Irodori-TTS](https://github.com/Aratako/Irodori-TTS) | bootstrap 固定 commit `59be938c99d48b5653ca5aa3fdd0cddaa56940e5` |
| [Irodori-TTS-v4.1-Small](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small) | base revision `2b28324dc263ed5e6638b3cf3dd94c82ead07b4b` |
| [Semantic-DACVAE-Japanese-32dim](https://huggingface.co/Aratako/Semantic-DACVAE-Japanese-32dim) | codec revision `47376ee24834d7a05a48ebabfe3cde29b3c5e214` |
| [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) | 選用的預設聲線與 reference clone |
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | 未經人工確認的 ASR 線索 |
| [SilentCipher](https://github.com/SesameAILabs/silentcipher) | 依固定 Irodori 上游的官方 watermark 路徑 |

程式依賴與各模型、聲音、角色、素材的權利範圍不同。引用名稱或公開來源並不授予再散布或訓練權；請只提供自己有權使用的輸入。第三方角色與作品權利仍屬各權利人。

`benchmarks/prompts.jsonl` 是自行編寫的測試句，未收錄真實對話或來源逐字稿。本儲存庫沒有替第三方內容或模型附加概括授權；各上游條款分別適用。
