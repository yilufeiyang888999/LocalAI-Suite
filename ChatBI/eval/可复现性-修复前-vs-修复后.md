# 逐题比对 · 修复前-vs-修复后

| — | 甲 | 乙 |
|---|---|---|
| 文件 | `盲测结果-60题-微调后v3-Q4KM-精简.json` | `盲测结果-60题-修复后-v3.json` |
| 题数 | 60 | 60 |
| 标注 | 微调后v3-Q4KM-精简 | 修复后-v3 |
| 模型（调用方声明） | DS-R1-7B-Q4KM+lora-v3 | DS-R1-7B-Q4KM+lora-v3 |
| 引擎自报 | D:\workspace\10-LocalAI-Installer\models\DS-R1-Qwen-7B-GGUF\base-Q4_K_M.gguf | D:\workspace\10-LocalAI-Installer\models\DS-R1-Qwen-7B-GGUF\base-Q4_K_M.gguf |
| 提示词指纹 | `ecb40bc153da1b81` | `4e5cf5932ca194dc` |

🔴 **提示词指纹不同 ⇒ 换了提示词，不是同一口径**：`ecb40bc153da1b81` vs `4e5cf5932ca194dc`。数字差异不能归因于被测量。

结论：**不可比，已中止**（不输出「差异率」以免被当成可解释的数字）。
