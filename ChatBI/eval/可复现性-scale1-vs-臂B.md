# 逐题比对 · scale1-vs-臂B

| — | 甲 | 乙 |
|---|---|---|
| 文件 | `盲测结果-60题-scale1-同进程.json` | `盲测结果-60题-微调后v3-Q4KM-精简.json` |
| 题数 | 60 | 60 |
| 标注 | scale1-同进程 | 微调后v3-Q4KM-精简 |
| 模型（调用方声明） | DS-R1-7B-Q4KM+lora-v3 | DS-R1-7B-Q4KM+lora-v3 |
| 引擎自报 | D:\workspace\10-LocalAI-Installer\models\DS-R1-Qwen-7B-GGUF\base-Q4_K_M.gguf | D:\workspace\10-LocalAI-Installer\models\DS-R1-Qwen-7B-GGUF\base-Q4_K_M.gguf |
| 提示词指纹 | `ecb40bc153da1b81` | `ecb40bc153da1b81` |

## 结论

✅ **60 题逐题一致**（比对字段：意图 / 类型 / SQL / 解读 / 结果 / 分类）—— 该配置下输出**可复现**。
