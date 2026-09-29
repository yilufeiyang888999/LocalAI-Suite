# Qwen3.8-27B 加载失败：双引擎对照记录

> 场景：Ubuntu（P100 台式机）启动 `Qwen3.8-27B-UD-Q3_K_XL.gguf` 报
> `missing tensor 'blk.64.ssm_conv1d.weight'`。
> 结论：**引擎太老**，与模型文件、显存、参数都无关。
> 记录时间：2026-09-24　文件 SHA256：`8C2A45FF85E7...A67A5E`

---

## 一、失败侧：Ubuntu，build 8211

```
build: 8211 (a0ed91a44) with GNU 10.5.0 for Linux x86_64
llama_model_loader: - kv 0:  general.architecture str = qwen35
llama_model_loader: - kv 17: qwen35.block_count u32 = 65
llama_model_loader: - kv 34: qwen35.full_attention_interval u32 = 4
llama_model_loader: - kv 28: qwen35.nextn_predict_layers u32 = 1
...
print_info: n_layer = 65
...
load_tensors: loading model tensors, this can take a while...
llama_model_load: error loading model: missing tensor 'blk.64.ssm_conv1d.weight'
llama_model_load_from_file_impl: failed to load model
```

元数据全部解析正确（架构、层数、SSM 参数都认出来了），**卡在加载张量**这一步。
说明「认识这个架构」和「能装出这个模型」是两件事。

## 二、成功侧：笔记本，build 10853（同一份文件）

```
build      : b10853-9dcf84e5a
...
0.01.109.428 W model has unused tensor blk.64.attn_norm.weight (size = 20480 bytes) -- ignoring
0.01.109.459 W model has unused tensor blk.64.attn_q.weight (size = 51609600 bytes) -- ignoring
0.01.109.466 W model has unused tensor blk.64.attn_k.weight (size = 5570560 bytes) -- ignoring
0.01.109.472 W model has unused tensor blk.64.attn_v.weight (size = 5570560 bytes) -- ignoring
0.01.109.496 W model has unused tensor blk.64.attn_output.weight (size = 25804800 bytes) -- ignoring
0.01.109.509 W model has unused tensor blk.64.attn_k_norm.weight (size = 1024 bytes) -- ignoring
0.01.109.516 W model has unused tensor blk.64.ffn_gate.weight (size = 73113600 bytes) -- ignoring
0.01.109.523 W model has unused tensor blk.64.ffn_down.weight (size = 73113600 bytes) -- ignoring
0.01.109.529 W model has unused tensor blk.64.ffn_up.weight (size = 73113600 bytes) -- ignoring
0.01.109.537 W model has unused tensor blk.64.nextn.eh_proj.weight (size = 43008000 bytes) -- ignoring
0.01.109.544 W model has unused tensor blk.64.nextn.enorm.weight (size = 20480 bytes) -- ignoring
0.01.109.551 W model has unused tensor blk.64.nextn.hnorm.weight (size = 20480 bytes) -- ignoring
0.01.109.570 W model has unused tensor blk.64.nextn.shared_head_norm.weight (size = 20480 bytes) -- ignoring
...
0.08.085.100 I srv  llama_server: model loaded
0.08.085.108 I srv  llama_server: listening on http://127.0.0.1:8199
```

`model loaded` + `listening`，8 秒加载完成（`-ngl 0` 纯 CPU，mmap）。

## 三、根因

`blk.64` 是 **MTP 块**（多 token 预测头），不是 SSM 层。上面的日志里它**整层 15 个张量
全部被判为 unused 并忽略** —— 包括 attn_*、ffn_*、nextn.*。

厂商对层数的声明不一致，但描述的是同一个模型：

| 来源 | block_count | MTP 头位置 |
|---|---|---|
| Unsloth / bartowski / lmstudio | 65 | 内嵌在 `blk.64`（`nextn_predict_layers = 1`） |
| ggml-org | 64 | 单独的 `mtp-*.gguf` 文件 |

llama.cpp 的调和方式（`src/llama-hparams.cpp`）：`n_layer() = n_layer_all - n_layer_nextn`，
判层类型的条件是 `is_recr_impl[i] = (i < n_layer()) && ((i + 1) % interval != 0)`。

**老引擎缺了 `i < n_layer()` 这道保护**，于是对 `i = 64` 也去套 SSM 的模板找
`ssm_conv1d.weight`，而文件里 `blk.64` 是全注意力 + MTP 头结构 → missing tensor → 退出。

上游修复：commit **`82dbc4f01`「load MTP tensors only if they are really used」**。

## 四、连带结论（两条要改口径的）

### 4.1 参与推理的是 16 层全注意力，不是 17 层

- 主干 = `block_count - nextn` = 65 − 1 = **64 层**（`blk.0` ~ `blk.63`）
- 全注意力层 = `(i + 1) % 4 == 0`，即 `blk.3, 7, 11, ..., 63` = **16 层**
- `blk.64` 整层不参与推理

KV 真值（此前按「17 层」算，高估 6.25%）：

| KV 档 | 每 token | 算法 |
|---|---|---|
| f16 | **64 KiB** | 16 层 × 2（K+V） × 4 头 × 256 维 × 2 字节 |
| q8_0 | **34 KiB** | 同上 × 1.0625 |
| q4_0 | **18 KiB** | 同上 × 0.5625 |

### 4.2 被忽略的权重约 0.327 GiB

`blk.64` 的 15 个张量合计 **351,008,768 字节 = 0.327 GiB**，不加载。
所以实际权重 ≈ 12.244 − 0.327 = **11.92 GiB**，不是 12.24 GiB。

P100 16 GiB 重算：

```
16.00  标称
-0.40  预留
-11.92  权重（已扣 MTP 块）
-1.59  计算缓冲
= 2.09 GiB 留给 KV
  → q8_0（34 KiB/token）约 64K token
  → f16（64 KiB/token）约 34K token
```

原来按 17 层算出的「q8_0 上限 48K」偏保守；**脚本现用的 49152 仍是安全的**（有余量）。

## 五、附带发现：`llama-server --version` 打到 stderr

```
$ llama-server --version
（stdout 为空）
stderr> version: 0.4.0-dev (build 10853, commit 9dcf84e5a)
stderr> built with Clang 20.1.8 for Windows x86_64
```

写脚本时若用 `2>/dev/null`，版本就永远是「无法读取」。要用 `2>&1` 再抓。
