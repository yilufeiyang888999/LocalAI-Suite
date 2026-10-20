# 本地AI模型一键部署助手 开发技术文档

**版本**：v1.3  
**更新日期**：2026-10-20  
**修订说明**：按 0.1.7 实现校准。显存预算改为「GGUF 元数据实算 + 驱动模式分档开销」；Windows 后端改写为「瘦启动器 + ggml DLL 隔离池」的实测结论；速度估算改为按世代与权重分档的 roofline 系数；新增第十章，记录 0.1.1→0.1.7 之间被真机验证过的实现口径（MTP 层扣除、混合注意力 KV、TCC 陷阱、跨架构编译、服务生命周期与日志首行）。前九章保留 2026-06 的设计原貌，**与实现冲突处以第十章为准**。

---

## 一、技术选型

### 1.1 整体选型思路

面向桌面端、轻量、高性能、低打包体积、原生系统交互能力强，优先选择 Tauri 技术栈，推理层基于成熟的 llama.cpp 封装，屏蔽底层硬件差异。

| 层级 | 技术选型 | 选型理由 |
|------|----------|----------|
| 桌面端框架 | Tauri 2.x | 1. 基于 Rust + Web 前端，打包体积小（安装包＜200MB）<br>2. 原生性能，硬件检测、进程管理效率高<br>3. 跨平台支持 Windows/macOS/Linux<br>4. 内存占用远低于 Electron |
| 前端技术栈 | Vue 3 + TypeScript + Vite + vue-i18n | 1. 组件化开发效率高<br>2. 生态成熟，UI 组件丰富<br>3. 构建产物轻量<br>4. 预留国际化能力，首版仅内置中文文案，避免后期为英文等语言重构文案体系 |
| 后端逻辑 | Rust（Tauri 内置） | 1. 内存安全、性能优异<br>2. 原生系统调用能力强<br>3. 方便封装 llama.cpp 与系统硬件检测 |
| 推理引擎 | llama.cpp | 1. 全硬件平台支持，CPU/GPU 均有优化<br>2. GGUF 格式生态最完善<br>3. 启动速度快，资源占用低 |
| 模型数据源 | ModelScope + hf-mirror | 1. 国内访问速度快<br>2. 开源模型覆盖全面<br>3. 提供标准 API 可查询模型元数据 |
| 本地存储 | SQLite | 1. 轻量嵌入式数据库，无需额外服务<br>2. 存储模型元数据、下载任务、配置信息 |
| 日志系统 | tracing + tracing-subscriber | 1. Rust 生态标准日志框架<br>2. 支持结构化日志与分级过滤<br>3. 可按模块/级别灵活配置 |

### 1.2 关键依赖库

| 功能模块 | 依赖库 | 说明 |
|---------|--------|------|
| 硬件检测（通用） | `sysinfo` | 跨平台获取 CPU、内存、磁盘、基础显卡信息 |
| 硬件检测（Windows 显卡补充） | `wmi` + `winapi` | 通过 WMI 接口获取 NVIDIA/AMD 显卡显存、驱动版本；`nvidia-smi` 命令补充详细参数 |
| 下载模块 | `reqwest` + `tokio` | 异步 HTTP 客户端，支持 Range 断点续传、多线程分片下载 |
| 哈希校验 | `sha2` + `hex` | `sha2` crate 提供 SHA-256 算法，`hex` 用于哈希值编解码（替代模糊的 `sha256` 库） |
| 进程管理 | `tokio::process` | 异步管理 llama.cpp 服务生命周期 |
| 端口检测 | `portpicker` | 自动分配空闲端口 |
| 数据库 | `sqlx` + `sqlite` | 异步 SQLite 访问，编译期查询检查 |
| 配置管理 | `config` + `serde` | 分层配置读取（默认、用户配置、环境变量） |
| 序列化 | `serde` + `serde_json` | 数据序列化与 JSON 处理 |
| 错误处理 | `thiserror` | 定义结构化错误类型 |
| 日志记录 | `tracing` + `tracing-subscriber` + `tracing-appender` | 分级日志、按天切割、文件滚动 |
| 签名验证 | `ed25519-dalek` | 模型清单文件的 Ed25519 签名验证 |

---

## 二、整体架构设计

采用四层分层架构，自上而下依次为表现层、业务逻辑层、引擎适配层、数据层，各层职责单一，边界清晰。

```
┌───────────────────────────────────────────────────┐
│ 表现层（前端）                                      │
│ 引导向导、模型管理、聊天界面、设置面板、故障提示      │
└───────────────────┬───────────────────────────────┘
                    │ IPC 通信（Tauri Command + Event）
┌───────────────────▼───────────────────────────────┐
│ 业务逻辑层（Rust）                                  │
│ 硬件检测、推荐算法、下载管理、服务管控、故障诊断      │
│ 聊天会话管理、模型管理、设置持久化、卸载清理          │
└───────────────────┬───────────────────────────────┘
                    │ 统一抽象接口
┌───────────────────▼───────────────────────────────┐
│ 引擎适配层                                          │
│ llama.cpp 统一封装、多硬件后端适配、OpenAI 兼容API   │
│ 后端按需下载管理器                                   │
└───────────────────┬───────────────────────────────┘
                    │
┌───────────────────▼───────────────────────────────┐
│ 数据层                                              │
│ SQLite 配置库、本地模型文件、系统缓存、日志文件       │
└───────────────────────────────────────────────────┘
```

### 2.1 各层职责说明

1. **表现层**：负责所有用户交互与界面渲染，不包含业务逻辑，通过 Tauri IPC 调用后端能力。
2. **业务逻辑层**：核心业务实现，包含硬件检测、推荐算法、下载调度、服务管理、故障诊断、聊天会话管理、模型管理、设置持久化、卸载清理等模块。**所有网络访问均优先使用国内源（ModelScope/hf-mirror），确保国内用户网络开箱即用。**
3. **引擎适配层**：封装底层推理引擎，对外提供统一的启动、停止、健康检查接口，屏蔽不同硬件、不同引擎的实现差异；同时对外提供标准 OpenAI 兼容 HTTP 接口。管理后端引擎的按需下载与版本控制，**后端引擎下载地址使用国内镜像源。**
4. **数据层**：持久化存储配置、模型元数据、下载任务、聊天会话，管理本地文件目录结构。**模型清单优先从国内源获取和更新。**

### 2.2 国内网络环境适配

鉴于国内网络环境无法直接访问 HuggingFace 等海外模型托管平台，本项目采用以下策略确保网络开箱即用：

**模型下载源优先级**：
1. **ModelScope**：国内模型托管平台，优先使用
2. **hf-mirror**：HuggingFace 国内镜像
3. **自定义国内镜像**：自建模型托管服务

**清单更新源**：
- 默认从国内镜像源拉取模型清单更新
- 支持用户手动指定清单更新地址

**后端引擎下载源**：
- CUDA/ROCm/SYCL/CPU 后端引擎托管在国内 CDN
- 支持 GitHub Releases 作为备用源（用户可手动开启）

**网络超时与重试策略**：
- 所有网络请求设置合理超时时间（连接超时10秒，读取超时60秒）
- 失败后自动切换备用源
- 指数退避重试（1s → 2s → 4s），最多重试3次

---

## 三、核心模块技术实现

### 3.1 硬件检测模块 [对应需求：3.2]

#### 3.1.1 功能目标

跨平台统一获取硬件信息，输出标准化 `HardwareInfo` 结构体。

#### 3.1.2 统一数据结构

```rust
#[derive(Debug, Clone, Serialize)]
pub struct HardwareInfo {
    pub gpu_list: Vec<GpuInfo>,
    pub gpu_groups: Vec<GpuGroup>,
    pub gpu_name: String,
    pub gpu_vram_gb: f64,
    pub gpu_count: usize,
    pub has_gpu: bool,
    pub homogeneous: bool,
    pub backend: BackendType,
    pub unified_memory: bool,
    pub cpu: CpuInfo,
    pub memory: MemoryInfo,
    pub disk: DiskInfo,
}

#[derive(Debug, Clone, Serialize)]
pub struct GpuInfo {
    pub index: usize,
    pub vendor: GpuVendor, // NVIDIA / AMD / Intel
    pub name: String,
    pub vram_gb: f64,      // 显存大小（GB）
    pub driver_version: String,
    pub bandwidth_gbs: Option<u32>, // 显存带宽（GB/s），通过型号查表获取
}

#[derive(Debug, Clone, Serialize)]
pub struct GpuGroup {
    pub name: String,
    pub vram_each: f64,
    pub count: usize,
    pub indices: Vec<usize>,
    pub vram_total: f64,
}

#[derive(Debug, Clone, Serialize)]
pub enum GpuVendor {
    NVIDIA,
    AMD,
    Intel,
}

#[derive(Debug, Clone, Serialize)]
pub enum BackendType {
    CUDA,
    ROCm,
    Metal,
    CPUx86,
    CPUarm,
}

#[derive(Debug, Clone, Serialize)]
pub struct CpuInfo {
    pub name: String,
    pub cores: usize,
    pub threads: usize,
    pub arch: String, // x86_64 / arm64 / arm
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryInfo {
    pub total_gb: f64,
    pub available_gb: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct DiskInfo {
    pub install_path: String,
    pub available_gb: f64,
}
```

#### 3.1.3 依赖库分工策略

| 信息类型 | 首选库 | 补充方式 | 说明 |
|---------|--------|---------|------|
| CPU 型号/核心/线程 | `sysinfo` | — | 跨平台统一 |
| CPU 架构 | `sysinfo` / `platform` | — | x86_64 / arm64 / arm |
| 内存总容量/可用 | `sysinfo` | — | 跨平台统一 |
| 磁盘可用空间 | `sysinfo` | — | 取安装目标分区 |
| 显卡型号（基础） | `sysinfo` | — | 仅基础识别 |
| 显存容量/驱动版本 | `wmi` (Windows) | `nvidia-smi` 补充 | Windows 平台精准获取 |
| 显存带宽 | 硬编码查表 | GPU型号匹配 | 通过显卡型号查询带宽表 |
| GPU 分组 | 自研逻辑 | — | 将同构GPU分组，支持异构系统 |
| macOS 硬件信息 | `system_profiler` | — | V1.5 阶段实现 |
| Linux 硬件信息 | `/proc` + `lspci` | `nvidia-smi` / `rocm-smi` | V1.5 阶段实现 |

**分工原则**：`sysinfo` 负责跨平台通用信息（CPU、内存、磁盘），保证一致性与性能；`wmi` / `nvidia-smi` 仅用于补充 `sysinfo` 无法覆盖的显卡细节（显存、驱动版本）；显存带宽通过硬编码查表获取，用于速度估算。

#### 3.1.4 显存带宽查表机制

通过显卡型号匹配预定义的带宽表，用于后续模型推理速度估算：

```rust
// NVIDIA GPU 显存带宽表（GB/s）
const NVIDIA_BANDWIDTH: &[(&str, u32)] = &[
    ("RTX 5090", 1792), ("RTX 5080", 960), ("RTX 5070 Ti", 896), ("RTX 5070", 672),
    ("RTX 4090", 1008), ("RTX 4080 Super", 736), ("RTX 4080", 717), ("RTX 4070 Ti Super", 672),
    ("RTX 3090 Ti", 1008), ("RTX 3090", 936), ("RTX 3080 Ti", 912), ("RTX 3080", 760),
    ("RTX 2080 Ti", 616), ("RTX 2080", 448),
    ("GTX 1660 Ti", 288), ("GTX 1660 Super", 336), ("GTX 1660", 192),
];

// AMD GPU 显存带宽表（GB/s）
const AMD_BANDWIDTH: &[(&str, u32)] = &[
    ("RX 7900 XTX", 960), ("RX 7900 XT", 800), ("RX 7800 XT", 624),
    ("RX 6950 XT", 576), ("RX 6900 XT", 512), ("RX 6800 XT", 512),
];

// Apple Silicon 统一内存带宽（GB/s）
const APPLE_BANDWIDTH: &[(&str, u32)] = &[
    ("M1 Ultra", 800), ("M1 Max", 400), ("M1 Pro", 200),
    ("M2 Ultra", 800), ("M2 Max", 400), ("M2 Pro", 200),
    ("M3 Ultra", 800), ("M3 Max", 400), ("M3 Pro", 150),
];
```

#### 3.1.5 GPU分组机制

vLLM 的张量并行仅支持同构GPU，因此需要将相同型号的GPU分组：

```rust
fn group_gpus(gpus: &[GpuInfo]) -> Vec<GpuGroup> {
    // 按（名称，四舍五入显存）分组
    // 返回分组列表，按总显存降序排列
}
```

**异构GPU处理策略**：

- 默认选择总可用显存最大且后端支持最完整的GPU分组作为推荐依据。
- 若最大显存分组不是性能最优分组（如一张大显存旧卡 + 一张小显存新卡），推荐结果页显示"检测到多张不同显卡"，允许用户在高级入口中手动选择运行显卡。
- 推理服务启动时仅使用当前选定分组，不跨不同型号GPU做张量并行，避免稳定性与性能不可控。
- MVP阶段仅自动使用单张 NVIDIA 主显卡；多卡手动选择与同构多卡利用在 V1.0 后评估上线。


#### 3.1.6 分平台实现

- **Windows 平台**：`sysinfo` 获取 CPU/内存/磁盘 → `wmi` crate 调用 WMI 接口获取显卡信息 → `nvidia-smi` 命令补充 NVIDIA 显卡详细参数 → 查表获取显存带宽。
- **macOS 平台（V1.5）**：调用 `system_profiler` 命令解析硬件信息 → 查表获取 Apple Silicon 统一内存带宽。
- **Linux 平台（V1.5）**：读取 `/proc` 系统文件 + `lspci` + `nvidia-smi` / `rocm-smi` 组合获取 → 查表获取显存带宽。

#### 3.1.7 降级策略与性能保障

- **降级策略**：检测失败时不抛出异常，返回部分可用信息（如 `gpu_list` 为空），由业务层引导用户手动选择硬件档位。
- **超时机制**：WMI 查询设置 2 秒超时，超时后降级为 `sysinfo` 基础信息 + 手动选择档位。
- **性能验证**：通过 `tracing` 在 `hardware_detect` 函数入口/出口打点，记录耗时。若持续 >3s，触发告警日志。

#### 3.1.8 异步化处理

硬件检测应在后台线程执行，不阻塞 UI：

- 使用 `tokio::spawn` 将检测任务提交到后台线程
- 检测过程中向前端发送进度事件（如"正在检测显卡信息..."）
- 前端显示进度提示，用户可取消检测（取消后使用缓存数据或手动选择档位）
- 检测完成后通过 Tauri Event 通知前端结果

---

### 3.2 模型推荐算法模块 [对应需求：3.3]

#### 3.2.1 推荐策略

采用「硬件档位过滤 + 四维评分排序」的两级策略，参考 Odysseus Cookbook 的成熟设计，实现精准且效果可控的模型推荐。

**第一步：硬件档位过滤**

根据最大显存筛选出所有可运行的模型，超出显存上限的直接排除。

**显存预算（0.1.7 实现口径）**：

```
可用预算   = 空闲显存 − 模型权重 − 显存开销
最大上下文 = (可用预算 − 滑窗层固定 KV) ÷ 每 token KV 用量
```

- **模型权重**：取 GGUF 文件的**真实字节数 ÷ 1024³**（GiB），不用「参数数 × BPP」推算。清单里的 `file_size_mb` 是十进制 MB，换算必须 `× 1e6 ÷ 1024³`；直接除 1024 会偏高约 4.9%，与 `nvidia-smi` 的可用显存相减后会带进系统偏差。
- **显存开销**：按驱动模式分档，由 `service::显存开销_gib()` 统一给出 —— WDDM（有显示输出）**1.84 GiB**，TCC（Tesla 计算模式，无显示输出）**0.65 GiB**。该值 2026-10-18 由三次实测反推校准，此前的 1.3 偏低。
- **KV 缓存**：读 GGUF 元数据实算，不用「0.000008 × 活跃参数 × ctx」的经验式（该式在混合注意力模型上实测低估 7.8× 到高估 2.1×，现仅作解析失败时的兜底）。
- **滑窗层固定 KV**：滑动窗口层的占用上限是 `min(ctx, 窗口)`，超过窗口后**不随上下文增长**。必须先扣掉这部分固定值，再对每 token 用量做除法，否则算出的上下文会虚高。
- **下载前的粗筛**：模型尚未下载时没有 GGUF 可读，仍用「参数数 × BPP + 估算 KV + 开销」排序（落点 `models.rs::estimate_vram_gb()`，KV 按 8K 上下文估）；**下载完成后一律用真实文件大小与真实元数据覆盖**，清单里的 `vram_required_gb` 只是估算值。

> 层结构如何影响 KV、MTP 层为什么要先扣掉，见 §10.2。

**第二步：适配度等级评估**

| 等级 | 条件 | 说明 |
|------|------|------|
| `perfect` | 推荐显存 ≤ GPU VRAM | 完美适配，模型可完全加载到GPU |
| `good` | GPU VRAM ≥ 需求 × 1.2 | 良好适配，有充足余量 |
| `marginal` | 需求在显存范围内但余量不足 | 边缘适配，可能需要CPU卸载 |
| `too_tight` | 需求超过可用内存 | 无法运行 |

**运行模式**：`gpu`（全GPU）/ `cpu_offload`（部分卸载到CPU）/ `cpu_only`（纯CPU）

**第三步：四维评分排序**

对过滤后的模型进行综合打分，分数越高排序越靠前。评分权重根据使用场景动态调整：

| 使用场景 | 质量权重 | 速度权重 | 适配权重 | 上下文权重 |
|---------|---------|---------|---------|-----------|
| 通用 | 45% | 30% | 15% | 10% |
| 写作 | 50% | 20% | 15% | 15% |
| 代码 | 50% | 20% | 15% | 15% |
| 推理 | 55% | 15% | 15% | 15% |
| 聊天 | 40% | 35% | 15% | 10% |

**各维度评分逻辑**：

- **质量评分**：基于参数规模（越大越高）、架构Bonus（Qwen3.6+9分，Qwen3.5+8分等）、量化惩罚（Q4_K_M-5分等）、中文能力（中文优化模型+2-3分）
- **速度评分**：基于显卡带宽、模型大小、量化类型计算 tok/s，目标速度根据场景调整（通用40 t/s，推理25 t/s）
- **适配评分**：基于显存占用比例（≤50%得60-100分，50%-80%得100分，80%-90%得70分，>90%得50分）
- **上下文评分**：基于模型支持的上下文长度（≥目标得100分，≥目标/2得70分，否则得30分）

**第四步：版本号排序**

同分数模型按版本号排序，新版本优先。通过从模型名称中提取版本号（如 `M2.7` → `2.7`，`Qwen3.6-35B` → `3.6`）实现。

**输出规则**

- 排序第1位为「推荐首选」
- 第2-3位为「备选方案」
- 最多展示3个，避免选择困难

**推荐透明度展示**

为提高用户信任度，在推荐结果中展示推荐理由和评分明细：

**推荐理由（大白话）**：
- "推荐此模型是因为您的显存为8GB，该模型可流畅运行"
- "此模型在写作场景下表现优异，中文能力评分95分"

**评分明细展示**：

| 维度 | 得分 | 权重 | 说明 |
|------|------|------|------|
| 质量 | 85 | 50% | Qwen3架构+2分，Q5_K_M量化-5分 |
| 速度 | 78 | 20% | 基于RTX 4060带宽计算，约35 t/s |
| 适配 | 90 | 15% | 显存占用3.5GB，占比44% |
| 上下文 | 80 | 15% | 支持128K上下文，满足写作需求 |

**前端展示形式**：
- 推荐理由以卡片形式展示在模型卡片下方
- 评分明细默认折叠，用户点击"查看详情"展开
- 各项得分以进度条形式可视化展示

#### 3.2.2 量化类型参数表

参考 Odysseus Cookbook 的量化参数定义，支持多种量化格式：

| 量化类型 | BPP值 | 速度倍率 | 质量惩罚 | 说明 |
|---------|-------|---------|---------|------|
| f16 | 2.0 | 0.75 | 0 | 全精度，质量最好，速度最慢 |
| f8 | 1.0 | 1.0 | 1 | FP8量化，平衡质量与速度 |
| q8_0 | 0.875 | 1.1 | 2 | GGUF Q8_0量化 |
| q6_K | 0.625 | 1.15 | 3 | GGUF Q6_K量化 |
| q5_K_M | 0.5 | 1.2 | 5 | GGUF Q5_K_M量化（推荐） |
| q5_K_S | 0.5 | 1.25 | 6 | GGUF Q5_K_S量化 |
| q4_K_M | 0.4 | 1.3 | 7 | GGUF Q4_K_M量化（推荐） |
| q4_K_S | 0.4 | 1.35 | 8 | GGUF Q4_K_S量化 |
| q3_K_M | 0.32 | 1.4 | 10 | GGUF Q3_K_M量化 |
| q2_K | 0.26 | 1.5 | 15 | GGUF Q2_K量化 |
| AWQ | 0.4 | 1.3 | 4 | AWQ量化 |
| GPTQ | 0.4 | 1.25 | 5 | GPTQ量化 |

#### 3.2.3 架构Bonus机制

对最新架构的模型给予额外加分：

| 架构 | Bonus分数 | 说明 |
|------|----------|------|
| Qwen3.6 | +9 | Qwen3.6系列最新模型 |
| Qwen3.5 | +8 | Qwen3.5系列 |
| Qwen3-Next | +6 | Qwen3-Next系列 |
| Qwen3 | +4 | Qwen3系列 |
| Qwen2.5 | +2 | Qwen2.5系列 |
| Llama 3.3 | +4 | Llama 3.3系列 |
| Llama 3.2 | +3 | Llama 3.2系列 |

#### 3.2.4 模型清单管理

参考 Odysseus Cookbook 的模型清单管理机制，实现基于国内源的模型元数据管理。

**清单数据来源**：

| 来源 | 说明 | 优先级 |
|------|------|-------|
| ModelScope | 国内模型托管平台，优先使用 | 最高 |
| hf-mirror | HuggingFace国内镜像 | 次高 |
| 自定义国内镜像 | 自建模型托管服务 | 中等 |

**清单更新机制**：

- **内置清单**：模型元数据以内置JSON清单形式随安装包分发
- **在线更新**：软件启动时检测清单版本，支持在线拉取最新清单
- **更新频率**：默认每天检查一次更新，用户可手动触发更新

**清单格式**：

```json
{
  "version": "2026-06-25",
  "models": [
    {
      "id": "qwen3-7b-chat-q4_k_m",
      "name": "Qwen3-7B-Chat-Q4_K_M",
      "author": "Qwen",
      "base_model": "Qwen3-7B",
      "quant_type": "GGUF/Q4_K_M",
      "quant_bpp": 0.4,
      "parameter_count": "7B",
      "parameters_raw": 7173395456,
      "is_moe": false,
      "vram_required_gb": 4.5,
      "file_size_mb": 3800,
      "context_length": 131072,
      "architecture": "Qwen3",
      "is_gguf": true,
      "model_format": "gguf",
      "release_date": "2026-06-01",
      "download_url": "https://www.modelscope.cn/api/v1/models/qwen/Qwen3-7B-Chat-GGUF/repo/files?path=qwen3-7b-chat-q4_k_m.gguf",
      "mirror_urls": [
        "https://hf-mirror.com/Qwen/Qwen3-7B-Chat-GGUF/resolve/main/qwen3-7b-chat-q4_k_m.gguf"
      ],
      "sha256": "abc123...",
      "supported_gpu": "NVIDIA,AMD,Intel,CPU",
      "license": "Apache-2.0",
      "recommended_scene": ["聊天", "写作", "代码"],
      "chinese_score": 95,
      "speed_rating": 5
    }
  ],
  "signature": "base64-encoded-ed25519-signature"
}
```

**清单更新脚本**：

参考 Odysseus 的 `add_hwfit_models.py`，编写模型清单更新脚本：

```python
# scripts/update_model_list.py
# 功能：从国内源自动扫描模型并生成清单

# 自动扫描来源：
# 1. ModelScope 指定作者的模型（如 Qwen、DeepSeek、Yi 等）
# 2. hf-mirror 指定作者的模型
# 3. 手动添加的特定模型（通过 EXTRA_REPOS 配置）

# 自动解析：
# - 从模型名称提取参数数量、量化类型
# - 从 config.json 提取上下文长度、架构等信息
# - 计算显存需求、文件大小等
```

**缓存机制**：

- **硬件检测缓存**：缓存24小时，避免重复检测
- **模型清单缓存**：缓存到内存，提高推荐速度
- **支持手动刷新**：提供"重新检测硬件"和"刷新模型列表"按钮

---

### 3.3 下载管理模块 [对应需求：3.4.1 ~ 3.4.2]

#### 3.3.1 核心能力

多线程断点续传、多镜像源自动切换、哈希校验、进度实时回调。

#### 3.3.2 实现方案

**镜像源测速**

启动下载前对所有可用镜像源执行 HEAD 请求测速，选择延迟最低的源。

**断点续传**

通过 HTTP Range 请求实现，已下载部分写入 `.tmp` 临时文件，重启后读取文件偏移量继续下载。

**下载状态机**：

```
pending → downloading → verifying → completed
    ↓         ↓             ↓          ↓
paused   failed        failed     cancelled
    ↓         ↓
downloading completed
```

| 状态 | 说明 | 触发条件 |
|------|------|---------|
| `pending` | 任务已创建，等待下载 | 用户点击下载 |
| `downloading` | 正在下载 | 开始下载 |
| `paused` | 已暂停 | 用户点击暂停 |
| `verifying` | 正在检查文件是否完整 | 下载完成后开始 SHA256 校验 |
| `failed` | 下载失败 | 网络错误、超时、校验失败 |
| `completed` | 下载完成 | 文件下载并校验通过 |
| `cancelled` | 已取消 | 用户点击取消 |

**状态转换规则**：
- `pending` → `downloading`：开始下载
- `downloading` → `paused`：用户暂停
- `downloading` → `failed`：下载失败
- `downloading` → `verifying`：文件下载完成，开始校验
- `verifying` → `completed`：SHA256 校验通过
- `verifying` → `failed`：SHA256 校验失败，按规则重下损坏分片或整文件
- `paused` → `downloading`：用户继续
- `failed` → `downloading`：自动重试或用户重新下载
- 任何状态 → `cancelled`：用户取消（删除临时文件）

**多线程分片**

大文件自动切分为多个分片并发下载，单文件默认4线程。

**异常重试**

网络错误自动重试，重试间隔指数退避（1s → 2s → 4s），最多重试3次；单源失败后自动切换备用镜像源。

**完整性校验**

下载完成后比对 SHA256 哈希值。哈希值来源：由签名后的模型清单文件提供。校验不一致时自动重新下载损坏分片（若服务器支持 Range）或整文件。

**进度回调**

通过 Tauri Event 向前端推送实时进度，事件结构：

```rust
#[derive(Clone, Serialize)]
pub struct DownloadProgress {
    pub task_id: i64,
    pub model_id: String,
    pub downloaded_bytes: u64,
    pub total_bytes: u64,
    pub speed_bps: u64,
    pub status: DownloadStatus,
    pub current_step: String, // 如"正在下载模型文件"
}
```

---

### 3.4 推理服务管理模块 [对应需求：3.5]

#### 3.4.1 核心能力

统一管理 llama.cpp 服务生命周期，自动生成最优启动参数，端口自动分配，健康状态监控。

#### 3.4.2 智能启动参数配置（Profiles模块）

参考 Odysseus Cookbook 的 profiles 模块，实现基于硬件和模型的智能参数配置。

**输入**：硬件信息、模型文件、用户偏好  
**输出**：多种配置方案（Quality/Balanced/Speed），包含完整的 llama.cpp 启动参数

**智能参数计算逻辑**：

```rust
pub struct ModelProfile {
    pub name: String,           // Quality / Balanced / Speed
    pub description: String,    // 大白话描述
    pub n_gpu_layers: i32,      // GPU卸载层数
    pub n_cpu_moe_layers: i32,  // MoE模型CPU卸载层数
    pub ctx_size: i32,          // 上下文窗口
    pub rope_freq_base: f64,    // RoPE频率基数（上下文扩展用）
    pub rope_freq_scale: f64,   // RoPE频率缩放
    pub kv_cache_type: String,  // KV缓存类型（q4_0/q8_0/f16）
    pub threads: i32,           // CPU线程数
    pub batch_size: i32,        // 批处理大小
    pub n_parallel: i32,        // 并发数（默认1）
}
```

**参数自动计算规则**：

| 参数 | 实际计算逻辑 | 说明 |
|------|---------|------|
| `n_gpu_layers` | 预算够 → `999`（全量卸载）；不够 → 按余量分 20 / 12 / 8 层 | 全量用 `999` 而非 `-1`：两者语义相同，前者在日志里更直观 |
| `ctx_size` | 目标取模型原生上下文，上限由「(空闲显存 − 权重 − 开销 − 滑窗固定 KV) ÷ 每 token KV」反解 | fp16 优先；只差 5% 以内仍开 fp16，不为这点长度把整段 KV 降成 q8 |
| `kv_cache_type` | 默认 fp16；fp16 装不下目标才降 `q8_0` | 该构建在 KV 量化时自动启用 FlashAttention，不显式传 `--flash-attn` |
| `parallel` | 固定 1 | 桌面单用户；llama.cpp 默认 4 槽会把 KV 乘 4 倍，是小显存卡溢出到共享内存的主因 |
| `n_cpu_moe_layers` | 设计保留，当前未使用 | 内置清单中没有 MoE 模型 |

**为什么不能只按「显存充足就满层卸载」二分**：若连最小可用上下文（2048）都放不下仍硬给 2048，实际占用会超出预算，llama.cpp 把 KV 溢出到共享内存 —— 结果是「启动成功但龟速」，比明确走部分卸载更糟。部分卸载是有意的降级，不是失败。

**配置方案生成**：

| 方案 | 特点 | 适用场景 |
|------|------|---------|
| **Quality** | 最大化模型质量，完整上下文，全精度KV缓存 | 写作、推理等对质量要求高的场景 |
| **Balanced** | 平衡质量与速度，适当压缩上下文和KV缓存 | 日常聊天、通用问答 |
| **Speed** | 最大化推理速度，小上下文，低精度KV缓存 | 快速问答、代码补全 |

**参数与构建号**：

llama.cpp 的版本标识是构建号（如 `b10853`），不是 `0.x` 语义化版本，**不存在「按版本切换参数名」的规则**。当前固定使用长名 `--n-gpu-layers`，`--ngl` 是它的官方缩写，二者等价；统一用长名，便于日志阅读与检索。

引擎版本**钉死不追 master**：上游在某次改动后默认加载 MTP 张量，会让同一模型多吃约 0.327 GiB 显存。

**额外参数**：

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `--host` | `127.0.0.1` | 仅本机访问 |
| `--port` | 自动检测空闲端口 | 默认从8000开始递增 |
| `--parallel` | 1 | 单用户场景避免资源浪费 |
| `--no-mmap` | 未设置 | 内存充足时启用mmap加速加载 |
| `--mlock` | 未设置 | 内存充足时锁定到内存 |

**视觉模型特殊处理**：

- 视觉模型额外预留约1GB显存用于图像处理
- 自动添加 `--vision` 参数（如适用）

#### 3.4.3 进程生命周期管理

- 使用 `tokio::process` 异步管理子进程。
- 软件退出时自动优雅终止推理服务（发送 SIGTERM，超时后 SIGKILL），避免残留进程。
- 异常崩溃自动重启，最多重启3次，连续失败则进入故障诊断。

#### 3.4.4 健康检查

**健康检查接口演进记录**：早期方案曾考虑使用独立 `/health` 接口；为减少额外适配并保持与 llama.cpp OpenAI 兼容服务一致，最终统一使用 `GET /v1/models` 作为健康检查接口。

- 启动后每5秒调用 **`GET /v1/models`** 接口检测服务可用性。
- 连续3次失败判定为服务异常，自动触发故障诊断。
- 若诊断后进入"安全模式"：以纯CPU运行最低配参数（`ngl=0`, `ctx-size=2048`），确保用户至少能进行基础对话。
- **安全模式进入条件**：同一模型连续3次健康检查失败，且故障诊断无法通过A级自动修复恢复服务。
- **安全模式退出条件**：用户手动切换模型、重新启动当前模型且健康检查连续2次成功，或用户在故障提示中点击"恢复显卡加速"并启动成功后退出安全模式。
- 安全模式状态需写入当前会话配置，不写入永久默认配置，避免下次启动永久降级。

#### 3.4.5 OpenAI 兼容接口

llama.cpp 原生提供 OpenAI 兼容 API，直接复用，对外统一接口标准：

- `GET /v1/models`：获取可用模型列表（同时作为健康检查接口）
- `POST /v1/chat/completions`：聊天补全接口
- `POST /v1/completions`：文本补全接口

默认监听 `127.0.0.1`，不对外网开放。

---

### 3.5 故障自诊断模块 [对应需求：3.7]

#### 3.5.1 实现方案

采用「检查项 + 错误码 + 修复动作」的规则引擎模式，可扩展。修复动作按自动化程度分为 A/B/C 三级，与需求文档 2.3 保持一致。

#### 3.5.2 检查项定义

每个检查项实现统一接口：

```rust
pub trait DiagnoseCheck {
    fn check(&self) -> DiagnoseResult;
    fn fix(&self) -> Result<(), FixError>;
    fn description(&self) -> &str;
    fn auto_level(&self) -> AutoLevel; // A级/B级/C级
}

pub enum AutoLevel {
    FullAuto,      // A级：全自动
    SemiAuto,      // B级：半自动（需用户确认）
    ManualGuide,   // C级：人工引导
}

pub struct DiagnoseResult {
    pub issue_id: String,
    pub title: String,        // 大白话标题
    pub description: String,  // 大白话描述
    pub auto_level: AutoLevel,
    pub fixable: bool,
}
```

#### 3.5.3 内置检查清单（按执行顺序）

| 顺序 | 检查项 | 自动化等级 | 对应需求 |
|------|--------|-----------|---------|
| 1 | 容器环境检测 | C级 | 新增 |
| 2 | 磁盘空间检查 | A级 | 3.7.2 |
| 3 | 模型文件完整性检查 | A级 | 3.7.2 |
| 4 | 端口占用检查 | A级 | 3.7.2 |
| 5 | 显卡驱动版本检查 | C级 | 3.7.2 |
| 6 | 运行依赖库检查 | B级 | 3.7.2 |
| 7 | 显存可用性检查 | A级 | 3.7.2 |

#### 3.5.4 容器环境检测

参考 Odysseus Cookbook 的容器环境检测逻辑，在容器环境下给出非阻塞警告：

**检测方式**：
- 检查 `/proc/1/cgroup` 文件（Linux）
- 检查环境变量 `CONTAINER_ID`、`DOCKER_CONTAINER`、`KUBERNETES_SERVICE_HOST`
- 检查文件系统是否为容器典型路径

**处理逻辑**：
- 检测到容器环境但未检测到GPU时，给出非阻塞警告："检测到您可能在容器环境中运行，GPU加速可能不可用。如需使用显卡加速，请确保容器已正确配置GPU直通。"
- 提供手动硬件选择入口，允许用户手动指定硬件配置
- 不阻断正常流程，仅作为提示信息

#### 3.5.5 执行流程

1. 启动模型前串行执行所有检查项。
2. 发现异常则中断启动，展示问题描述与修复按钮（按钮文案根据 `auto_level` 动态生成：A级显示"一键修复"，B级显示"帮我安装"，C级显示"查看解决步骤"）。
3. 用户点击后执行对应 `fix` 方法。
4. 修复完成后自动重试启动。

---

### 3.6 聊天会话管理模块 [对应需求：3.6]

#### 3.6.1 功能目标

管理多会话生命周期、消息持久化、角色切换、参数调节，为前端提供统一的会话操作 IPC 接口。

#### 3.6.2 数据结构设计

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatSession {
    pub id: String,
    pub name: String,
    pub model_id: String,
    pub role_preset: RolePreset,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub id: String,
    pub session_id: String,
    pub role: MessageRole, // user / assistant
    pub content: String,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RolePreset {
    DailyAssistant,
    Writing,
    Coding,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatSettings {
    pub temperature: f32, // 0.0 ~ 1.0，对应"创意程度"滑块
}
```

#### 3.6.3 系统提示词注入

角色切换时，系统自动在请求开头注入对应 System Prompt，用户无感知：

- **日常助手**："你是一个友好的AI助手，用通俗易懂的中文回答用户问题。"
- **文案写作**："你是一个专业的中文写作助手，擅长撰写文案、报告、邮件。"
- **代码助手**："你是一个编程助手，擅长解释代码、生成代码、排查Bug。"

#### 3.6.4 前端调用路径

聊天界面**不通过 IPC 调用**，直接通过 HTTP 请求访问本地 llama.cpp 服务的 OpenAI 兼容接口（`POST /v1/chat/completions`）。会话管理（增删改查）通过 IPC Command 与 Rust 后端交互。

---

### 3.7 模型管理模块 [对应需求：3.4.3、3.4.4]

#### 3.7.1 功能目标

管理已安装模型列表、运行状态、本地导入，为模型管理面板提供数据与操作接口。

#### 3.7.2 模型状态机

```
[已安装/已停止] --启动--> [启动中] --健康检查通过--> [运行中]
     ↑                                              |
     └--------------停止/切换模型-------------------┘
```

#### 3.7.3 本地模型导入

支持导入用户本地已有的 `.gguf` 模型文件：

1. 前端选择文件路径，通过 IPC 传递给后端。
2. 后端解析 GGUF 文件头，提取参数数量、上下文长度、量化类型等元数据：
   - 若解析成功：用真实文件大小（GiB）+ 驱动模式对应的开销 + 该文件层结构算出的 KV，得出显存需求
   - 若解析失败：按 `文件字节数 ÷ 1024³ × 1.1 + 开销` 估算，并提示用户手动确认参数规模
3. 弹出表单让用户填写模型名称、选择场景标签，允许用户手动调整参数规模和上下文长度。
4. 生成唯一 `model_id`，写入 `model_metadata` 表（标记为 `is_imported = true`）。

**GGUF文件头解析**：
- 读取 GGUF 文件前 128KB，解析 magic number、version、tensor count、tensor metadata
- 从 tensor metadata 中提取参数数量（通过 tensor 尺寸计算）和量化类型（通过 tensor name 判断）
- 支持 GGUF v1 和 v2 格式

#### 3.7.4 已安装模型更新

模型清单更新后，模型管理模块负责检测已安装模型是否存在新版本：

1. 根据 `base_model`、`architecture`、`quant_type`、`release_date` 和版本号提取结果匹配同系列新版模型。
2. 在模型管理面板显示"发现新版本"，展示新版文件大小、推荐理由和许可证变化。
3. 用户确认后创建下载任务，下载完成进入 `verifying` 状态并执行 SHA256 校验。
4. 校验通过后停止旧模型服务，将新版标记为当前默认版本；旧模型文件默认保留，用户可手动删除。
5. 更新失败或新版启动失败时回滚到旧模型记录，保持聊天会话和配置不变。

数据库中通过 `supersedes_model_id`（可选字段）记录新版与旧版关系，避免覆盖历史模型记录。

---

### 3.8 系统设置与持久化模块 [对应需求：3.9]

#### 3.8.1 配置分层

采用 `config` crate 实现三层配置合并（优先级从高到低）：

1. **运行时配置**：内存中的临时覆盖（如当前会话状态）。
2. **用户配置**：`data/config.toml`，用户通过设置面板修改的持久化配置。
3. **默认配置**：打包在应用内的 `default-config.toml`。

#### 3.8.2 关键配置项

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub install_dir: PathBuf,           // 模型与数据存储路径
    pub auto_start: bool,               // 开机自启
    pub download_speed_limit: Option<u64>, // 下载限速（bytes/s）
    pub mirror_source: MirrorSource,    // 镜像源选择
    pub proxy: Option<String>,          // HTTP 代理
    pub log_level: String,              // 日志等级
    pub advanced_params: Option<String>, // 自定义启动参数
}
```

#### 3.8.3 开机自启实现（Windows）

- 通过写入 Windows 注册表 `HKEY_CURRENT_USER\Software\Microsoft\Windows\CurrentVersion\Run` 实现。
- 以普通用户权限运行，不需要管理员权限。
- 关闭自启时删除对应注册表项。

#### 3.8.4 多实例数据隔离

虽然产品不提供多用户/多项目管理能力，但需要避免同一台机器安装多个同类工具或多个版本时发生 SQLite 与模型目录冲突：

- 默认数据目录使用应用名称子目录：`%LOCALAPPDATA%\LocalAI-Installer\data`。
- 绿色版默认使用程序所在目录下的 `data/`，同时允许用户在设置中切换到独立数据目录。
- 配置文件中保存 `instance_id`，首次启动生成 UUID，用于区分日志、下载任务和数据库锁。
- 启动时检测数据库锁文件，若已有同实例运行中，提示用户切换到已打开窗口；不同 `instance_id` 的安装互不影响。

#### 3.8.5 UAC 提权处理

- 默认安装到用户目录（`%LOCALAPPDATA%\LocalAI-Installer`），无需管理员权限。
- 用户修改安装目录到系统目录（如 `C:\Program Files`）时，Tauri 通过 `tauri-plugin-shell` 或自定义权限请求触发 UAC 弹窗，明确提示"需要管理员权限才能修改此目录"。

---

### 3.9 卸载与数据管理模块 [对应需求：3.10]

#### 3.9.1 卸载程序技术方案

- **Windows**：使用打包工具内置的卸载逻辑（Tauri 的 NSIS / MSI 打包）。
- 卸载时读取用户选择（保留数据 / 彻底删除 / 强制卸载），调用 Rust 后端 `uninstall_app` Command：
  - 「保留模型文件」：仅删除程序文件，保留 `data/` 和 `models/` 目录。
  - 「彻底删除」：删除整个安装目录。
  - 「强制卸载」：先停止由本工具启动的推理服务与下载任务，必要时终止仍占用本工具目录的子进程，再执行删除；仅处理本工具记录的进程ID和安装目录下的文件，不扫描或终止无关进程。

#### 3.9.2 数据重置

- 设置中提供「重置软件」按钮，调用 `reset_app_data` Command。
- 清空 SQLite 中所有配置表和聊天会话表，但保留 `models/` 目录下的模型文件。
- 重置前二次确认，防止误操作。

#### 3.9.3 数据导出与迁移（V1.5）

- 导出内容：配置（脱敏）、对话记录、模型文件路径映射。
- 导出格式：ZIP 压缩包，内含 `config.json`、`chat_sessions.db`、路径映射文件。
- 导入时校验版本兼容性，不兼容时提示用户。

---

### 3.10 后端按需下载管理器 [对应需求：3.5.1]

#### 3.10.1 功能目标

基础安装包≤200MB，仅包含当前硬件所需后端。非当前硬件的后端（CUDA/ROCm/SYCL）在首次检测到对应硬件时后台自动下载。

#### 3.10.2 后端的真实构成：瘦启动器 + DLL

安装包里每个后端由两部分组成：

| 层 | 文件 | 作用 |
|---|---|---|
| 启动器 | `llama-server-cuda.exe` / `llama-server-vulkan.exe` / `llama-server-cpu.exe` | **三个文件 SHA256 完全相同**，是同一个瘦启动器换了名字 |
| 计算后端 | `ggml-cuda.dll` / `ggml-vulkan.dll` / `ggml-cpu-*.dll` | 真正的计算实现；`ggml-cuda.dll` 单个约 565 MB |

运行时由启动器在**自己所在目录**里找 `ggml-<name>-*.dll` 或 `ggml-<name>.dll`，找到就 `dlopen` 并调用它的 `ggml_backend_init`。**走哪条计算路径由目录里存在哪个 DLL 决定，与启动器的文件名无关。**

当前构建号 `b10853`，三套包（CUDA / Vulkan / CPU）同版本。ROCm 与 SYCL 尚未提供。

**包内容与下载源**：每个后端包携带文件名、SHA256 与构建号，按 `ModelScope 主源 → gh-proxy → GitHub Releases 原站` 的顺序尝试，前一源失败自动切下一个，三源皆失败才提示用户检查网络。检测到新硬件且本机没有对应后端时后台静默下载，完成后提示「检测到新显卡，已为你准备好加速引擎」。`engine/versions.json` 记录最近一次准备的 `current_backend` / `version` / `executable` / `source`；GPU 引擎与 CPU 兜底是**独立准备**的，各自另有构建号记录 —— 共用一个文件会互相覆盖。

#### 3.10.3 后端选择与隔离

- **「切后端」不是换 exe，而是换「哪个后端 DLL 留在 `engine/backends/` 根目录」。**
- 目标后端需要的 DLL 留在根目录，其余后端 DLL 全部移进 `engine/backends/.pool/<后端名>/`；切换时先还原池子再按新目标隔离。
- 共用底座（`ggml.dll` / `ggml-base.dll` / `llama.dll` / `llama-server-impl.dll` 等）不属于任何一个后端，**不参与隔离**。
- 只移动 GPU 后端 DLL 是不够的：只要 `ggml-cuda.dll` 还在根目录，**即使启动 `llama-server-vulkan.exe` 也照样初始化 CUDA**（2026-09-29 虚拟机实测）。

#### 3.10.4 目录结构

```
engine/
├── backends/
│   ├── ggml.dll                # 共用底座，不隔离
│   ├── llama.dll
│   ├── llama-server-impl.dll
│   ├── llama-server-{cuda,vulkan,cpu}.exe   # 同一启动器的三个名字
│   ├── ggml-cuda.dll           # 仅当目标是 CUDA 时留在根目录
│   ├── ggml-vulkan.dll
│   ├── ggml-cpu-*.dll
│   └── .pool/
│       ├── CUDA/               # 被移走的非当前后端 DLL
│       └── Vulkan/
└── current_backend.txt         # 记录用户选择，用于还原与展示
```

`current_backend.txt` 记录的是**用户的选择**，不是「哪条计算路径真的生效」——后者只能由启动日志与显存占用证明。

#### 3.10.5 实测结论与判据（2026-09-29）

1. **CUDA / Vulkan / CPU 三个 exe 的 SHA256 全同** = 同一个二进制换名。所以「我换了 exe」不等于「我换了后端」。
2. **判据只能看证据，不能看文件名**：
   - 启动日志 `device_info:` 列表里出现 `CUDA0` / `Vulkan0` 才算真上卡；**只有 `CPU` 一行就是退到了 CPU**。
   - `nvidia-smi` 的 `memory.used` 若只有个位数 MiB 而推理已在跑 = 权重没进显存。
   - 速度反推：`有效带宽 ≈ tok/s × 权重GiB`。结果落在内存带宽量级（P100 显存 732 GB/s，实测反推约 14.7 GB/s）即为纯 CPU。
3. **TCC 卡（Tesla 计算模式，无显示输出）走 Vulkan 会静默退 CPU。** Vulkan 是图形 API，设备枚举需要呈现能力，TCC 下枚举不到设备 —— 表现是「启动成功、慢十倍、不报错」。选后端前先看 `nvidia-smi` 的 `driver_model.current`：`TCC` 卡只能走 CUDA 或 CPU。

---

## 四、数据模型设计

采用 SQLite 本地存储，核心表结构如下。

**字段命名与界面文案约束**：数据表中的 `quant_*`、`context_length`、`backend` 等字段均为内部字段名，前端不得直接把字段名或技术值展示给用户。表现层必须通过文案映射表转换为"压缩版本"、"AI能记住的上下文长度"、"显卡加速"等大白话表达，和需求文档附录B保持一致。

### 4.1 model_metadata（模型元数据表）

| 字段名               | 类型      | 说明           |
| ----------------- | ------- | ------------ |
| id                | TEXT    | 模型唯一ID       |
| name              | TEXT    | 模型展示名称       |
| author            | TEXT    | 模型作者         |
| base_model        | TEXT    | 基座模型         |
| quant_level       | TEXT    | 量化等级（内部字段，不直接展示到界面） |
| quant_type        | TEXT    | 量化类型（内部字段，如 GGUF/Q4_K_M/AWQ/GPTQ/FP8；界面映射为"压缩版本"等大白话） |
| quant_bpp         | REAL    | 量化BPP值（内部计算字段，字节/参数） |
| parameter_count   | TEXT    | 参数数量（如"35B"） |
| parameters_raw    | INTEGER | 原始参数数量（整数） |
| active_parameters | INTEGER | MoE活跃参数数量（可选） |
| is_moe            | INTEGER | 是否MoE模型（0/1） |
| vram_required_gb  | REAL    | 最低显存需求（GB）   |
| file_size_mb      | INTEGER | 文件大小（MB）     |
| context_length    | INTEGER | 上下文长度（tokens） |
| min_ram_gb        | REAL    | 最低内存需求（GB）   |
| recommended_ram_gb| REAL    | 推荐内存需求（GB）   |
| recommended_scene | TEXT    | 推荐场景（JSON数组） |
| chinese_score     | INTEGER | 中文评分（0-100）  |
| speed_rating      | INTEGER | 速度评级（1-5）    |
| architecture      | TEXT    | 模型架构（Qwen3/Qwen2/Llama3等） |
| is_gguf           | INTEGER | 是否GGUF格式（0/1） |
| gguf_sources      | TEXT    | GGUF下载源（JSON数组） |
| model_format      | TEXT    | 模型格式（gguf/awq/gptq/fp8等） |
| release_date      | TEXT    | 发布日期（YYYY-MM-DD） |
| download_url      | TEXT    | 主下载地址（国内源优先） |
| mirror_urls       | TEXT    | 镜像地址（JSON数组，国内源优先） |
| sha256            | TEXT    | 文件哈希         |
| supported_gpu     | TEXT    | 支持的显卡厂商      |
| license           | TEXT    | 开源协议         |
| is_imported       | INTEGER | 是否本地导入（0/1） |
| local_path        | TEXT    | 本地文件路径（导入模型用） |
| supersedes_model_id | TEXT  | 被此版本替代的旧模型ID（模型更新用，可空） |
| hf_downloads      | INTEGER | HuggingFace下载量（用于口碑评分） |
| hf_likes          | INTEGER | HuggingFace点赞数（用于口碑评分） |

### 4.2 download_tasks（下载任务表）

| 字段名              | 类型       | 说明                                             |
| ---------------- | -------- | ---------------------------------------------- |
| id               | INTEGER  | 任务ID                                           |
| model_id         | TEXT     | 关联模型ID                                         |
| save_path        | TEXT     | 保存路径                                           |
| total_bytes      | INTEGER  | 总大小                                            |
| downloaded_bytes | INTEGER  | 已下载大小                                          |
| status           | TEXT     | 状态：pending/downloading/paused/verifying/failed/completed/cancelled |
| current_source   | TEXT     | 当前使用的镜像源URL                                  |
| source_history   | TEXT     | 已尝试的镜像源（JSON数组）                            |
| create_time      | DATETIME | 创建时间                                           |
| update_time      | DATETIME | 更新时间                                           |

### 4.3 chat_sessions（聊天会话表）

| 字段名       | 类型       | 说明                  |
| ----------- | --------- | --------------------- |
| id          | TEXT      | 会话ID                |
| name        | TEXT      | 会话名称              |
| model_id    | TEXT      | 关联模型ID            |
| role_preset | TEXT      | 角色预设              |
| created_at  | DATETIME  | 创建时间              |
| updated_at  | DATETIME  | 更新时间              |

### 4.4 chat_messages（聊天消息表）

| 字段名      | 类型       | 说明                  |
| ----------- | --------- | --------------------- |
| id          | TEXT      | 消息ID                |
| session_id  | TEXT      | 所属会话ID            |
| role        | TEXT      | user / assistant      |
| content     | TEXT      | 消息内容              |
| created_at  | DATETIME  | 创建时间              |

### 4.5 system_config（系统配置表）

存储用户设置、端口配置、默认模型等全局配置，以 Key-Value 形式存储：

| 字段名 | 类型 | 说明 |
|-------|------|------|
| key   | TEXT | 配置键 |
| value | TEXT | 配置值（JSON序列化） |

---

## 五、接口设计

### 5.1 前端 IPC 接口（Tauri Command）

前端通过调用 Rust 后端命令实现所有功能，核心接口：

| 接口名                    | 功能        | 参数       | 返回值          | 对应需求 |
| ---------------------- | --------- | -------- | ------------ | -------- |
| get_hardware_info      | 获取硬件检测结果  | 无        | HardwareInfo | 3.2 |
| get_recommend_models   | 获取推荐模型列表  | 场景、硬件档位  | 模型列表（首选+备选）  | 3.3 |
| start_download         | 开始下载模型    | model_id | 任务ID         | 3.4.1 |
| pause_download         | 暂停下载      | task_id  | 成功/失败        | 3.4.1 |
| cancel_download        | 取消下载      | task_id  | 成功/失败        | 3.4.1 |
| get_download_progress  | 获取下载进度    | task_id  | DownloadProgress | 3.4.1 |
| get_installed_models   | 获取已安装模型列表 | 无      | 模型列表         | 3.4.3 |
| start_model_service    | 启动模型服务    | model_id | 服务端口         | 3.5 |
| stop_model_service     | 停止模型服务    | model_id | 成功/失败        | 3.5 |
| get_service_status     | 获取服务运行状态  | model_id | 运行状态、端口      | 3.5 |
| import_local_model     | 导入本地模型    | path, name, scenes | model_id | 3.4.3 |
| update_installed_model  | 更新已安装模型 | model_id, target_version_id | 新模型ID/失败原因 | 3.4.4 |
| uninstall_model        | 卸载模型      | model_id | 成功/失败        | 3.4.3 |
| run_diagnose           | 执行故障诊断    | model_id | DiagnoseResult列表 | 3.7 |
| fix_issue              | 执行修复      | issue_id | 修复结果         | 3.7 |
| get_third_party_config | 获取第三方对接配置 | app_name | 配置信息         | 3.8 |
| create_chat_session    | 创建聊天会话    | model_id, role_preset | session_id | 3.6 |
| get_chat_sessions      | 获取会话列表    | 无       | 会话列表         | 3.6 |
| delete_chat_session    | 删除聊天会话    | session_id | 成功/失败      | 3.6 |
| get_chat_messages      | 获取会话消息    | session_id | 消息列表       | 3.6 |
| clear_chat_context     | 清空会话上下文  | session_id | 成功/失败      | 3.6 |
| get_settings           | 获取系统设置    | 无       | AppConfig       | 3.9 |
| update_settings        | 更新系统设置    | AppConfig | 成功/失败       | 3.9 |
| export_logs            | 导出日志包     | 无       | ZIP文件路径      | 4.5 |
| reset_app_data         | 重置软件数据    | 无       | 成功/失败        | 3.10 |
| uninstall_app          | 卸载应用       | keep_data: bool | 成功/失败 | 3.10 |

### 5.2 Tauri Event（前端订阅）

| 事件名 | 说明 | 数据 |
|-------|------|------|
| download_progress | 下载进度推送 | DownloadProgress |
| service_status_changed | 服务状态变更 | { model_id, status, port } |
| diagnose_result | 诊断结果推送 | DiagnoseResult |

### 5.3 对外 OpenAI 兼容接口

推理服务对外提供标准 HTTP 接口，与 llama.cpp 原生接口保持一致：

- `GET /v1/models`：获取可用模型列表（同时作为健康检查接口）
- `POST /v1/chat/completions`：聊天补全接口
- `POST /v1/completions`：文本补全接口

默认监听 `127.0.0.1`，不对外网开放。

---

## 六、目录结构与打包

### 6.1 安装后目录结构

```
LocalAI-Installer/
├── app.exe                     # 主程序（Tauri）
├── engine/                     # 推理引擎目录
│   ├── current_backend.txt     # 用户选择的后端，用于还原与展示
│   ├── versions.json           # 最近一次准备的构建号（另有按后端分开的记录）
│   └── backends/
│       ├── ggml.dll            # 共用底座，不参与隔离
│       ├── llama.dll
│       ├── llama-server-{cuda,vulkan,cpu}.exe   # 同一启动器的三个名字
│       ├── ggml-cuda.dll       # 仅当目标是 CUDA 时留在根目录
│       ├── ggml-cpu-*.dll
│       └── .pool/              # 非当前后端 DLL 的暂存区
│           ├── CUDA/
│           └── Vulkan/
├── models/                     # 模型文件目录，按 model_id 分子目录
│   └── qwen3_8b_chat/
│       ├── Qwen3-8B-Q4_K_M.gguf
│       └── Qwen3-8B-Q4_K_M.gguf.tmp   # 下载中；完成后去掉 .tmp
├── data/                       # 数据目录
│   ├── app.db                  # SQLite 元数据库（附 -wal / -shm）
│   ├── config.json             # 用户配置
│   ├── catalog.json            # 模型清单缓存
│   └── logs/                   # 日志目录
└── resources/                  # 前端资源
```

### 6.2 打包发布

- **Windows**：打包为 MSI 安装包 + ZIP 绿色免安装版，支持数字签名。
- **更新机制**：内置增量更新，新版本发布后自动提示升级。
- **分发渠道**：提供直链下载，支持国内镜像站分发。

---

## 七、日志系统 [对应需求：4.5、3.9.3]

### 7.1 技术选型

使用 `tracing` + `tracing-subscriber` + `tracing-appender`：

- `tracing`：结构化日志记录，支持 Span 上下文跟踪。
- `tracing-subscriber`：日志订阅与格式化，支持按级别过滤。
- `tracing-appender`：非阻塞文件写入，按天切割日志文件。

### 7.2 配置

```rust
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};
use tracing_appender::rolling::{RollingFileAppender, Rotation};

let file_appender = RollingFileAppender::new(Rotation::DAILY, "data/logs", "localai");
let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);

tracing_subscriber::registry()
    .with(EnvFilter::new("info")) // 默认 INFO 级别
    .with(tracing_subscriber::fmt::layer().with_writer(non_blocking))
    .init();
```

### 7.3 脱敏规则

日志中禁止输出以下敏感信息。优先在结构化字段写入前脱敏，避免依赖全文正则误伤正常调试信息；文本匹配仅作为导出日志包前的兜底清理。

| 敏感类型 | 处理方式 | 示例 |
|---------|---------|------|
| 用户对话内容 | 在结构化日志字段层禁止写入原文，统一记录长度、消息ID和会话ID哈希 | `message_content` 不入日志，仅记录 `content_len=120` |
| 绝对文件路径 | 结构化字段写入前脱敏，保留文件名，替换路径为 `.../` | `C:\Users\Alice\models\a.gguf` → `.../a.gguf` |
| IP 地址 | 默认不对服务监听地址字段脱敏；仅对来自用户配置、代理地址、外部请求来源等可能包含隐私的字段脱敏 | `proxy_host=192.168.1.2` → `proxy_host=[REDACTED_IP]` |

- 文本正则脱敏规则不得全局替换固定本地监听地址 `127.0.0.1`，否则会影响健康检查、端口诊断等日志可读性。

### 7.4 一键导出日志包

- 收集最近7天日志文件，打包为 ZIP。
- 打包前执行脱敏处理。
- 导出的日志包不包含任何用户对话内容或文件路径。

---

## 八、安全设计 [对应需求：4.5]

### 8.1 网络安全

- 推理服务默认仅监听 `127.0.0.1`，不暴露到公网。
- 界面中明确提示用户"AI服务仅在本机可用"。

### 8.2 下载安全

- 所有下载通道强制使用 HTTPS，`reqwest` 默认校验服务器证书链。
- 模型文件哈希值由签名后的清单文件提供，防止中间人篡改。

### 8.3 清单签名验证

- 使用 Ed25519 非对称签名，公钥硬编码在应用内。
- **公钥轮换机制**：应用内置主公钥和至少一个备用公钥。清单签名中包含 `key_id` 与 `valid_from`，客户端按 `key_id` 选择公钥验证。需要轮换时，先发布由旧主公钥签名的"公钥轮换清单"，其中包含新公钥、启用时间和吊销列表；客户端验证通过后写入本地可信公钥缓存。若主公钥泄露且无法信任在线轮换清单，则必须通过应用更新包分发新公钥，并拒绝旧 `key_id` 的清单。
- 本地可信公钥缓存只允许追加由当前可信公钥签名的新公钥或吊销记录，不允许用户配置任意公钥，避免被恶意源替换。
- 清单文件格式：
  ```json
  {
    "version": "2026-06-25",
    "key_id": "manifest-key-2026-a",
    "models": [...],
    "signature": "base64-encoded-ed25519-signature"
  }
  ```
- 更新清单时，客户端先用公钥验证签名，通过后解析模型列表。

### 8.4 权限安全

- 软件默认以普通用户权限运行。
- 需要管理员权限的操作（修改系统目录、安装运行库）必须通过 Tauri 权限 API 二次弹窗确认。

### 8.5 进程安全

- 推理服务以普通用户权限运行，不获取管理员权限。

### 8.6 依赖安全

- 定期使用 `cargo audit` 扫描依赖漏洞。
- CI/CD 流程中集成安全扫描。

---

## 九、性能测试与验证方案 [对应需求：4.1]

### 9.1 测试基准环境

- CPU：i5-8400
- 内存：16GB DDR4
- 磁盘：SATA SSD
- 系统：Windows 11 64位
- 显卡：NVIDIA GTX 1660Ti

### 9.2 性能指标验证方法

| 指标 | 验证方法 | 不达标预案 |
|------|---------|-----------|
| 硬件检测耗时 ≤3秒 | `tracing` 打点记录耗时，连续100次取P95 | WMI超时降至1.5秒，进一步超时直接降级 |
| 服务启动 ≤10秒 | 记录 `start_model_service` 到首次 `/v1/models` 成功响应 | 启用更轻量后端、减少 ngl 层数 |
| 冷启动 ≤2秒 | 记录进程启动到前端首屏渲染完成 | 延迟加载非核心模块 |
| 下载速度 ≥带宽80% | 对比 `reqwest` 下载速度与 SpeedTest 结果 | 增加线程数、启用多源并发 |

### 9.3 持续监控

- 在 `tracing` 中定义性能相关的 Span，收集到本地日志。
- 用户反馈问题时，通过导出日志包分析性能瓶颈。

---

## 十、0.1.1 → 0.1.7 实现校准（2026-10-20）

> 前九章是 2026-06 的设计稿。本章记录 0.1.1 到 0.1.7 之间**已经被真机验证过的实现口径**，凡与前面章节冲突，以本章为准。

### 10.1 交付基线

| 项 | 值 |
|---|---|
| 版本号 | 0.1.7 —— `package.json` / `src-tauri/tauri.conf.json` / `src-tauri/Cargo.toml` 三处必须同号 |
| 构建命令 | `npm run tauri build -- --bundles nsis`；漏掉 `--bundles nsis` 不会失败，但产物不符合发布口径 |
| 安装包 | `本地AI模型一键部署助手_0.1.7_x64-setup.exe` |
| 测试门槛 | `cargo test --lib` 281 通过 / 0 失败 / 2 忽略；`cargo check` 零 warning |
| 引擎版本 | llama.cpp **b10853**，CUDA / Vulkan / CPU 三套包同构建号 |

### 10.2 层结构决定 KV，KV 决定能开多长上下文

显存的大头常常不是权重：Qwen3-1.7B 在 40960 上下文下，KV 约 **4.4 GiB**（112 KB/token），而权重只有 **1.7 GiB** —— KV 是权重的 2.5 倍。所以「这个模型能不能跑」不能只看文件大小。

| 模型 | 注意力结构 | KV 口径（fp16） |
|---|---|---|
| Qwen3-1.7B | 全注意力（28 层） | 112 KB/token |
| Qwen3-8B | 全注意力（36 层） | 144 KB/token |
| Spark-X2.5-4B | 9 层全注意力 + 27 层滑窗（窗口 512） | 滑窗层封顶后为固定值 |
| Ornith-1.5-9B | 8 层全注意力 + 24 层 SSM | 32 KiB/token |
| Qwen3.8-27B | 16 层全注意力 + 48 层 SSM（`full_attention_interval = 4`） | 64 KB/token（f16），q8_0 约半量 |

**混合注意力的 KV 增速只有常规模型的约 1/4**，这是 8G 卡上开长上下文的唯一出路。

四条实现要点：

1. **MTP 层必须扣掉再除。** `block_count` 包含 MTP 层，但 llama.cpp **不加载、不推理、不占 KV**（启动日志里的 `unused tensor blk.N.* ... ignoring` 就是它）。走 `full_attention_interval` 分支时，先用 `nextn_predict_layers` 做饱和减法再除以间隔：Qwen3.8 是 17→**16**、Ornith 是 9→**8**。不扣会高估一层，预算算小 —— 方向保守，但白浪费显存。
2. **滑窗层的 KV 是常数。** 占用上限是 `min(ctx, 窗口)`，超过窗口后不再增长。必须先扣固定项再做除法，否则上下文虚高（Spark 27 层 × 512 窗口 ≈ 54 MiB）。
3. **旧的 KV 经验式不能当主路径。** `0.000008 × 活跃参数 × ctx` 在混合注意力模型上实测**低估 7.8× 到高估 2.1×**，现在只在读不到 GGUF 时兜底。
4. **同一格式的多份解析必须交叉验证。** GGUF 解析在 `gguf.rs`、探测脚本、规划脚本里各有一份实现。任一处只错一点都不会报错，只会给出一个看起来合理的错数字；基准表要把「全注意力层数」「MTP 层数」「滑窗层数」直接列出来，让偏差看得见。

### 10.3 速度估算：系数按世代与权重分档

公式为 `tok/s ≈ 带宽 ÷ 权重GiB × 效率系数`，但**效率系数不是常数**：

| 分档 | 系数 | 实测锚点 |
|---|---|---|
| NVIDIA 张量加速卡（Turing 及以后） | 0.60 | RTX 4060 Laptop 8B-Q4 实测 35 t/s（理论 52） |
| NVIDIA Pascal，≤ 4 GiB 权重 | 0.26 | P100 1.7B-Q8 实测 110–113 t/s（理论 428） |
| NVIDIA Pascal，> 4 GiB 权重 | 0.18 | P100 27B-Q3 实测 10.5–11.4 t/s（理论 61.4） |
| Apple Silicon（Metal） | 0.55 | M2 Max 8B-Q4 约 40 t/s |
| AMD / Intel（Vulkan 路径） | 0.35 | 折中保守值 |

- 单一系数 0.7 在 P100 上把 11 t/s 估成 43 t/s（**虚高 3.8 倍**），但它对新卡其实站得住（实测到理论值的 67%）——**不能一刀降掉，否则把新卡也压低**。分档后，5 个实测锚点下的平均绝对误差从 85% 降到 16%。
- Pascal 还要按权重再分档：同一张 P100 两个实测点，1.7B 达 25.9%、27B 只有 17.9%。小模型每 token 的固定开销（采样、logits、kernel launch）占比更大，所以更贴近 roofline 上限；一个 0.20 会把 1.7B 低估 1.3 倍，而 8–16 GiB 卡上最关心的恰是它。
- **系数只能按代码里的单位标定**：`weight_gb` 是 GiB、`bandwidth_gbs` 是十进制 GB/s，1.7B 的理论值是 `732 ÷ 1.708 = 428`，不是 `732 ÷ 1.834 = 399`。按 GB 标定会凭空多算 7%。

### 10.4 跨架构编译与驱动门槛

| 目标 | 约束 |
|---|---|
| RTX 50 系（Blackwell，`sm_120`） | 编译需 CUDA ≥ 12.8；支持 bf16 与 FlashAttention-2 |
| Tesla P100（Pascal，`sm_60`） | 官方预编译包**不含** `sm_60`；不支持 bf16 / FA-2 |
| 一台机器出双架构包 | `-DCMAKE_CUDA_ARCHITECTURES=60-real;120a-real` |

P100 用官方 CUDA 包的唯一解是**升驱动**（539.72 → 582.78）：R580 是最后支持 Pascal 的分支。同一份 27B-Q3 因此从不可用变为 10.5–11.4 t/s。

### 10.5 服务生命周期与判据

- **切模型 = 先停旧服务、再拉新进程。** 旧进程还占着端口时，新进程的健康检查会被**旧引擎的 200** 直接骗过，日志写「新模型已运行」而实际跑的是旧模型。判据：`端口 pid == child.id()`。
- **启动健康检查要连续 2 次成功**：warmup 期单次 200 不算稳定，不能当作已就绪。
- **停止要等到端口让出，而不是等 `kill` 返回。** `kill` 返回只代表信号已发出，socket 还要几十毫秒才拆。实测切模型场景从「已请求结束」到「stopped」**668 ms**，800 ms 预算只剩 132 ms —— 这个预算不许收紧。
- **`reattach` 与 `start` 是两条路。** 端口健康就直接接管（`reattach`），不进启动主流程，隔离、引擎、驱动、显存校验全部跳过。**测新版必须先停再启**，否则测到的是旧进程。

### 10.6 日志：首行如实报「实际生效级别」

首行固定输出四项，含义各不相同：

| 字段 | 含义 |
|---|---|
| `settings_level` | 设置里存的级别（用户意图） |
| `effective_level` | **实际生效**的级别 —— 判断 DEBUG 开没开只看这个 |
| `level_source` | 级别来自哪里：`settings` 还是 `RUST_LOG` |
| `rust_log_ignored` | 仅在 `RUST_LOG` 非法回落时出现，附被忽略的原文 |

早先只打设置值：配上非法的 `RUST_LOG` 时，日志写「INFO」而实际级别已被环境变量改掉 —— 判据拿「配置值」当「生效值」。现在设置值与生效值来自同一份判断函数，不会再各说各话。

### 10.7 旁车（sidecar）配置

模型目录下的旁车 JSON 用于覆盖启动参数。解析失败时**不能带着半截状态继续**：早期版本在解析失败处用 `?` 早退，进程没起来、状态却停在 `Starting`，两个按钮都变灰，只能重开 APP。

现在的要求：任何早退路径都必须把状态复位成 `Stopped`，且失败原因要能显示出来。判据不是「日志里有复位那一行」，而是**能再点一次启动、并且真的走进启动主体**。
