//! GGUF 文件头的元数据解析。
//!
//! 只读取元数据段与张量信息段，不加载张量数据，因此对 8GB 的文件也是毫秒级完成。
//!
//! 用途有两个：
//!
//! 1. **导入本地模型时填充真实参数。** 此前 `import_local_model` 对任何导入文件都写死
//!    `context_length: 4096` —— 于是 Spark-X2.5-4B 原生的 1M 上下文被当成了 4K，
//!    Ornith-1.5-9B 的 256K 被当成了 4K。
//! 2. **为 KV 缓存估算提供层结构。** 混合注意力模型（状态空间层 / 滑动窗口层）的 KV
//!    不随上下文线性增长，只有全注意力层才增长。用总层数算会高估数倍。
//!
//! 交叉验证：`scripts/gguf-probe.py` 是同一算法的 Python 版探针，两者对同一批模型
//! 必须给出相同结果。改本文件后请同步跑一次探针比对。
//!
//! 格式要点（踩过的坑）：
//!
//! - 所有整数为小端。
//! - 字符串是 u64 长度前缀 + UTF-8 字节，**没有结尾的 `\0`**。
//! - 数组是 u32 元素类型 + u64 元素个数 + 元素数据。
//! - **张量的维度是 u64 而非 u32。** 按 u32 读会让每个张量少读 `4 * n_dims` 字节，
//!   偏移逐项累积后必然错位，表现为「读到天文数字的字符串长度」。

// 本模块按项目约定使用中文标识符，snake_case / upper_case 检查在此不适用。
#![allow(non_snake_case, non_upper_case_globals)]

use std::{
    fs::File,
    io::{BufReader, Read, Seek, SeekFrom},
    path::Path,
};

use thiserror::Error;

const GGUF_MAGIC: &[u8; 4] = b"GGUF";

/// 1 GiB 的字节数。显存口径统一走 1024 进制（`nvidia-smi` 报的是 MiB），
/// 不要用 10⁹ —— 与可用显存相减时会引入约 7% 的系统偏差。
pub const 每GiB字节: f64 = 1024.0 * 1024.0 * 1024.0;

/// 字符串长度上限，防止损坏文件让程序分配巨量内存。
const MAX_STRING_BYTES: u64 = 16 * 1024 * 1024;

/// 元数据类型码（GGUF 规范）
mod 类型码 {
    pub const UINT8: u32 = 0;
    pub const INT8: u32 = 1;
    pub const UINT16: u32 = 2;
    pub const INT16: u32 = 3;
    pub const UINT32: u32 = 4;
    pub const INT32: u32 = 5;
    pub const FLOAT32: u32 = 6;
    pub const BOOL: u32 = 7;
    pub const STRING: u32 = 8;
    pub const ARRAY: u32 = 9;
    pub const UINT64: u32 = 10;
    pub const INT64: u32 = 11;
    pub const FLOAT64: u32 = 12;
}

/// 定长类型的字节数；返回 None 表示该类型不是定长（字符串 / 数组）。
fn 定长字节数(类型: u32) -> Option<u64> {
    Some(match 类型 {
        类型码::UINT8 | 类型码::INT8 | 类型码::BOOL => 1,
        类型码::UINT16 | 类型码::INT16 => 2,
        类型码::UINT32 | 类型码::INT32 | 类型码::FLOAT32 => 4,
        类型码::UINT64 | 类型码::INT64 | 类型码::FLOAT64 => 8,
        _ => return None,
    })
}

#[derive(Debug, Error)]
pub enum GgufError {
    #[error("读取模型文件失败：{0}")]
    Io(#[from] std::io::Error),
    #[error("这不是 GGUF 文件（文件头标识不匹配）")]
    BadMagic,
    #[error("不支持的 GGUF 版本 {0}（本程序支持 v2 / v3）")]
    UnsupportedVersion(u32),
    #[error("元数据类型码 {0} 不受支持")]
    UnsupportedType(u32),
    #[error("文件结构异常：{0}")]
    Malformed(String),
}

/// 从 GGUF 头部解析出来的元数据。字段全部可选：缺失时由调用方决定兜底值。
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GgufMetadata {
    /// 架构标识，如 `qwen3` / `qwen35` / `spark2_5`
    pub architecture: String,
    /// 模型自报名称
    pub name: Option<String>,
    /// 文件里的张量个数
    pub tensor_count: u64,
    /// 参数量：由张量维度累加得到（元数据段常不提供 `general.parameter_count`）
    pub parameter_count: u64,
    /// `general.file_type`：llama.cpp 的量化档位枚举
    pub file_type: Option<u32>,
    /// 总层数
    pub block_count: Option<u32>,
    /// 模型训练时的原生上下文长度
    pub context_length: Option<u32>,
    /// 嵌入维度
    pub embedding_length: Option<u32>,
    /// 注意力头数
    pub head_count: Option<u32>,
    /// KV 头数（GQA/MQA 下小于 head_count）
    pub head_count_kv: Option<u32>,
    /// 单个 key 向量的维度
    pub key_length: Option<u32>,
    /// 单个 value 向量的维度
    pub value_length: Option<u32>,
    /// MoE 专家总数
    pub expert_count: Option<u32>,
    /// 每个 token 激活的专家数
    pub expert_used_count: Option<u32>,
    /// 状态空间混合模型：每 N 层包含 1 层全注意力（如 Qwen3.5）
    pub full_attention_interval: Option<u32>,
    /// 多 token 预测（MTP）附加层数，键名 `nextn_predict_layers`。
    ///
    /// 这些层**不计入主干**：llama.cpp 加载时会连同其张量一起跳过
    /// （日志里的 `unused tensor blk.64.* ... ignoring`），既不参与推理、
    /// 也不产生 KV。但它们**会被算进 `block_count`**，所以只要拿
    /// `block_count` 去推层结构，就必须先把这部分减掉。
    pub mtp_layers: Option<u32>,
    /// 滑动窗口大小
    pub sliding_window: Option<u32>,
    /// 每层是否使用滑动窗口；`false` 表示该层是全注意力
    pub sliding_window_pattern: Vec<bool>,
    /// (ggml 张量类型码, 张量个数) 的统计，用于核对文件与标称量化档是否一致
    pub tensor_type_histogram: Vec<(u32, u64)>,
}

impl GgufMetadata {
    /// 全注意力层数 —— 只有这些层的 KV 随上下文线性增长。
    ///
    /// 判定优先序：
    /// 1. 有 `sliding_window_pattern` 时按数组精确计数；
    /// 2. 有 `full_attention_interval` 时按「每 N 层 1 层全注意力」向上取整，
    ///    **且先扣除 MTP 层**（`nextn_predict_layers`）—— 那些层不参与推理；
    /// 3. 都没有则为纯全注意力，即全部层。
    pub fn full_attention_layers(&self) -> u32 {
        let 总层数 = self.block_count.unwrap_or(0);
        if 总层数 == 0 {
            return 0;
        }

        let 模式长度 = self.sliding_window_pattern.len() as u32;
        if 模式长度 > 0 {
            let 全注意力 = self
                .sliding_window_pattern
                .iter()
                .filter(|是否滑窗| !**是否滑窗)
                .count() as u32;
            // 模式数组若未覆盖全部层，剩余层按全注意力保守计算，宁可高估显存。
            let 未覆盖 = 总层数.saturating_sub(模式长度);
            return 全注意力.saturating_add(未覆盖);
        }

        if let Some(间隔) = self.full_attention_interval.filter(|值| *值 > 1) {
            // ⚠️ 必须用「参与推理的主干层数」去除，而不是 block_count。
            //
            // `block_count` 把 MTP 层（`nextn_predict_layers`）也算在内，但
            // llama.cpp 根本不加载那些层 —— 它们不产生 KV，却会把这里的除法
            // 推高一档。实测两例：
            //   Qwen3.8-27B：block_count=65、MTP=1 → 主干 64 层 → 16 层全注意力
            //                用 65 除 4 向上取整得 17，**高估 1 层（6.25%）**
            //   Ornith-1.5-9B：block_count=33、MTP=1 → 主干 32 层 → 8 层
            //                用 33 除 4 向上取整得 9，**高估 1 层（12.5%）**
            // 张量清单亦佐证：Qwen3.8 的 SSM 张量只出现在
            // blk.0/1/2、4/5/6、…、60/61/62，跳过 3/7/…/63 共 16 层。
            //
            // 高估的方向是「保守」的 —— 不会撑爆显存，但会让规划器把
            // 上下文上限算小，白白浪费显存。所以照实扣。
            let 主干层数 = 总层数.saturating_sub(self.mtp_layers.unwrap_or(0));
            if 主干层数 == 0 {
                // MTP 层数异常（≥ 总层数）时退化为原口径，宁可高估不可低估。
                return (总层数 + 间隔 - 1) / 间隔;
            }
            // 向上取整：宁可高估一层，也不要在临界点上撑爆显存。
            return (主干层数 + 间隔 - 1) / 间隔;
        }

        总层数
    }

    /// 使用滑动窗口的层数。
    pub fn sliding_window_layers(&self) -> u32 {
        let 总层数 = self.block_count.unwrap_or(0);
        if 总层数 == 0 {
            return 0;
        }
        let 滑窗 = self
            .sliding_window_pattern
            .iter()
            .filter(|是否滑窗| **是否滑窗)
            .count() as u32;
        if !self.sliding_window_pattern.is_empty() {
            return 滑窗;
        }
        // 只给了窗口大小却没给逐层模式时，按「除全注意力层外都是滑窗层」处理。
        if self.sliding_window.is_some() && self.full_attention_layers() < 总层数 {
            return 总层数 - self.full_attention_layers();
        }
        0
    }

    /// 单个注意力层、单个 token 占用的 KV 字节数。
    fn 每层每token字节(&self, 元素字节: f64) -> f64 {
        let kv头数 = self
            .head_count_kv
            .or(self.head_count)
            .unwrap_or(0);
        if kv头数 == 0 {
            return 0.0;
        }
        let key维度 = self.key_length.unwrap_or(128) as f64;
        let value维度 = self.value_length.unwrap_or(key维度 as u32) as f64;
        kv头数 as f64 * (key维度 + value维度) * 元素字节
    }

    /// 每 token 的 KV 字节数（只计随上下文增长的全注意力层）。
    pub fn kv_bytes_per_token(&self, 元素字节: f64) -> f64 {
        self.full_attention_layers() as f64 * self.每层每token字节(元素字节)
    }

    /// 给定上下文长度时的 KV 缓存总量（GiB）。
    ///
    /// 滑动窗口层的占用有上限 `min(ctx, window)`，不随上下文无限制增长。
    ///
    /// 单位是 GiB（1024³）而非 GB（10⁹）：显存侧的每一个量最终都要和
    /// `nvidia-smi` 报的可用显存相减，而它报的是 MiB —— 混用两种进制会在
    /// 预算里引入约 7% 的系统偏差。全项目显存一律走 GiB。
    pub fn kv_gb(&self, ctx: u64, 元素字节: f64) -> f64 {
        let 每层 = self.每层每token字节(元素字节);
        let mut 字节 = self.full_attention_layers() as f64 * 每层 * ctx as f64;
        if let Some(窗口) = self.sliding_window {
            let 有效窗口 = ctx.min(窗口 as u64) as f64;
            字节 += self.sliding_window_layers() as f64 * 每层 * 有效窗口;
        }
        字节 / 每GiB字节
    }

    /// 不随上下文增长的那部分 KV 占用（GiB）。
    ///
    /// 滑动窗口层的占用上限是 `min(ctx, 窗口)`：上下文一旦超过窗口，这部分
    /// 就固定下来，与 ctx 再无关系。预算反解必须先扣掉它 —— 否则按每 token
    /// 线性反解会把 ctx 算大（Spark-X2.5-4B 的 27 层滑窗 × 512 窗口
    /// = 54 MiB，约占 8G 卡 KV 预算的 1.8%）。
    ///
    /// 注意：该式只在 `ctx ≥ 窗口` 时精确。窗口通常只有 512～4096，而预算
    /// 反解出的 ctx 至少 2048，所以实际不会落进高估的那一小段区间。
    pub fn kv_fixed_gb(&self, 元素字节: f64) -> f64 {
        match self.sliding_window {
            Some(窗口) => {
                self.sliding_window_layers() as f64
                    * self.每层每token字节(元素字节)
                    * 窗口 as f64
                    / 每GiB字节
            }
            None => 0.0,
        }
    }

    /// 实测的每参数字节数（文件大小 ÷ 参数量）。用于替代硬编码的 `quant_bpp`。
    pub fn bytes_per_param(&self, 文件字节: u64) -> Option<f64> {
        if self.parameter_count == 0 {
            return None;
        }
        Some(文件字节 as f64 / self.parameter_count as f64)
    }

    /// 面向用户的参数量标签，如 `9.2B` / `620M`。
    pub fn parameter_label(&self) -> Option<String> {
        if self.parameter_count == 0 {
            return None;
        }
        let 十亿 = self.parameter_count as f64 / 1e9;
        Some(if 十亿 >= 1.0 {
            format!("{十亿:.1}B")
        } else {
            format!("{:.0}M", self.parameter_count as f64 / 1e6)
        })
    }

    /// `general.file_type` 对应的量化档位名。
    pub fn file_type_label(&self) -> Option<&'static str> {
        Some(match self.file_type? {
            0 => "F32",
            1 => "F16",
            2 => "Q4_0",
            3 => "Q4_1",
            4 => "Q4_1_SOME_F16",
            7 => "Q8_0",
            8 => "Q5_0",
            9 => "Q5_1",
            10 => "Q2_K",
            11 => "Q3_K_S",
            12 => "Q3_K_M",
            13 => "Q3_K_L",
            14 => "Q4_K_S",
            15 => "Q4_K_M",
            16 => "Q5_K_S",
            17 => "Q5_K_M",
            18 => "Q6_K",
            32 => "BF16",
            _ => return None,
        })
    }

    /// 是否走 MoE 结构。
    pub fn is_moe(&self) -> bool {
        matches!(self.expert_count, Some(数量) if 数量 > 1)
    }

    /// MoE 模型每个 token 实际激活的参数量估算（按激活专家占比折算）。
    pub fn active_parameters(&self) -> Option<u64> {
        match (self.expert_count, self.expert_used_count) {
            (Some(总数), Some(激活数)) if 总数 > 0 && 激活数 > 0 && 激活数 < 总数 => {
                Some(self.parameter_count / 总数 as u64 * 激活数 as u64)
            }
            _ => None,
        }
    }
}

/// 解析一个 GGUF 文件，只读元数据与张量信息。
pub fn parse_file(path: &Path) -> Result<GgufMetadata, GgufError> {
    let 文件 = File::open(path)?;
    let mut 读取器 = BufReader::new(文件);
    解析(&mut 读取器)
}

/// 判断某个元数据键是否与模型结构相关 —— 只有这些需要真正解析。
///
/// 词表（`tokenizer.ggml.tokens` / `merges`）可达十几万项，跳过它们是把
/// 解析耗时从秒级降到毫秒级的关键。用后缀匹配以兼容各架构的键名前缀。
fn 是结构键(键: &str) -> bool {
    const 关注的后缀: &[&str] = &[
        "block_count",
        "context_length",
        "embedding_length",
        "attention.head_count",
        "attention.head_count_kv",
        "attention.key_length",
        "attention.value_length",
        "attention.sliding_window",
        "attention.sliding_window_pattern",
        "expert_count",
        "expert_used_count",
        "full_attention_interval",
        "nextn_predict_layers",
    ];
    关注的后缀.iter().any(|后缀| 键.ends_with(后缀))
}

fn 解析<R: Read + Seek>(读取器: &mut R) -> Result<GgufMetadata, GgufError> {
    let mut 魔数 = [0u8; 4];
    读取器.read_exact(&mut 魔数)?;
    if &魔数 != GGUF_MAGIC {
        return Err(GgufError::BadMagic);
    }

    let 版本 = 读u32(读取器)?;
    if 版本 < 2 {
        return Err(GgufError::UnsupportedVersion(版本));
    }

    let 张量数 = 读u64(读取器)?;
    let 元数据项数 = 读u64(读取器)?;

    let mut 元数据 = GgufMetadata {
        tensor_count: 张量数,
        ..Default::default()
    };

    for _ in 0..元数据项数 {
        let 键 = 读字符串(读取器)?;
        let 类型 = 读u32(读取器)?;

        if !是结构键(&键) && 键 != "general.architecture" && 键 != "general.name"
            && 键 != "general.file_type"
        {
            跳过值(读取器, 类型)?;
            continue;
        }

        match 键.as_str() {
            "general.architecture" => 元数据.architecture = 读字符串(读取器)?,
            "general.name" => 元数据.name = Some(读字符串(读取器)?),
            "general.file_type" => 元数据.file_type = Some(读u32(读取器)?),
            其他 => 把结构值写进元数据(&mut 元数据, 其他, 读取器, 类型)?,
        }
    }

    // 张量信息段：累加维度得到参数量，并统计量化类型分布。
    元数据.parameter_count = 读张量段落(读取器, 张量数, &mut 元数据.tensor_type_histogram)?;

    Ok(元数据)
}

/// 把结构相关的值写进 `元数据`。键名用后缀匹配，兼容各架构前缀。
fn 把结构值写进元数据<R: Read + Seek>(
    元数据: &mut GgufMetadata,
    键: &str,
    读取器: &mut R,
    类型: u32,
) -> Result<(), GgufError> {
    if 键.ends_with("attention.sliding_window_pattern") {
        // 逐层布尔数组：true 表示该层用滑动窗口，false 表示全注意力。
        let 个数 = 读数组头(读取器, 类型, 类型码::BOOL)?;
        元数据.sliding_window_pattern = Vec::with_capacity(个数 as usize);
        for _ in 0..个数 {
            let mut 字节 = [0u8; 1];
            读取器.read_exact(&mut 字节)?;
            元数据.sliding_window_pattern.push(字节[0] != 0);
        }
        return Ok(());
    }

    // 其余都是整数标量。有的模型把 head_count_kv 写成数组（每层不同），
    // 此时取最大值 —— 宁可高估 KV，也不要在临界点上撑爆显存。
    let 值 = match 类型 {
        类型码::UINT32 => 读u32(读取器)? as u64,
        类型码::INT32 => 读u32(读取器)? as i32 as u64,
        类型码::UINT64 => 读u64(读取器)?,
        类型码::INT64 => 读u64(读取器)? as i64 as u64,
        类型码::ARRAY => {
            let 元素类型 = 读u32(读取器)?;
            let 个数 = 读u64(读取器)?;
            let 字节数 = 定长字节数(元素类型)
                .ok_or(GgufError::UnsupportedType(元素类型))?;
            let mut 最大 = 0u64;
            for _ in 0..个数 {
                let 当前 = match 字节数 {
                    1 => {
                        let mut 缓冲 = [0u8; 1];
                        读取器.read_exact(&mut 缓冲)?;
                        缓冲[0] as u64
                    }
                    4 => 读u32(读取器)? as u64,
                    8 => 读u64(读取器)?,
                    其他 => {
                        return Err(GgufError::Malformed(format!(
                            "数组元素字节数异常：{其他}"
                        )))
                    }
                };
                最大 = 最大.max(当前);
            }
            最大
        }
        其他 => return Err(GgufError::UnsupportedType(其他)),
    };

    let 值32 = u32::try_from(值).unwrap_or(u32::MAX);
    if 键.ends_with("block_count") {
        元数据.block_count = Some(值32);
    } else if 键.ends_with("context_length") && !键.contains("rope") {
        元数据.context_length = Some(值32);
    } else if 键.ends_with("embedding_length") {
        元数据.embedding_length = Some(值32);
    } else if 键.ends_with("attention.head_count_kv") {
        元数据.head_count_kv = Some(值32);
    } else if 键.ends_with("attention.head_count") {
        元数据.head_count = Some(值32);
    } else if 键.ends_with("attention.key_length") {
        元数据.key_length = Some(值32);
    } else if 键.ends_with("attention.value_length") {
        元数据.value_length = Some(值32);
    } else if 键.ends_with("attention.sliding_window") {
        元数据.sliding_window = Some(值32);
    } else if 键.ends_with("expert_count") {
        元数据.expert_count = Some(值32);
    } else if 键.ends_with("expert_used_count") {
        元数据.expert_used_count = Some(值32);
    } else if 键.ends_with("full_attention_interval") {
        元数据.full_attention_interval = Some(值32);
    } else if 键.ends_with("nextn_predict_layers") {
        // 多 token 预测层数。这些层不参与推理、不产生 KV，但计在 block_count 里，
        // 推算层结构时必须扣除（见 full_attention_layers 的注释）。
        元数据.mtp_layers = Some(值32);
    }
    Ok(())
}

/// 读张量信息段，返回参数量（各张量维度之积的和）。
fn 读张量段落<R: Read + Seek>(
    读取器: &mut R,
    张量数: u64,
    直方图: &mut Vec<(u32, u64)>,
) -> Result<u64, GgufError> {
    let mut 总参数 = 0u64;
    for _ in 0..张量数 {
        let _名字 = 读字符串(读取器)?;
        let 维数 = 读u32(读取器)?;
        if 维数 > 8 {
            return Err(GgufError::Malformed(format!("张量维度数异常：{维数}")));
        }
        let mut 本张量参数 = 1u64;
        for _ in 0..维数 {
            // 关键：维度是 u64，不是 u32。
            let 维度 = 读u64(读取器)?;
            本张量参数 = 本张量参数.saturating_mul(维度);
        }
        let 张量类型 = 读u32(读取器)?;
        let _偏移 = 读u64(读取器)?;

        总参数 = 总参数.saturating_add(本张量参数);
        match 直方图.iter_mut().find(|(类型, _)| *类型 == 张量类型) {
            Some((_, 个数)) => *个数 += 1,
            None => 直方图.push((张量类型, 1)),
        }
    }
    Ok(总参数)
}

/// 读一个数组的头部，校验元素类型并返回元素个数。
fn 读数组头<R: Read>(读取器: &mut R, 类型: u32, 期望元素类型: u32) -> Result<u64, GgufError> {
    if 类型 != 类型码::ARRAY {
        return Err(GgufError::Malformed(format!(
            "期望数组类型，实际类型码为 {类型}"
        )));
    }
    let 元素类型 = 读u32(读取器)?;
    if 元素类型 != 期望元素类型 {
        return Err(GgufError::Malformed(format!(
            "数组元素类型不符：期望 {期望元素类型}，实际 {元素类型}"
        )));
    }
    let 个数 = 读u64(读取器)?;
    Ok(个数)
}

/// 跳过不需要的值。数组走长度计算而非逐项读取，避免读入十几万条词表。
fn 跳过值<R: Read + Seek>(读取器: &mut R, 类型: u32) -> Result<(), GgufError> {
    match 类型 {
        类型码::STRING => {
            let 长度 = 读u64(读取器)?;
            读取器.seek(SeekFrom::Current(长度 as i64))?;
        }
        类型码::ARRAY => {
            let 元素类型 = 读u32(读取器)?;
            let 个数 = 读u64(读取器)?;
            if 元素类型 == 类型码::STRING {
                // 变长元素，只能逐个跳过。用 seek 而非 read，避免分配字符串。
                for _ in 0..个数 {
                    let 长度 = 读u64(读取器)?;
                    读取器.seek(SeekFrom::Current(长度 as i64))?;
                }
            } else if let Some(字节数) = 定长字节数(元素类型) {
                读取器.seek(SeekFrom::Current((字节数 * 个数) as i64))?;
            } else {
                return Err(GgufError::UnsupportedType(元素类型));
            }
        }
        其他 => {
            let 字节数 = 定长字节数(其他).ok_or(GgufError::UnsupportedType(其他))?;
            读取器.seek(SeekFrom::Current(字节数 as i64))?;
        }
    }
    Ok(())
}

fn 读u32<R: Read>(读取器: &mut R) -> Result<u32, GgufError> {
    let mut 缓冲 = [0u8; 4];
    读取器.read_exact(&mut 缓冲)?;
    Ok(u32::from_le_bytes(缓冲))
}

fn 读u64<R: Read>(读取器: &mut R) -> Result<u64, GgufError> {
    let mut 缓冲 = [0u8; 8];
    读取器.read_exact(&mut 缓冲)?;
    Ok(u64::from_le_bytes(缓冲))
}

fn 读字符串<R: Read>(读取器: &mut R) -> Result<String, GgufError> {
    let 长度 = 读u64(读取器)?;
    if 长度 > MAX_STRING_BYTES {
        return Err(GgufError::Malformed(format!(
            "字符串长度异常：{长度} 字节，疑似文件损坏或解析错位"
        )));
    }
    let mut 缓冲 = vec![0u8; 长度 as usize];
    读取器.read_exact(&mut 缓冲)?;
    Ok(String::from_utf8_lossy(&缓冲).into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    // ---- 构造测试用 GGUF 字节流 ----

    fn 追加字符串(数据: &mut Vec<u8>, 文本: &str) {
        数据.extend_from_slice(&(文本.len() as u64).to_le_bytes());
        数据.extend_from_slice(文本.as_bytes());
    }

    fn 追加字符串KV(数据: &mut Vec<u8>, 键: &str, 值: &str) {
        追加字符串(数据, 键);
        数据.extend_from_slice(&类型码::STRING.to_le_bytes());
        追加字符串(数据, 值);
    }

    fn 追加u32KV(数据: &mut Vec<u8>, 键: &str, 值: u32) {
        追加字符串(数据, 键);
        数据.extend_from_slice(&类型码::UINT32.to_le_bytes());
        数据.extend_from_slice(&值.to_le_bytes());
    }

    fn 追加布尔数组KV(数据: &mut Vec<u8>, 键: &str, 值: &[bool]) {
        追加字符串(数据, 键);
        数据.extend_from_slice(&类型码::ARRAY.to_le_bytes());
        数据.extend_from_slice(&类型码::BOOL.to_le_bytes());
        数据.extend_from_slice(&(值.len() as u64).to_le_bytes());
        for 项 in 值 {
            数据.push(if *项 { 1 } else { 0 });
        }
    }

    fn 追加张量(数据: &mut Vec<u8>, 名称: &str, 维度: &[u64], 类型: u32) {
        追加字符串(数据, 名称);
        数据.extend_from_slice(&(维度.len() as u32).to_le_bytes());
        for 值 in 维度 {
            数据.extend_from_slice(&值.to_le_bytes());
        }
        数据.extend_from_slice(&类型.to_le_bytes());
        数据.extend_from_slice(&0u64.to_le_bytes());
    }

    /// 造一份结构完整的 GGUF：混合注意力 + 两个张量。
    fn 造混合注意力GGUF() -> Vec<u8> {
        let mut 数据 = Vec::new();
        数据.extend_from_slice(GGUF_MAGIC);
        数据.extend_from_slice(&3u32.to_le_bytes()); // 版本
        数据.extend_from_slice(&2u64.to_le_bytes()); // 张量数
        数据.extend_from_slice(&8u64.to_le_bytes()); // 元数据项数

        追加字符串KV(&mut 数据, "general.architecture", "qwen35");
        追加字符串KV(&mut 数据, "general.name", "Ornith-1.5-9B");
        追加u32KV(&mut 数据, "qwen35.block_count", 33);
        追加u32KV(&mut 数据, "qwen35.context_length", 262144);
        追加u32KV(&mut 数据, "qwen35.attention.head_count", 16);
        追加u32KV(&mut 数据, "qwen35.attention.head_count_kv", 4);
        追加u32KV(&mut 数据, "qwen35.attention.key_length", 256);
        追加u32KV(&mut 数据, "qwen35.attention.value_length", 256);

        追加张量(&mut 数据, "token_embd.weight", &[4096, 151936], 12);
        追加张量(&mut 数据, "output.weight", &[4096, 151936], 14);
        数据
    }

    #[test]
    fn parses_architecture_and_scalar_fields() {
        let 数据 = 造混合注意力GGUF();
        let mut 读取器 = Cursor::new(数据);
        let 元数据 = 解析(&mut 读取器).expect("应当解析成功");

        assert_eq!(元数据.architecture, "qwen35");
        assert_eq!(元数据.name.as_deref(), Some("Ornith-1.5-9B"));
        assert_eq!(元数据.block_count, Some(33));
        assert_eq!(元数据.context_length, Some(262144));
        assert_eq!(元数据.head_count, Some(16));
        assert_eq!(元数据.head_count_kv, Some(4));
        assert_eq!(元数据.key_length, Some(256));
        assert_eq!(元数据.value_length, Some(256));
    }

    #[test]
    fn accumulates_parameter_count_from_tensor_dimensions() {
        let 数据 = 造混合注意力GGUF();
        let mut 读取器 = Cursor::new(数据);
        let 元数据 = 解析(&mut 读取器).unwrap();

        // 两个 4096 × 151936 的张量
        assert_eq!(元数据.parameter_count, 2 * 4096 * 151936);
        assert_eq!(元数据.parameter_label().as_deref(), Some("1.2B"));
        assert_eq!(元数据.tensor_count, 2);
    }

    #[test]
    fn tensor_dimensions_are_read_as_u64() {
        // 维度值超过 u32 上限时必须仍然正确 —— 这正是当初按 u32 读会错位的原因。
        let mut 数据 = Vec::new();
        数据.extend_from_slice(GGUF_MAGIC);
        数据.extend_from_slice(&3u32.to_le_bytes());
        数据.extend_from_slice(&1u64.to_le_bytes());
        数据.extend_from_slice(&0u64.to_le_bytes()); // 无元数据
        追加张量(&mut 数据, "big.weight", &[5_000_000_000], 1);

        let mut 读取器 = Cursor::new(数据);
        let 元数据 = 解析(&mut 读取器).unwrap();
        assert_eq!(元数据.parameter_count, 5_000_000_000);
    }

    #[test]
    fn full_attention_layers_uses_interval_for_ssm_hybrid() {
        let mut 元数据 = GgufMetadata {
            block_count: Some(33),
            ..Default::default()
        };
        // 33 层、每 4 层 1 层全注意力 → 向上取整 9 层
        assert_eq!(元数据.full_attention_layers(), 33);

        元数据.full_attention_interval = Some(4);
        assert_eq!(元数据.full_attention_layers(), 9);
        assert_eq!(元数据.sliding_window_layers(), 0);

        // 声明了 1 层 MTP 后，主干只剩 32 层 → 32 ÷ 4 = 8 层全注意力。
        // 这是 Ornith-1.5-9B 的真实口径：block_count=33 里的 blk.32 是 MTP 层，
        // llama.cpp 不加载它，自然也不为它分配 KV。
        元数据.mtp_layers = Some(1);
        assert_eq!(元数据.full_attention_layers(), 8);
    }

    #[test]
    fn mtp_layers_are_ignored_for_interval_based_layer_count() {
        // Qwen3.8-27B 的真实结构：block_count=65（含 1 层 MTP）、间隔 4。
        //
        // 修正前：65 ÷ 4 向上取整 = 17 层 → 68 KiB/token（高估 6.25%）
        // 修正后：64 ÷ 4 = 16 层 → 64 KiB/token
        //
        // 交叉佐证（scripts/张量清单.md）：SSM 张量只出现在
        // blk.0/1/2、4/5/6、…、60/61/62，跳过的正是 3/7/…/63 共 16 层。
        let 元数据 = GgufMetadata {
            block_count: Some(65),
            full_attention_interval: Some(4),
            mtp_layers: Some(1),
            head_count_kv: Some(4),
            key_length: Some(256),
            value_length: Some(256),
            ..Default::default()
        };

        assert_eq!(元数据.full_attention_layers(), 16);
        // 4 头 × (256+256) × 2 字节 × 16 层 = 64 KiB/token
        assert_eq!(元数据.kv_bytes_per_token(2.0), 65_536.0);

        // MTP 层数异常（≥ 总层数）时应退化为原口径，宁可高估不可低估。
        let 异常 = GgufMetadata {
            block_count: Some(65),
            full_attention_interval: Some(4),
            mtp_layers: Some(65),
            ..Default::default()
        };
        assert_eq!(异常.full_attention_layers(), 17);
    }

    #[test]
    fn full_attention_layers_uses_pattern_for_sliding_window() {
        let 模式: Vec<bool> = (0..36).map(|序号| (序号 + 1) % 4 != 0).collect();
        let 元数据 = GgufMetadata {
            block_count: Some(36),
            sliding_window: Some(512),
            sliding_window_pattern: 模式,
            ..Default::default()
        };
        // 每 4 层里有 1 层是全注意力 → 9 层
        assert_eq!(元数据.full_attention_layers(), 9);
        assert_eq!(元数据.sliding_window_layers(), 27);
    }

    #[test]
    fn kv_per_token_matches_real_model_measurements() {
        // 本机 6 个真实 GGUF 文件的实测层结构（详见 CLAUDE.md「实机校准」章）。
        //
        // KV 只取决于层结构，与参数量无关 —— 注意 Qwen3-1.7B 的每 token KV
        // (112 KiB) 反而 **大于** Ornith-9B (32 KiB)，因为后者 32 层主干里只有
        // 8 层是全注意力。这正是「按参数量估算 KV」会同时出现高估与低估的原因，
        // 也是这个测试存在的意义：公式一旦被改回按参数量算，这里立刻会红。
        //
        // (名称, 层数, KV头, 头维度, 期望全注意力层, 期望每token字节, 全注意力间隔, 是否逐层滑窗模式, MTP层数)
        //
        // MTP 层数单列一栏：它决定「用 block_count 推层结构时要先扣掉几层」。
        // Ornith-1.5 是 33 层含 1 层 MTP → 主干 32 → 8 层全注意力（早前算成 9）。
        type 用例 = (&'static str, u32, u32, u32, u32, f64, Option<u32>, bool, Option<u32>);
        let 用例集: &[用例] = &[
            ("Ornith-1.5-9B Q4_K_M", 33, 4, 256, 8, 32_768.0, Some(4), false, Some(1)),
            ("Ornith-1.5-9B Q6_K", 33, 4, 256, 8, 32_768.0, Some(4), false, Some(1)),
            ("Ornith-1.5-9B Q8_0", 33, 4, 256, 8, 32_768.0, Some(4), false, Some(1)),
            ("Spark-X2.5-4B BF16", 36, 4, 256, 9, 36_864.0, None, true, None),
            ("Qwen3-1.7B Q8_0", 28, 8, 128, 28, 114_688.0, None, false, None),
            ("Qwen3-8B Q4_K_M", 36, 8, 128, 36, 147_456.0, None, false, None),
        ];

        for (名称, 层数, kv头, 头维度, 期望全注意力, 期望字节, 间隔, 用滑窗模式, mtp) in 用例集 {
            let 元数据 = GgufMetadata {
                block_count: Some(*层数),
                head_count_kv: Some(*kv头),
                key_length: Some(*头维度),
                value_length: Some(*头维度),
                full_attention_interval: *间隔,
                mtp_layers: *mtp,
                // Spark 用逐层布尔数组表达滑窗；Ornith 用间隔表达；Qwen3 两者皆无。
                sliding_window_pattern: if *用滑窗模式 {
                    (0..*层数).map(|序号| (序号 + 1) % 4 != 0).collect()
                } else {
                    Vec::new()
                },
                ..Default::default()
            };

            assert_eq!(
                元数据.full_attention_layers(),
                *期望全注意力,
                "{名称} 的全注意力层数不符"
            );
            let 每token = 元数据.kv_bytes_per_token(2.0);
            assert!(
                (每token - 期望字节).abs() < 1.0,
                "{名称} 每 token KV 不符：实际 {每token}，期望 {期望字节}"
            );
        }
    }

    #[test]
    fn gib_conversion_is_binary_not_decimal() {
        // 显存口径必须是 1024 进制：`nvidia-smi` 报的是 MiB，所有显存估算最终
        // 都要和它相减。用 10⁹ 会让每一项差约 7%，在 8G 卡上就是半个 GiB 的
        // 误判 —— 足够把「能全量跑」错判成「要卸载」。
        assert_eq!(每GiB字节, 1024.0 * 1024.0 * 1024.0);
        assert!((每GiB字节 - 1e9).abs() > 7e7, "不能退化成十进制的 10⁹");
    }

    #[test]
    fn kv_gb_caps_sliding_window_layers() {
        // 滑动窗口层的占用不随上下文增长，超过窗口后应固定。
        let 元数据 = GgufMetadata {
            block_count: Some(36),
            head_count_kv: Some(4),
            key_length: Some(256),
            value_length: Some(256),
            sliding_window: Some(512),
            sliding_window_pattern: (0..36).map(|序号| (序号 + 1) % 4 != 0).collect(),
            ..Default::default()
        };
        let 短 = 元数据.kv_gb(512, 2.0);
        let 长 = 元数据.kv_gb(131_072, 2.0);
        // 窗口已经吃满，两者差异只应来自 9 层全注意力的增长。
        let 增长 = 长 - 短;
        let 预期 = 9.0 * 4.0 * (256.0 + 256.0) * 2.0 * (131_072.0 - 512.0) / 每GiB字节;
        assert!((增长 - 预期).abs() < 0.01, "实际增长 {增长}，预期 {预期}");
    }

    #[test]
    fn kv_fixed_gb_covers_only_sliding_window_layers() {
        // 滑窗层占用的固定部分：27 层 × 4096 字节/层/token × 512 窗口。
        let 滑窗模型 = GgufMetadata {
            block_count: Some(36),
            head_count_kv: Some(4),
            key_length: Some(256),
            value_length: Some(256),
            sliding_window: Some(512),
            sliding_window_pattern: (0..36).map(|序号| (序号 + 1) % 4 != 0).collect(),
            ..Default::default()
        };
        let 固定 = 滑窗模型.kv_fixed_gb(2.0);
        let 预期 = 27.0 * 4096.0 * 512.0 / 每GiB字节;
        assert!((固定 - 预期).abs() < 1e-9, "实际 {固定}，预期 {预期}");
        // 54 MiB 量级 —— 数值不大，但它在「预算 ÷ 每 token」的反解里
        // 必须先扣掉，否则算出的 ctx 会虚高。
        assert!(
            (固定 * 1024.0 - 54.0).abs() < 0.5,
            "实际约 {} MiB",
            固定 * 1024.0
        );

        // 纯全注意力（Qwen3）与状态空间混合（Ornith 无滑窗字段）都应为 0。
        let 纯全注意力 = GgufMetadata {
            block_count: Some(36),
            head_count_kv: Some(8),
            key_length: Some(128),
            value_length: Some(128),
            ..Default::default()
        };
        assert_eq!(纯全注意力.kv_fixed_gb(2.0), 0.0);

        let 状态空间 = GgufMetadata {
            block_count: Some(33),
            head_count_kv: Some(4),
            key_length: Some(256),
            value_length: Some(256),
            full_attention_interval: Some(4),
            ..Default::default()
        };
        assert_eq!(状态空间.kv_fixed_gb(2.0), 0.0);
    }

    #[test]
    fn kv_gb_equals_per_token_times_ctx_plus_fixed() {
        // 预算反解用的就是这个分解式：kv = per_token × ctx + fixed。
        // 它必须与 kv_gb 的逐层实现吻合，否则显存预算和 KV 报告这两条
        // 路径会给出互相矛盾的结论。
        let 元数据 = GgufMetadata {
            block_count: Some(36),
            head_count_kv: Some(4),
            key_length: Some(256),
            value_length: Some(256),
            sliding_window: Some(512),
            sliding_window_pattern: (0..36).map(|序号| (序号 + 1) % 4 != 0).collect(),
            ..Default::default()
        };
        let 每token = 元数据.kv_bytes_per_token(2.0) / 每GiB字节;
        let 固定 = 元数据.kv_fixed_gb(2.0);
        for ctx in [2048u64, 16_384, 131_072] {
            let 分解式 = 每token * ctx as f64 + 固定;
            let 逐层式 = 元数据.kv_gb(ctx, 2.0);
            assert!(
                (分解式 - 逐层式).abs() < 1e-9,
                "ctx={ctx} 分解式 {分解式} ≠ 逐层式 {逐层式}"
            );
        }
    }

    #[test]
    fn parses_sliding_window_pattern_from_bytes() {
        // 数组解析是最容易出错的一段（u32 元素类型 + u64 元素个数 + 数据），
        // 而滑动窗口模式决定了一半以上的 KV 估算，必须有字节级测试兜底。
        let mut 数据 = Vec::new();
        数据.extend_from_slice(GGUF_MAGIC);
        数据.extend_from_slice(&3u32.to_le_bytes());
        数据.extend_from_slice(&0u64.to_le_bytes()); // 无张量
        数据.extend_from_slice(&4u64.to_le_bytes()); // 4 个元数据项

        追加字符串KV(&mut 数据, "general.architecture", "spark2_5");
        追加u32KV(&mut 数据, "spark2_5.block_count", 8);
        追加u32KV(&mut 数据, "spark2_5.attention.sliding_window", 512);
        追加布尔数组KV(
            &mut 数据,
            "spark2_5.attention.sliding_window_pattern",
            &[true, true, true, false, true, true, true, false],
        );

        let mut 读取器 = Cursor::new(数据);
        let 元数据 = 解析(&mut 读取器).unwrap();

        assert_eq!(元数据.architecture, "spark2_5");
        assert_eq!(元数据.block_count, Some(8));
        assert_eq!(元数据.sliding_window, Some(512));
        assert_eq!(元数据.sliding_window_pattern.len(), 8);
        // 每 4 层里最后 1 层是全注意力
        assert_eq!(元数据.full_attention_layers(), 2);
        assert_eq!(元数据.sliding_window_layers(), 6);
    }

    #[test]
    fn skips_tokenizer_arrays_without_reading_them_into_memory() {
        // 词表数组可达十几万项。跳过路径必须只在长度上做文章 —— 这一段若写错，
        // 解析会「成功」但偏移全乱，后续字段全是垃圾值。
        let mut 数据 = Vec::new();
        数据.extend_from_slice(GGUF_MAGIC);
        数据.extend_from_slice(&3u32.to_le_bytes());
        数据.extend_from_slice(&0u64.to_le_bytes());
        数据.extend_from_slice(&3u64.to_le_bytes());

        追加字符串KV(&mut 数据, "general.architecture", "qwen3");
        // 一个 3 项的字符串数组，随后必须还能正确读出 block_count
        追加字符串(&mut 数据, "tokenizer.ggml.tokens");
        数据.extend_from_slice(&类型码::ARRAY.to_le_bytes());
        数据.extend_from_slice(&类型码::STRING.to_le_bytes());
        数据.extend_from_slice(&3u64.to_le_bytes());
        for 词 in ["alpha", "beta", "gamma"] {
            追加字符串(&mut 数据, 词);
        }
        追加u32KV(&mut 数据, "qwen3.block_count", 36);

        let mut 读取器 = Cursor::new(数据);
        let 元数据 = 解析(&mut 读取器).unwrap();
        assert_eq!(元数据.architecture, "qwen3");
        assert_eq!(
            元数据.block_count,
            Some(36),
            "跳过大数组后偏移必须仍然对齐"
        );
    }

    #[test]
    fn rejects_non_gguf_file() {
        let 数据 = b"NOTGGUF............".to_vec();
        let mut 读取器 = Cursor::new(数据);
        assert!(matches!(
            解析(&mut 读取器),
            Err(GgufError::BadMagic)
        ));
    }

    #[test]
    fn reports_unsupported_version() {
        let mut 数据 = Vec::new();
        数据.extend_from_slice(GGUF_MAGIC);
        数据.extend_from_slice(&1u32.to_le_bytes()); // v1 不受支持
        数据.extend_from_slice(&0u64.to_le_bytes());
        数据.extend_from_slice(&0u64.to_le_bytes());
        let mut 读取器 = Cursor::new(数据);
        assert!(matches!(
            解析(&mut 读取器),
            Err(GgufError::UnsupportedVersion(1))
        ));
    }

    #[test]
    fn truncated_file_errors_without_panicking() {
        let 数据 = 造混合注意力GGUF();
        for 截断长度 in [8usize, 24, 60, 数据.len() - 1] {
            let 残缺 = 数据[..截断长度].to_vec();
            let mut 读取器 = Cursor::new(残缺);
            assert!(解析(&mut 读取器).is_err(), "长度 {截断长度} 应当报错");
        }
    }

    #[test]
    fn detects_moe_and_scales_active_parameters() {
        let 元数据 = GgufMetadata {
            parameter_count: 100_000_000_000,
            expert_count: Some(128),
            expert_used_count: Some(8),
            ..Default::default()
        };
        assert!(元数据.is_moe());
        // 100B 总参数 / 128 专家 × 激活 8 个
        assert_eq!(元数据.active_parameters(), Some(6_250_000_000));
    }

    #[test]
    fn file_type_maps_to_quant_label() {
        let 元数据 = GgufMetadata {
            file_type: Some(15),
            ..Default::default()
        };
        assert_eq!(元数据.file_type_label(), Some("Q4_K_M"));

        let 未知 = GgufMetadata {
            file_type: Some(999),
            ..Default::default()
        };
        assert_eq!(未知.file_type_label(), None);
    }

    #[test]
    fn bytes_per_param_reflects_measured_quantization() {
        // 用 Qwen3-8B-Q4_K_M 的真实文件（5_027_783_488 字节）与真实参数量（8.191B）：
        // 实测 0.614 字节/参数，而旧代码硬编码 0.4 —— 权重会被低估约 1.5 倍。
        let 元数据 = GgufMetadata {
            parameter_count: 8_191_000_000,
            ..Default::default()
        };
        let 实测 = 元数据.bytes_per_param(5_027_783_488).unwrap();
        assert!((实测 - 0.6138).abs() < 0.001, "实际为 {实测}");
    }

    // ---- 真实文件交叉验证 ----

    fn 收集gguf文件(目录: &Path, 收集: &mut Vec<std::path::PathBuf>) {
        let Ok(条目集) = std::fs::read_dir(目录) else {
            return;
        };
        for 条目 in 条目集.flatten() {
            let 路径 = 条目.path();
            if 路径.is_dir() {
                收集gguf文件(&路径, 收集);
            } else if 路径
                .extension()
                .and_then(|扩展名| 扩展名.to_str())
                .map(|扩展名| 扩展名.eq_ignore_ascii_case("gguf"))
                .unwrap_or(false)
            {
                收集.push(路径);
            }
        }
    }

    /// 用磁盘上的真实模型文件做端到端校验。
    ///
    /// 默认忽略（CI 上没有模型文件）。手动运行：
    ///
    /// ```text
    /// GGUF_SAMPLE_DIR=<模型目录> cargo test --lib cross_check -- --ignored --nocapture
    /// ```
    ///
    /// 断言值来自 `scripts/gguf-probe.py` 对同一批文件的结论 —— 两个实现必须一致，
    /// 否则说明其中一方的解析发生了偏移。这是本模块唯一能发现「静默错位」的检查，
    /// 因为错位后仍可能解析出「看起来合理」的数字。
    #[test]
    #[ignore = "需要真实模型文件，设置 GGUF_SAMPLE_DIR 后手动运行"]
    fn cross_check_against_real_model_files() {
        let 目录 = std::env::var("GGUF_SAMPLE_DIR")
            .expect("请设置 GGUF_SAMPLE_DIR 指向包含 .gguf 的目录");
        let mut 文件列表 = Vec::new();
        收集gguf文件(Path::new(&目录), &mut 文件列表);
        assert!(!文件列表.is_empty(), "在 {目录} 下没找到任何 .gguf 文件");
        文件列表.sort();

        // (文件名, 架构, 层数, 原生上下文, KV 头数, 滑窗层数, 全注意力层数, MTP 层数)
        // 期望值来自 scripts/gguf-probe.py 在 2026-09-24 的实测输出。
        // 滑窗层数单列一栏：它决定「不随上下文增长」的那部分 KV，是预算
        // 反解里必须先扣掉的一块，不能只靠全注意力层数间接推断。
        //
        // 全注意力层数也单列：它是「KV 随 ctx 增长」的唯一乘数，一度算错过
        // （用含 MTP 的 block_count 去除间隔，Qwen3.8 从 16 高估到 17）。
        // MTP 层数同样单列，它是那次错误的成因，必须能被这个基准直接看见。
        const 基准: &[(&str, &str, u32, u32, u32, u32, u32, u32)] = &[
            // Ornith-1.5：33 层含 1 层 MTP → 主干 32 → 32÷4 = 8 层全注意力。
            ("Ornith-1.5-9B-Q4_K_M.gguf", "qwen35", 33, 262_144, 4, 0, 8, 1),
            ("Ornith-1.5-9B-Q6_K.gguf", "qwen35", 33, 262_144, 4, 0, 8, 1),
            ("Ornith-1.5-9B-Q8_0.gguf", "qwen35", 33, 262_144, 4, 0, 8, 1),
            // Spark 是 3:1 滑窗:全注意力 —— 36 层里 27 层滑窗、9 层全注意力，无 MTP。
            ("Spark-X2.5-4B-Q6_K.gguf", "spark2_5", 36, 1_048_576, 4, 27, 9, 0),
            ("Spark-X2.5-4B.gguf", "spark2_5", 36, 1_048_576, 4, 27, 9, 0),
            ("Qwen3-1.7B-Q8_0.gguf", "qwen3", 28, 40_960, 8, 0, 28, 0),
            ("Qwen3-8B-Q4_K_M.gguf", "qwen3", 36, 40_960, 8, 0, 36, 0),
            // Qwen3.8-27B 是 4:1 SSM:全注意力 —— 65 层含 1 层 MTP → 主干 64 层，
            // 其中 16 层全注意力（blk.3,7,…,63）、48 层 SSM。
            // 张量清单佐证：SSM 张量出现在 blk.0/1/2、4/5/6、…、60/61/62。
            // SSM 层不产生随 ctx 增长的 KV（其恒定状态见探针报告的注意事项），滑窗层数记 0。
            ("Qwen3.8-27B-UD-Q3_K_XL.gguf", "qwen35", 65, 262_144, 4, 0, 16, 1),
        ];

        for 路径 in &文件列表 {
            let 文件名 = 路径.file_name().unwrap().to_string_lossy().to_string();
            let 元数据 = parse_file(路径).unwrap_or_else(|错误| panic!("{文件名} 解析失败：{错误}"));
            let 期望 = 基准
                .iter()
                .find(|(名字, ..)| *名字 == 文件名)
                .unwrap_or_else(|| panic!("{文件名} 不在交叉验证基准里，请先跑 scripts/gguf-probe.py"));

            let 固定 = 元数据.kv_fixed_gb(2.0);
            println!(
                "{文件名}\n  架构={} 层数={:?} 上下文={:?} KV头={:?} 参数={} 量化={:?} \
                 \n  全注意力层={} 滑窗层={} 固定KV={固定:.4}GiB bpp={:?}",
                元数据.architecture,
                元数据.block_count,
                元数据.context_length,
                元数据.head_count_kv,
                元数据.parameter_label().unwrap_or_default(),
                元数据.file_type_label(),
                元数据.full_attention_layers(),
                元数据.sliding_window_layers(),
                元数据.bytes_per_param(std::fs::metadata(路径).unwrap().len()),
            );

            assert_eq!(元数据.architecture, 期望.1, "{文件名} 架构不符");
            assert_eq!(元数据.block_count, Some(期望.2), "{文件名} 层数不符");
            assert_eq!(元数据.context_length, Some(期望.3), "{文件名} 上下文不符");
            assert_eq!(元数据.head_count_kv, Some(期望.4), "{文件名} KV 头数不符");
            assert_eq!(
                元数据.sliding_window_layers(),
                期望.5,
                "{文件名} 滑窗层数不符"
            );
            assert_eq!(
                元数据.full_attention_layers(),
                期望.6,
                "{文件名} 全注意力层数不符（KV 随 ctx 增长的唯一乘数）"
            );
            assert_eq!(
                元数据.mtp_layers.unwrap_or(0),
                期望.7,
                "{文件名} MTP 层数不符（这些层不参与推理，但计在 block_count 里）"
            );
            assert!(
                元数据.parameter_count > 1_000_000_000,
                "{文件名} 参数量异常：{}",
                元数据.parameter_count
            );

            // 固定 KV 必须与滑窗层数自洽：没有滑窗层就必须是 0，有就必须为正。
            // 这个量不随 ctx 变化，是最容易被漏算的一项，让它在基准表里显式留痕。
            if 期望.5 == 0 {
                assert_eq!(固定, 0.0, "{文件名} 无滑窗层，固定 KV 应为 0");
            } else {
                assert!(固定 > 0.0, "{文件名} 有 {} 层滑窗却算出 0 固定 KV", 期望.5);
            }
        }

        println!("交叉验证通过，共 {} 个文件", 文件列表.len());
    }
}
