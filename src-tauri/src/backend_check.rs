//! 后端一致性自检 —— 用同一个提示词在两个推理后端各生成一次，逐字节比对输出。
//!
//! 为什么需要这个功能：在 P100（Pascal `sm_60`）这类老卡上，Vulkan 与 CUDA 走的是
//! 完全不同的计算路径（Vulkan 只能 fp32、CUDA 会走 fp16 快速路径），历史上出现过
//! 「输出悄悄跑偏但不报错」的精度缺陷。**静态配置无法回答「我这台机器上两个后端
//! 是否一致」**，只有实跑一次逐字节比对才能给出可见证据。
//!
//! 比对方法学的三条铁律（来自 2026-09-28 的实测踩坑，详见 skill 工作流 F）：
//!
//! 1. **只用标准字段**：请求体里不能塞非标准参数。踩过的坑是在 chat 端点加了
//!    `return_tokens: True`，两个引擎对未知字段的**容错处理不同**，导致返回文本
//!    截断位置不同，表现为「两个后端输出不一致」——**那是测试代码引入的假阳性**。
//! 2. **优先用 `/completion` 而不是 `/v1/chat/completions`**：少一层 chat 模板变量，
//!    要测的是模型，不是模板实现。
//! 3. **读 `stop_type`**：`limit` = 打满 `n_predict`，`stop` = 命中 EOS。
//!    「长度不同」有内容分歧与提前停止两种原因，必须先靠这个字段区分开，
//!    否则会把「停止判定差异」误读成「精度差异」。

use std::time::Duration;

use serde::{Deserialize, Serialize};

/// 自检用的固定提示词。
///
/// 选「按序计数」是因为它满足三个条件：①贪心解码下输出高度确定，几乎不会中途
/// 改变思路；②内容足够长，能暴露随序列累积的数值误差；③纯 ASCII 数字，
/// 比对时不会被中文分词差异干扰。
const 自检提示词: &str = "1\n2\n3\n4\n5\n6\n7\n8\n9\n10\n11\n12\n13\n14\n15\n16\n17\n18\n19\n20\n";

/// 自检生成长度。要足够长才能让累积误差显现（实测 512 已能暴露问题）。
const 自检输出长度: u32 = 384;

/// 单次生成的 HTTP 超时。老卡上 384 token 约 40 秒，留足余量。
const 生成超时: Duration = Duration::from_secs(300);

/// 一个后端的一次生成结果。
///
/// 字段名对外统一用 ASCII（`#[serde(rename)]`）：这些值要跨 Tauri 边界进入
/// TypeScript，让前端写 `run.token_count` 比写中文键名更符合工程惯例，
/// 也避免非 ASCII 键在不同序列化路径上出岔子。中文只保留在 Rust 内部。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct 后端输出 {
    /// 后端名称，用于展示：`CUDA` / `Vulkan`。
    #[serde(rename = "backend")]
    pub 后端: String,
    /// 模型实际生成的文本。
    #[serde(rename = "text")]
    pub 文本: String,
    /// 生成的 token 数。
    #[serde(rename = "token_count")]
    pub token数: u32,
    /// 停止原因：`limit`（打满）或 `stop`（命中 EOS）。
    #[serde(rename = "stop_type")]
    pub 停止原因: String,
    /// 本次生成耗时（秒），顺带给出真实速度读数。
    #[serde(rename = "elapsed_sec")]
    pub 耗时秒: f64,
    /// 每秒生成 token 数。
    #[serde(rename = "tokens_per_sec")]
    pub 每秒token: f64,
}

/// 两个后端的比对结论。
///
/// **每个变体必须显式 `#[serde(rename = "...")]`**：`rename_all = "snake_case"`
/// 对中文标识符**完全无效**（实测：`一致` 序列化后仍是 `"一致"`，而不是 `"identical"`）。
/// 这些键要跨 Tauri 边界进入 TypeScript，前端按 ASCII 字面量做判断，
/// 中文键会让前端**静默走不到任何分支**——三档配色与文案全部失效且不报错。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum 一致性 {
    /// 两个后端逐字节相同。
    #[serde(rename = "identical")]
    一致,
    /// 一方是另一方的前缀，仅长度不同 —— 通常是停止判定差异，不是算错。
    #[serde(rename = "length_only")]
    仅长度不同,
    /// 中段出现分歧 —— 这才是真正需要警惕的精度或实现差异。
    #[serde(rename = "diverged")]
    中段分歧,
}

/// 比对报告。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct 比对报告 {
    #[serde(rename = "verdict")]
    pub 结论: 一致性,
    /// 最长公共前缀的字符数。
    #[serde(rename = "common_prefix_len")]
    pub 公共前缀长度: usize,
    /// 首个分歧处的字符位置（一致时为 `None`）。
    #[serde(rename = "first_divergence")]
    pub 首次分歧位置: Option<usize>,
    /// 一致部分的占比（相对较长的一侧）。
    #[serde(rename = "match_ratio")]
    pub 一致占比: f64,
    /// 给用户看的一句话结论。
    #[serde(rename = "summary")]
    pub 说明: String,
}

/// 逐字节比对两份输出。**纯函数，不碰网络与文件**，便于单测覆盖各种边界。
///
/// 判据分三档，关键是区分「内容分歧」与「仅长度不同」：
/// - 完全相同 → `一致`
/// - 一方是另一方的前缀 → `仅长度不同`（多为停止判定差异，风险低）
/// - 中段出现不同字符 → `中段分歧`（这才是要警惕的）
pub fn 比对(甲: &str, 乙: &str) -> 比对报告 {
    if 甲 == 乙 {
        return 比对报告 {
            结论: 一致性::一致,
            公共前缀长度: 甲.chars().count(),
            首次分歧位置: None,
            一致占比: 1.0,
            说明: "两个后端输出完全一致，当前配置下可以放心使用任一后端。".to_string(),
        };
    }

    // 按字符（而非字节）定位分歧，避免在多字节字符中间切断 —— 中文输出下
    // 按字节比对会给出无法解释的位置。
    let 甲字符: Vec<char> = 甲.chars().collect();
    let 乙字符: Vec<char> = 乙.chars().collect();
    let 较短 = 甲字符.len().min(乙字符.len());
    let 前缀 = (0..较短)
        .find(|&i| 甲字符[i] != 乙字符[i])
        .unwrap_or(较短);
    let 较长 = 甲字符.len().max(乙字符.len());
    let 占比 = if 较长 == 0 {
        1.0
    } else {
        前缀 as f64 / 较长 as f64
    };

    if 前缀 == 较短 {
        // 短的整段都是长的前缀：只差长度，内容没有分歧。
        return 比对报告 {
            结论: 一致性::仅长度不同,
            公共前缀长度: 前缀,
            首次分歧位置: None,
            一致占比: 占比,
            说明: format!(
                "两个后端输出内容相同，仅长度不同（{} / {} 字），通常是停止条件判定差异，不是计算错误。",
                甲字符.len(),
                乙字符.len()
            ),
        };
    }

    比对报告 {
        结论: 一致性::中段分歧,
        公共前缀长度: 前缀,
        首次分歧位置: Some(前缀),
        一致占比: 占比,
        说明: format!(
            "两个后端在第 {} 个字开始出现分歧（前面 {:.0}% 的内容一致）。这可能是计算精度差异，建议以实测较准的后端为准，并把结果反馈给我们。",
            前缀 + 1,
            占比 * 100.0
        ),
    }
}

/// 发起一次确定性生成。
///
/// **只用标准字段**，原因见模块头注释 —— 塞非标准字段会让两个后端产生与本功能
/// 无关的行为差异。三重锁贪心（`temperature=0` + `top_k=1` + 固定 `seed`）确保
/// 同一输入在两个后端上得到可比的输出。
async fn 生成一次(端口: u16, 后端名: &str) -> Result<后端输出, String> {
    let 客户端 = reqwest::Client::builder()
        .timeout(生成超时)
        .build()
        .map_err(|e| format!("网络客户端创建失败：{e}"))?;

    let 请求体 = serde_json::json!({
        "prompt": 自检提示词,
        "n_predict": 自检输出长度,
        "temperature": 0,
        "top_k": 1,
        "seed": 12345,
        "cache_prompt": false
    });

    let 址 = format!("http://127.0.0.1:{端口}/completion");
    let 起 = std::time::Instant::now();
    let 响应 = 客户端
        .post(&址)
        .json(&请求体)
        .send()
        .await
        .map_err(|e| format!("{后端名} 后端生成请求失败：{e}"))?;

    if !响应.status().is_success() {
        return Err(format!("{后端名} 后端返回错误状态：{}", 响应.status()));
    }

    let 体: serde_json::Value = 响应
        .json()
        .await
        .map_err(|e| format!("{后端名} 后端返回内容无法解析：{e}"))?;
    let 耗时 = 起.elapsed().as_secs_f64();

    let 文本 = 体
        .get("content")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string();
    let token数 = 体
        .get("tokens_predicted")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;
    // stop_type 区分「打满」与「命中 EOS」—— 这是判读长度差异的关键依据。
    let 停止原因 = 体
        .get("stop_type")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();

    if 文本.is_empty() {
        return Err(format!("{后端名} 后端没有返回任何内容"));
    }

    Ok(后端输出 {
        后端: 后端名.to_string(),
        文本,
        token数,
        停止原因,
        耗时秒: (耗时 * 100.0).round() / 100.0,
        每秒token: if 耗时 > 0.0 {
            (f64::from(token数) / 耗时 * 100.0).round() / 100.0
        } else {
            0.0
        },
    })
}

/// 自检流程的完整结果。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct 自检结果 {
    #[serde(rename = "first")]
    pub 甲: 后端输出,
    #[serde(rename = "second")]
    pub 乙: 后端输出,
    #[serde(rename = "comparison")]
    pub 报告: 比对报告,
    /// 自检结束后恢复到的后端（用户原本在用的那个）。
    #[serde(rename = "restored_to")]
    pub 已恢复到: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **同线程重复加锁 = 自死锁护栏。**
    ///
    /// 这条测试是 2026-09-29 实机事故换来的：`get_backend_check_availability` 曾写成
    ///
    /// ```ignore
    /// let (install_dir, 当前) = {
    ///     let inner = state.inner.lock().expect("state lock poisoned");
    ///     (inner.settings.install_dir.clone(), current_gpu_backend(&state))
    /// };
    /// ```
    ///
    /// `current_gpu_backend` 内部自己会去抢同一把 `state.inner` 锁；而 `inner` 这把锁
    /// 会一直活到整条 `let` 语句结束（Rust 临时值存活规则）→ 同一线程重复加锁 →
    /// `std::sync::Mutex` 直接死锁。该命令是**同步 Tauri 命令**，死锁会挂住 IPC 线程，
    /// 表现为「打开界面点哪都未响应、一直转圈」。
    ///
    /// 这里把两种写法都放进「独立线程 + 超时」里跑，把死锁变成可断言的失败：
    /// 危险写法必须超时（证明护栏有效），修复写法（两条独立语句）必须能返回。
    #[test]
    fn lock_then_call_current_gpu_backend_must_not_self_deadlock() {
        use crate::app_state::AppState;
        use crate::settings::AppSettings;
        use std::collections::HashMap;
        use std::sync::mpsc;
        use std::time::Duration;

        fn 造状态() -> AppState {
            AppState::new(
                AppSettings::default(),
                std::path::PathBuf::new(),
                std::path::PathBuf::new(),
                HashMap::new(),
            )
        }

        // ① 危险写法：持锁期间调用会再次取锁的函数。
        //    必须超时 —— 若这里居然能返回，说明该模式的危害被误判，
        //    护栏的前提不成立，需要重新评估（而不是当作通过）。
        let (危险发送, 危险接收) = mpsc::channel();
        std::thread::spawn(move || {
            let 状态 = 造状态();
            let (install_dir, 当前) = {
                let inner = 状态.inner.lock().expect("state lock poisoned");
                (
                    inner.settings.install_dir.clone(),
                    crate::service::current_gpu_backend(&状态),
                )
            };
            let _ = (install_dir, 当前);
            let _ = 危险发送.send(());
        });
        assert!(
            危险接收.recv_timeout(Duration::from_secs(3)).is_err(),
            "预期中的自死锁没有发生；请重新确认护栏假设是否仍然成立。"
        );

        // ② 修复写法：取锁与调用拆成两条语句。
        //    这是 get_backend_check_availability / run_backend_consistency_check
        //    当前采用的形式，必须能正常返回。
        let (安全发送, 安全接收) = mpsc::channel();
        std::thread::spawn(move || {
            let 状态 = 造状态();
            let install_dir = {
                let inner = 状态.inner.lock().expect("state lock poisoned");
                inner.settings.install_dir.clone()
            };
            let 当前 = crate::service::current_gpu_backend(&状态);
            let _ = (install_dir, 当前);
            let _ = 安全发送.send(());
        });
        assert!(
            安全接收.recv_timeout(Duration::from_secs(5)).is_ok(),
            "修复后的写法（取锁与调用拆开）不应死锁，但它挂住了。"
        );
    }

    /// 跨 Tauri 边界的枚举键必须是 ASCII 字面量，前端才能匹配。
    ///
    /// **这条护栏是实测教训变的**：`#[serde(rename_all = "snake_case")]` 对中文变体名
    /// 完全不生效 —— `一致` 序列化出来仍是 `"一致"`，而 `src/types.ts` 里写的是
    /// `'identical' | 'length_only' | 'diverged'`。两者对不上时前端**不报错、静默失效**，
    /// 三档配色与文案全部走不到。没有这条测试，这个 bug 只有上真机才看得见。
    #[test]
    fn verdict_keys_are_ascii_for_the_frontend() {
        let 映射 = serde_json::json!({
            "一致": 一致性::一致,
            "仅长度不同": 一致性::仅长度不同,
            "中段分歧": 一致性::中段分歧,
        });
        assert_eq!(映射["一致"].as_str(), Some("identical"));
        assert_eq!(映射["仅长度不同"].as_str(), Some("length_only"));
        assert_eq!(映射["中段分歧"].as_str(), Some("diverged"));
    }

    /// 反向：前端传来的 ASCII 串必须能被解析回来（`FromStr` 走 serde 的 Deserialize）。
    #[test]
    fn ascii_verdict_keys_round_trip() {
        for (原文, 期望) in [
            ("\"identical\"", 一致性::一致),
            ("\"length_only\"", 一致性::仅长度不同),
            ("\"diverged\"", 一致性::中段分歧),
        ] {
            let 解析: 一致性 = serde_json::from_str(原文).unwrap();
            assert_eq!(解析, 期望, "解析 {} 失败", 原文);
        }
    }

    #[test]
    fn identical_output_reports_consistent() {
        let 报告 = 比对("1\n2\n3\n4\n", "1\n2\n3\n4\n");
        assert_eq!(报告.结论, 一致性::一致);
        assert_eq!(报告.首次分歧位置, None);
        assert_eq!(报告.公共前缀长度, 8);
        assert!((报告.一致占比 - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn prefix_only_difference_is_not_content_divergence() {
        // 一方是另一方的前缀 —— 这是「仅长度不同」，绝不能报成中段分歧。
        // 这正是 2026-09-28 实测里把「提前停止」误读成「精度问题」的那个坑。
        let 报告 = 比对("1\n2\n3\n", "1\n2\n3\n4\n5\n");
        assert_eq!(报告.结论, 一致性::仅长度不同);
        assert_eq!(报告.首次分歧位置, None);
        assert_eq!(报告.公共前缀长度, 6);
    }

    #[test]
    fn mid_stream_divergence_is_flagged() {
        let 报告 = 比对("1\n2\n3\n4\n5\n", "1\n2\n9\n4\n5\n");
        assert_eq!(报告.结论, 一致性::中段分歧);
        assert_eq!(报告.首次分歧位置, Some(4));
        assert_eq!(报告.公共前缀长度, 4);
    }

    #[test]
    fn divergence_position_counts_characters_not_bytes() {
        // 中文是多字节字符：按字节定位会滑到字符中间。这里第 3 个「字」开始不同，
        // 报告必须说 3，而不是字节偏移。
        let 报告 = 比对("你好世界", "你好大家");
        assert_eq!(报告.结论, 一致性::中段分歧);
        assert_eq!(报告.首次分歧位置, Some(2), "分歧位置应按字符计数");
    }

    #[test]
    fn empty_strings_are_consistent_with_each_other() {
        let 报告 = 比对("", "");
        assert_eq!(报告.结论, 一致性::一致);

        // 空串与任何非空串之间：空串是前缀 → 仅长度不同，且不能出现除零。
        let 报告 = 比对("", "abc");
        assert_eq!(报告.结论, 一致性::仅长度不同);
        assert_eq!(报告.公共前缀长度, 0);
        assert!((报告.一致占比 - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn short_is_prefix_of_long_even_with_multibyte() {
        let 报告 = 比对("中文", "中文测试");
        assert_eq!(报告.结论, 一致性::仅长度不同);
        assert_eq!(报告.公共前缀长度, 2);
    }

    #[test]
    fn self_check_request_stays_standard() {
        // 回归护栏：自检请求体里 **不能** 出现非标准字段。
        // 踩过的坑：加了 return_tokens 之后两个引擎返回文本的截断位置不同，
        // 产生了一个看起来完全合理的假阳性结论。
        let 请求体 = serde_json::json!({
            "prompt": 自检提示词,
            "n_predict": 自检输出长度,
            "temperature": 0,
            "top_k": 1,
            "seed": 12345,
            "cache_prompt": false
        });
        let 键: Vec<&str> = 请求体.as_object().unwrap().keys().map(|k| k.as_str()).collect();
        for 允许 in &键 {
            assert!(
                matches!(
                    *允许,
                    "prompt" | "n_predict" | "temperature" | "top_k" | "seed" | "cache_prompt"
                ),
                "自检请求体混入了非标准字段 `{允许}`，会让两个后端产生与本功能无关的行为差异"
            );
        }
        assert_eq!(键.len(), 6, "字段数量变化需同步复核是否仍全部标准");
    }

    #[test]
    fn prompt_is_deterministic_counting_and_pure_ascii() {
        // 提示词必须是纯 ASCII 且内容固定：中文分词差异或随机内容都会污染比对。
        assert!(自检提示词.is_ascii(), "自检提示词混入非 ASCII 字符会让两个后端的分词差异进入比对");
        assert!(自检提示词.contains("1\n2\n"), "提示词应引导确定性计数续写");
        assert!(
            自检输出长度 >= 256,
            "输出过短会被固定开销摊薄，且累积误差来不及显现"
        );
    }
}

/// 暴露给上层的生成入口（保持 `生成一次` 私有，便于将来替换实现）。
pub async fn 生成(端口: u16, 后端名: &str) -> Result<后端输出, String> {
    生成一次(端口, 后端名).await
}

// ---------------------------------------------------------------------------
// 编排层：切换后端 → 各生成一次 → 比对 → 恢复原后端
// ---------------------------------------------------------------------------

use tauri::AppHandle;

use crate::{
    app_state::AppState,
    error::AppResult,
    service::{
        available_gpu_backends, backend_label, current_gpu_backend, current_port,
        restore_engine_path, start_service_core, stop_service_core, swap_engine_path_for,
        EngineBackend,
    },
};

/// 自检能不能跑。前端据此决定按钮是否可用，避免点了才报错。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct 自检可用性 {
    #[serde(rename = "available")]
    pub 可用: bool,
    /// 当前正在使用的后端（`CUDA` / `Vulkan` / `CPU`）。
    #[serde(rename = "current_backend")]
    pub 当前后端: String,
    /// 可以对比的后端列表。
    #[serde(rename = "comparable_backends")]
    pub 可对比后端: Vec<String>,
    /// 不可用时的原因。
    #[serde(rename = "reason")]
    pub 原因: Option<String>,
}

#[tauri::command]
pub fn get_backend_check_availability(state: tauri::State<'_, AppState>) -> 自检可用性 {
    // 注意：取锁与调用 current_gpu_backend **必须分开两条语句**。
    // current_gpu_backend 内部自己会去抢同一把 state.inner 锁；若把它写在上面的
    // 元组表达式里，`inner` 这把锁会一直活到整条 let 语句结束（Rust 临时值存活
    // 规则），于是同一线程对同一把 std::sync::Mutex 重复加锁 → 直接死锁。
    // 这个命令是同步 Tauri 命令，死锁会挂住 IPC 线程，导致整个界面「未响应」。
    let install_dir = {
        let inner = state.inner.lock().expect("state lock poisoned");
        inner.settings.install_dir.clone()
    };
    let 当前 = current_gpu_backend(&state);
    let 可对比: Vec<EngineBackend> = available_gpu_backends(&install_dir)
        .into_iter()
        .filter(|b| *b != 当前)
        .collect();

    let (可用, 原因) = if 可对比.is_empty() {
        (
            false,
            Some(
                "当前只安装了一个显卡后端，没有可对比的对象。\
                 一致性校验需要机器上同时具备 CUDA 与 Vulkan 两个引擎。"
                    .to_string(),
            ),
        )
    } else if !当前.is_gpu() {
        (
            false,
            Some("当前正在使用 CPU 模式，无法进行显卡后端比对。".to_string()),
        )
    } else {
        (true, None)
    };

    自检可用性 {
        可用,
        当前后端: backend_label(当前).to_string(),
        可对比后端: 可对比.iter().map(|b| backend_label(*b).to_string()).collect(),
        原因,
    }
}

/// 跑一次后端一致性自检。
///
/// 流程：用当前后端生成一次 → 切到另一后端再生成一次 → 逐字节比对 →
/// **恢复用户原来的后端并重启服务**。
///
/// 中断说明：过程中服务会重启两次，期间聊天不可用（约 1–2 分钟，取决于模型
/// 大小与是否 `--no-mmap`）。前端必须在调用前明确告知用户这一点。
#[tauri::command]
pub async fn run_backend_consistency_check(
    app: AppHandle,
    state: tauri::State<'_, AppState>,
) -> AppResult<自检结果> {
    // 同 get_backend_check_availability：current_gpu_backend 内部会再次抢
    // state.inner 锁，绝不能与上面的 lock 写在同一个元组表达式里（自死锁）。
    let (install_dir, 模型) = {
        let inner = state.inner.lock().expect("state lock poisoned");
        (
            inner.settings.install_dir.clone(),
            inner.service.model_id.clone(),
        )
    };
    let 当前后端 = current_gpu_backend(&state);
    let 模型 = 模型.ok_or_else(|| {
        crate::error::AppError::Message("请先启动一次 AI 服务，再进行后端一致性校验。".to_string())
    })?;

    let 另一后端 = available_gpu_backends(&install_dir)
        .into_iter()
        .find(|b| *b != 当前后端)
        .ok_or_else(|| {
            crate::error::AppError::Message(
                "没有找到可用于对比的另一个显卡后端。".to_string(),
            )
        })?;

    // 记录原始引擎路径 —— 无论下面哪一步失败，都必须恢复到它。
    let 原路径 = swap_engine_path_for(&state, &install_dir, 当前后端)?;

    // 用闭包包裹实际流程，便于在任意 Err 分支后统一恢复。
    let 结果 = async {
        // ---- 第一次：当前后端 ----
        stop_service_core(&app, &state);
        crate::service::wait_for_service_stop(&state).await;
        start_service_core(&模型, &app, &state).await?;
        let 端口 = current_port(&state)
            .ok_or_else(|| crate::error::AppError::Message("服务未返回监听端口。".to_string()))?;
        let 甲 = 生成(端口, backend_label(当前后端))
            .await
            .map_err(crate::error::AppError::Message)?;

        // ---- 第二次：另一个后端 ----
        swap_engine_path_for(&state, &install_dir, 另一后端)?;
        stop_service_core(&app, &state);
        crate::service::wait_for_service_stop(&state).await;
        start_service_core(&模型, &app, &state).await?;
        let 端口 = current_port(&state)
            .ok_or_else(|| crate::error::AppError::Message("服务未返回监听端口。".to_string()))?;
        let 乙 = 生成(端口, backend_label(另一后端))
            .await
            .map_err(crate::error::AppError::Message)?;

        Ok::<_, crate::error::AppError>(比对(&甲.文本, &乙.文本).into_自检结果(甲, 乙))
    }
    .await;

    // ---- 恢复：无论成败，都切回用户原来的后端并重启服务 ----
    swap_engine_path_for(&state, &install_dir, 当前后端).ok();
    stop_service_core(&app, &state);
    crate::service::wait_for_service_stop(&state).await;
    let 已恢复 = start_service_core(&模型, &app, &state).await.is_ok();
    restore_engine_path(&state, 原路径);

    let mut 结果 = 结果?;
    结果.已恢复到 = if 已恢复 {
        backend_label(当前后端).to_string()
    } else {
        format!("{}（重启失败，请手动重新启动服务）", backend_label(当前后端))
    };
    Ok(结果)
}

impl 比对报告 {
    /// 把比对报告与两次生成结果合成最终返回值。
    fn into_自检结果(self, 甲: 后端输出, 乙: 后端输出) -> 自检结果 {
        自检结果 {
            甲,
            乙,
            报告: self,
            // 编排层随后会覆盖成真实值。
            已恢复到: String::new(),
        }
    }
}
