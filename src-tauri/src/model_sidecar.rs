//! 模型旁车启动配置：把「微调产物」接进启动链路的唯一入口。
//!
//! ## 为什么需要它（一次真实事故）
//!
//! 微调出来的 adapter 不能单独使用，它有两个**必须与训练时保持一致**的部署条件：
//!
//! 1. **对话模板**。训练数据的提示尾部是 `<｜Assistant｜></think>\n`（think 段**已闭合**），
//!    而 `DeepSeek-R1-Distill` 的 GGUF 里自带的模板强制 `<｜Assistant｜><think>\n`
//!    （**打开** think 段且永不闭合）。两边不一致时，引擎的 reasoning 解析会把模型输出
//!    **整段判进 `reasoning_content`**，`content` 变成空串 —— 下游只读 `content` 的服务
//!    于是拿到空回答，走兜底。实测代价：**同一份 adapter**，模板一致时增益 `+39.1 pp`，
//!    不一致时只剩 `+4.3 pp`（等于白训）。
//! 2. **LoRA 挂载**。adapter 是独立文件，必须用 `--lora` / `--lora-scaled` 显式挂上。
//!
//! 这两件事都只是 llama.cpp 的**启动参数**。APP 原先一个都不传 ⇒
//! 「实验室里修好了、生产上照旧」。本模块让「训练契约」能**跟着模型包一起交付**。
//!
//! ## 文件契约
//!
//! 与 GGUF **同目录、同主名**，扩展名换成 `.launch.json`：
//!
//! ```text
//! models/DS-R1-Qwen-7B-GGUF/
//!   base-Q4_K_M.gguf              ← 模型
//!   base-Q4_K_M.launch.json       ← 本模块读的文件
//!   lora-v3-f16.gguf              ← 适配器
//!   tpl-prefill-closed.jinja      ← 对话模板
//! ```
//!
//! ```json
//! {
//!   "对话模板文件": "tpl-prefill-closed.jinja",
//!   "适配器文件": "lora-v3-f16.gguf",
//!   "适配器缩放": 1.0
//! }
//! ```
//!
//! 三个字段都可省略；写的路径**相对模型所在目录**（写绝对路径也行）。
//! `适配器缩放` 缺省为 `1.0`。**写 `0` 是合法的** —— 它让引擎加载 adapter 但不生效
//! （`--lora-scaled FNAME:0`），正好是**同一个进程**里做「纯基座 vs 挂 adapter」
//! 对照的白送手段（连模型加载都共用），不需要第二个模型文件。
//!
//! ## 🔴 引用的文件缺失必须报错，不许跳过
//!
//! 配置里写了某个文件、那个文件却不在 ⇒ **启动直接失败并指名是哪个文件**。
//! 理由与「降级 / 跳过 / 未跑完必须与『通过』可区分」是同一条：
//! 跳过之后服务照常起来、界面显示「运行中」，而 adapter 事实上没生效
//! （或反过来模板没换），**表现与正常完全一样** —— 这正是本模块要消灭的那种静默偏离。
//!
//! 同理，配置文件里出现**不认识的字段**也报错（`deny_unknown_fields`）：
//! 手写 JSON 时打错一个字（`适配器文`），静默忽略的结果就是「配置写了却没生效」。

use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::error::{AppError, AppResult};

/// 旁车配置文件的扩展名（落成 `<模型主名>.launch.json`）。
pub const 旁车扩展名: &str = "launch.json";

/// 与 GGUF 同目录、同主名的旁车配置文件路径。
pub fn 旁车路径(模型路径: &Path) -> PathBuf {
    let mut 路径 = 模型路径.to_path_buf();
    路径.set_extension(旁车扩展名);
    路径
}

/// 这个模型是否自带旁车配置（**只看文件在不在，不解析内容**）。
///
/// 给「模型列表」用：让「这个模型带微调包」在界面上看得见。
/// 刻意不做解析 —— 列表页不该因为某个配置文件写坏了而整页拉不出来；
/// 真正解析、真正报错的地方是启动那一刻（`读取旁车参数`）。
pub fn 有旁车配置(模型路径: &Path) -> bool {
    旁车路径(模型路径).is_file()
}

/// 旁车配置的内容。
///
/// 字段名**就是** JSON 键名 —— 本项目的数据文件一律用中文键
/// （`ChatBI/eval/题目.json`、`指标口径定义-v1.0.yaml` 等），
/// 所以这里不加 `serde(rename)`，字段名与文件内容一眼对得上。
#[derive(Debug, Clone, Default, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct 模型旁车配置 {
    /// 对话模板文件（传 `--chat-template-file`）。
    pub 对话模板文件: Option<String>,
    /// LoRA 适配器文件（传 `--lora`；缩放不为 1 时传 `--lora-scaled`）。
    pub 适配器文件: Option<String>,
    /// 适配器缩放；缺省视为 `1.0`。**`0` 合法**（加载但不生效 = 纯基座）。
    pub 适配器缩放: Option<f64>,
}

/// 把旁车配置翻译成 llama.cpp 启动参数（**纯函数**，不碰文件系统之外的东西，
/// 便于单测把每条分支都钉住）。
///
/// 返回顺序固定：先对话模板、后适配器，便于对账脚本逐字比对。
///
/// 三种情况返回 `Err`（都是「写了却没生效」的前身，宁可起不来也别静默）：
/// 字段是空串 / 引用的文件不存在 / 只写了缩放却没写适配器文件。
pub fn 推导旁车参数(模型目录: &Path, 配置: &模型旁车配置) -> AppResult<Vec<(String, String)>> {
    let mut 参数: Vec<(String, String)> = Vec::new();

    if let Some(原始) = 配置.对话模板文件.as_deref() {
        let 路径 = 解析引用文件(模型目录, 原始, "对话模板文件")?;
        参数.push(("--chat-template-file".to_string(), 路径));
    }

    match 配置.适配器文件.as_deref() {
        Some(原始) => {
            let 路径 = 解析引用文件(模型目录, 原始, "适配器文件")?;
            // 缺省 1.0。用「是否恰好等于 1.0」而不是「有没有写」来选标志：
            // 显式写 1.0 与不写应当走同一条路（`--lora`），否则同一份模型会
            // 因为配置文件多做了一件没差别的事而走两条不同的命令行。
            let 缩放 = 配置.适配器缩放.unwrap_or(1.0);
            if 缩放 == 1.0 {
                参数.push(("--lora".to_string(), 路径));
            } else {
                // `--lora-scaled FNAME:SCALE`。缩放 0 = 加载但不生效（纯基座）。
                参数.push(("--lora-scaled".to_string(), format!("{}:{}", 路径, 缩放)));
            }
        }
        None => {
            if 配置.适配器缩放.is_some() {
                return Err(AppError::Message(
                    "模型配置里写了「适配器缩放」，却没有写「适配器文件」。\
                     请补上适配器文件，或把缩放这一项删掉。"
                        .to_string(),
                ));
            }
        }
    }

    Ok(参数)
}

/// 解析配置里引用的文件：相对路径按**模型所在目录**展开，并当场确认文件存在。
fn 解析引用文件(模型目录: &Path, 原始: &str, 字段名: &str) -> AppResult<String> {
    let 名字 = 原始.trim();
    if 名字.is_empty() {
        return Err(AppError::Message(format!(
            "模型配置里的「{字段名}」是空的。请填上文件名，或者把这一项删掉。"
        )));
    }
    let 候选 = Path::new(名字);
    let 全路径 = if 候选.is_absolute() {
        候选.to_path_buf()
    } else {
        模型目录.join(候选)
    };
    if !全路径.is_file() {
        return Err(AppError::Message(format!(
            "模型配置里写的「{字段名}」找不到：{}。\
             请确认文件确实放在模型目录里，或把这一项改成正确的路径。",
            全路径.display()
        )));
    }
    Ok(全路径.to_string_lossy().to_string())
}

/// 读取并翻译某个模型文件旁的旁车配置。
///
/// * 没有旁车文件 ⇒ `Ok(vec![])` —— 绝大多数模型的正常情况，不做任何附加；
/// * 文件在、但读不动或不是合法 JSON ⇒ `Err`（**不许静默当成没有**，
///   否则用户改坏了配置文件，只会看到「我的适配器怎么没生效」）。
pub fn 读取旁车参数(模型路径: &Path) -> AppResult<Vec<(String, String)>> {
    let 路径 = 旁车路径(模型路径);
    if !路径.is_file() {
        return Ok(Vec::new());
    }

    let 文本 = std::fs::read_to_string(&路径).map_err(|错误| {
        AppError::Message(format!(
            "模型配置 {} 读不出来：{}。请确认这个文件没有被其它程序占用。",
            路径.display(),
            错误
        ))
    })?;
    let 配置: 模型旁车配置 = serde_json::from_str(&文本).map_err(|错误| {
        AppError::Message(format!(
            "模型配置 {} 不是合法的 JSON：{}。\
             这个文件只认「对话模板文件」「适配器文件」「适配器缩放」三个字段。",
            路径.display(),
            错误
        ))
    })?;

    // 相对路径的基准是**模型所在目录**（不是进程工作目录 —— 那会随启动方式变）。
    let 模型目录 = 模型路径.parent().unwrap_or_else(|| Path::new("."));
    let 参数 = 推导旁车参数(模型目录, &配置)?;

    if 参数.is_empty() {
        tracing::info!(
            配置 = %路径.display(),
            "模型旁车配置存在，但两个字段都没写（无对话模板、无适配器）"
        );
    } else {
        // ⚠️ 这里**只陈述「配置里有什么」，不宣称动作**。
        //
        // 本函数的调用方有两个：`service.rs::start_service_core`（真起进程、真把参数
        // 加进命令行）与 `service.rs::recover_service` 的**接管路径**（没有起进程、
        // 没有往任何命令行里加参数，只是拿这个结果当期望值去跟引擎对账）。
        //
        // 2026-10-16 实机踩到：原先这里写「已按模型旁车配置附加 llama.cpp 启动参数」，
        // 接管路径也会打这一行 ⇒ 日志宣称了一个**没有发生**的动作（那次根本没起进程）。
        // 与「日志只记发生过的事 / 判据不许把调用方的标签当状态」同族。
        // 「已并入本次启动命令行」那句现在只由 `start_service_core` 自己说。
        tracing::info!(
            配置 = %路径.display(),
            参数 = ?参数,
            "读到模型旁车配置（自定义对话模板 / LoRA 适配器）"
        );
    }
    Ok(参数)
}

// ─────────────────────────────────────────────────────────────────────────────
// 启动之后：**问引擎**要回执
// ─────────────────────────────────────────────────────────────────────────────
//
// 为什么非要有这一段：`--lora` 与上面推导出来的参数，都是**我们自己填进命令行的
// 标签** —— 写进命令行 ≠ 引擎接受了。本项目为此栽过（把「调用方填的标签」当成
// 「引擎实测值」，§9.5 第 ⑤ 类错误）。所以服务起来之后必须再问引擎两件事：
//
//   * `GET /lora-adapters`            —— 引擎**自己报**的适配器（文件名 + 缩放）
//   * `GET /props` 的 `chat_template` —— 引擎**实际生效**的对话模板
//
// 本段只做**纯解析与比对**（不联网、不打日志），联网与日志在 `service.rs`。
// 这样每条分支都能被单测钉住，验证时不必真起一个引擎。
//
// 🔴 判据纪律：**取不到 ≠ 没生效**。三态（一致 / 不一致 / 取不到）必须可区分 ——
// 否则「端点还没就绪」会被读成「适配器白挂」，反而把结论带反。

/// 从**最终传给引擎的附加参数**里，取出某个标志的全部取值。
///
/// 同时认 `--flag 值` 与 `--flag=值` 两种写法（`service.rs` 里的 `extra_has`
/// 也认两种，口径保持一致）。
///
/// 为什么从「最终参数」取、而不是从旁车配置里读：用户可以在设置里手写同名标志
/// （用户优先、盖过旁车配置）⇒ **只有最终参数才是事实**。
fn 取标志的值(参数: &[String], 想要的: &str) -> Vec<String> {
    let mut 结果 = Vec::new();
    let mut 下标 = 0usize;
    while let Some(词) = 参数.get(下标) {
        let (标志, 同行值) = match 词.split_once('=') {
            Some((左, 右)) => (左, Some(右.to_string())),
            None => (词.as_str(), None),
        };
        if 标志 == 想要的 {
            let 值 = match 同行值 {
                Some(右) => Some(右),
                None => {
                    下标 += 1;
                    参数.get(下标).cloned()
                }
            };
            if let Some(值) = 值 {
                if !值.is_empty() {
                    结果.push(值);
                }
            }
        }
        下标 += 1;
    }
    结果
}

/// `--lora-scaled` 的取值形如 `路径:缩放`，拆成两半。
///
/// 🔴 按**最后一个冒号**切：Windows 路径自带 `D:`，从第一个冒号切会把盘符
/// 当成缩放值（`D` 解析失败 → 整条被丢掉 → 变成「没有期望」→ **静默放过**）。
fn 拆分路径与缩放(值: &str) -> Option<(String, f64)> {
    let 位置 = 值.rfind(':')?;
    let (路径, 缩放段) = 值.split_at(位置);
    if 路径.is_empty() {
        return None;
    }
    let 缩放 = 缩放段[1..].parse::<f64>().ok()?;
    Some((路径.to_string(), 缩放))
}

/// 取路径的文件名（比对用语）：`D:\a\b.gguf` 与 `/a/b.gguf` 都给 `b.gguf`。
fn 文件名(路径: &str) -> String {
    路径.rsplit(['/', '\\']).next().unwrap_or(路径).to_string()
}

/// 两个路径指向的文件名是否相同。**不区分大小写** —— Windows 上盘符与目录名
/// 的大小写可能不同（`D:\` 与 `d:\`），那不是「挂错了版本」。
fn 同名(左: &str, 右: &str) -> bool {
    文件名(左).to_lowercase() == 文件名(右).to_lowercase()
}

/// 从最终附加参数里解析出**期望的适配器**：`(文件路径, 缩放)`。
///
/// 缩放 `None` = 用的是 `--lora`（不带缩放）⇒ 按缺省 `1.0` 比对。
///
/// 🔴 `--lora-scaled` 的取值拆不开时（比如忘了写 `:缩放`）**也不许静默丢掉**：
/// 丢掉会让「期望 1 个、实际 0 个」变成「期望 0 个、无需核对」，正好放过了出问题的那种。
pub fn 期望的适配器(附加参数: &[String]) -> Vec<(String, Option<f64>)> {
    let mut 结果: Vec<(String, Option<f64>)> = Vec::new();
    for 值 in 取标志的值(附加参数, "--lora") {
        结果.push((值, None));
    }
    for 值 in 取标志的值(附加参数, "--lora-scaled") {
        match 拆分路径与缩放(&值) {
            Some((路径, 缩放)) => 结果.push((路径, Some(缩放))),
            None => 结果.push((值, None)),
        }
    }
    结果
}

/// 从最终附加参数里解析出**期望的对话模板文件路径**（没传就是 `None`）。
pub fn 期望的对话模板(附加参数: &[String]) -> Option<String> {
    取标志的值(附加参数, "--chat-template-file").into_iter().next()
}

/// 解析引擎 `GET /lora-adapters` 的返回体：`[{"id":0,"path":"…","scale":1.0}]`。
/// 只要 `(path, scale)`；`scale` 可能是整数（`1`），`Option<f64>` 能直接收。
pub fn 解析适配器回执(正文: &str) -> AppResult<Vec<(String, Option<f64>)>> {
    #[derive(Deserialize)]
    struct 条目 {
        #[serde(default)]
        path: String,
        #[serde(default)]
        scale: Option<f64>,
    }
    let 条目们: Vec<条目> = serde_json::from_str(正文).map_err(|错误| {
        AppError::Message(format!("引擎的适配器回执不是合法 JSON：{错误}"))
    })?;
    Ok(条目们.into_iter().map(|条| (条.path, 条.scale)).collect())
}

/// 解析引擎 `GET /props` 的返回体，取它**实际生效**的对话模板。
pub fn 解析模板回执(正文: &str) -> AppResult<String> {
    #[derive(Deserialize)]
    struct 属性 {
        #[serde(default)]
        chat_template: Option<String>,
    }
    let 属性: 属性 = serde_json::from_str(正文)
        .map_err(|错误| AppError::Message(format!("引擎的属性回执不是合法 JSON：{错误}")))?;
    属性.chat_template.ok_or_else(|| {
        AppError::Message("引擎没有报出 chat_template（这个版本可能不支持）".to_string())
    })
}

/// 从 `/props` 回执里取**引擎自报的上下文长度**。
///
/// 🔴 **为什么接管态必须问这一条**（P2-14，2026-10-16 P100 实机抓出）：
/// 接管（`reattached to previous llama-server`）**不重算启动参数**，沿用的是上一个
/// 会话留下的那个进程。若它当初是用兜底默认值（24 层 / 4096 上下文）起的，接管之后
/// 日志里「接管到的服务跑的确实是这个模型」照样通过 —— 于是
/// **「接管到参数正确的服务」与「接管到参数已过时的服务」在日志里同形**。
///
/// 当天那台 P100 就是这样：14:35 接管到的其实是 07:18 用 24 层 / 4096 起的进程，
/// 而回执只说了「模型对」—— 用户因此一度以为上一轮的修复没生效。
/// （项目自带的 `scripts/llama-server-check.py` 早就用 `n_ctx` 判这件事，
/// 注释里甚至写着「n_ctx 若小于原生上下文，通常是服务没重启」，只是没接进 APP。）
///
/// 判据只认**引擎自报**的值，不认我们自己填进命令行的那个数字 ——
/// 后者属于「把调用方填的标签当引擎实测值」，是本项目翻过五次车的那类错误。
///
/// 字段路径按**实测**取：`default_generation_settings.n_ctx`；顶层 `n_ctx`
/// 作为兜底（有的版本直接放在根上）。两处都没有就**报错**，不猜。
pub fn 解析上下文回执(正文: &str) -> AppResult<u32> {
    #[derive(Deserialize)]
    struct 代次设置 {
        #[serde(default)]
        n_ctx: Option<u32>,
    }
    #[derive(Deserialize)]
    struct 属性 {
        #[serde(default)]
        default_generation_settings: Option<代次设置>,
        #[serde(default)]
        n_ctx: Option<u32>,
    }
    let 属性: 属性 = serde_json::from_str(正文)
        .map_err(|错误| AppError::Message(format!("引擎的属性回执不是合法 JSON：{错误}")))?;
    属性.default_generation_settings.and_then(|设置| 设置.n_ctx)
        .or(属性.n_ctx)
        .ok_or_else(|| {
            AppError::Message("引擎的属性回执里没有 n_ctx（这个版本可能不支持）".to_string())
        })
}

/// 把「期望的适配器」与「引擎自报的适配器」对上，返回**问题清单**（空 = 一致）。
///
/// 三条判据分开报，因为**病因不同**：
///   * 条数不符   —— 有一个根本没加载；
///   * 文件名不符 —— 加载的是**另一个版本**（最危险：数字可能全对、方向全反）；
///   * 缩放不符   —— 加载了但力度不对（`0` 与 `1` 是「纯基座」与「真生效」的区别）。
pub fn 核对适配器(期望: &[(String, Option<f64>)], 实际: &[(String, Option<f64>)]) -> Vec<String> {
    let mut 问题: Vec<String> = Vec::new();
    if 期望.len() != 实际.len() {
        问题.push(format!(
            "条数不符：命令行里传了 {} 个适配器，引擎只报了 {} 个",
            期望.len(),
            实际.len()
        ));
    }
    // 🔴 引擎一个都没报时，「条数不符」已经把话说完（且写明了是 0）⇒ **不再逐条点名**。
    //    否则同一个病因报两遍，日志里看着像两个问题（本函数的单测当场抓到了这一点）。
    //    逐条点名只在「报了一些、但缺了某一个」时才有额外信息量。
    if 实际.is_empty() {
        return 问题;
    }
    for (路径, 缩放) in 期望 {
        let 名字 = 文件名(路径);
        let Some((_, 实际缩放)) = 实际.iter().find(|(其他, _)| 同名(其他, &名字)) else {
            问题.push(format!(
                "引擎没报出适配器「{名字}」：命令行里传了，引擎侧看不到"
            ));
            continue;
        };
        // 缺省都按 1.0 算，与 `推导旁车参数` 的口径一致。
        let 想要 = 缩放.unwrap_or(1.0);
        let 得到 = 实际缩放.unwrap_or(1.0);
        if (想要 - 得到).abs() > 1e-9 {
            问题.push(format!(
                "适配器「{名字}」的缩放不符：命令行 {想要:?}，引擎报 {得到:?}"
            ));
        }
    }
    问题
}

/// 把「适配器列表」渲染成**给人看**的日志片段。
///
/// 为什么需要它：`{:?}` 打出来是 `[("…lora-v3-f16.gguf", Some(1.0))]` ——
/// `Some(…)` 是 Rust 的 `Option` 内部表示，不是给「配错了微调包的人」看的东西。
/// 日志里冒出 `Some` 会让人以为出了问题（2026-10-16 老李本机实测提出）。
///
/// 🔴 **抹平只抹外层**：缩放缺失渲染成「缩放未报」，**不许也写成 `1.0`**。
/// 判据里 `None` 与 `Some(1.0)` 按同一个值（缺省 1.0）比对 —— 但那说的是**比对**；
/// 在**诊断**上两者不是一件事：「引擎没报缩放」是引擎**没说话**，
/// 写成 `1.0` 等于替引擎声称了一个它没说过的值（§9.5 第⑤类错误：把「调用方填的标签」当「引擎实测值」）。
pub fn 描述适配器(列表: &[(String, Option<f64>)]) -> String {
    let 项: Vec<String> = 列表
        .iter()
        .map(|(路径, 缩放)| match 缩放 {
            Some(值) => format!("(\"{路径}\", {值:?})"),
            None => format!("(\"{路径}\", 缩放未报)"),
        })
        .collect();
    format!("[{}]", 项.join(", "))
}

/// 核对引擎**实际生效**的模板与我们交给它的那份文件，返回问题清单（空 = 一致）。
///
/// 两边都先 `trim_end`：模板文件末尾的换行是否被引擎原样保留**不由我们决定**
/// （实测这一份 2091 字符逐字节相同，但那是**这一份**），所以只排除「尾空白」
/// 这一种噪声，其余任何差异都报出来。
pub fn 核对对话模板(期望文件内容: &str, 引擎模板: &str) -> Vec<String> {
    if 期望文件内容.trim_end() == 引擎模板.trim_end() {
        return Vec::new();
    }
    let mut 问题 = vec![format!(
        "引擎实际生效的对话模板与文件不一致：文件 {} 字符、引擎报 {} 字符",
        期望文件内容.chars().count(),
        引擎模板.chars().count()
    )];

    // 补一条**最有行动价值**的线索：本模块存在的理由就是这一个字面量。
    // 🔴 比的是完整的 `</think>`，不是 `think` —— 后者会把 `<think>`（**打开**思考段）
    //    也判成命中，而这两者恰恰是要区分开的那一对（本项目的体检脚本第一版就栽在这）。
    const 闭合字面: &str = "</think>";
    let 文件标了闭合 = 期望文件内容.contains(闭合字面);
    let 引擎标了闭合 = 引擎模板.contains(闭合字面);
    if 文件标了闭合 && !引擎标了闭合 {
        问题.push(
            "交给引擎的模板标了 `</think>`（think 已闭合），引擎报回来的模板里却没有这个字面量 \
             ⇒ 生效的不是这一份；模型的整段输出会被判进 reasoning_content、content 变空串"
                .to_string(),
        );
    } else if !文件标了闭合 && 引擎标了闭合 {
        问题.push("引擎报回来的模板里有 `</think>`，而交给它的文件里没有 ⇒ 生效的不是这一份".to_string());
    }
    问题
}

/// 解析引擎 `GET /v1/models` 的返回体，取它**自报**的模型（路径或别名）。
///
/// 为什么接管场景下第一件事是问这个：接管的是**上一个会话留下的进程**，
/// APP 手里只有一个 `model_id` 字符串，**没有任何东西能证明端口上跑的就是它**。
/// 「端口在听 + `/health` 回 200」回答不了这个问题 —— 那是本项目
/// 「『活着』≠『能干这件事』」那条判据的直接应用。
///
/// APP 不传 `--alias`（全仓库无此参数），所以引擎自报的通常就是**模型文件路径**。
pub fn 解析模型回执(正文: &str) -> AppResult<String> {
    #[derive(Deserialize)]
    struct 条目 {
        #[serde(default)]
        id: String,
    }
    #[derive(Deserialize)]
    struct 列表 {
        #[serde(default)]
        data: Vec<条目>,
    }
    let 列表: 列表 = serde_json::from_str(正文)
        .map_err(|错误| AppError::Message(format!("引擎的模型列表回执不是合法 JSON：{错误}")))?;
    // 跳过 id 为空的条目：`data` 存在但内容是空壳，与「一个模型都没有」同义。
    let 首个 = 列表
        .data
        .into_iter()
        .find(|条| !条.id.is_empty())
        .ok_or_else(|| AppError::Message("引擎的模型列表是空的（一个模型都没报出来）".to_string()))?;
    Ok(首个.id)
}

/// 核对「APP 让它加载的模型」与「引擎自报的模型」，返回问题清单（空 = 一致）。
///
/// 只比**文件名**（见 `同名`）：引擎自报的可能是完整路径也可能是别名，
/// 路径分隔符与盘符大小写都可能不同（`D:\a\B.gguf` vs `d:/a/b.gguf`），
/// 那些不是「跑错了模型」。
///
/// 🔴 引擎报**别名**（不带 `.gguf` 后缀）时这里**也会报** —— 这是刻意的：
/// 报出来的意思是「无法确认端口上跑的是哪个模型」。把无法确认算成一致，
/// 正是本项目反复栽的那个坑。真到那一天，要么去掉别名、要么在这里加别名映射。
pub fn 核对模型(期望路径: &str, 引擎自报: &str) -> Vec<String> {
    if 同名(期望路径, 引擎自报) {
        return Vec::new();
    }
    vec![format!(
        "端口上跑的模型可能不是它：APP 期望「{}」，引擎自报「{}」\
         （只比文件名，故引擎报别名时这里也会报 —— 那说明无法确认，不是已确认）",
        文件名(期望路径),
        引擎自报
    )]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 每个测试用一个独立临时目录，避免互相踩；退出时**不**清理（临时目录，
    /// 留着便于失败时人工翻看），目录名带进程号与计数器保证不重名。
    fn 造临时目录(标签: &str) -> PathBuf {
        use std::sync::atomic::{AtomicU32, Ordering};
        static 计数: AtomicU32 = AtomicU32::new(0);
        let 序号 = 计数.fetch_add(1, Ordering::SeqCst);
        let 目录 = std::env::temp_dir().join(format!(
            "localai-旁车测试-{}-{}-{}",
            std::process::id(),
            标签,
            序号
        ));
        std::fs::create_dir_all(&目录).expect("临时目录应能创建");
        目录
    }

    fn 写文件(目录: &Path, 名字: &str, 内容: &str) -> PathBuf {
        let 路径 = 目录.join(名字);
        std::fs::write(&路径, 内容).expect("测试文件应能写入");
        路径
    }

    fn 解析(json: &str) -> 模型旁车配置 {
        serde_json::from_str(json).expect("测试用的 JSON 应当合法")
    }

    #[test]
    fn 旁车路径与模型同目录且主名相同() {
        let 模型 = PathBuf::from("/tmp/某目录/base-Q4_K_M.gguf");
        let 旁车 = 旁车路径(&模型);
        assert_eq!(
            旁车,
            PathBuf::from("/tmp/某目录/base-Q4_K_M.launch.json"),
            "旁车文件必须与模型同目录、同主名，只换扩展名"
        );
    }

    #[test]
    fn 主名里本来就带点时只换最后一个扩展名() {
        // 量化档名里常有下划线、点（如 `Qwen3-8B-Q4_K_M.gguf`）——
        // 只有**最后**一个点后面是扩展名，前面的必须原样保留。
        let 模型 = PathBuf::from("/tmp/m/Qwen3-8B-Instruct-v2.gguf");
        assert_eq!(
            旁车路径(&模型),
            PathBuf::from("/tmp/m/Qwen3-8B-Instruct-v2.launch.json")
        );
    }

    #[test]
    fn 没有旁车文件时返回空参数() {
        let 目录 = 造临时目录("无旁车");
        let 模型 = 写文件(&目录, "base.gguf", "假的");
        let 参数 = 读取旁车参数(&模型).expect("没有旁车文件是正常情况，不该报错");
        assert!(参数.is_empty(), "没有旁车文件就不该附加任何参数");
    }

    #[test]
    fn 只写对话模板时只产出chat_template_file() {
        let 目录 = 造临时目录("仅模板");
        let 模型 = 写文件(&目录, "base.gguf", "假的");
        写文件(&目录, "tpl.jinja", "{{'<｜Assistant｜></think>\n'}}");
        写文件(
            &目录,
            "base.launch.json",
            r#"{"对话模板文件": "tpl.jinja"}"#,
        );

        let 参数 = 读取旁车参数(&模型).expect("配置合法");
        assert_eq!(
            参数,
            vec![(
                "--chat-template-file".to_string(),
                目录.join("tpl.jinja").to_string_lossy().to_string()
            )],
            "只写模板时不应出现任何 LoRA 参数"
        );
    }

    #[test]
    fn 适配器缩放缺省为一走lora() {
        let 目录 = 造临时目录("缺省缩放");
        let 模型 = 写文件(&目录, "base.gguf", "假的");
        写文件(&目录, "lora.gguf", "假的");
        写文件(&目录, "base.launch.json", r#"{"适配器文件": "lora.gguf"}"#);

        let 参数 = 读取旁车参数(&模型).expect("配置合法");
        assert_eq!(参数.len(), 1, "只写适配器时只该有一个参数");
        assert_eq!(参数[0].0, "--lora", "缺省缩放必须走 --lora");
    }

    #[test]
    fn 显式写缩放一也走lora() {
        // 反例护栏：显式写 1.0 与不写必须落到同一个标志上，
        // 否则「同一份模型」会因为配置文件多写了一个无差别的字段而换命令行。
        let 目录 = 造临时目录("显式缩放一");
        let 模型 = 写文件(&目录, "base.gguf", "假的");
        写文件(&目录, "lora.gguf", "假的");
        写文件(
            &目录,
            "base.launch.json",
            r#"{"适配器文件": "lora.gguf", "适配器缩放": 1.0}"#,
        );

        let 参数 = 读取旁车参数(&模型).expect("配置合法");
        assert_eq!(参数[0].0, "--lora", "显式 1.0 不该退化成 --lora-scaled");
    }

    #[test]
    fn 缩放为零时走lora_scaled并保留零() {
        // 缩放 0 = 加载 adapter 但不生效（纯基座）。这是「同一进程内做
        // 纯基座/挂 adapter 对照」的白送手段，不能被当成「没写」而吃掉。
        let 目录 = 造临时目录("缩放零");
        let 模型 = 写文件(&目录, "base.gguf", "假的");
        写文件(&目录, "lora.gguf", "假的");
        写文件(
            &目录,
            "base.launch.json",
            r#"{"适配器文件": "lora.gguf", "适配器缩放": 0}"#,
        );

        let 参数 = 读取旁车参数(&模型).expect("配置合法");
        assert_eq!(参数[0].0, "--lora-scaled", "缩放不为 1 必须走 --lora-scaled");
        assert!(
            参数[0].1.ends_with(":0"),
            "缩放必须原样带过去（0 是合法值、不许被当成缺省），实际 {}",
            参数[0].1
        );
    }

    #[test]
    fn 模板与适配器同时写时两者都在且模板在前() {
        let 目录 = 造临时目录("模板加适配器");
        let 模型 = 写文件(&目录, "base.gguf", "假的");
        写文件(&目录, "tpl.jinja", "模板");
        写文件(&目录, "lora.gguf", "假的");
        写文件(
            &目录,
            "base.launch.json",
            r#"{"对话模板文件": "tpl.jinja", "适配器文件": "lora.gguf", "适配器缩放": 0.5}"#,
        );

        let 参数 = 读取旁车参数(&模型).expect("配置合法");
        assert_eq!(参数.len(), 2, "两件事都该派生出一个参数");
        assert_eq!(参数[0].0, "--chat-template-file", "顺序固定：模板在前");
        assert_eq!(参数[1].0, "--lora-scaled");
        assert!(参数[1].1.ends_with(":0.5"), "缩放值要原样带过去");
    }

    #[test]
    fn 绝对路径可以直接用() {
        let 目录 = 造临时目录("绝对路径");
        let 模型 = 写文件(&目录, "base.gguf", "假的");
        let 模板 = 写文件(&目录, "tpl.jinja", "模板");
        写文件(
            &目录,
            "base.launch.json",
            &format!(r#"{{"对话模板文件": {:?}}}"#, 模板.to_string_lossy()),
        );

        let 参数 = 读取旁车参数(&模型).expect("绝对路径合法");
        assert_eq!(参数[0].1, 模板.to_string_lossy().to_string());
    }

    // ---- 反例：每一条都是「写了却没生效」的前身，必须报错而不是静默 ----

    #[test]
    fn 引用的模板文件不存在时报错并指名文件() {
        let 目录 = 造临时目录("模板缺失");
        let 模型 = 写文件(&目录, "base.gguf", "假的");
        写文件(
            &目录,
            "base.launch.json",
            r#"{"对话模板文件": "不存在的模板.jinja"}"#,
        );

        let 错误 = 读取旁车参数(&模型).expect_err("文件不存在必须报错，不许静默跳过");
        let 文本 = 错误.to_string();
        assert!(
            文本.contains("不存在的模板.jinja"),
             "报错必须指名是哪个文件，实际：{文本}"
        );
    }

    #[test]
    fn 引用的适配器不存在时报错() {
        let 目录 = 造临时目录("适配器缺失");
        let 模型 = 写文件(&目录, "base.gguf", "假的");
        写文件(&目录, "base.launch.json", r#"{"适配器文件": "缺.gguf"}"#);

        let 错误 = 读取旁车参数(&模型).expect_err("适配器缺失必须报错");
        assert!(错误.to_string().contains("缺.gguf"), "实际：{错误}");
    }

    #[test]
    fn 只写缩放不写适配器时报错() {
        let 目录 = 造临时目录("孤立缩放");
        let 模型 = 写文件(&目录, "base.gguf", "假的");
        写文件(&目录, "base.launch.json", r#"{"适配器缩放": 0.5}"#);

        let 错误 = 读取旁车参数(&模型).expect_err("孤立的缩放是配置写错了，必须报错");
        assert!(
            错误.to_string().contains("适配器文件"),
             "报错要告诉用户缺的是哪一项，实际：{错误}"
        );
    }

    #[test]
    fn 字段写成空串时报错() {
        let 目录 = 造临时目录("空串");
        let 模型 = 写文件(&目录, "base.gguf", "假的");
        写文件(&目录, "base.launch.json", r#"{"对话模板文件": "   "}"#);

        let 错误 = 读取旁车参数(&模型).expect_err("空串就是「没填」，必须报错");
        assert!(
            错误.to_string().contains("对话模板文件"),
            "实际：{错误}"
        );
    }

    #[test]
    fn 不认识的字段被拒绝() {
        // 手写 JSON 打错一个字，静默忽略的结果就是「配置写了却没生效」。
        let 目录 = 造临时目录("错别字");
        let 模型 = 写文件(&目录, "base.gguf", "假的");
        写文件(&目录, "lora.gguf", "假的");
        写文件(&目录, "base.launch.json", r#"{"适配器文": "lora.gguf"}"#);

        let 错误 = 读取旁车参数(&模型).expect_err("字段名打错必须报错");
        assert!(
            错误.to_string().contains("JSON"),
            "报错要说清是 JSON 字段的问题，实际：{错误}"
        );
    }

    #[test]
    fn json损坏时报错而不是当成没有配置() {
        let 目录 = 造临时目录("坏JSON");
        let 模型 = 写文件(&目录, "base.gguf", "假的");
        写文件(&目录, "base.launch.json", "{ 这不是 JSON");

        let 错误 = 读取旁车参数(&模型).expect_err("坏 JSON 必须报错，不能当成「没有旁车」");
        assert!(错误.to_string().contains("base.launch.json"), "实际：{错误}");
    }

    #[test]
    fn 推导函数本身也能直接调用() {
        // 纯函数入口的独立用例：不经过文件，直接钉「配置 → 参数」的映射。
        let 目录 = 造临时目录("纯函数");
        写文件(&目录, "lora.gguf", "假的");
        let 配置 = 解析(r#"{"适配器文件": "lora.gguf", "适配器缩放": 2.0}"#);
        let 参数 = 推导旁车参数(&目录, &配置).expect("配置合法");
        assert_eq!(参数[0].0, "--lora-scaled");
        assert!(参数[0].1.ends_with(":2"), "实际 {}", 参数[0].1);
    }

    #[test]
    fn 空配置产出空参数() {
        let 目录 = 造临时目录("空配置");
        let 参数 = 推导旁车参数(&目录, &模型旁车配置::default()).expect("空配置合法");
        assert!(参数.is_empty());
    }

    // ── 启动后的回执核对（期望值解析 / 回执解析 / 比对） ──────────────────

    fn 串(词: &[&str]) -> Vec<String> {
        词.iter().map(|词| 词.to_string()).collect()
    }

    #[test]
    fn 从最终参数里取出lora与模板() {
        let 参数 = 串(&[
            "--ctx-size", "4096", "--parallel", "1",
            "--chat-template-file", "D:/m/tpl.jinja",
            "--lora", "D:/m/lora-v3-f16.gguf",
        ]);
        let 适配器 = 期望的适配器(&参数);
        assert_eq!(适配器.len(), 1);
        assert!(适配器[0].0.ends_with("lora-v3-f16.gguf"));
        assert_eq!(适配器[0].1, None, "`--lora` 不带缩放");
        assert_eq!(期望的对话模板(&参数).as_deref(), Some("D:/m/tpl.jinja"));
    }

    #[test]
    fn 等号写法的标志也认得() {
        // `--flag=值` 与 `--flag 值` 都必须认（`extra_has` 也是这个口径）。
        let 参数 = 串(&["--lora=D:/m/lora.gguf", "--chat-template-file=D:/m/t.jinja"]);
        assert_eq!(期望的适配器(&参数).len(), 1);
        assert_eq!(期望的对话模板(&参数).as_deref(), Some("D:/m/t.jinja"));
    }

    #[test]
    fn lora_scaled的缩放按最后一个冒号拆所以盘符不被当成缩放() {
        // 🔴 从第一个冒号切会把 `D` 当缩放、解析失败、整条被丢掉 —— 那就变成
        //    「期望 0 个」，恰好放过了出问题的那种。这条用例就是钉住这一点。
        let 参数 = 串(&["--lora-scaled", "D:/m/lora-v3-f16.gguf:0.5"]);
        let 适配器 = 期望的适配器(&参数);
        assert_eq!(适配器.len(), 1, "不许被静默丢掉");
        assert_eq!(适配器[0].0, "D:/m/lora-v3-f16.gguf");
        assert_eq!(适配器[0].1, Some(0.5));
    }

    #[test]
    fn 反斜杠路径的lora_scaled也能拆() {
        let 参数 = 串(&["--lora-scaled", r"D:\Ivan-File\m\lora.gguf:0"]);
        let 适配器 = 期望的适配器(&参数);
        assert_eq!(适配器.len(), 1);
        assert_eq!(适配器[0].1, Some(0.0), "缩放 0 = 加载但不生效（纯基座），要保留");
    }

    #[test]
    fn lora_scaled拆不开时也不静默丢掉() {
        // 忘写 `:缩放` ⇒ 拆不开。丢掉会让「期望 1 个」变成「期望 0 个、无需核对」。
        let 参数 = 串(&["--lora-scaled", "D:/m/lora.gguf"]);
        assert_eq!(期望的适配器(&参数).len(), 1, "宁可报不一致，也不许当成没有");
    }

    #[test]
    fn 没有附加参数时期望为空() {
        assert!(期望的适配器(&[]).is_empty());
        assert_eq!(期望的对话模板(&[]), None);
        let 参数 = 串(&["--ctx-size", "4096", "--no-mmap"]);
        assert!(期望的适配器(&参数).is_empty(), "别把别的标志的值当适配器");
        assert_eq!(期望的对话模板(&参数), None);
    }

    #[test]
    fn 适配器回执能解析出文件名与缩放() {
        let 正文 = r#"[{"id":0,"path":"D:\\m\\lora-v3-f16.gguf","scale":1.0,"task_name":""}]"#;
        let 回执 = 解析适配器回执(正文).expect("合法回执");
        assert_eq!(回执.len(), 1);
        assert_eq!(文件名(&回执[0].0), "lora-v3-f16.gguf");
        assert_eq!(回执[0].1, Some(1.0));
    }

    #[test]
    fn 回执里的缩放是整数也能解析() {
        // 引擎写 `1` 而不是 `1.0` 时，serde 必须照样收成 f64。
        let 回执 = 解析适配器回执(r#"[{"path":"/m/l.gguf","scale":1}]"#).expect("合法回执");
        assert_eq!(回执[0].1, Some(1.0));
    }

    #[test]
    fn 空列表回执是合法的空结果() {
        // 「引擎报 0 个」本身是**有效回执**，不能被当成解析失败 ——
        // 否则「传了却没挂上」这个最该报的形态反而会隐身。
        let 回执 = 解析适配器回执("[]").expect("空列表合法");
        assert!(回执.is_empty());
    }

    #[test]
    fn 适配器回执损坏时报错() {
        assert!(解析适配器回执("{ 这不是 JSON").is_err());
        assert!(解析适配器回执(r#"{"path":"/m/l.gguf"}"#).is_err(), "对象不是列表");
    }

    #[test]
    fn 模板回执能取出chat_template() {
        let 正文 = r#"{"chat_template":"{{'<|Assistant|></think>\n'}}","n_ctx":4096}"#;
        assert!(解析模板回执(正文).expect("合法").contains("</think>"));
    }

    #[test]
    fn 模板回执里没有chat_template时报错而不是给空串() {
        // 给空串会让比对把「取不到」读成「模板不一致」，指向完全错误的方向。
        let 错误 = 解析模板回执(r#"{"n_ctx":4096}"#).expect_err("缺字段必须报错");
        assert!(错误.to_string().contains("chat_template"), "实际：{错误}");
    }

    #[test]
    fn 上下文回执从代次设置里取出n_ctx() {
        // 字段路径取自**实测**（scripts/llama-server-check.py 也是这么取的）。
        // 数值沿用 2026-10-16 P100 那次事故现场：引擎自报 4096，而它能开的其实是 262144。
        let 正文 = r#"{"default_generation_settings":{"n_ctx":4096},"total_slots":1}"#;
        assert_eq!(解析上下文回执(正文).expect("合法"), 4096);
    }

    #[test]
    fn 上下文回执也认顶层的n_ctx() {
        // 兜底：有的版本直接把 n_ctx 放在根上。少认一种写法 = 少一次判据。
        assert_eq!(解析上下文回执(r#"{"n_ctx":262144}"#).expect("合法"), 262144);
    }

    #[test]
    fn 两处都报了时以代次设置为准() {
        // 优先级必须钉死：两处都取、且都算「合法」时若选错，判据会静默报出另一个数。
        let 正文 = r#"{"n_ctx":4096,"default_generation_settings":{"n_ctx":262144}}"#;
        assert_eq!(
            解析上下文回执(正文).expect("合法"),
            262144,
            "代次设置是引擎当前的槽位设置，顶层那个可能是启动时的旧值"
        );
    }

    #[test]
    fn 上下文回执里没有n_ctx时报错而不是给0() {
        // 给 0（或任何默认值）会把「取不到」变成「上下文是 0」这个假结论 ——
        // 那正是本项目反复强调的「降级必须与通过可区分」。
        let 错误 = 解析上下文回执(r#"{"total_slots":1}"#).expect_err("缺字段必须报错");
        assert!(错误.to_string().contains("n_ctx"), "实际：{错误}");
    }

    #[test]
    fn 适配器全对时没有问题() {
        let 期望 = vec![("D:/m/lora-v3-f16.gguf".to_string(), None)];
        let 实际 = vec![(r"D:\m\lora-v3-f16.gguf".to_string(), Some(1.0))];
        assert!(
            核对适配器(&期望, &实际).is_empty(),
            "同一个文件的两种写法（斜杠、大小写）不该算不一致"
        );
    }

    #[test]
    fn 大小写与斜杠不同仍算同一个文件() {
        let 期望 = vec![("D:/M/LoRa.gguf".to_string(), Some(0.5))];
        let 实际 = vec![(r"d:\m\lora.GGUF".to_string(), Some(0.5))];
        assert!(核对适配器(&期望, &实际).is_empty());
    }

    #[test]
    fn 引擎一个适配器都没报时要报条数不符() {
        // 这是**最该报**的形态：命令行写了 `--lora`，引擎侧却是空的。
        let 期望 = vec![("D:/m/lora.gguf".to_string(), None)];
        let 问题 = 核对适配器(&期望, &[]);
        // **只报一条**：条数不符已经把话说完；再逐条点名等于同一病因报两遍，
        // 日志里看着像两个问题（本用例第一版就是按 2 条写的，被自己的断言抓出来）。
        assert_eq!(问题.len(), 1, "同一病因不许报两遍：{问题:?}");
        assert!(问题[0].contains("条数不符"), "实际：{问题:?}");
        assert!(问题[0].contains("只报了 0 个"), "要写清是 0：{问题:?}");
    }

    #[test]
    fn 报了一些但缺了某一个时才逐条点名() {
        // 「部分缺失」与「一个都没有」是两种形态：前者需要点名缺哪个。
        let 期望 = vec![
            ("D:/m/a.gguf".to_string(), None),
            ("D:/m/b.gguf".to_string(), None),
        ];
        let 实际 = vec![("D:/m/a.gguf".to_string(), Some(1.0))];
        let 问题 = 核对适配器(&期望, &实际);
        assert_eq!(问题.len(), 2, "条数不符 + 点名缺的那个：{问题:?}");
        assert!(问题.iter().any(|句| 句.contains("b.gguf")), "{问题:?}");
    }

    #[test]
    fn 挂错了版本要单独报文件名不符() {
        let 期望 = vec![("D:/m/lora-v3-f16.gguf".to_string(), None)];
        let 实际 = vec![("D:/m/lora-v2-f16.gguf".to_string(), Some(1.0))];
        let 问题 = 核对适配器(&期望, &实际);
        assert!(
            问题.iter().any(|句| 句.contains("lora-v3-f16.gguf")),
            "要指名期望的那一个，实际：{问题:?}"
        );
    }

    #[test]
    fn 缩放不符要单独报因为零和一不是一回事() {
        let 期望 = vec![("D:/m/lora.gguf".to_string(), Some(1.0))];
        let 实际 = vec![("D:/m/lora.gguf".to_string(), Some(0.0))];
        let 问题 = 核对适配器(&期望, &实际);
        assert_eq!(问题.len(), 1);
        assert!(问题[0].contains("缩放"), "实际：{问题:?}");
    }

    #[test]
    fn 渲染适配器时抹掉Some但保住缩放未报() {
        // 日志是给人看的：`[("…", Some(1.0))]` 里的 `Some` 是 Rust 内部表示（老李实测提出）。
        let 一个 = vec![(r"D:\m\lora-v3-f16.gguf".to_string(), Some(1.0))];
        assert_eq!(描述适配器(&一个), r#"[("D:\m\lora-v3-f16.gguf", 1.0)]"#);

        // 🔴 反例：`None` **不许**也渲染成 `1.0` —— 那是替引擎声称它没说过的话。
        let 缺缩放 = vec![("D:/m/lora.gguf".to_string(), None)];
        let 文本 = 描述适配器(&缺缩放);
        assert!(文本.contains("缩放未报"), "实际：{文本}");
        assert!(!文本.contains("1.0"), "不许替引擎补 1.0：{文本}");

        // 半整数缩放照原样，别把 0.5 抹成整数。
        let 半档 = vec![("D:/m/lora.gguf".to_string(), Some(0.5))];
        assert_eq!(描述适配器(&半档), r#"[("D:/m/lora.gguf", 0.5)]"#);

        assert_eq!(描述适配器(&[]), "[]");
    }

    #[test]
    fn 用lora时引擎报负一才算缩放不符() {
        // `--lora`（不带缩放）的期望是缺省 1.0；引擎报别的值就说明挂的不是我们想的那份。
        let 期望 = vec![("D:/m/lora.gguf".to_string(), None)];
        let 实际 = vec![("D:/m/lora.gguf".to_string(), Some(-1.0))];
        assert_eq!(核对适配器(&期望, &实际).len(), 1);
    }

    #[test]
    fn 模板一致时没有问题并容忍尾空白() {
        assert!(核对对话模板("甲\n乙\n", "甲\n乙").is_empty());
        assert!(核对对话模板("", "").is_empty());
    }

    #[test]
    fn 模板不一致时要点出丢失了闭合标记() {
        let 文件 = "{{'<|Assistant|></think>\\n'}}";
        let 引擎 = "{{'<|Assistant|><think>\\n'}}";
        let 问题 = 核对对话模板(文件, 引擎);
        assert_eq!(问题.len(), 2, "一条说不一致，一条指出闭合标记丢失：{问题:?}");
        assert!(问题[1].contains("reasoning_content"), "要说清后果：{问题:?}");
    }

    #[test]
    fn 判据不能只看think三个字() {
        // 🔴 反例：`<think>` 与 `</think>` 都含 "think"，粗判会把这一对判成同一个 ——
        //    而它俩恰恰是本模块要区分开的那一对。这里钉住「引擎丢了闭合标记必须报」。
        let 文件 = "<think> 甲 </think>";
        let 引擎 = "<think> 甲 <think>";
        let 问题 = 核对对话模板(文件, 引擎);
        assert!(!问题.is_empty(), "丢了 `</think>` 不许当「差不多」放过");
        assert!(问题[0].contains("不一致"));
    }

    #[test]
    fn 模板只有尾空白差异时不算不一致() {
        // 引擎是否原样保留文件末尾的换行不由我们决定 ⇒ 只把这一种噪声排除掉。
        let 文件 = "尾部 </think>\n\n";
        let 引擎 = "尾部 </think>";
        assert!(核对对话模板(文件, 引擎).is_empty());
    }

    // ── 模型身份：接管场景的第一层判据 ──────────────────────────────────────
    //
    // 接管（reattach）时 APP 手里只有一个 `model_id` 字符串，没有任何东西能证明
    // 端口上跑的就是它 ⇒ 必须问一次 `/v1/models`。

    #[test]
    fn 模型回执能取出自报的模型路径() {
        let 正文 = r#"{"object":"list","data":[{"id":"D:\\m\\base-Q4_K_M.gguf","object":"model"}]}"#;
        let 自报 = 解析模型回执(正文).expect("应能解析");
        assert_eq!(自报, r"D:\m\base-Q4_K_M.gguf");
    }

    #[test]
    fn 模型列表为空时报错而不是给空串() {
        // 空串会被拿去与期望值比对，报出的却是「模型对不上」——
        // **把病因说错**比不报更糟（本项目同族：判据把错钉成期望行为）。
        let 错 = 解析模型回执(r#"{"object":"list","data":[]}"#);
        assert!(错.is_err(), "一个模型都没报时必须报错，不能给空串");
    }

    #[test]
    fn 模型回执里id为空壳条目要被跳过() {
        let 正文 = r#"{"data":[{"id":""},{"id":"/m/real.gguf"}]}"#;
        assert_eq!(解析模型回执(正文).expect("应能解析"), "/m/real.gguf");
    }

    #[test]
    fn 模型回执损坏时报错() {
        assert!(解析模型回执("not json").is_err());
        // `data` 整个缺失 = 不是我们认识的格式，同样不许当「空列表」放过。
        assert!(解析模型回执(r#"{"object":"list"}"#).is_err());
    }

    #[test]
    fn 模型一致时不报问题且容忍路径写法差异() {
        // 引擎自报的路径分隔符与盘符大小写不由我们决定。
        assert!(核对模型(r"D:\m\Base-Q4_K_M.gguf", r"d:/m/base-q4_k_m.gguf").is_empty());
    }

    #[test]
    fn 模型不一致时要点名两边各是什么() {
        let 问题 = 核对模型(r"D:\m\base-Q4_K_M.gguf", r"D:\m\Spark-X2.5-4B-Q6_K.gguf");
        assert_eq!(问题.len(), 1);
        assert!(问题[0].contains("base-Q4_K_M.gguf"), "{问题:?}");
        assert!(问题[0].contains("Spark-X2.5-4B-Q6_K.gguf"), "{问题:?}");
    }

    #[test]
    fn 引擎报别名时也报无法确认而不是算一致() {
        // APP 目前不传 `--alias`，但**将来**有人加了别名这里就会亮 ——
        // 亮的意思是「无法确认」，不是「已确认」。把无法确认算成一致是本项目的坑。
        let 问题 = 核对模型(r"D:\m\base-Q4_K_M.gguf", "my-alias");
        assert_eq!(问题.len(), 1);
        assert!(问题[0].contains("无法确认"), "{问题:?}");
    }
}
