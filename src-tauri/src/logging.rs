use std::{
    io::{Read, Write},
    path::Path,
    time::{Duration, SystemTime},
};

use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

use crate::{
    app_state::AppState,
    error::{AppError, AppResult},
};

/// 这次启动的日志级别到底是谁说了算。
///
/// 必须如实报出来：首行写「INFO」而实际按 `RUST_LOG=debug` 在跑，会让所有
/// 「看到 DEBUG 行就算走到了、没看到就算没走到」的判据集体失准 —— 判据本身
/// 没错，是证据的来源被谎报了。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum 级别来源 {
    /// 设置里的 `log_level`（已按 `normalize_level` 归一）。
    配置,
    /// 环境变量 `RUST_LOG` 的原样指令，可带多条 directive。
    环境变量,
}

impl 级别来源 {
    /// 日志里用的 ASCII 字面量（日志要能 grep，不写中文）。
    pub fn as_str(self) -> &'static str {
        match self {
            Self::配置 => "settings",
            Self::环境变量 => "RUST_LOG",
        }
    }
}

/// 决定这次真正生效的过滤指令 —— `init` 与首行日志共用这一份判断，
/// 不另写一套等价逻辑（两处定义必然漂移，且漂移时不报错只给错报告）。
///
/// 规则与原先的 `EnvFilter::try_from_default_env()` 完全一致：`RUST_LOG` 存在
/// 且是合法指令时才盖过设置，否则回落到 `normalize_level(设置值)`。
/// 非法 `RUST_LOG` 的原文一并带出，由调用方单独 WARN —— 静默忽略等于又一次
/// 「看起来按你的要求跑了」。
///
/// 返回 `(来源, 生效指令, 被忽略的 RUST_LOG 原文)`。
pub fn 选择过滤(设置级别: &str, 环境变量: Option<&str>) -> (级别来源, String, Option<String>) {
    if let Some(raw) = 环境变量 {
        let trimmed = raw.trim();
        if !trimmed.is_empty() {
            if EnvFilter::try_new(trimmed).is_ok() {
                return (级别来源::环境变量, trimmed.to_string(), None);
            }
            return (
                级别来源::配置,
                normalize_level(设置级别).to_string(),
                Some(raw.to_string()),
            );
        }
    }
    (级别来源::配置, normalize_level(设置级别).to_string(), None)
}

/// `logging::init` 的产物：要留到进程结束的 writer guard，加上「实际生效了什么」。
///
/// `guard` 为 `None` 表示日志目录建不出来（只剩控制台输出），不是失败。
pub struct 日志初始化 {
    /// 只靠 `Drop` 生效（把非阻塞 writer 的缓冲刷出去），**永不读取** ——
    /// 所以显式允许 dead_code：一旦被当成无用字段优化掉，日志会静默丢尾巴。
    #[allow(dead_code)]
    pub guard: Option<WorkerGuard>,
    /// 设置文件里的原值，原样报出（非法值也照报，别替用户美化）。
    pub 设置级别: String,
    pub 级别来源: 级别来源,
    /// 真正交给 `EnvFilter` 的指令原文。
    pub 生效指令: String,
    /// `RUST_LOG` 设了但解析不了时的原文；`None` 表示没这回事。
    pub 环境变量无效: Option<String>,
}

/// Initialize logging: console + daily-rolling file under `data_dir/logs`.
///
/// The base level comes from the user's `log_level` setting (DEBUG/INFO/WARN/
/// ERROR); a `RUST_LOG` env var still overrides it for developers. The returned
/// guard flushes the non-blocking writer on drop, so callers must keep it alive
/// for the lifetime of the app.
pub fn init(data_dir: &Path, log_level: &str) -> 日志初始化 {
    let 环境变量 = std::env::var("RUST_LOG").ok();
    let (来源, 指令, 无效) = 选择过滤(log_level, 环境变量.as_deref());
    // `选择过滤` 保证这里一定合法：要么是归一后的单级别，要么是刚被
    // `EnvFilter::try_new` 验过的原文，因此 `new` 不会 panic。
    let filter = EnvFilter::new(&指令);
    let console = fmt::layer().with_target(false);

    let 结果 = |guard: Option<WorkerGuard>| 日志初始化 {
        guard,
        设置级别: log_level.to_string(),
        级别来源: 来源,
        生效指令: 指令.clone(),
        环境变量无效: 无效.clone(),
    };

    let logs_dir = data_dir.join("logs");
    if std::fs::create_dir_all(&logs_dir).is_err() {
        tracing_subscriber::registry()
            .with(filter)
            .with(console)
            .init();
        return 结果(None);
    }

    let appender = tracing_appender::rolling::daily(&logs_dir, "localai.log");
    let (non_blocking, guard) = tracing_appender::non_blocking(appender);
    let file = fmt::layer()
        .with_ansi(false)
        .with_target(false)
        .with_writer(non_blocking);

    tracing_subscriber::registry()
        .with(filter)
        .with(console)
        .with(file)
        .init();
    结果(Some(guard))
}

/// Map a user-facing log level (DEBUG/INFO/WARN/ERROR, any case) to the tracing
/// filter directive. Unrecognized values fall back to `info`.
fn normalize_level(level: &str) -> &'static str {
    match level.trim().to_ascii_uppercase().as_str() {
        "DEBUG" => "debug",
        "WARN" => "warn",
        "ERROR" => "error",
        _ => "info",
    }
}

/// Collect log files from the last 7 days, scrub each per §7.3, and package
/// them into a ZIP at `dest_path`. The export never contains user chat content
/// (never logged) or absolute file paths (masked here as a fallback).
#[tauri::command]
pub fn export_logs(dest_path: String, state: tauri::State<'_, AppState>) -> AppResult<String> {
    let data_dir = {
        let inner = state.inner.lock().expect("state lock poisoned");
        inner.data_dir.clone()
    };
    let logs_dir = data_dir.join("logs");

    let cutoff = SystemTime::now()
        .checked_sub(Duration::from_secs(7 * 24 * 3600))
        .unwrap_or(SystemTime::UNIX_EPOCH);

    let entries = std::fs::read_dir(&logs_dir).map_err(|_| {
        AppError::Message("暂时没有可导出的日志。".to_string())
    })?;

    let dest = std::path::PathBuf::from(&dest_path);
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let zip_file = std::fs::File::create(&dest)?;
    let mut zip = zip::ZipWriter::new(zip_file);
    let options = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);

    let mut added = 0usize;
    for entry in entries.filter_map(|entry| entry.ok()) {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let recent = entry
            .metadata()
            .and_then(|meta| meta.modified())
            .map(|modified| modified >= cutoff)
            .unwrap_or(true);
        if !recent {
            continue;
        }

        let mut raw = String::new();
        if std::fs::File::open(&path)
            .and_then(|mut file| file.read_to_string(&mut raw))
            .is_err()
        {
            continue;
        }
        let scrubbed = desensitize(&raw);
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("localai.log")
            .to_string();
        zip.start_file(name, options)
            .map_err(|_| AppError::Message("日志打包时遇到问题，请稍后重试。".to_string()))?;
        zip.write_all(scrubbed.as_bytes())?;
        added += 1;
    }

    zip.finish()
        .map_err(|_| AppError::Message("日志打包时遇到问题，请稍后重试。".to_string()))?;

    if added == 0 {
        let _ = std::fs::remove_file(&dest);
        return Err(AppError::Message("最近 7 天暂时没有可导出的日志。".to_string()));
    }

    Ok(dest.to_string_lossy().to_string())
}

/// Fallback text scrubbing applied before exporting logs. Masks absolute file
/// paths (keeping the filename) and non-local IP addresses, while deliberately
/// preserving `127.0.0.1` so health-check / port diagnostics stay readable.
pub fn desensitize(input: &str) -> String {
    redact_ips(&redact_paths(input))
}

fn redact_paths(input: &str) -> String {
    let chars: Vec<char> = input.chars().collect();
    let mut out = String::with_capacity(input.len());
    let mut i = 0;
    while i < chars.len() {
        if is_path_start(&chars, i) {
            let mut end = i;
            while end < chars.len() && !chars[end].is_whitespace() {
                end += 1;
            }
            let token: String = chars[i..end].iter().collect();
            out.push_str(&mask_path(&token));
            i = end;
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    out
}

fn is_path_start(chars: &[char], i: usize) -> bool {
    // Windows drive path: `C:\` or `C:/`. The drive letter must stand alone, so
    // a preceding letter means this is a URL scheme like `http://`, not a drive.
    let drive = i + 2 < chars.len()
        && chars[i].is_ascii_alphabetic()
        && (i == 0 || !chars[i - 1].is_ascii_alphabetic())
        && chars[i + 1] == ':'
        && (chars[i + 2] == '\\' || chars[i + 2] == '/');
    // UNC path: `\\server\...`
    let unc = i + 1 < chars.len() && chars[i] == '\\' && chars[i + 1] == '\\';
    drive || unc
}

fn mask_path(token: &str) -> String {
    let filename = token
        .rsplit(['\\', '/'])
        .next()
        .filter(|name| !name.is_empty());
    match filename {
        Some(name) => format!(".../{}", name),
        None => ".../".to_string(),
    }
}

fn redact_ips(input: &str) -> String {
    let chars: Vec<char> = input.chars().collect();
    let mut out = String::with_capacity(input.len());
    let mut i = 0;
    while i < chars.len() {
        if chars[i].is_ascii_digit() {
            if let Some((ip, next)) = try_ipv4(&chars, i) {
                if ip == "127.0.0.1" {
                    out.push_str(&ip);
                } else {
                    out.push_str("[REDACTED_IP]");
                }
                i = next;
                continue;
            }
        }
        out.push(chars[i]);
        i += 1;
    }
    out
}

fn try_ipv4(chars: &[char], start: usize) -> Option<(String, usize)> {
    // Don't match inside a longer number like `1234.5.6.7`.
    if start > 0 {
        let prev = chars[start - 1];
        if prev.is_ascii_digit() || prev == '.' {
            return None;
        }
    }

    let mut i = start;
    for octet in 0..4 {
        let digit_start = i;
        let mut digits = 0;
        while i < chars.len() && chars[i].is_ascii_digit() && digits < 3 {
            i += 1;
            digits += 1;
        }
        if digits == 0 {
            return None;
        }
        let value: u32 = chars[digit_start..i].iter().collect::<String>().parse().ok()?;
        if value > 255 {
            return None;
        }
        if octet < 3 {
            if i < chars.len() && chars[i] == '.' {
                i += 1;
            } else {
                return None;
            }
        }
    }
    // Reject a trailing 5th octet (e.g. `1.2.3.4.5`).
    if i < chars.len() && chars[i] == '.' {
        return None;
    }
    Some((chars[start..i].iter().collect(), i))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn masks_windows_path_keeping_filename() {
        assert_eq!(
            desensitize("saving model C:\\Users\\Alice\\models\\a.gguf ok"),
            "saving model .../a.gguf ok"
        );
        assert_eq!(
            desensitize("dir D:\\LocalAI-Installer\\models\\"),
            "dir .../"
        );
    }

    #[test]
    fn masks_unc_path() {
        assert_eq!(
            desensitize("open \\\\server\\share\\x.gguf now"),
            "open .../x.gguf now"
        );
    }

    #[test]
    fn preserves_local_loopback_and_urls() {
        let line = "llama_server: listening on http://127.0.0.1:15981/v1/models";
        assert_eq!(desensitize(line), line);
    }

    #[test]
    fn redacts_non_local_ip() {
        assert_eq!(desensitize("proxy 192.168.1.2 set"), "proxy [REDACTED_IP] set");
        assert_eq!(desensitize("upstream 10.0.0.5 down"), "upstream [REDACTED_IP] down");
    }

    #[test]
    fn leaves_version_numbers_alone() {
        assert_eq!(desensitize("driver 551.86 ready"), "driver 551.86 ready");
        assert_eq!(desensitize("build b9837 v2.2.7"), "build b9837 v2.2.7");
    }

    #[test]
    fn handles_path_and_ip_together() {
        assert_eq!(
            desensitize("load D:\\a\\b.txt from 10.1.2.3"),
            "load .../b.txt from [REDACTED_IP]"
        );
    }

    #[test]
    fn normalizes_log_levels() {
        assert_eq!(normalize_level("debug"), "debug");
        assert_eq!(normalize_level("WARN"), "warn");
        assert_eq!(normalize_level(" Error "), "error");
        // Unknown values fall back to info.
        assert_eq!(normalize_level("verbose"), "info");
    }

    #[test]
    fn reports_settings_level_when_no_env_override() {
        let (来源, 指令, 无效) = 选择过滤("DEBUG", None);
        assert_eq!(来源, 级别来源::配置);
        assert_eq!(来源.as_str(), "settings");
        // 设置里的写法不统一（大小写/空格），生效值必须是归一后的。
        assert_eq!(指令, "debug");
        assert!(无效.is_none());
    }

    #[test]
    fn env_override_beats_settings_and_is_reported_verbatim() {
        // 反例就是这次要修的谎报：设置写 INFO，实际按 debug 在跑。
        // 旧实现首行打的是「log_level=INFO」，没人看得出 DEBUG 行为什么会出现。
        let (来源, 指令, 无效) = 选择过滤("INFO", Some("debug"));
        assert_eq!(来源, 级别来源::环境变量);
        assert_eq!(来源.as_str(), "RUST_LOG");
        assert_eq!(指令, "debug");
        assert_ne!(指令, normalize_level("INFO"));
        assert!(无效.is_none());

        // 多 directive 的写法也必须原样保留，不能被拆成单级别报出去。
        let (来源, 指令, _) = 选择过滤("INFO", Some("localai=debug,hyper=warn"));
        assert_eq!(来源, 级别来源::环境变量);
        assert_eq!(指令, "localai=debug,hyper=warn");
    }

    #[test]
    fn invalid_env_falls_back_loudly() {
        // 非法指令不能静默吞掉：回落设置，同时把原文交出去让调用方 WARN。
        let (来源, 指令, 无效) = 选择过滤("WARN", Some("localai=nonsense"));
        assert_eq!(来源, 级别来源::配置);
        assert_eq!(指令, "warn");
        assert_eq!(无效.as_deref(), Some("localai=nonsense"));
    }

    #[test]
    fn blank_env_is_treated_as_unset() {
        // `RUST_LOG=""` 与没设等价（`try_from_default_env` 也是这个行为），
        // 不该被报成「无效环境变量」—— 否则每台机器都会白亮一条 WARN。
        for raw in ["", "   "] {
            let (来源, 指令, 无效) = 选择过滤("INFO", Some(raw));
            assert_eq!(来源, 级别来源::配置);
            assert_eq!(指令, "info");
            assert!(无效.is_none(), "空 RUST_LOG 不该被当成非法指令");
        }
    }
}
