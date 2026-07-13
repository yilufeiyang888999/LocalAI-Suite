//! 抑制子进程（llama-server、nvidia-smi、reg、taskkill）在 Windows 上弹出的控制台窗口。
use std::process::Command;

/// 让 `command` 启动的子进程不弹出控制台窗口（仅 Windows 生效，其它平台空操作）。
pub fn hide_console(command: &mut Command) {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        command.creation_flags(CREATE_NO_WINDOW);
    }
    #[cfg(not(windows))]
    let _ = command;
}
