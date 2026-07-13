; 安装前系统版本预检：本软件依赖仅 Windows 8+ 才有的系统 API，
; 且底层 Tauri 2 / WebView2 要求 Windows 10 及以上。低版本系统上主程序
; 会在启动时因缺少 kernel32.dll 入口点（如 GetSystemTimePreciseAsFileTime）而崩溃，
; 因此在安装阶段就拦截并给出人话提示，避免用户装完后撞到英文报错。
!include "WinVer.nsh"

!macro NSIS_HOOK_PREINSTALL
  ${IfNot} ${AtLeastWin10}
    MessageBox MB_OK|MB_ICONSTOP "本地AI模型一键部署助手 仅支持 Windows 10 / Windows 11（64 位）。$\r$\n$\r$\n检测到当前系统版本过低，无法继续安装。$\r$\n请在 Windows 10 或更高版本的系统上运行本安装程序。"
    Quit
  ${EndIf}
!macroend
