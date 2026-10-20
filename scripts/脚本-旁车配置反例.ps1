# 旁车配置反例测试（C-13 ~ C-16）—— 在 P100 的 Windows 11 测试机上运行
#
# 本脚本只做三件事：改坏配置 / 等你点启动 / 抓日志增量。
# 判定权不在脚本：它只做关键词粗筛，日志里没有报错就等于没做。
#
# 前提（缺一条就白跑）：
#   1. APP 里的 AI 服务必须是【停止】状态 —— 服务在跑时点「启动」会走接管，根本不会读配置；
#   2. 模型选 1.7B（那个有 launch.json 的），别换成 Ornith。
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File .\脚本-旁车配置反例.ps1
#   或指定配置路径：... -ConfigPath "C:\Model\models\qwen3_1_7b_light\Qwen3-1.7B-Q8_0.launch.json"

param(
    [string]$ConfigPath = "",
    [string]$LogDir = "$env:LOCALAPPDATA\LocalAI-Installer\data\logs"
)

$ErrorActionPreference = "Stop"

# ---- 无 BOM 写文件：APP 用 serde_json 解析，BOM 会让"正确配置"也启动失败 ----
function Write-NoBom {
    param([string]$Path, [string]$Text)
    $enc = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($Path, $Text, $enc)
}

function Find-Config {
    param([string]$Hint)
    if ($Hint -ne "") { return $Hint }
    $roots = @("C:\Model\models", "D:\Model\models", "$env:USERPROFILE\models")
    foreach ($root in $roots) {
        if (Test-Path $root) {
            $hit = Get-ChildItem -Path $root -Recurse -Filter "*.launch.json" -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($hit) { return $hit.FullName }
        }
    }
    return ""
}

function Latest-Log {
    param([string]$Dir)
    if (-not (Test-Path $Dir)) { return "" }
    $f = Get-ChildItem -Path $Dir -Filter "localai.log.*" -ErrorAction SilentlyContinue |
         Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($f) { return $f.FullName }
    return ""
}

function Size-Of {
    param([string]$Path)
    if ($Path -eq "" -or -not (Test-Path $Path)) { return 0 }
    return (Get-Item $Path).Length
}

function Read-NewLines {
    param([string]$Path, [long]$From)
    if ($Path -eq "" -or -not (Test-Path $Path)) { return @() }
    $fs = [System.IO.File]::Open($Path, 'Open', 'Read', 'ReadWrite')
    try {
        if ($fs.Length -le $From) { return @() }
        $fs.Seek($From, 'Begin') | Out-Null
        $sr = New-Object System.IO.StreamReader($fs, [System.Text.Encoding]::UTF8)
        $text = $sr.ReadToEnd()
        if ($text -eq "") { return @() }
        return ($text -split "`r?`n")
    } finally { $fs.Close() }
}

function Show-Verdict {
    param([string[]]$Lines)
    if ($null -eq $Lines -or $Lines.Count -eq 0) {
        Write-Host "    !! 日志没有新增行 —— 请把界面上的报错文字抄下来（这条不能算通过）" -ForegroundColor Yellow
        return -1
    }
    $hit = @($Lines | Where-Object { $_ -match "失败|错误|error|Error|ERROR|不存在|不是合法|无法|拒绝|invalid" })
    Write-Host ("    新增 " + $Lines.Count + " 行，其中疑似报错 " + $hit.Count + " 行：") -ForegroundColor Gray
    if ($hit.Count -gt 0) {
        foreach ($line in $hit) { Write-Host ("      " + $line) -ForegroundColor Red }
    } else {
        Write-Host "      （没匹配到报错关键词 —— 可能是静默放过，需人工确认）" -ForegroundColor Yellow
    }
    return $hit.Count
}

# ---------------- 主流程 ----------------

$path = Find-Config -Hint $ConfigPath
if ($path -eq "") {
    Write-Host "没找到 *.launch.json，请用 -ConfigPath 指定（例如 C:\Model\models\qwen3_1_7b_light\Qwen3-1.7B-Q8_0.launch.json）" -ForegroundColor Red
    exit 2
}
if (-not (Test-Path $path)) {
    Write-Host ("配置不存在：" + $path) -ForegroundColor Red
    exit 2
}

$original = Get-Content -Path $path -Raw -Encoding UTF8
$backup = $path + ".bak-" + (Get-Date -Format "yyyyMMdd-HHmmss")
Copy-Item -Path $path -Destination $backup -Force

Write-Host ("配置：" + $path)
Write-Host ("备份：" + $backup) -ForegroundColor Gray
Write-Host ("日志：" + (Latest-Log -Dir $LogDir)) -ForegroundColor Gray
Write-Host ""
Write-Host "请确认：APP 里 AI 服务是【停止】状态、当前模型是 1.7B。" -ForegroundColor Yellow
Read-Host "确认后按回车开始（会连做四例，每例都会等你点启动）" | Out-Null
Write-Host ""

# 给 C-14 用：字段名故意写错、值仍指向真实模板文件，这样失败原因才是"未知字段"
$tplName = ""
try {
    $obj = $original | ConvertFrom-Json
    if ($obj.'对话模板文件') { $tplName = $obj.'对话模板文件' }
} catch { }
if ($tplName -eq "") {
    Write-Host "!! 原配置里没有读到「对话模板文件」字段，C-14 会退化成「文件不存在」——请人工确认失败原因" -ForegroundColor Yellow
    $tplName = "tpl-prefill-closed.jinja"
}

$broken16 = [regex]::Replace($original.TrimEnd(), '\}\s*$', '')

$cases = @(
    @{ Id = "C-13"; Desc = "引用的文件不存在";            Body = "{`"对话模板文件`":`"tpl-不存在.jinja`"}";  Expect = "启动失败，并指名 tpl-不存在.jinja" },
    @{ Id = "C-14"; Desc = "字段名打错一个字";            Body = ("{`"对话模板文档`":`"" + $tplName + "`"}"); Expect = "启动失败（未知字段不许静默忽略）" },
    @{ Id = "C-15"; Desc = "只写缩放、不给适配器文件";     Body = "{`"适配器缩放`":0.5}";                      Expect = "启动失败（孤立缩放无法推导）" },
    @{ Id = "C-16"; Desc = "少一个右花括号";              Body = $broken16;                                   Expect = "启动失败，提示不是合法 JSON" }
)

$log = Latest-Log -Dir $LogDir
if ($log -eq "") {
    Write-Host "没找到日志文件（$LogDir）—— 仍可继续，只是抓不到增量，全靠界面文字" -ForegroundColor Yellow
}

$results = @()

try {
    foreach ($c in $cases) {
        Write-Host ("=== " + $c.Id + " · " + $c.Desc + " ===") -ForegroundColor Cyan
        Write-NoBom -Path $path -Text $c.Body
        Write-Host "    已写入坏配置。现在到 APP 里点「启动 AI 服务」，等界面出结果。" -ForegroundColor Yellow
        Write-Host ("    期望：" + $c.Expect)
        $before = Size-Of -Path $log
        Read-Host "    看到结果后按回车继续" | Out-Null
        $lines = Read-NewLines -Path $log -From $before
        $n = Show-Verdict -Lines $lines
        if ($n -gt 0) { $results += ($c.Id + "  " + $c.Desc + "  ->  " + $n + " 行报错（疑似挡住）") }
        elseif ($n -eq 0) { $results += ($c.Id + "  " + $c.Desc + "  ->  有日志但无报错关键词（需人工确认）") }
        else { $results += ($c.Id + "  " + $c.Desc + "  ->  日志无新增（不能算通过）") }
        # 立刻还原，避免下一例带着上一例的坏配置
        Write-NoBom -Path $path -Text $original
        Write-Host "    已还原正确配置" -ForegroundColor Green
        Write-Host ""
    }

    Write-Host "=== 汇总 ===" -ForegroundColor Cyan
    foreach ($r in $results) { Write-Host ("  " + $r) }
    Write-Host ""
}
finally {
    Write-NoBom -Path $path -Text $original
    Write-Host "最终配置已还原（内容与备份一致）。备份：" -ForegroundColor Green
    Write-Host ("  " + $backup)
    Write-Host ""
    Write-Host "四条里只要有一条「服务照样起来了 / 日志里毫无报错」，就是静默放过，算不通过。" -ForegroundColor Yellow
    Write-Host "把上面整段输出贴回问题单。" -ForegroundColor Yellow
}
