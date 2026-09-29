#!/bin/bash
# ==========================================================================
# P100 llama.cpp 引擎升级 —— B1 方案（官方 Vulkan 预编译包）部署脚本
#
# 执行者：AI 代跑（2026-09-24，老李临时离开期间）
# 方案出处：AI大模型应用开发实战训练营/30天学习计划/资产-本地模型部署/
#           P100引擎升级清单.md  步骤 1 + 步骤 2-B1
#
# 设计原则：**不碰 /opt/llama.cpp 的现有引擎一个字节**
#   1) 只读预检（GPU / 磁盘 / rpath / Vulkan ICD）
#   2) 备份旧引擎二进制目录 → build-b8211-bin（只新增，不删除任何东西）
#   3) 解压官方 Vulkan 包 → /opt/llama-vulkan（全新目录，与旧引擎并存）
#   4) 版本与设备识别验证
#
# 回退（任意时刻，都在 /opt 下各归各位）：
#   rm -rf /opt/llama-vulkan                                     # 撤掉新引擎
#   cd /opt/llama.cpp && cp -a build-b8211-bin/. build/bin/      # 旧引擎铺回（仅当动过）
# ==========================================================================
set -u

LOG=/tmp/p100-b1-deploy.log
: > "$LOG"
log() { echo -e "$@" | tee -a "$LOG"; }
hr()  { log "----------------------------------------------------------------"; }

log "P100 B1 部署 —— $(date '+%Y-%m-%d %H:%M:%S')"
hr

# ---------- 0. 只读预检 ----------
log "【0】只读预检"
log "  GPU 使用（已用显存 / 总显存 / 利用率 / 温度）："
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu \
    --format=csv,noheader 2>&1 | sed 's/^/    /' | tee -a "$LOG"

log "  GPU 上的计算进程："
APPS=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1)
if [ -n "$APPS" ]; then
    echo "$APPS" | sed 's/^/    /' | tee -a "$LOG"
else
    log "    （无）"
fi

log "  /opt 可用空间："
df -h /opt | tail -1 | sed 's/^/    /' | tee -a "$LOG"

log "  旧引擎 rpath/runpath（决定回退方式 1 是否可靠）："
RP=$(readelf -d /opt/llama.cpp/build/bin/llama-server 2>/dev/null | grep -iE 'rpath|runpath' || true)
if [ -n "$RP" ]; then
    echo "$RP" | sed 's/^/    /' | tee -a "$LOG"
else
    log "    （无）"
fi

log "  Vulkan ICD 配置 /usr/share/vulkan/icd.d/nvidia_icd.json："
cat /usr/share/vulkan/icd.d/nvidia_icd.json 2>&1 | sed 's/^/    /' | tee -a "$LOG"

log "  NVIDIA Vulkan 库是否已注册（ldconfig）："
NVL=$(ldconfig -p | grep -iE 'libGLX_nvidia|libnvidia-gl|libnvidia-egl' || true)
if [ -n "$NVL" ]; then
    echo "$NVL" | sed 's/^/    /' | tee -a "$LOG"
else
    log "    ⚠ 未找到 —— 可能出现 ggml_vulkan: no devices found"
fi

log "  当前 llm-switch.sh 版本："
grep -m1 -E '定稿版|v3' /opt/llama.cpp/llm-switch.sh 2>&1 | sed 's/^/    /' | tee -a "$LOG"
hr

# ---------- 1. 备份旧引擎 ----------
log "【1】备份旧引擎二进制目录（只新增，不动原件）"
if [ -d /opt/llama.cpp/build-b8211-bin ]; then
    log "    已存在 build-b8211-bin，跳过（保护既有备份，不覆盖）"
else
    cp -a /opt/llama.cpp/build/bin /opt/llama.cpp/build-b8211-bin
    log "    ✅ 已备份 → /opt/llama.cpp/build-b8211-bin"
fi
log "    备份内容抽查："
ls -la /opt/llama.cpp/build-b8211-bin/ 2>&1 | head -6 | sed 's/^/      /' | tee -a "$LOG"
hr

# ---------- 2. 解压 Vulkan 包 ----------
log "【2】解压官方 Vulkan 预编译包"
PKG=/opt/llama-b10853-bin-ubuntu-vulkan-x64.tar.gz
if [ ! -f "$PKG" ]; then
    log "    ❌ 找不到 $PKG，中止"
    exit 1
fi
log "    包内顶层结构（应带一层 llama-b10853/）："
tar tzf "$PKG" 2>&1 | head -3 | sed 's/^/      /' | tee -a "$LOG"

mkdir -p /opt/llama-vulkan
tar xzf "$PKG" -C /opt/llama-vulkan --strip-components=1
log "    ✅ 解压完成，/opt/llama-vulkan 内容（前 12 项）："
ls /opt/llama-vulkan 2>&1 | head -12 | sed 's/^/      /' | tee -a "$LOG"
hr

# ---------- 3. 验证新引擎 ----------
log "【3】验证新引擎"
cd /opt/llama-vulkan || exit 1

log "    3.1 --version（期望 version: 10853）："
LD_LIBRARY_PATH=/opt/llama-vulkan ./llama-server --version 2>&1 | head -5 \
    | sed 's/^/      /' | tee -a "$LOG"

log "    3.2 --list-devices（期望 Vulkan0: Tesla P100-PCIE-16GB）："
LD_LIBRARY_PATH=/opt/llama-vulkan ./llama-server --list-devices 2>&1 | head -20 \
    | sed 's/^/      /' | tee -a "$LOG"
hr

log "完成 $(date '+%H:%M:%S')   —— 完整日志：$LOG"

# ---------- 4. 结尾自检 ----------
log ""
log "【4】收尾核对"
log "    旧引擎仍在原位（build/bin 未被改动）："
ls -la /opt/llama.cpp/build/bin/llama-server 2>&1 | sed 's/^/      /' | tee -a "$LOG"
log "    旧引擎版本："
LD_LIBRARY_PATH=/opt/llama.cpp/build/bin /opt/llama.cpp/build/bin/llama-server --version 2>&1 | head -3 \
    | sed 's/^/      /' | tee -a "$LOG"
hr
