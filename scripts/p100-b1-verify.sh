#!/bin/bash
# ==========================================================================
# B1 验证：用官方 Vulkan 引擎（b10853）加载 Qwen3.8-27B
#
# 目的（对应升级清单「缺口 3」的未知数）：
#   核心    Qwen3.8-27B 能否加载成功（missing tensor 是否消失）
#   未知数2 Vulkan 后端是否支持 q8_0 KV 量化
#   未知数3 P100 上 Vulkan 的 tok/s（prompt / predict）
#   未知数4 -ngl 99 是否真把层放上卡（nvidia-smi 实测 vs 账面）
#
# 分两阶段，每阶段独立启停：
#   阶段 A：-c 4096（小上下文，快速确认加载问题已解决）
#   阶段 B：-c 49152 q8_0（脚本实际配置，测显存与速度）
#
# 不做任何持久化改动；跑完即释放显存。
# ==========================================================================
set -u

ENGINE=/opt/llama-vulkan/llama-server
EDIR=/opt/llama-vulkan
MODEL=/opt/llama.cpp/models/Qwen3.8-27B-UD-Q3_K_XL.gguf
PORT=8999
LOG=/tmp/p100-b1-verify.log

: > "$LOG"
log() { echo -e "$@" | tee -a "$LOG"; }
hr()  { log "----------------------------------------------------------------"; }

log "B1 验证 —— $(date '+%Y-%m-%d %H:%M:%S')"
log "引擎：$ENGINE"
log "模型：$MODEL"
hr

# 确保没有残留进程（预检已确认 GPU 空闲，这里只是收尾）
pkill -9 -f "llama-server" 2>/dev/null
sleep 3

# ------------------------------------------------------------------
# 通用函数：启动 → 等待 → 抓关键日志 → 报告显存
#   参数：$1=上下文  $2=KV档  $3=阶段名
# ------------------------------------------------------------------
跑阶段() {
    local CTX="$1" KVT="$2" NAME="$3"
    local SRVLOG="/tmp/p100-b1-server-${CTX}.log"

    hr
    log "【$NAME】ctx=$CTX  KV=$KVT"
    hr

    : > "$SRVLOG"
    LD_LIBRARY_PATH="$EDIR" nohup "$ENGINE" \
        -m "$MODEL" \
        -ngl 99 -c "$CTX" \
        -ctk "$KVT" -ctv "$KVT" \
        --host 127.0.0.1 --port "$PORT" \
        --parallel 1 --no-mmap --metrics \
        > "$SRVLOG" 2>&1 &
    local PID=$!
    log "  进程 PID=$PID，日志：$SRVLOG"

    # 最多等 600 秒（27B 从磁盘读 12 GiB 约 30~90 秒）
    local i=0 OK=0
    while [ "$i" -lt 120 ]; do
        if grep -q "server is listening" "$SRVLOG" 2>/dev/null; then OK=1; break; fi
        if ! kill -0 "$PID" 2>/dev/null; then break; fi
        sleep 5
        i=$((i + 1))
        if [ $((i % 6)) -eq 0 ]; then log "    ...等待中 $((i * 5))s"; fi
    done

    if [ "$OK" -eq 1 ]; then
        log "  ✅ 服务已监听（耗时约 $((i * 5)) 秒）"
    else
        if kill -0 "$PID" 2>/dev/null; then ALIVE="是"; else ALIVE="否"; fi
        log "  ❌ 未监听：进程存活=$ALIVE"
    fi

    # ---- 关键日志行 ----
    log ""
    log "  ▸ 加载相关（missing tensor 应消失，改出现 unused tensor ignoring）："
    grep -iE "missing tensor|unused tensor|error|failed|abort" "$SRVLOG" 2>/dev/null \
        | head -20 | sed 's/^/      /' | tee -a "$LOG"
    [ -s "$SRVLOG" ] || log "      （日志为空）"

    log "  ▸ 后端与设备："
    grep -iE "ggml_vulkan|Vulkan devices|device memory|Found [0-9]+ Vulkan|backend" "$SRVLOG" 2>/dev/null \
        | head -10 | sed 's/^/      /' | tee -a "$LOG"

    log "  ▸ offload 与上下文："
    grep -iE "offloaded|offloading|n_ctx|context size|KV self size|kv cache|flash_attn|flash attention" "$SRVLOG" 2>/dev/null \
        | head -15 | sed 's/^/      /' | tee -a "$LOG"

    log "  ▸ KV 量化是否被支持（未知数 2）："
    if grep -qi "V cache quantization requires flash attention" "$SRVLOG" 2>/dev/null; then
        log "      ⚠ 出现「V cache quantization requires flash attention」→ Vulkan 下 q8_0 KV 不可用"
    elif [ "$OK" -eq 1 ]; then
        log "      ✅ 未报该错 → q8_0 KV 在 Vulkan 下可用"
    else
        log "      （未跑到该检查点）"
    fi

    # ---- 显存实测（未知数 4）----
    log "  ▸ nvidia-smi 实测显存："
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>&1 \
        | sed 's/^/      /' | tee -a "$LOG"
    log "  ▸ GPU 计算进程："
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1 \
        | sed 's/^/      /' | tee -a "$LOG"

    # ---- 速度实测（未知数 3）----
    if [ "$OK" -eq 1 ]; then
        log ""
        log "  ▸ 推理测速（未知数 3）："
        local T0 T1
        T0=$(date +%s)
        curl -s -m 300 -X POST "http://127.0.0.1:${PORT}/completion" \
            -H "Content-Type: application/json" \
            -d '{"prompt":"请用三句话介绍本地私有化部署大模型的价值。","n_predict":128,"temperature":0}' \
            -o /tmp/p100-b1-completion.json 2>/dev/null
        T1=$(date +%s)
        log "      墙钟耗时：$((T1 - T0)) 秒（含 128 token 生成）"
        if [ -s /tmp/p100-b1-completion.json ]; then
            python3 - <<'PYEOF' 2>&1 | sed 's/^/      /' | tee -a "$LOG"
import json
try:
    d = json.load(open('/tmp/p100-b1-completion.json'))
    t = d.get('timings', {})
    print("prompt  tokens:", t.get('prompt_n'), " 速度:", round(t.get('prompt_per_second', 0), 2), "tok/s")
    print("predict tokens:", t.get('predicted_n'), " 速度:", round(t.get('predicted_per_second', 0), 2), "tok/s")
    txt = d.get('content', '')
    print("生成样例（前 120 字）:", txt[:120].replace("\n", " "))
except Exception as e:
    print("解析失败:", e)
PYEOF
        else
            log "      （无响应内容）"
        fi

        log "  ▸ /props 核对 n_ctx："
        curl -s -m 30 "http://127.0.0.1:${PORT}/props" 2>/dev/null \
            | python3 -c "import sys,json; d=json.load(sys.stdin); print('n_ctx =', d.get('default_generation_settings',{}).get('n_ctx')); print('model =', d.get('model_path','')[-50:])" 2>&1 \
            | sed 's/^/      /' | tee -a "$LOG"
    fi

    # ---- 收尾：释放显存 ----
    log ""
    log "  释放显存（kill $PID）..."
    kill "$PID" 2>/dev/null
    sleep 5
    kill -9 "$PID" 2>/dev/null
    sleep 3
    nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>&1 | sed 's/^/      释放后占用：/' | tee -a "$LOG"
}

# 阶段 A：小上下文，快速确认加载问题已解决
跑阶段 4096 q8_0 "阶段 A —— 快速确认加载"

# 阶段 B：脚本实际配置
跑阶段 49152 q8_0 "阶段 B —— 脚本实际配置"

hr
log "验证完成 $(date '+%H:%M:%S')   完整日志：$LOG"
hr
log "所有 llama-server 进程已释放："
pgrep -af llama-server 2>&1 | sed 's/^/  /' | tee -a "$LOG" || log "  （无残留）"
nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>&1 | sed 's/^/  最终显存占用：/' | tee -a "$LOG"
