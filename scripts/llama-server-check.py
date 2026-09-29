#!/usr/bin/env python3
"""llama-server 服务诊断（HTTP 层）

用途：APP 启动推理服务后，从外部核对服务状态，并把「客户端请求体的问题」与
「server 自身的问题」一刀切开。对应 CLAUDE.md「已知上游问题：内置 WebUI 的
DRY 参数被 server 拒收」一节里那套三步排查法。

它只做 HTTP 层的事，不解析 GGUF —— 模型层结构请跑 scripts/gguf-probe.py，
避免出现第三份 GGUF 实现。

用法：
    python scripts/llama-server-check.py                 # 默认端口 8000，只读状态
    python scripts/llama-server-check.py 8080            # 指定端口
    python scripts/llama-server-check.py 8000 --对照      # 追加三组对照请求（会真的推理）

无第三方依赖，纯标准库。
"""

import json
import sys
import urllib.error
import urllib.request

sys.stdout.reconfigure(errors="replace")

超时秒 = 120

# 已知的上游报错特征 -> 结论与规避建议。
# 判别方式：报错文本里出现这个片段，就说明是上游自己没对齐，与 APP 无关。
已知上游报错 = {
    "dry_penalty_last_n": (
        "llama.cpp 上游自相矛盾：CLI 帮助把 `--dry-penalty-last-n` 的 -1 定义为"
        "「扫全上下文」的合法值，但 HTTP 请求体校验只接受 0 及以上。\n"
        "     内置 WebUI 会把浏览器里保存的那个值（-1）原样发出来，于是撞上自己的校验。\n"
        "     **APP 与外部 API 不受影响** —— 它们的请求体里没有这个字段。\n"
        "     规避：① 用 APP 内置聊天；② WebUI → 设置 → Sampling Penalties →\n"
        "     「DRY penalty last N」改成 0（DRY 总开关是 dry_multiplier，默认 0 = 禁用，改 0 无损）；\n"
        "     ③ 清掉浏览器对本站点的存储，让设置回到 server 的默认值。"
    ),
}

# 默认生成参数里值得关注的几项（其余为常规采样参数，诊断价值低）
关注参数 = [
    "dry_multiplier",
    "dry_penalty_last_n",
    "repeat_last_n",
    "repeat_penalty",
    "temperature",
    "top_k",
    "top_p",
    "min_p",
]


def 取(url, body=None, timeout=超时秒):
    """发 GET（body 为 None）或 POST，返回 (状态码, 解析后的 JSON 或原始文本)。"""
    data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method="POST" if data else "GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        原文 = e.read().decode("utf-8", "replace")
        try:
            return e.code, json.loads(原文)
        except json.JSONDecodeError:
            return e.code, 原文
    except Exception as e:  # 连接层失败（服务没起来 / 端口不对）
        return None, f"{type(e).__name__}: {e}"


def 取报错文本(结果):
    """/v1/chat/completions 的报错藏在 error.message 里。"""
    if isinstance(结果, dict) and isinstance(结果.get("error"), dict):
        return str(结果["error"].get("message", ""))
    return str(结果) if not isinstance(结果, dict) else ""


def 诊断报错(文本):
    """把已知的上游报错翻译成结论 + 规避建议。返回 True 表示已识别。"""
    for 特征, 说明 in 已知上游报错.items():
        if 特征 in 文本:
            print(f"     >>> 命中已知上游问题（{特征}）")
            print(f"     {说明}")
            return True
    return False


def 打印状态(端口):
    """读 /props，打印服务视角的配置。"""
    状态, 结果 = 取(f"http://127.0.0.1:{端口}/props")
    print(f"=== GET /props（端口 {端口}）===")
    if 状态 is None:
        print(f"  连不上：{结果}")
        print("  → 确认 APP 里推理服务已启动，且端口与此处一致（设置里的 service_port）。")
        return None
    if 状态 != 200:
        print(f"  HTTP {状态}：{结果}")
        return None

    设置 = 结果.get("default_generation_settings") or {}
    参数 = 设置.get("params") or {}
    上下文 = 设置.get("n_ctx")

    print(f"  构建      : {结果.get('build_info')}")
    print(f"  模型      : {结果.get('model_path')}")
    print(f"  量化      : {结果.get('model_ftype')}")
    print(f"  slots     : {结果.get('total_slots')}   （桌面单人用应为 1，>1 会让 KV ×N）")
    print(f"  n_ctx     : {上下文}")
    print()
    print("  默认采样参数（关注项）：")
    for 名 in 关注参数:
        if 名 in 参数:
            print(f"    {名:<22} = {参数[名]}")

    # server 自己的 dry_penalty_last_n 若为合法值，说明它不是报错来源，
    # -1 必然来自客户端（浏览器保存的设置 / 调用方自己传的）。
    if 参数.get("dry_penalty_last_n") is not None:
        值 = 参数["dry_penalty_last_n"]
        判定 = "合法" if isinstance(值, int) and 0 <= 值 <= 2147483647 else "**越界，server 自身有问题**"
        print()
        print(f"  → server 自报的 dry_penalty_last_n = {值}（{判定}）")
        if isinstance(值, int) and 值 >= 0:
            print("     既然 server 默认值是合法的，WebUI 报 -1 就只能是它自己带了值过来。")

    print()
    print("  提示：n_ctx 若小于该模型的原生上下文，通常是服务没重启（改完参数要重启才生效）。")
    print("        对比模型原生值请跑 scripts/gguf-probe.py。")
    return 结果


def 对照请求(端口, 模型名="local"):
    """三组对照：不带该字段 / 带 -1 / 带 0。

    这一组的作用是**切开「客户端发的值」与「server 默认值」** ——
    只有带 -1 的那组才会复现 WebUI 上的报错。
    """
    print(f"=== 三组对照请求（端口 {端口}，会真的推理，max_tokens=2）===")
    基础 = {"model": 模型名, "messages": [{"role": "user", "content": "好"}], "max_tokens": 2, "stream": False}

    组 = [
        ("不带该字段（APP / 外部 API 的形态）", dict(基础)),
        ("dry_penalty_last_n = -1（WebUI 的形态）", {**基础, "dry_penalty_last_n": -1}),
        ("dry_penalty_last_n = 0（建议改成这个）", {**基础, "dry_penalty_last_n": 0}),
    ]

    命中上游问题 = False
    for 名称, body in 组:
        状态, 结果 = 取(f"http://127.0.0.1:{端口}/v1/chat/completions", body)
        if 状态 is None:
            print(f"  [{名称}] 连接失败：{结果}")
            continue
        if 状态 == 200:
            print(f"  [{名称}] 200 OK")
        else:
            print(f"  [{名称}] HTTP {状态}：{取报错文本(结果)}")
            命中上游问题 = 诊断报错(取报错文本(结果)) or 命中上游问题

    print()
    if 命中上游问题:
        print("  结论：报错只由 -1 引起，与 APP 无关。按上面的规避建议处理即可。")
    else:
        print("  结论：三组均未复现该报错。若浏览器里仍报错，请带上浏览器请求的完整报错再查。")


def 主函数():
    参数表 = [a for a in sys.argv[1:] if not a.startswith("-")]
    端口 = int(参数表[0]) if 参数表 else 8000
    做对照 = "--对照" in sys.argv

    结果 = 打印状态(端口)
    if 结果 is not None and 做对照:
        对照请求(端口)


if __name__ == "__main__":
    主函数()
