# -*- coding: utf-8 -*-
"""
知识库服务 · HTTP 层冒烟
========================

为什么不复用 `探测-知识库服务.py`：那个脚本**直接 import `回答()`**，不走网络。
它能验"逻辑对不对"，但验不到这一层的东西：

  · CORS 头在不在（Tauri webview 里前端直连本服务，缺了它只会看到一个
    笼统的失败 —— 既没有状态码也没有响应体，排查时会误判成"服务没起来"）；
  · OPTIONS 预检能不能过；
  · 中文问句经 HTTP 传过去会不会变样；
  · `--生成端` 参数有没有真的进到 `/health` 的 `deps`。

用法：
    python scripts/探测-知识库HTTP.py                    # 打本机 8092
    python scripts/探测-知识库HTTP.py --端口 8094
    python scripts/探测-知识库HTTP.py --题 P01 L03 O03   # 只打指定题
"""

import json
import os
import sys
import urllib.error
import urllib.request

脚本目录 = os.path.dirname(os.path.abspath(__file__))
根目录 = os.path.dirname(脚本目录)

端口 = 8092
指定题 = []

# 模拟 Tauri webview 的 Origin（打包是 tauri://localhost，dev 是 http://localhost:1420）
来源 = "http://localhost:1420"

# 取三题代表：库内普通题 / 库内表格题 / 库外题
默认题 = ["P01", "L03", "O03"]


def 读题集():
    with open(os.path.join(根目录, "eval", "题目.json"), encoding="utf-8") as 文件:
        数据 = json.load(文件)
    列表 = 数据["题目"] if isinstance(数据, dict) else 数据
    return {x["编号"]: x for x in 列表}


def 请求(路径, 方法="GET", 体=None, 带来源=True):
    """返回 (状态码, 响应头, 原始字节)。4xx/5xx 不抛，交给调用方打印。"""
    数据 = json.dumps(体, ensure_ascii=False).encode("utf-8") if 体 is not None else None
    头 = {"Content-Type": "application/json"}
    if 带来源:
        头["Origin"] = 来源
    if 方法 == "OPTIONS":
        头["Access-Control-Request-Method"] = "POST"
        头["Access-Control-Request-Headers"] = "content-type"
    请求对象 = urllib.request.Request(
        "http://127.0.0.1:%d%s" % (端口, 路径), data=数据, headers=头, method=方法)
    try:
        with urllib.request.urlopen(请求对象, timeout=180) as 响应:
            return 响应.status, dict(响应.headers), 响应.read()
    except urllib.error.HTTPError as 错:
        return 错.code, dict(错.headers), 错.read()


def 主流程():
    global 端口, 指定题
    sys.stdout.reconfigure(errors="replace")

    if "--端口" in sys.argv:
        端口 = int(sys.argv[sys.argv.index("--端口") + 1])
    if "--题" in sys.argv:
        指定题 = sys.argv[sys.argv.index("--题") + 1:]
    题号列表 = 指定题 or 默认题

    print("=" * 92)
    print(" 知识库服务 · HTTP 层冒烟    127.0.0.1:%d    Origin: %s" % (端口, 来源))
    print("=" * 92)

    # ---------- ① CORS 预检 ----------
    print("\n① OPTIONS /ask（预检）")
    码, 头, _ = 请求("/ask", 方法="OPTIONS")
    跨域 = 头.get("Access-Control-Allow-Origin")
    print("   状态码 %s（期望 204）" % 码)
    print("   Access-Control-Allow-Origin: %s" % (跨域 or "❌ 缺失"))
    print("   Allow-Methods: %s ｜ Allow-Headers: %s"
          % (头.get("Access-Control-Allow-Methods"), 头.get("Access-Control-Allow-Headers")))
    预检过 = 码 == 204 and 跨域 == "*"

    # ---------- ② /health ----------
    print("\n② GET /health")
    码, 头, 体 = 请求("/health", 方法="GET")
    健康 = json.loads(体.decode("utf-8"))
    依赖 = 健康.get("deps", {})
    print("   状态码 %s（期望 200）" % 码)
    print("   Access-Control-Allow-Origin: %s" % (头.get("Access-Control-Allow-Origin") or "❌ 缺失"))
    print("   %d 份文档 / %d 块 / 池深 %d / top_k %d ｜ 类2软=%s"
          % (健康["docs"], 健康["blocks"], 健康["pool"], 健康["top_k"], 健康["class2_soft"]))
    print("   检索链 %s" % 健康["chain"])
    print("   deps.rerank   = %s" % 依赖.get("rerank"))
    print("   deps.generate = %s   ← 必须与 APP 设置里的 service_port 一致" % 依赖.get("generate"))
    健康过 = 码 == 200 and 健康.get("status") == "ok"

    # ---------- ③ /ask 逐题 ----------
    题集 = 读题集()
    结果行 = []
    for 号 in 题号列表:
        题 = 题集.get(号)
        if 题 is None:
            print("\n③ 跳过 %s（题目集里没有）" % 号)
            continue
        问句 = 题.get("问") or 题.get("问句")
        print("\n③ POST /ask  [%s] %s" % (号, 问句))
        码, 头, 体 = 请求("/ask", 方法="POST", 体={"question": 问句, "top_k": 3})
        if 码 != 200:
            print("   状态码 %s ❌  %s" % (码, 体.decode("utf-8", "replace")[:200]))
            结果行.append((号, 码, "—", 0, "—"))
            continue
        出 = json.loads(体.decode("utf-8"))
        引用 = 出.get("citations", [])
        print("   状态码 200 ｜ CORS %s" % (头.get("Access-Control-Allow-Origin") or "❌ 缺失"))
        print("   最高分 %+.2f   拒答 %s   类型 %s"
              % (出["top_score"], "是" if 出["refused"] else "否", 出["refusal_type"] or "—"))
        print("   答案：%s" % (出["answer"] or "(空)").replace("\n", " ")[:110])
        for i, 条 in enumerate(引用, 1):
            print("     依据%d 【%s】%s  得分 %+.2f" % (i, 条["doc_no"], 条["section"], 条["score"]))
        if 出.get("off_library_refs"):
            print("     线索指向库外：%s" % "、".join(出["off_library_refs"]))
        # 中文往返检查：问句经 HTTP 回来必须与发出的一致
        回环 = "✅" if 出["question"] == 问句 else "❌ 中文在 HTTP 层被改动（发出 %r 收到 %r）" % (问句, 出["question"])
        print("   问句回环 %s ｜ 耗时 %s s" % (回环, 出["timings"].get("total")))
        结果行.append((号, 码, "是" if 出["refused"] else "否",
                       len(引用), (出["answer"] or "")[:28].replace("\n", " ")))

    # ---------- 汇总 ----------
    print("\n" + "=" * 92)
    print(" 汇总")
    print("=" * 92)
    print(" %-6s %-6s %-6s %-6s %s" % ("题", "状态", "拒答", "引用数", "答案开头"))
    print(" " + "-" * 88)
    for 行 in 结果行:
        print(" %-6s %-6s %-6s %-6s %s" % 行)

    全过 = 预检过 and 健康过 and all(r[1] == 200 for r in 结果行)
    print()
    print(" 预检 %s ｜ /health %s ｜ /ask %s  →  总判定 %s"
          % ("✅" if 预检过 else "❌", "✅" if 健康过 else "❌",
             "✅" if all(r[1] == 200 for r in 结果行) else "❌",
             "✅ 通过" if 全过 else "❌ 未通过"))
    print("=" * 92)
    return 0 if 全过 else 1


if __name__ == "__main__":
    sys.exit(主流程())
