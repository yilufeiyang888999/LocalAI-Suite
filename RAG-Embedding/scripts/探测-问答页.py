# -*- coding: utf-8 -*-
"""Web 问答页（`web/index.html`）的判据 —— HTTP 层 + JS/DOM 静态一致性。

为什么必须走网络、不能直接调函数：
  `do_GET` 的路由、状态码、`Content-Type`、`Cache-Control`、404 的可用路径列表
  —— 这些只有真发一次 HTTP 请求才看得到。

为什么要做 JS/DOM 静态检查：
  沙箱里没有浏览器，`getElementById("xxx")` 拿到 null 这种错要等用户打开页面才暴露。
  用文本比对（引用的 id 是否都声明了）把它提前到本地 —— 这也是
  「判据要能区分『通过』和『没检查』」的落实。

为什么要查「答案未拼 innerHTML」：
  模型输出是不可信文本。只要有一处用 innerHTML 拼接，页面就能被语料里的内容注入。

为什么还要测「客户端半途断连」（第 8 组）：
  浏览器刷新 / 关标签页 / 预连接回收，以及任何"只读响应头就走"的探针，都会在服务端
  写响应体写一半时把 socket 掐掉。默认 `socketserver` 会为此打一整页 Traceback ——
  项目里真的被吓过一次（2026-10-09）。所以这里把两件事都做成判据：
    ① 服务**吞掉**断连并计数（`/health` 的 `client_aborts` 要涨）；
    ② 断连风暴过**之后**，服务照旧把完整页面一字不差地发出来。
  🔴 缺了 ①，「扛住了」和「根本没触发到」在控制台上长得一模一样 —— 等于没判据。

用法：
  python scripts\\探测-问答页.py            # 默认端口 8092
  python scripts\\探测-问答页.py 8093       # 指定端口（服务须已启动）
"""
import io
import json
import os
import re
import socket
import struct
import sys
import time
import urllib.error
import urllib.request

端口 = int(sys.argv[1]) if len(sys.argv) > 1 else 8092
基 = "http://127.0.0.1:%d" % 端口
通过 = 失败 = 0


def 记(名, 条件, 补充=""):
    global 通过, 失败
    if 条件:
        通过 += 1
        print("  ✅ %-26s %s" % (名, 补充))
    else:
        失败 += 1
        print("  ❌ %-26s %s" % (名, 补充))


def 等端口(秒=25):
    """服务启动要时间（建基座 + 依赖自检），轮询而不是死等 sleep。"""
    终 = time.time() + 秒
    while time.time() < 终:
        try:
            urllib.request.urlopen(基 + "/health", timeout=3).read()
            return True
        except Exception:                                      # noqa: BLE001
            time.sleep(0.6)
    return False


print("=" * 88)
print("  Web 问答页 · 判据（端口 %d）" % 端口)
print("=" * 88)

if not 等端口():
    print("  ❌ 服务未就绪（先起 `知识库服务.py`），放弃")
    sys.exit(2)

# ---------- 1. GET / ----------
print("\n[1] GET /  （问答页本体）")
try:
    响 = urllib.request.urlopen(基 + "/", timeout=15)
    体 = 响.read().decode("utf-8")
    类 = 响.headers.get("Content-Type") or ""
    缓 = 响.headers.get("Cache-Control") or ""
    print("      状态 %d ｜ %s ｜ Cache-Control: %s ｜ %d 字节" % (响.status, 类, 缓, len(体)))
    记("200 且是 HTML", 响.status == 200 and "text/html" in 类)
    记("charset=utf-8", "charset=utf-8" in 类)
    记("no-store（防旧缓存）", "no-store" in 缓)
    记("含页面标题", "知识库问答" in 体)
    记("含 /ask 调用", "/ask" in 体)
    记("含 /health 调用", "/health" in 体)
    记("体积合理", len(体) > 3000, "%d 字节" % len(体))
except Exception as 错:                                        # noqa: BLE001
    记("GET /", False, repr(错))

# ---------- 2. /index.html 等价 ----------
print("\n[2] GET /index.html （应与 / 等价）")
try:
    甲 = urllib.request.urlopen(基 + "/", timeout=15).read()
    乙 = urllib.request.urlopen(基 + "/index.html", timeout=15).read()
    记("与 / 逐字节相同", 甲 == 乙)
except Exception as 错:                                        # noqa: BLE001
    记("/index.html", False, repr(错))

# ---------- 3. /favicon.ico ----------
print("\n[3] GET /favicon.ico （浏览器自动请求，应 204 而非 404）")
try:
    # 204 的 body 是空的，但**仍要读完再关**：这个探针自己不能变成"只连不读"的客户端，
    # 否则又给服务端送一个 RST（就是第 8 组测的那件事）。
    with urllib.request.urlopen(基 + "/favicon.ico", timeout=8) as 响:
        码 = 响.status
        响.read()
    记("状态 204", 码 == 204, "实际 %d" % 码)
except urllib.error.HTTPError as 错:
    记("状态 204", 错.code == 204, "实际 %d" % 错.code)
except Exception as 错:                                        # noqa: BLE001
    记("/favicon.ico", False, repr(错))

# ---------- 4. 未知路径 ----------
print("\n[4] GET /nope （应 404 且列出可用路径）")
try:
    urllib.request.urlopen(基 + "/nope", timeout=8)
    记("应 404 却成功", False)
except urllib.error.HTTPError as 错:
    体 = 错.read().decode("utf-8")
    记("404", 错.code == 404, "实际 %d" % 错.code)
    记("列出可用路径", "/ask" in 体 and "/health" in 体, 体.replace("\n", " ")[:80])
except Exception as 错:                                        # noqa: BLE001
    记("/nope", False, repr(错))

# ---------- 5. /health 未被破坏 ----------
print("\n[5] GET /health （旧接口不能因加页面而坏）")
try:
    h = json.loads(urllib.request.urlopen(基 + "/health", timeout=10).read())
    记("status ok", h.get("status") == "ok")
    记("含 class3 指纹", isinstance(h.get("class3"), dict),
       json.dumps(h.get("class3"), ensure_ascii=False)[:60])
    记("class3 标注类2 信号兜底", (h.get("class3") or {}).get("uses_class2_signal") is True,
       "缺该字段 = 跑的是「一次重写」的旧判据")
    记("docs/blocks 有值", bool(h.get("docs")) and bool(h.get("blocks")),
       "%s 份 / %s 块" % (h.get("docs"), h.get("blocks")))
    for k in ("score_threshold", "class2_soft", "chain", "deps"):
        记("字段 %s" % k, k in h)
except Exception as 错:                                        # noqa: BLE001
    记("/health", False, repr(错))

# ---------- 6. /ask 未被破坏（库内 + 库外各一题） ----------
print("\n[6] POST /ask （页面走的就是这条链）")
for 名, 问句, 期望拒答 in (("库内题", "宁波恒力不良率", False),
                          ("库外题", "电镀废液具体怎么处理", True)):
    try:
        体 = json.dumps({"question": 问句, "top_k": 3}, ensure_ascii=False).encode("utf-8")
        请 = urllib.request.Request(基 + "/ask", data=体,
                                    headers={"Content-Type": "application/json"})
        起 = time.time()
        d = json.loads(urllib.request.urlopen(请, timeout=300).read())
        耗 = time.time() - 起
        实拒 = bool(d.get("refused"))
        记("%s 拒答态符合预期" % 名, 实拒 == 期望拒答,
           "refused=%s type=%s %.2fs" % (实拒, d.get("refusal_type"), 耗))
        记("%s 有引用或拒答说明" % 名,
           bool(d.get("citations")) or bool(d.get("refusal_reason")),
           (d.get("answer") or "")[:44].replace("\n", " "))
        if not 期望拒答:
            记("库内题有 3 条引用", len(d.get("citations") or []) == 3,
               "doc_no=%s" % [c.get("doc_no") for c in d.get("citations") or []])
    except Exception as 错:                                    # noqa: BLE001
        记(名, False, repr(错))

# ---------- 7. JS / DOM 静态一致性 ----------
print("\n[7] JS / DOM 静态一致性（沙箱无浏览器，用文本比对提前抓 null 元素）")
路 = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "web", "index.html"))
页 = io.open(路, encoding="utf-8").read()
用到的 = set(re.findall(r'\$\("([^"]+)"\)', 页)) | set(re.findall(r'getElementById\("([^"]+)"\)', 页))
声明的 = set(re.findall(r'\bid="([^"]+)"', 页))
缺 = sorted(用到的 - 声明的)
记("getElementById 无悬空", not 缺, "缺 %s" % 缺 if 缺 else "%d 个 id 全部找到" % len(用到的))
记("声明数一致", len(声明的) == len(用到的), "声明 %d 个" % len(声明的))

# XSS：模型输出必须走 textContent，不能拼 innerHTML
拼串 = re.findall(r"innerHTML\s*=\s*[^;]*\b(?:answer|excerpt|section|doc_no|refusal_reason)\b", 页)
记("答案/引用未拼 innerHTML", not 拼串, str(拼串) if 拼串 else "全部 textContent")
记("无外链资源（离线可用）", not re.search(r'(src|href)\s*=\s*["\']https?://', 页))

# 提问后清空输入框（2026-10-10 修复的回归）：清空语句必须在 `提问()` 内、且在 fetch 之前。
# 静态检查只能证明「写了这行」，真正的 DOM 行为仍以浏览器人工复验为准 —— 两级判据，缺一不可。
清空 = re.search(r"问题框\.value\s*=\s*\"\"", 页)
清在提问前 = bool(清空) and 页.find("问题框.value = \"\"") < 页.find("fetch(\"/ask\"")
记("提问后清空输入框", 清在提问前, "清空须在 fetch 之前（失败/超时也不残留）")

# ---------- 8. 客户端半途断连 ----------
print("\n[8] 客户端提前断开（浏览器预连接 / 刷新 / 关标签页都会发生）")


def 取页():
    with urllib.request.urlopen(基 + "/", timeout=15) as 响:
        return 响.status, 响.read()


def 取断连计数():
    with urllib.request.urlopen(基 + "/health", timeout=10) as 响:
        return json.loads(响.read()).get("client_aborts")


def 硬断():
    """发完请求**立刻** RST（SO_LINGER=0），模拟浏览器/探针丢下 socket 就走。

    🔴 只发不读是关键：若改成「先读一点再断」，14 KB 的 body 很可能已被内核收下，
    服务端的 `sendall` 直接成功、**根本触发不到** —— 那就成了「看起来通过、其实没测」。
    """
    s = socket.create_connection(("127.0.0.1", 端口), timeout=5)
    s.sendall(b"GET / HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")
    for 格式 in ("HH", "ii"):        # `struct linger`：Windows 是 4 字节，POSIX 惯写 8 字节
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack(格式, 1, 0))
            break
        except OSError:
            continue
    s.close()


try:
    基线码, 基线体 = 取页()
    记("断连前取到完整基线", 基线码 == 200 and len(基线体) > 3000, "%d 字节" % len(基线体))

    起前 = 取断连计数()
    记("有 client_aborts 计数（判据的支点）", isinstance(起前, int),
       "缺该字段 = 服务端还是「照默认打完整堆栈」的旧壳")

    for _ in range(5):
        try:
            硬断()
        except Exception:                                      # noqa: BLE001
            pass
    time.sleep(0.8)                 # 等服务端那几个请求线程把异常走完
    终后 = 取断连计数()
    记("断连被吞掉且计数上涨", isinstance(起前, int) and isinstance(终后, int) and 终后 > 起前,
       "client_aborts %s → %s（5 次 RST）" % (起前, 终后))

    # 主判据：断连风暴之后，服务必须**照旧**把完整页面一字不差地发出来。
    现码, 现体 = 取页()
    记("断连后服务照常发整页", 现码 == 200 and 现体 == 基线体,
       "200 ｜ %d 字节（与基线逐字节相同）" % len(现体))
    记("/health 断连后仍 ok", 取断连计数() is not None)
except Exception as 错:                                        # noqa: BLE001
    记("客户端断连容错", False, repr(错))

print("\n" + "=" * 88)
print("  通过 %d ｜ 失败 %d" % (通过, 失败))
print("  总判定：%s" % ("✅ 全部通过" if 失败 == 0 else "❌ 有失败项"))
print("=" * 88)
sys.exit(0 if 失败 == 0 else 1)
