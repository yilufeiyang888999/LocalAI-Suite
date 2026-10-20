# -*- coding: utf-8 -*-
"""
全链路自检单（作品一 · D4/D5 收尾 · 可复跑）
=============================================

一条命令回答一个问题：**「现在这套东西还是好的吗？」**

设计原则（都是踩过的坑换来的）
------------------------------
1. **三态，不是二态**：每项只可能是
      ✅ 通过  —— 判据成立
      ❌ 失败  —— 判据不成立
      ⏭ 跳过  —— **没检查**（工具不在 / 上游不可达 / 需要 --全量）
   「跳过」必须显式出现，绝不能让它看起来像「通过」——
   判据若不能区分「通过」和「没检查」，就等于没有判据。
2. **`/health` 活着 ≠ 链路通**：llama-server 有「健康但请求挂起」的先例，
   所以必须**打一次真请求**才算数（本单的「三题冒烟」就是干这个的）。
3. **读旧产物不算证据**：`--全量` 跑验收后，会核对 `知识库服务-验收.json`
   的 mtime 是否**新于本次启动**，否则判 ❌（防「把上次的结论当这次的结果」）。
4. **基线分两类，别混**：语料规模（docs/blocks）从 `eval/语料基线.json` 读 —— 新增文档是
   合法变更，刷新走 `--写基线`（它会与服务 /health 交叉验证）；配置类（池 50 / Top-3 /
   阈值 -3.0 / 类2）写死断言，**它们变了才叫口径漂移**。
   语料或配置一变就会红 —— **红了是设计意图，不是脚本太严**。

与 `探测-知识库HTTP.py` 的分工
-------------------------------
  · `探测-知识库HTTP.py`：**深度单测**，打印每一项的原始细节，供人工看。
  · `自检-全链路.py`（本脚本）：**编排 + 断言**，只给结论，供批量复跑。
  两者都走网络，但本脚本会额外检查「配置是否被改偏」「产物是否新鲜」。

用法：
    python scripts/自检-全链路.py                    # 快检（约 10 s）
    python scripts/自检-全链路.py --全量             # + 66 题端到端验收（约 3 min）
    python scripts/自检-全链路.py --含APP            # + 前端类型检查 + Rust 单测
    python scripts/自检-全链路.py --全量 --含APP     # 全跑

退出码：0 = 无失败；1 = 至少一项 ❌（跳过不算失败）
"""

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

脚本目录 = os.path.dirname(os.path.abspath(__file__))
根目录 = os.path.dirname(脚本目录)
工程根 = os.path.dirname(根目录)
APP目录 = os.path.join(工程根, "Local-AI")
输出目录 = os.path.join(根目录, "eval")

# ---------- 地址（与 知识库服务.py / README 一致）----------
知识库地址 = "http://127.0.0.1:8092"
重排地址 = "http://127.0.0.1:8085"
生成地址 = "http://127.0.0.1:8000"
向量地址 = "http://127.0.0.1:8082"

# ---------- 冒烟题：库内普通 / 库内表格 / 库外（沿用 探测-知识库HTTP.py 的默认三题）----------
# (题号, 说明, 是否应当拒答)
冒烟题 = [
    ("P01", "库内·普通", False),
    ("L03", "库内·表格", False),
    ("O03", "库外", True),
]

# ---------- 基线（语料或配置变了就该红）----------
# 🔴 **语料规模不写死在这里** —— 它是会合法变化的量（新增文档就会变）。
#    唯一事实来源 = `eval/语料基线.json`，由 `自检-新增语料.py --写基线` **显式**刷新
#    （刷新时必与服务 /health 交叉验证）。这里只读它，不改它。
#    配置类（池深 / Top-K / 阈值 / 类2）才是真正该写死的常量 —— 它们变了才叫口径漂移。
基线 = {
    "pool": 50, "top_k": 3,
    "score_threshold": -3.0, "class2_soft": True, "class2_scope": 1,
}

# 语料基线（读不到就显式「跳过」，绝不装作通过）
语料基线 = None
try:
    with open(os.path.join(输出目录, "语料基线.json"), encoding="utf-8") as _文:
        _语 = json.load(_文)
    语料基线 = {"文档数": _语["文档数"], "块数": _语["块数"]}
except Exception:                                                # noqa: BLE001
    语料基线 = None

通过, 失败, 跳过 = "通过", "失败", "跳过"
图标 = {通过: "✅", 失败: "❌", 跳过: "⏭"}

结果 = []          # (阶段, 检查项, 状态, 详情)


# ============================ 基础工具 ============================

def 显示宽(文本):
    """中文/全角/emoji 按 2 宽计 —— f-string 的 `<12` 对中文会错位。"""
    return sum(2 if ord(c) > 0x2E7F else 1 for c in 文本)


def 补宽(文本, 宽):
    return 文本 + " " * max(0, 宽 - 显示宽(文本))


def 记(阶段, 名称, 状态, 详情=""):
    结果.append((阶段, 名称, 状态, 详情))
    print("  %s %s%s" % (图标[状态], 补宽(名称, 32), 详情), flush=True)


def HTTP(地址, 路径, 方法="GET", 体=None, 超时=10, 带来源=False):
    """返回 (状态码, 响应头小写字典型, 体文本)。

    连不上 → (None, {}, 错误串)。响应头**键一律小写**，避免取不到大小写变体。
    """
    数据 = json.dumps(体, ensure_ascii=False).encode("utf-8") if 体 is not None else None
    头 = {"Content-Type": "application/json"}
    if 带来源:
        头["Origin"] = "http://localhost:1420"        # Tauri dev 的 Origin
        if 方法 == "OPTIONS":
            头["Access-Control-Request-Method"] = "POST"
            头["Access-Control-Request-Headers"] = "content-type"
    请求 = urllib.request.Request(地址 + 路径, data=数据, headers=头, method=方法)
    try:
        with urllib.request.urlopen(请求, timeout=超时) as 响应:
            return 响应.status, {k.lower(): v for k, v in 响应.headers.items()}, \
                响应.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as 错:              # 4xx/5xx 不抛，交给调用方判
        return 错.code, {k.lower(): v for k, v in 错.headers.items()}, \
            错.read().decode("utf-8", "replace")
    except Exception as 错:                            # noqa: BLE001
        return None, {}, "%s: %s" % (type(错).__name__, 错)


def 子进程(命令, cwd=None, 超时=600, shell=False):
    """返回 (退出码, 输出文本)。退出码 None = 超时/启动失败。"""
    try:
        进程 = subprocess.run(命令, cwd=cwd, capture_output=True,
                              timeout=超时, shell=shell)
        文本 = (进程.stdout or b"").decode("utf-8", "replace") \
            + (进程.stderr or b"").decode("utf-8", "replace")
        return 进程.returncode, 文本
    except subprocess.TimeoutExpired:
        return None, "超时（%d s）" % 超时
    except Exception as 错:                            # noqa: BLE001
        return None, "%s: %s" % (type(错).__name__, 错)


# ============================ 一、依赖服务 ============================

def 查依赖():
    print("\n【一】依赖服务")
    for 名, 址, 必需 in (("重排 · bge-reranker-Q8_0", 重排地址, True),
                        ("生成端 · Spark-X2.5-4B", 生成地址, True),
                        ("向量 · bge-m3（本服务不用）", 向量地址, False)):
        码, _, 体 = HTTP(址, "/health", 超时=5)
        if 码 == 200:
            记("依赖", 名, 通过, 址)
        elif 必需:
            记("依赖", 名, 失败, "%s 不可达 —— %s" % (址, 体[:56]))
        else:
            记("依赖", 名, 跳过, "%s 不可达（知识库服务起手走 BM25，不需要它）" % 址)


# ============================ 二、知识库服务 ============================

def 查知识库():
    """返回：服务是否就绪（供末尾人工步骤判断）。"""
    print("\n【二】知识库服务（%s）" % 知识库地址)
    码, _, 体 = HTTP(知识库地址, "/health", 超时=5)
    if 码 != 200:
        记("知识库", "存活 /health", 失败,
           "连不上（%s）→ 先在终端跑：python scripts\\知识库服务.py" % 体[:48])
        return False
    记("知识库", "存活 /health", 通过, "200")

    健 = json.loads(体)
    不符 = ["%s=%s（基线 %s）" % (k, 健.get(k), v)
            for k, v in 基线.items() if 健.get(k) != v]
    if 不符:
        记("知识库", "配置与生产默认一致", 失败, "；".join(不符))
    else:
        记("知识库", "配置与生产默认一致", 通过,
           "池 50 / Top-3 / 阈值 -3.0 / 类2软")

    # 语料规模**单独一条** —— 它来自基线 json，不是写死常量（新增文档会让它合法变化）
    if 语料基线 is None:
        记("知识库", "语料规模与基线一致", 跳过,
           "eval/语料基线.json 读不到 → **未检查**（先跑 `自检-新增语料.py --写基线`）")
    else:
        差 = ["%s=%s（基线 %s）" % (k, 健.get(k), v) for k, v in
              (("docs", 语料基线["文档数"]), ("blocks", 语料基线["块数"]))
              if 健.get(k) != v]
        if 差:
            记("知识库", "语料规模与基线一致", 失败,
               "；".join(差) + " → 语料确实变了就重跑 `--写基线` 并记账")
        else:
            记("知识库", "语料规模与基线一致", 通过,
               "%d 文档 / %d 块（与 eval/语料基线.json 一致）"
               % (语料基线["文档数"], 语料基线["块数"]))

    # CORS 预检 —— webview 直连本服务的硬前提
    码, 头, _ = HTTP(知识库地址, "/ask", 方法="OPTIONS", 带来源=True, 超时=5)
    跨域 = 头.get("access-control-allow-origin")
    if 码 == 204 and 跨域 == "*":
        记("知识库", "CORS 预检", 通过, "204 + Allow-Origin: *")
    else:
        记("知识库", "CORS 预检", 失败, "状态 %s ｜ Allow-Origin %s" % (码, 跨域))

    # Web 问答页 —— 页面是**独立文件**（web/index.html）：文件被移走、路由被改坏，都不会
    # 在 /health 上体现，只有真请求一次才发现。所以这条不能省（同「健康 ≠ 链路通」）。
    码, _, 体 = HTTP(知识库地址, "/", 超时=8)
    if 码 == 200 and len(体) > 3000:
        记("知识库", "Web 问答页 GET /", 通过, "%d 字节 HTML" % len(体))
    else:
        记("知识库", "Web 问答页 GET /", 失败,
           "状态 %s（%d 字节）→ 检查 web/index.html 是否存在" % (码, len(体)))

    查三题()
    return True


def 查三题():
    with open(os.path.join(输出目录, "题目.json"), encoding="utf-8") as 文件:
        数据 = json.load(文件)
    列表 = 数据["题目"] if isinstance(数据, dict) else 数据
    题集 = {x.get("编号"): x for x in 列表}

    全对, 明细 = True, []
    for 号, 说明, 应拒 in 冒烟题:
        题 = 题集.get(号)
        if 题 is None:
            明细.append("%s 题目集里没有" % 号)
            全对 = False
            continue
        问句 = 题.get("问") or 题.get("问句")
        码, _, 体 = HTTP(知识库地址, "/ask", 方法="POST",
                       体={"question": 问句, "top_k": 3}, 超时=180)
        if 码 != 200:
            明细.append("%s HTTP %s" % (号, 码))
            全对 = False
            continue
        出 = json.loads(体)
        真拒 = bool(出.get("refused"))
        回环好 = 出.get("question") == 问句        # 中文经 HTTP 往返不变形
        好 = (真拒 == 应拒) and 回环好
        全对 = 全对 and 好
        明细.append("%s[%s] %s%s %+.2f %.1fs"
                    % (号, 说明, "拒" if 真拒 else "答",
                       "" if 回环好 else "(中文回环✗)",
                       出.get("top_score", 0.0),
                       出.get("timings", {}).get("total", 0.0)))
    记("知识库", "三题真请求（P01/L03/O03）", 通过 if 全对 else 失败,
       " ｜ ".join(明细))


# ============================ 三、重排可复现 ============================

def 打分(址, 问句, 段落):
    """返回与 段落 等长同序的分数表；失败返回 None。"""
    码, _, 体 = HTTP(址, "/rerank", 方法="POST",
                   体={"query": 问句, "documents": 段落}, 超时=120)
    if 码 != 200:
        return None
    表 = [None] * len(段落)
    for 项 in json.loads(体)["results"]:
        表[int(项["index"])] = float(项["relevance_score"])
    return 表 if all(x is not None for x in 表) else None


def 查重排可复现():
    """同一批 (问句, 段) 连打两次，逐位必须完全相同。

    Q8_0 走整数点积 → 对槽数免疫；若这条红了，先看是不是忘了 `--parallel 1`
    （FP16 多槽会漂移 2.5e-02 量级）。
    """
    print("\n【三】重排单槽可复现")
    段 = ["本条为可复现性自检的固定文本，长度贴近真实语料块以走相同计算路径。"
         "钝化体系由六价铬改为三价铬，涉及一号线前处理与镀层工序。" * 2
         + "（第 %d 段）" % i for i in range(8)]
    问句 = "电镀线的钝化体系在七月份发生了哪些变更"
    甲 = 打分(重排地址, 问句, 段)
    if 甲 is None:
        记("检索", "同一批连打两次逐位相同", 跳过, "重排服务不可达，无法检查")
        return
    乙 = 打分(重排地址, 问句, 段)
    差 = max(abs(a - b) for a, b in zip(甲, 乙))
    记("检索", "同一批连打两次逐位相同", 通过 if 差 == 0.0 else 失败,
       "最大分量差 %.3e（%d 段）" % (差, len(段)))


# ============================ 四、端到端验收（--全量）============================

def 查验收():
    print("\n【四】端到端验收（66 题，约 3 分钟）")
    产物 = os.path.join(输出目录, "知识库服务-验收.json")
    旧mtime = os.path.getmtime(产物) if os.path.exists(产物) else 0.0
    起 = time.time()

    码, 出 = 子进程([sys.executable, os.path.join(脚本目录, "验收-知识库服务.py")],
                   超时=1800)
    if 码 != 0:
        记("验收", "验收脚本", 失败, "退出码 %s %s" % (码, 出[-80:].replace("\n", " ")))
        return

    # 🔴 新鲜度护栏：不核对 mtime 就可能把**上次**的结论当**这次**的结果
    if not os.path.exists(产物) or os.path.getmtime(产物) <= max(旧mtime, 起 - 0.5):
        记("验收", "产物新鲜度", 失败, "json 未更新 —— 判据可能来自旧结果")
        return
    记("验收", "产物新鲜度", 通过, os.path.basename(产物))

    with open(产物, encoding="utf-8") as 文件:
        汇 = json.load(文件)["汇总"]
    库内数, 库外数 = 汇["库内题数"], 汇["库外题数"]

    for 名, 表, 分母 in (("库内误拒", 汇["库内拒答"], 库内数),
                        ("库外漏网", 汇["库外漏网"], 库外数),
                        ("引用问题题", 汇["引用问题题"], 库内数 + 库外数)):
        if 表:
            记("验收", 名, 失败, "%d/%d —— %s" % (len(表), 分母, "、".join(map(str, 表))))
        else:
            记("验收", 名, 通过, "0/%d" % 分母)

    记("验收", "编造数", 通过 if 汇["编造数"] == 0 else 失败,
       "%d/%d 个数值" % (汇["编造数"], 汇["数值总数"]))


# ============================ 五、APP 侧（--含APP）============================

def 查APP():
    print("\n【五】APP 侧静态检查")
    if not os.path.isdir(APP目录):
        记("APP", "工程目录", 跳过, "未找到 %s" % APP目录)
        return

    if not shutil.which("npx"):
        记("APP", "前端类型 vue-tsc --noEmit", 跳过, "npx 不在 PATH")
    else:
        码, 出 = 子进程("npx --no-install vue-tsc --noEmit", cwd=APP目录,
                     超时=300, shell=True)
        if 码 is None:
            记("APP", "前端类型 vue-tsc --noEmit", 失败, 出[:80])
        elif 码 == 0 and not 出.strip():
            记("APP", "前端类型 vue-tsc --noEmit", 通过, "零输出 = 零类型错误")
        else:
            记("APP", "前端类型 vue-tsc --noEmit", 失败,
               "退出码 %s ｜ %s" % (码, 出.strip().splitlines()[-1][:64] if 出.strip() else ""))

    if not shutil.which("cargo"):
        记("APP", "Rust 单测 cargo test --lib", 跳过, "cargo 不在 PATH")
    else:
        print("  ⏳ cargo test 要编译，可能几分钟；若 APP 正跑 tauri:dev，"
              "target 目录被占会锁冲突。", flush=True)
        码, 出 = 子进程(["cargo", "test", "--manifest-path",
                       os.path.join("src-tauri", "Cargo.toml"), "--lib"],
                      cwd=APP目录, 超时=1800)
        行 = [x for x in 出.splitlines() if "test result:" in x]
        if 码 == 0 and 行 and "0 failed" in 行[-1]:
            记("APP", "Rust 单测 cargo test --lib", 通过, 行[-1].strip())
        else:
            记("APP", "Rust 单测 cargo test --lib", 失败,
               (行[-1].strip() if 行 else "退出码 %s ｜ %s" % (码, 出[-80:].replace("\n", " "))))


# ============================ 汇总 + 人工步骤 ============================

def 打印汇总():
    print("\n" + "=" * 96)
    print(" 汇总")
    print("=" * 96)
    print("  " + 补宽("阶段", 10) + 补宽("检查项", 34) + 补宽("结果", 8) + "详情")
    print("  " + "-" * 92)
    for 阶段, 名, 状, 详 in 结果:
        print("  " + 补宽(阶段, 10) + 补宽(名, 34)
              + 补宽(图标[状] + 状, 8) + 详[:44])
    过 = sum(1 for x in 结果 if x[2] == 通过)
    失 = sum(1 for x in 结果 if x[2] == 失败)
    跳 = sum(1 for x in 结果 if x[2] == 跳过)
    print()
    print(" 通过 %d ｜ 失败 %d ｜ 跳过 %d（**未检查**，不计入通过）" % (过, 失, 跳))
    print(" 总判定：%s" % ("❌ 有失败项 —— 见上表" if 失 else
                        ("✅ 全部通过" if not 跳 else "✅ 已检查项全通过（另有 %d 项未检查）" % 跳)))
    print("=" * 96)


def 人工步骤(就绪):
    print("\n 需要人做的一步（脚本无法代劳）—— APP 真机确认")
    if 就绪:
        print("   服务端此刻已就绪，可以直接开始：")
    else:
        print("   ⚠ 知识库服务还没起来，先起服务再做：python scripts\\知识库服务.py")
    print("     1. 打开 Local-AI（tauri:dev 或安装包）")
    print("     2. 切到「知识库问答」模式")
    print("     3. 问：东莞精诚为什么 7 月起不良率突然升高")
    print("        → 期望：一段答案 + 3 张引用卡片（QR-QC-07-01 / JC-GY-2026-011 / QR-QC-07-01）")
    print("     4. 再问：电镀废液具体怎么处理")
    print("        → 期望：拒答「资料不足…QW-DD-05…不在本知识库中」")


def 主流程():
    global 结果
    sys.stdout.reconfigure(errors="replace")
    全量 = "--全量" in sys.argv
    含APP = "--含APP" in sys.argv

    print("=" * 96)
    print(" 全链路自检单    知识库服务 %s    %s"
          % (知识库地址, time.strftime("%Y-%m-%d %H:%M:%S")))
    print("=" * 96)
    起 = time.time()

    查依赖()
    就绪 = 查知识库()
    查重排可复现()
    if 全量:
        查验收()
    if 含APP:
        查APP()

    打印汇总()
    人工步骤(就绪)
    print("\n 用时 %.1f s" % (time.time() - 起))

    return 1 if any(x[2] == 失败 for x in 结果) else 0


if __name__ == "__main__":
    sys.exit(主流程())
