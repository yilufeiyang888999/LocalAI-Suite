# -*- coding: utf-8 -*-
"""ChatBI · 编排服务（包 5）· 端口 8096

**一个入口，三条链路。** 这是「制造业经营分析 Agent」对外的唯一门面 ——
Local-AI APP 只认这一个端口，背后走 RAG（8092）还是问数（8095）由编排层决定。

路由：
  GET  /            统一助手页（路由依据 / 轨迹 / 产物都可视化）
  GET  /health      状态（含三条链路各自的可达性 —— 缺这条就分不清「链路挂了」
                    和「路由没走到」）
  GET  /catalog     三条链路 + 路由词表 + 可执行动作（页面用）
  POST /ask         {"问题": "...", "角色": "操作员|主管|管理员",
                     "强制链路": "..."（可选，调试）}
                    角色缺省 = **操作员**（最少权限）；越权返回 **HTTP 403**
  POST /route       {"问题": "..."}   —— 只返回路由判定，不调链路（省钱、可批量测）

🔴 沿用作品一 / 问数服务的三条纪律（不要重新发明）：
  1. **关键路径禁写控制台**：落盘在调用线程、上屏交后台线程。
  2. **HTTP 路由一律 ASCII**：中文路径会被百分号编码，比字面量永远不等。
  3. `/health` 必须带 `log_file`。

🔴 包 5 新增的一条：
  4. **编排层自己不做任何业务判断** —— 它只做「选链路 + 串步骤」。
     口径、拒答、护栏全在各自的链路里（多一层判断就多一处口径分叉）。

🔴 包 6 新增的一条：
  5. **权限的对外表现是 HTTP 403，不是 body 里的一个字段。**
     状态码是「不用读 body 就知道被拒了」的判据；
     「读 body 看有没有被拒」这种判据，漏写一处就静默当成功。
"""

import argparse
import json
import os
import queue
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

本目录 = os.path.dirname(os.path.abspath(__file__))
项目目录 = os.path.dirname(本目录)
if 本目录 not in sys.path:
    sys.path.insert(0, 本目录)

import 动作                                              # noqa: E402
import 审计                                              # noqa: E402
import 权限                                              # noqa: E402
import 编排                                              # noqa: E402
import 拒答                                              # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:                                    # noqa: BLE001
        pass

服务版本 = "1.1"                    # 1.1 = 包 6（权限白名单 + 审计日志）
默认端口 = 8096


# ---------------------------------------------------------------- 日志

日志文件 = None
日志队列 = queue.Queue(maxsize=2000)
控制台丢弃数 = 0
日志写锁 = threading.Lock()


def 开日志文件():
    global 日志文件
    try:
        目录 = os.path.join(项目目录, "logs")
        os.makedirs(目录, exist_ok=True)
        日志文件 = os.path.join(目录, "编排-%s.log" % time.strftime("%Y%m%d"))
        return 日志文件
    except OSError:
        日志文件 = None
        return None


def _落盘(行0):
    if 日志文件 is None:
        return
    try:
        with 日志写锁:
            with open(日志文件, "a", encoding="utf-8") as 文:
                文.write(行0)
    except OSError:
        pass


def 上屏线程():
    while True:
        行0 = 日志队列.get()
        if 行0 is None:
            return
        try:
            sys.stdout.write(行0)
            sys.stdout.flush()
        except Exception:                                # noqa: BLE001
            pass


def 丢弃线程():
    while True:
        if 日志队列.get() is None:
            return


def 日志(内容, 加时间=True):
    global 控制台丢弃数
    行0 = ("  %s  %s\n" % (time.strftime("%H:%M:%S"), 内容)) if 加时间 else (内容 + "\n")
    _落盘(行0)
    try:
        日志队列.put_nowait(行0)
    except queue.Full:
        控制台丢弃数 += 1


# ---------------------------------------------------------------- 探活

def 链路状态():
    """三条链路各探一次。**每条单独报** —— 合成一个「服务正常」就没有判据了。"""
    出 = {}
    for 名, 地址 in (("知识问答", 编排.RAG地址), ("数据问数", 编排.问数地址)):
        基 = 地址.rsplit("/", 1)[0]
        try:
            应 = 编排.代理绕过.open(基 + "/health", timeout=3).read()
            出[名] = {"可达": True, "地址": 地址, "bytes": len(应)}
        except Exception as 错:                            # noqa: BLE001
            出[名] = {"可达": False, "地址": 地址,
                      "错误": "%s: %s" % (type(错).__name__, str(错)[:80])}
    try:
        os.makedirs(动作.产物目录, exist_ok=True)
        可写 = os.access(动作.产物目录, os.W_OK)
        出["动作执行"] = {"可达": True, "产物目录": 动作.产物目录, "可写": 可写}
    except OSError as 错:
        出["动作执行"] = {"可达": False, "错误": str(错)[:80]}
    return 出


# ---------------------------------------------------------------- 页面

页面文件 = os.path.join(项目目录, "web", "助手.html")


def 载入页面():
    try:
        with open(页面文件, encoding="utf-8") as 文:
            return 文.read()
    except OSError:
        return None


class 处理器(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    页面内容 = None

    def log_message(self, 格式, *参数):                    # noqa: A003
        日志("· %s %s" % (self.command, urllib.parse.unquote(self.path)))

    def _发跨域头(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _发(self, 码, 对象):
        体 = json.dumps(对象, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(码)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(体)))
        self._发跨域头()
        self.end_headers()
        self.wfile.write(体)

    def _发页面(self):
        if 处理器.页面内容 is None:
            self._发(503, {"error": "助手页未就绪", "hint": "web/助手.html 缺失或读取失败"})
            return
        体 = 处理器.页面内容.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(体)))
        self.send_header("Cache-Control", "no-store")
        self._发跨域头()
        self.end_headers()
        self.wfile.write(体)

    def do_OPTIONS(self):                                # noqa: N802
        self.send_response(204)
        self._发跨域头()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):                                    # noqa: N802
        路径 = urllib.parse.unquote(urllib.parse.urlparse(self.path).path)
        if 路径 in ("/", "/index.html", "/助手页"):
            self._发页面()
            return
        if 路径 == "/favicon.ico":
            self.send_response(204)
            self._发跨域头()
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if 路径 in ("/health", "/状态"):
            try:
                出 = {"status": "ok", "版本": 服务版本, "端口": self.server.server_address[1],
                      "链路": 链路状态(), "log_file": 日志文件,
                      "console_dropped": 控制台丢弃数,
                      # 包 6：审计文件路径要能从这里拿到 —— 否则「服务有审计」这件事
                      # 只能靠读代码相信（§9.5：拿「不报错」当「已检查过」的证据）。
                      "审计": 审计.统计(),
                      "权限": {"默认角色": 权限.默认角色,
                               "角色": [x["角色"] for x in 权限.角色清单()],
                               "白名单问题": 权限.校验配置()},
                      "路由确定性": {"动作动词": len(编排.动作动词),
                                     "单据名": len(编排.动作名词),
                                     "制度词": len(编排.制度词),
                                     "时间词": len(编排.时间词),
                                     # 动作嫌疑的**两侧**都要能回读：
                                     # 只报「嫌疑词表」而不报「否掉它的闸」，
                                     # 「为什么这句没被改判」就只能读代码相信。
                                     "处置动词": len(编排.处置动词),
                                     "问数信号": len(编排.问数信号)},
                      "动作": [x["动作"] for x in 动作.动作清单()]}
                self._发(200, 出)
            except Exception as 错:                        # noqa: BLE001
                self._发(500, {"error": str(错)})
            return
        if 路径 in ("/catalog", "/目录"):
            try:
                self._发(200, {
                    "服务版本": 服务版本,
                    "链路": [
                        {"名称": "知识问答", "依据": "制度文档（RAG）",
                         "示例": "采购流程怎么规定的？"},
                        {"名称": "数据问数", "依据": "供应链与生产数据库（语义层）",
                         "示例": "上月哪家供应商不良率最高？"},
                        {"名称": "动作执行", "依据": "查询结果 → 业务单据",
                         "示例": "给东莞精诚出一份质量整改通知"},
                    ],
                    "动作": 动作.动作清单(),
                    "角色": 权限.角色清单(),
                    "默认角色": 权限.默认角色,
                    "路由词表": {
                        "动作动词": list(编排.动作动词),
                        "单据名": list(编排.动作名词),
                        "制度词": list(编排.制度词),
                        "时间词": list(编排.时间词),
                        "处置动词": list(编排.处置动词),
                        "问数信号": list(编排.问数信号),
                    },
                })
            except Exception as 错:                        # noqa: BLE001
                self._发(500, {"error": str(错)})
            return
        self._发(404, {"error": "没有这个路径",
                       "paths": ["/", "/health", "/catalog", "POST /ask", "POST /route"]})

    def do_POST(self):                                   # noqa: N802
        路径 = urllib.parse.unquote(urllib.parse.urlparse(self.path).path)
        长 = int(self.headers.get("Content-Length") or 0)
        try:
            体 = json.loads(self.rfile.read(长).decode("utf-8")) if 长 else {}
        except (ValueError, UnicodeDecodeError) as 错:
            self._发(400, {"error": "请求体不是合法 JSON：%s" % 错})
            return
        问题 = (体.get("问题") or 体.get("question") or "").strip()
        try:
            if 路径 in ("/route", "/路由"):
                if not 问题:
                    self._发(400, {"error": "缺少「问题」"})
                    return
                self._发(200, 编排.路由(问题))
                return
            if 路径 in ("/ask", "/问"):
                if not 问题:
                    self._发(400, {"error": "缺少「问题」"})
                    return
                强制 = 体.get("强制链路") or None
                if 强制 and 强制 not in 编排.链路:
                    self._发(400, {"error": "强制链路只能是：%s" % "、".join(编排.链路)})
                    return
                # 🔴 角色缺省 = **最少权限**（权限.默认角色），不是「不检查」。
                #    入口默认放行是最常见的权限事故形态：测试期没人传角色，
                #    看着一切正常，上线才发现白名单从来没被触发过。
                角色 = 体.get("角色") or 权限.默认角色
                if 角色 not in 权限.角色表:
                    self._发(400, {"error": "未知角色「%s」（可用：%s）"
                                            % (角色, "、".join(权限.角色表))})
                    return
                起 = time.time()
                应 = 编排.执行(问题, 强制链路=强制, 角色=角色)
                应["总耗时秒"] = round(time.time() - 起, 2)
                日志("问：%s  →  %s（%s，%s s）角色=%s%s"
                     % (问题[:36], 应.get("类型"), 应.get("路由", {}).get("依据"),
                        应["总耗时秒"], 角色, "  🚫被拒" if 应.get("被拒") else ""))
                # 🔴 越权回 **403**：状态码本身就得说明「你被拒了」。
                #    只把被拒写在 body 里，前端要「记得去查 body 里有没有被拒」——
                #    那是「漏写就静默当成功」的判据，不能作为权限的对外表现。
                self._发(403 if 应.get("被拒") else 200, 应)
                return
            self._发(404, {"error": "没有这个路径"})
        except Exception as 错:                            # noqa: BLE001
            日志("🔴 请求异常：%s: %s" % (type(错).__name__, 错))
            self._发(500, {"error": "%s: %s" % (type(错).__name__, 错)})


# ---------------------------------------------------------------- 启动

def 启动自检():
    坏 = 0
    日志("=" * 74, 加时间=False)
    日志("ChatBI 编排服务 v%s —— 一个入口，三条链路" % 服务版本, 加时间=False)
    日志("=" * 74, 加时间=False)
    日志("路由：POST /ask（走编排）｜ POST /route（只判路由）｜ GET /health /catalog /")
    日志("动作产物目录：%s" % 动作.产物目录)
    日志("审计日志：%s" % 审计.文件())
    白名单问题 = 权限.校验配置()
    坏 += 1 if 白名单问题 else 0
    日志("权限：%s ｜ 默认角色 %s ｜ 白名单 %s"
         % ("、".join(权限.角色表), 权限.默认角色,
            "✅ 与真实动作对齐" if not 白名单问题 else "🔴 " + "；".join(白名单问题)))
    for 名, 状 in 链路状态().items():
        好 = 状.get("可达")
        坏 += 0 if 好 else 1
        日志("%s 链路 %s：%s" % ("✅" if 好 else "🔴", 名,
                              "可达" if 好 else 状.get("错误", "")))
    return 坏


def 主():
    global 默认端口
    sys.stdout.reconfigure(errors="replace")
    参 = argparse.ArgumentParser(description="ChatBI 编排服务（包 5）")
    参.add_argument("--端口", type=int, default=默认端口)
    参.add_argument("--无上屏", action="store_true", help="不往控制台写（自检用）")
    参 = 参.parse_args()
    默认端口 = 参.端口

    开日志文件()
    threading.Thread(target=丢弃线程 if 参.无上屏 else 上屏线程,
                     name="日志-上屏", daemon=True).start()
    处理器.页面内容 = 载入页面()
    if 处理器.页面内容 is None:
        日志("⚠️ 助手页未就绪（%s 缺失）—— HTTP 接口不受影响" % 页面文件)

    坏 = 启动自检()
    服 = ThreadingHTTPServer(("127.0.0.1", 默认端口), 处理器)
    服.daemon_threads = True
    日志("已监听 http://127.0.0.1:%d/  （链路探活失败 %d 项；链路失败不影响服务启动，"
         "失败的请求会在 /ask 里明确报错）" % (默认端口, 坏))
    try:
        服.serve_forever()
    except KeyboardInterrupt:
        日志("收到 Ctrl+C，退出")
    finally:
        服.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(主())
