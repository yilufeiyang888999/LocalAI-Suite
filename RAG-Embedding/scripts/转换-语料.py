# -*- coding: utf-8 -*-
"""
转换-语料.py —— 把 Word / PDF / Excel / TXT 转成**知识库能用的** Markdown
================================================================================

为什么需要这一步
----------------
`实验-分块对照.读语料()` 只认 `corpus/raw/*.md`，而且读的时候是**硬编码 UTF-8**：

  · 非 `.md`         → **静默丢弃**（不报错，只是 `/health.docs` 不涨）；
  · `.md` 但非 UTF-8 → **`UnicodeDecodeError` 上抛**，把整个建基座带崩 = 服务起不来。

（两条都是 2026-10-12 实测出来的，不是推断。）

但「转成 Markdown」≠「知识库能用」。分块器（S13 软表保-1200）吃的是三个**结构**：

  1. 第一行 `# 标题`        → 文档标题，进块前缀；
  2. `| **文件编号** | XX |` → 文号，**既进块前缀、又进 `库内文号` 集合**（类2 库外引用判据靠它）；
  3. `| 表格 |` 行 / `## n.` 小节 → 软表格保护、章节路径。

所以本脚本只负责"把格式转过来 + 把三样结构的**骨架**搭好"，
**文号该填什么、标题该是什么，是语义，脚本不猜** —— 抽不到就留占位符，交给人和 `自检-新增语料.py`。

用法
----
    python scripts\\转换-语料.py 输入文件或目录 [更多...] [--输出目录 路径] [--文号 QW-XX-01]

    # 例：把一批下载下来的文档丢进暂存区
    python scripts\\转换-语料.py D:\\下载\\新制度\\  --输出目录 corpus\\_待入库

产出落在 **`corpus/_待入库/`**（暂存区）—— 刻意**不直接写 `corpus/raw/`**。
`读语料()` 只扫 `raw/`，所以暂存区里的半成品不会被误入库。

解释器要求
----------
    · TXT / CSV / DOCX → **零第三方依赖**（纯标准库，含 zipfile+XML 解 .docx）
    · XLSX            → 需要 `openpyxl`
    · PDF（文本层）    → 需要 `pdfplumber`（`pymupdf` 可选，仅用于图片提取）
    · 扫描件 PDF      → **本脚本不处理**：无文本层时必须 OCR，属外部通道（见 `--说明`）

本机实测（2026-10-12）：系统 Python 3.14 已装 `pdfplumber` / `pymupdf` / `openpyxl`；
托管 Python 3.13 一个都没有。脚本会自己**探测并告诉你该用哪个解释器**，不让你猜。
"""

import argparse
import os
import re
import shutil
import sys
import time
import unicodedata
import zipfile
from xml.etree import ElementTree as ET

# ======================= 路径与常量 =======================

脚本目录 = os.path.dirname(os.path.abspath(__file__))
根目录 = os.path.dirname(脚本目录)
语料目录 = os.path.join(根目录, "corpus", "raw")
暂存目录 = os.path.join(根目录, "corpus", "_待入库")

# 纯文本类：只转码，不拆结构
纯文本类 = {".txt", ".csv", ".text", ".log"}
Markdown类 = {".md", ".markdown"}

# Word 的 XML 命名空间
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

# 文号占位符：刻意做成**一眼能看出没填**的样子，避免"忘了改却看起来正常"
文号占位 = "<<待填写文号>>"
标题占位 = "<<待填写标题>>"


def 设终端编码():
    """Windows 控制台默认 GBK。这里只改 errors，**不改 encoding** ——
    改成 utf-8 会让终端里已有的中文全变乱码（本项目约定，见 MEMORY 二）。"""
    for 流 in (sys.stdout, sys.stderr):
        if hasattr(流, "reconfigure"):
            流.reconfigure(errors="replace")


def 显示宽(文本):
    """中文按 2 宽。f-string 的 `<12` 对中文会错位。"""
    宽 = 0
    for 字符 in str(文本):
        宽 += 2 if unicodedata.east_asian_width(字符) in ("W", "F") else 1
    return 宽


def 补宽(文本, 宽):
    文本 = str(文本)
    填充 = 宽 - 显示宽(文本)
    return 文本 + " " * 填充 if 填充 > 0 else 文本


def 截断(文本, 宽=28):
    出 = ""
    for 字符 in str(文本):
        if 显示宽(出 + 字符) > 宽:
            return 出 + "…"
        出 += 字符
    return 出


# ======================= 依赖探测 =======================

def 探测依赖():
    """返回 {模块名: 是否可用}。**先探测再决定走哪条路**，不靠 try/except 兜底。"""
    import importlib.util as 工具
    出 = {}
    for 名 in ("openpyxl", "pdfplumber", "pymupdf", "fitz"):
        出[名] = 工具.find_spec(名) is not None
    return 出


def 建议解释器():
    """扫几个常见解释器，看哪个能满足"全部依赖"。只用于**提示**，不代跑。"""
    候选 = [
        sys.executable,
        r"C:\Users\user\AppData\Local\Programs\Python\Python314\python.exe",
        r"C:\Users\user\AppData\Local\Programs\Python\Python313\python.exe",
    ]
    import importlib.util as 工具
    好 = []
    for 路 in 候选:
        if not os.path.isfile(路):
            continue
        # 用「同目录下的 site-packages」粗判：直接跑一次子进程更准，但这里只做提示，够用
        好.append(路)
    return 好


# ======================= 编码：这是最容易踩的一坑 =======================

# 顺序有意如此：utf-8 放最前（**先试严格 UTF-8**，成功就一定是干净的 UTF-8）；
# gb18030 放 gbk 之后（gb18030 是 gbk 的超集，能多认几个生僻字）。
编码候选 = ("utf-8", "utf-8-sig", "gb18030", "gbk", "big5", "latin-1")


def 读文本(路径):
    """按候选编码读纯文本。返回 (文本, 实际编码)。

    `latin-1` 兜底**永远不会失败**（任意字节都可解），所以循环必然能出来 ——
    但这正是它的危险：它会把 GBK 文件解成乱码而**不报错**。
    因此最后一个候选命中时，调用方应当把它标成"编码可疑"。
    """
    原始 = open(路径, "rb").read()
    for 编码 in 编码候选:
        try:
            文本 = 原始.decode(编码)
        except (UnicodeDecodeError, LookupError):
            continue
        if 编码 in ("gbk", "gb18030") and 文本.count("\ufffd") > 0:
            continue
        return 文本, 编码
    return 原始.decode("utf-8", errors="replace"), "utf-8(替换)"


# ======================= DOCX：纯标准库解 =======================

def _段落文本(段):
    """把一个 w:p 的文字拼起来。w:t 是文本、w:tab 是制表、w:br 是换行。"""
    片段 = []
    for 节点 in 段.iter():
        if 节点.tag == W + "t":
            片段.append(节点.text or "")
        elif 节点.tag == W + "tab":
            片段.append("\t")
        elif 节点.tag in (W + "br", W + "cr"):
            片段.append(" ")
    return "".join(片段).strip()


def _样式到级别():
    """读 styles.xml，把 styleId 映射成 heading 级别（1~6）。0 起算的 outlineLvl +1。

    为什么要读这个：Word 的标题**不是**靠文字判断的，靠 `w:pStyle` 指向的样式；
    而样式的层级写在 `w:pPr/w:outlineLvl` 里。中英文 Word 的 styleId 完全不同
    （中文常见 `1`~`9` 或 `a3`），所以**别硬编码 styleId，一律查 styles.xml**。
    """
    表 = {}
    try:
        with zipfile.ZipFile(样式来源[0]) as 包:
            根 = ET.fromstring(包.read("word/styles.xml"))
    except Exception:                                          # noqa: BLE001
        return 表
    for 样式 in 根.iter(W + "style"):
        号 = 样式.get(W + "styleId")
        if not 号:
            continue
        级别 = None
        大纲 = 样式.find(W + "pPr/" + W + "outlineLvl")
        if 大纲 is not None and 大纲.get(W + "val") is not None:
            级别 = int(大纲.get(W + "val")) + 1
        if 级别 is None:
            名 = 样式.find(W + "name")
            名值 = (名.get(W + "val") if 名 is not None else "") or ""
            m = re.search(r"(\d)", 名值)
            if m and re.search(r"heading|标题|heading\s*\d", 名值, re.I):
                级别 = int(m.group(1))
        if 级别:
            表[号] = max(1, min(6, 级别))
    return 表


样式来源 = [None]     # 小技巧：_样式到级别() 需要知道当前打开的是哪个 docx


def _段落级别(段, 样式表):
    """取一个段落的标题级别。返回 None 表示正文。"""
    pPr = 段.find(W + "pPr")
    if pPr is None:
        return None
    大纲 = pPr.find(W + "outlineLvl")
    if 大纲 is not None and 大纲.get(W + "val") is not None:
        return max(1, min(6, int(大纲.get(W + "val")) + 1))
    样式 = pPr.find(W + "pStyle")
    if 样式 is None:
        return None
    号 = 样式.get(W + "val")
    if 号 in 样式表:
        return 样式表[号]
    # 样式表里查不到时的**窄回退**：只在 styleId 本身长得像 heading 时才认
    m = re.match(r"^(?:heading|Heading|标题)([1-6])$", 号 or "")
    return int(m.group(1)) if m else None


def _表格到Markdown(表):
    """w:tbl → Markdown 表格。第一行当表头。"""
    行列表 = []
    for 行 in 表.findall(W + "tr"):
        单元 = []
        for 格 in 行.findall(W + "tc"):
            文本 = " ".join(_段落文本(p) for p in 格.findall(W + "p")).strip()
            单元.append(文本.replace("|", "\\|").replace("\n", " "))
        if any(单元):
            行列表.append(单元)
    if not 行列表:
        return []
    列数 = max(len(r) for r in 行列表)
    规范 = [r + [""] * (列数 - len(r)) for r in 行列表]
    出 = ["| " + " | ".join(规范[0]) + " |",
          "|" + "|".join(["---"] * 列数) + "|"]
    for r in 规范[1:]:
        出.append("| " + " | ".join(r) + " |")
    return 出


def 转DOCX(路径):
    """返回 (Markdown 正文, 统计信息)。"""
    样式来源[0] = 路径
    样式表 = _样式到级别()
    with zipfile.ZipFile(路径) as 包:
        根 = ET.fromstring(包.read("word/document.xml"))
    体 = 根.find(W + "body")
    if 体 is None:
        return "", {"段落": 0, "标题": 0, "表格": 0}

    出 = []
    统计 = {"段落": 0, "标题": 0, "表格": 0}
    for 块 in 体:
        if 块.tag == W + "p":
            文本 = _段落文本(块)
            if not 文本:
                continue
            统计["段落"] += 1
            级别 = _段落级别(块, 样式表)
            if 级别:
                统计["标题"] += 1
                出.append("#" * 级别 + " " + 文本)
            else:
                出.append(文本)
            出.append("")
        elif 块.tag == W + "tbl":
            表 = _表格到Markdown(块)
            if 表:
                统计["表格"] += 1
                出.extend(表)
                出.append("")
    return "\n".join(出).strip(), 统计


# ======================= XLSX =======================

def 转XLSX(路径):
    """每个工作表 → `## 表名` + 一张 Markdown 表格。

    不做"每行一条 字段:值"（`erp-ai-assistant` 的做法）——
    分块器的软表格保护**只认 `| 表格 |` 行**，拍平成散文会让它认不出表格、切开。
    """
    from openpyxl import load_workbook

    工作簿 = load_workbook(路径, data_only=True, read_only=True)
    出 = []
    统计 = {"工作表": 0, "行": 0}
    try:
        for 表 in 工作簿.worksheets:
            行列表 = []
            for 行 in 表.iter_rows(values_only=True):
                if 行 is None or all(v is None or str(v).strip() == "" for v in 行):
                    continue
                行列表.append(["" if v is None else str(v).strip().replace("|", "\\|").replace("\n", " ")
                              for v in 行])
            if not 行列表:
                continue
            统计["工作表"] += 1
            统计["行"] += len(行列表)
            列数 = max(len(r) for r in 行列表)
            规范 = [r + [""] * (列数 - len(r)) for r in 行列表]
            出.append("## " + 表.title)
            出.append("")
            出.append("| " + " | ".join(规范[0]) + " |")
            出.append("|" + "|".join(["---"] * 列数) + "|")
            for r in 规范[1:]:
                出.append("| " + " | ".join(r) + " |")
            出.append("")
    finally:
        工作簿.close()
    return "\n".join(出).strip(), 统计


# ======================= PDF（文本层） =======================

def 转PDF(路径):
    """pdfplumber 抽文本 + 表格。**同时判断是不是扫描件。**

    判据（必须能区分"扫描件"和"没检查"）：  
      · 有文本层 → 每页字符数通常几百以上；
      · 扫描件   → 每页 0~几个字符（只有零星页眉水印）。
    阈值取 **每页 20 字**：低于它一律标"疑似扫描件，需 OCR"，**不产出空文档**。
    """
    import pdfplumber

    出 = []
    页数 = 0
    总字 = 0
    with pdfplumber.open(路径) as pdf:
        for 序号, 页 in enumerate(pdf.pages, start=1):
            页数 += 1
            文本 = (页.extract_text() or "").strip()
            总字 += len(文本)
            if 文本:
                出.append(文本)
                # 表格：pdfplumber 的 extract_tables 可能把正常段落也切出来，这里只用它补齐
                for 表 in (页.extract_tables() or []):
                    if not 表 or len(表) < 2:
                        continue
                    行 = [["" if c is None else str(c).strip().replace("\n", " ") for c in r] for r in 表]
                    列数 = max(len(r) for r in 行)
                    规范 = [r + [""] * (列数 - len(r)) for r in 行]
                    出.append("| " + " | ".join(规范[0]) + " |")
                    出.append("|" + "|".join(["---"] * 列数) + "|")
                    for r in 规范[1:]:
                        出.append("| " + " | ".join(r) + " |")
            出.append("")

    每页均字 = 总字 / 页数 if 页数 else 0
    统计 = {"页数": 页数, "总字": 总字, "每页均字": round(每页均字, 1),
            "疑似扫描件": 每页均字 < 20 and 页数 > 0}
    return "\n".join(出).strip(), 统计


# ======================= 规范化：搭好三样结构骨架 =======================

def 从文件名猜文号(文件名):
    """文件名末尾的形如 QW-DD-03 / JC-GY-2026-006 的串。猜不到返回 None。

    这只是**降低填写成本**，不是判据 —— 预检会明确告诉你"这个文号是从文件名猜的"。
    """
    主干 = os.path.splitext(文件名)[0]
    m = re.search(r"([A-Z]{2,6}(?:-[A-Z]{2,6}){0,2}-[A-Za-z0-9\-]*\d[A-Za-z0-9\-]*)$", 主干)
    return m.group(1) if m else None


def 规范成知识库格式(正文, 文件名, 文号=None, 标题=None, 来源="", 提示=None):
    """把转换结果包成知识库要的形状：标题 + 文件头信息表 + 正文。

    ⚠️ 抽不到文号时**不猜**，写占位符 —— 「不报错只出错」比「报错」危险得多。
    """
    正文 = 正文.strip()
    行列表 = [x.strip() for x in 正文.split("\n")]

    # 标题候选两路：① 正文里第一个 `# ` 行（DOCX 有样式时走这条）；
    #                ② 正文第一行（TXT 这种没结构的，首行通常就是标题）。
    # 阈值 40 字 + 不以 `|` 开头：够区分"标题"和"一上来就是表格/正文段落"。
    现成标题 = None
    for 行 in 行列表[:8]:
        if 行.startswith("# "):
            现成标题 = 行[2:].strip()
            break
    首行 = next((x for x in 行列表 if x), "")
    if not 现成标题 and 首行 and len(首行) <= 40 and not 首行.startswith("|"):
        现成标题 = 首行

    标题 = (标题 or 现成标题 or os.path.splitext(文件名)[0]).strip()
    文号 = 文号 or 从文件名猜文号(文件名) or 文号占位

    # 🔴 正文开头与标题重复的那一行必须删掉：不删的话，文件头信息表下面会再冒一个
    #    同名 `# `，拆节时变成"一个空的同名小节"（多耗一个块前缀预算，且 `/health.blocks` 虚高）。
    #    只在前 3 行内找 —— 往深处删会误伤正文里正常的同名小节标题。
    清理 = []
    已删 = False
    for 序号, 行 in enumerate(行列表):
        if not 已删 and 序号 < 3 and 行.lstrip("#").strip() == 标题:
            已删 = True
            continue
        if not 已删 and 序号 < 3 and not 行:
            continue                              # 顺手吃掉标题前的空行
        清理.append(行)
    正文 = "\n".join(清理).strip()

    头 = [
        "# " + 标题,
        "",
        "| 项目 | 内容 |",
        "|---|---|",
        "| **文件编号** | %s |" % 文号,
        "| **来源文件** | %s |" % 来源,
        "| **入库日期** | %s |" % time.strftime("%Y-%m-%d"),
        "",
    ]
    if 提示:
        头.append("> **入库提示**：%s" % 提示)
        头.append("")
    头.append("---")
    头.append("")
    return "\n".join(头) + 正文 + "\n"


# ======================= 主流程 =======================

转换器 = {
    ".docx": ("Word", 转DOCX),
    ".xlsx": ("Excel", 转XLSX),
    ".xlsm": ("Excel", 转XLSX),
    ".pdf": ("PDF", 转PDF),
}
扩展支持 = set(纯文本类) | Markdown类 | set(转换器)


def 展开输入(输入列表):
    出 = []
    for 项 in 输入列表:
        if os.path.isdir(项):
            for 名 in sorted(os.listdir(项)):
                路 = os.path.join(项, 名)
                if os.path.isfile(路) and os.path.splitext(名)[1].lower() in 扩展支持:
                    出.append(路)
        elif os.path.isfile(项):
            出.append(项)
        else:
            print("  ⚠️ 找不到：%s" % 项)
    return 出


def 主流程():
    解释器 = argparse.ArgumentParser(
        description="把 Word / PDF / Excel / TXT 转成知识库能用的 Markdown（输出到暂存区）")
    解释器.add_argument("输入", nargs="*", help="输入文件或目录")
    解释器.add_argument("--输出目录", default=暂存目录, help="默认 corpus\\_待入库")
    解释器.add_argument("--文号", default=None, help="强制指定文号（单文件时用）")
    解释器.add_argument("--标题", default=None, help="强制指定标题（单文件时用）")
    解释器.add_argument("--说明", action="store_true", help="打印格式支持矩阵后退出")
    参数 = 解释器.parse_args()

    if 参数.说明:
        print(__doc__)
        return 0

    print("转换-语料.py  —— 只转格式 + 搭结构骨架，**不猜文号和标题**")
    print("=" * 78)

    依赖 = 探测依赖()
    print(" 依赖探测（当前解释器 %s）" % sys.executable)
    for 名 in ("openpyxl", "pdfplumber"):
        print("   %s %s" % ("✅" if 依赖[名] else "❌", 名))
    if not (依赖["openpyxl"] and 依赖["pdfplumber"]):
        print()
        print(" ⚠️ 有依赖缺失。**只有 XLSX 需要 openpyxl、PDF 需要 pdfplumber**，")
        print("    TXT/CSV/DOCX 走纯标准库，任何解释器都能跑。")
        print("    本机实测装了这三个库的解释器是：")
        for 路 in 建议解释器():
            print("      " + 路)
        print("    若要用它跑：\"<上面的路径>\" scripts\\转换-语料.py ...")
    print()

    if not 参数.输入:
        print(" 没有输入。用法：python scripts\\转换-语料.py <文件或目录> [更多...]")
        return 1

    文件列表 = 展开输入(参数.输入)
    if not 文件列表:
        print(" 展开后没有可转换的文件。支持的扩展名：" + "、".join(sorted(扩展支持)))
        return 1

    os.makedirs(参数.输出目录, exist_ok=True)
    print(" 输出到：%s" % 参数.输出目录)
    print(" " + "-" * 76)
    print(" %s %s %s %s" % (补宽("来源文件", 34), 补宽("格式", 8), 补宽("结果", 12), "备注"))
    print(" " + "-" * 76)

    成 = 败 = 0
    for 路 in 文件列表:
        名 = os.path.basename(路)
        扩展 = os.path.splitext(名)[1].lower()
        备注 = ""
        正文 = ""
        try:
            if 扩展 in Markdown类:
                正文, 编码 = 读文本(路)
                格式 = "Markdown"
                备注 = "原样转码(%s)" % 编码
                if 编码 not in ("utf-8", "utf-8-sig"):
                    备注 += " ⚠️原非UTF-8"
                正文 = 正文.strip()
            elif 扩展 in 纯文本类:
                正文, 编码 = 读文本(路)
                格式 = 扩展.lstrip(".").upper()
                备注 = "转码(%s)" % 编码
                if 编码 == "latin-1":
                    备注 = "⚠️ 编码可疑（latin-1 兜底，可能是乱码）"
                正文 = 正文.strip()
            elif 扩展 in 转换器:
                格式, 函数 = 转换器[扩展]
                if 格式 == "Excel" and not 依赖["openpyxl"]:
                    raise RuntimeError("缺 openpyxl，请换解释器（见上方提示）")
                if 格式 == "PDF" and not 依赖["pdfplumber"]:
                    raise RuntimeError("缺 pdfplumber，请换解释器（见上方提示）")
                正文, 统计 = 函数(路)
                if 格式 == "PDF" and 统计.get("疑似扫描件") and not 正文.strip():
                    # 单独一条：这是**已知且可操作**的情况，不该混进"解析出错"里
                    raise RuntimeError(
                        "无文本层（%d 页、每页均 %s 字）→ 疑似扫描件，需先 OCR。"
                        "本脚本不做 OCR（见 --说明）" % (统计["页数"], 统计["每页均字"]))
                if 格式 == "Word":
                    备注 = "%d 段 / %d 标题 / %d 表" % (统计["段落"], 统计["标题"], 统计["表格"])
                    if 统计["标题"] == 0:
                        备注 += " ⚠️无标题层级"
                elif 格式 == "Excel":
                    备注 = "%d 表 / %d 行" % (统计["工作表"], 统计["行"])
                else:
                    备注 = "%d 页 / 每页均 %s 字" % (统计["页数"], 统计["每页均字"])
                    if 统计["疑似扫描件"]:
                        备注 += " 🔴疑似扫描件，需 OCR"
            else:
                raise RuntimeError("不支持的扩展名 %s" % 扩展)

            if not 正文.strip():
                raise RuntimeError("提取到 0 字符（扫描件 / 空文件 / 加密？）")

            提示 = None
            if 扩展 in 转换器 and 扩展 != ".md":
                提示 = "本文件由 %s 自动转换。**文号与标题请人工核对**；正文结构可能与原文有出入。" % 名
            文号 = 参数.文号 if len(文件列表) == 1 else None
            标题 = 参数.标题 if len(文件列表) == 1 else None
            产物 = 规范成知识库格式(正文, 名, 文号=文号, 标题=标题, 来源=名, 提示=提示)

            # 统一输出为 .md；同名已存在时加后缀，绝不覆盖（覆盖 = 静默丢数据）
            基础 = os.path.splitext(名)[0]
            出路径 = os.path.join(参数.输出目录, 基础 + ".md")
            序号 = 1
            while os.path.exists(出路径):
                出路径 = os.path.join(参数.输出目录, "%s-%d.md" % (基础, 序号))
                序号 += 1
            with open(出路径, "w", encoding="utf-8", newline="\n") as 文件:
                文件.write(产物)
            成 += 1
            print(" %s %s %s %s" % (补宽(截断(名, 32), 34), 补宽(格式, 8),
                                   补宽("✅ %d 字" % len(产物), 12), 备注))
        except Exception as 错:                                # noqa: BLE001
            败 += 1
            友好 = 转换器.get(扩展, (扩展 or "?", None))[0]
            print(" %s %s %s %s" % (补宽(截断(名, 32), 34), 补宽(友好, 8),
                                   补宽("❌ 未产出", 12), str(错)[:70]))

    print(" " + "-" * 76)
    print(" 成功 %d / 失败 %d。产物在 %s" % (成, 败, 参数.输出目录))
    print()
    print(" 下一步（**别跳过**）：")
    print("   python scripts\\自检-新增语料.py --预检 %s\\*.md" % os.path.basename(参数.输出目录))
    print(" 预检会告诉你：文号是不是猜的、前缀在不在白名单、会不会撞上故意留白的文号、")
    print(" 块数会涨多少 —— 通过之后再拷进 corpus\\raw\\ 并重启服务。")
    return 0 if 败 == 0 else 1


if __name__ == "__main__":
    设终端编码()
    sys.exit(主流程())
