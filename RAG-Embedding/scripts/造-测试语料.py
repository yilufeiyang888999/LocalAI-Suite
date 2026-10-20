# -*- coding: utf-8 -*-
"""造两份 PDF 测试样本，验证「扩充语料 → 影子预检 → 入库」整条链路。

用法：python scripts\\造-测试语料.py [输出目录]      # 默认 corpus\\_待入库

| 样本 | 期望预检结果 |
| --- | --- |
| ① 19-润滑油管理补充规定-QW-SB-02.pdf | 六项**全绿**，可直接入库（体系文号 + 表格 + 完整正文） |
| ② 客户提供的设备润滑通则-2025版.pdf | 第 ③ 项 ⚠️（非体系文号，按参考资料处理）、**不阻断** |

🔴 两个实测踩过的坑（改这个脚本时别踩回去）：
  1. `insert_textbox` 在矩形高度不足时**静默不写** —— 表格单元格里就是这么丢的，
     画了 10 条线、文字一个都没有。单元格一律用 `insert_text`（单行、坐标精确）。
  2. 段落 `insert_textbox` 返回值 < 0 表示没放下，必须换页重排 + 断言，不能忽略。
"""
import os
import sys

import fitz  # pymupdf

出目录 = sys.argv[1] if len(sys.argv) > 1 else r"D:/workspace/7-AI-Code/RAG-Embedding/corpus/_待入库"
os.makedirs(出目录, exist_ok=True)

字体 = "china-s"          # pymupdf 内置 CJK 字体，实测 pdfplumber 能正确抽回
左 = 72.0
右 = 523.0
底 = 842.0 - 72.0
段前 = 7.0


class 稿:
    """按段落往下排，到页尾自动换页。"""

    def __init__(self):
        self.文档 = fitz.open()
        self.页 = self.文档.new_page(width=595, height=842)
        self.y = 80.0

    def 换页(self):
        self.页 = self.文档.new_page(width=595, height=842)
        self.y = 80.0

    def 文(self, 文本, 字号=10.5, 前=段前, 加粗=False):
        顶 = self.y + 前
        矩形 = fitz.Rect(左, 顶, 右, 底)
        余 = self.页.insert_textbox(矩形, 文本, fontsize=字号, fontname=字体,
                                  align=0, lineheight=1.28)
        if 余 < 0:                                   # 本页放不下 → 换页重排
            self.换页()
            矩形 = fitz.Rect(左, self.y, 右, 底)
            余 = self.页.insert_textbox(矩形, 文本, fontsize=字号, fontname=字体,
                                      align=0, lineheight=1.28)
            if 余 < 0:
                raise RuntimeError("段落单页放不下（会被静默丢掉）：%s" % 文本[:20])
        self.y = 矩形.y1 - max(余, 0.0)

    def 表(self, 数据, 列宽, 行高=21.0, 字号=10.0):
        总宽 = sum(列宽)
        需要 = 行高 * len(数据) + 8
        if self.y + 需要 > 底:
            self.换页()
        顶 = self.y + 6
        xs = [左]
        for 宽 in 列宽:
            xs.append(xs[-1] + 宽)
        for i in range(len(数据) + 1):               # 横线
            yy = 顶 + i * 行高
            self.页.draw_line(fitz.Point(左, yy), fitz.Point(左 + 总宽, yy),
                            color=(0.35, 0.35, 0.35), width=0.6)
        for xx in xs:                                # 竖线
            self.页.draw_line(fitz.Point(xx, 顶), fitz.Point(xx, 顶 + len(数据) * 行高),
                            color=(0.35, 0.35, 0.35), width=0.6)
        for r, 行 in enumerate(数据):
            cx = 左
            for c, 单元 in enumerate(行):
                # 🔴 这里必须用 insert_text（单行、坐标精确）：
                #    insert_textbox 在单元格高度不足时会**静默不写**（实测踩过）。
                基线 = 顶 + r * 行高 + 行高 * 0.68
                self.页.insert_text(fitz.Point(cx + 5, 基线), 单元,
                                  fontsize=字号, fontname=字体)
                cx += 列宽[c]
        self.y = 顶 + len(数据) * 行高

    def 存(self, 路径):
        self.文档.save(路径, deflate=True)
        self.文档.close()
        return 路径


# ============ ① 体系文号样本 ============
a = 稿()
a.文("润滑油管理补充规定", 字号=16)
a.文("文件编号：QW-SB-02", 字号=10)
a.文("1. 目的")
a.文("为规范车间设备润滑油的选型、领用与更换记录，减少因润滑不良造成的设备故障和非计划停机，"
    "结合现行设备管理制度执行中的实际情况，特制定本补充规定。")
a.文("2. 适用范围")
a.文("本规定适用于一号车间、二号车间全部转动设备的日常润滑管理。本规定未涉及的内容，"
    "仍按设备维护保养程序的现行规定执行。")
a.文("3. 加油周期与油品要求")
a.文("关键设备的加油周期与推荐油品见表 1。表中「运行小时」指设备实际通电运行累计时长，"
    "非自然日累计。")
a.表([["设备类型", "加油周期", "推荐油品牌号"],
      ["CNC 加工中心", "500 运行小时", "L-HM46"],
      ["数控车床", "800 运行小时", "L-HM32"],
      ["液压站", "2000 运行小时", "L-HM68"],
      ["空压机", "4000 运行小时", "L-DAB100"]],
     [150.0, 150.0, 151.0])
a.文("3.1 油品领用")
a.文("润滑油领用须填写领用单，由设备管理员签字后到油料库领取。单次领用量不得超过该设备"
    "三个月的正常消耗量；确有超领需要的，须经设备科科长批准。")
a.文("4. 更换与记录")
a.文("4.1 油品更换须记录更换日期、油品牌号、更换数量和执行人，记录保存期限不少于三年，"
    "并按月汇总报设备科备案。")
a.文("4.2 发现油品乳化、变黑或含有明显金属屑时，须立即停机检查，并在两小时内上报设备科；"
    "未经确认不得擅自开机。")
a.文("5. 考核")
a.文("未按规定周期加油的，每发现一次扣责任班组当月绩效 2 分；由此造成设备损坏的，"
    "按设备事故管理的相关规定处理。")
a.文("本规定自发布之日起执行，由设备科负责解释。")
路1 = a.存(os.path.join(出目录, "19-润滑油管理补充规定-QW-SB-02.pdf"))

# ============ ② 参考资料样本（无体系文号）============
b = 稿()
b.文("客户提供的设备润滑通则（2025 版）", 字号=16)
b.文("本文件为客户随订单提供的技术要求附件，供本单位设备维护部门参照执行。"
    "文件本身不带本单位体系文号。")
b.文("1. 通则要求")
b.文("所有转动设备的轴承部位应采用锂基润滑脂，工作温度长期高于 120 摄氏度的部位"
    "应改用合成高温脂。")
b.文("2. 与本单位规定的差异")
b.文("客户通则要求的加油周期比本单位现行规定缩短约三分之一。若两者冲突，"
    "以客户技术协议约定为准，并在设备履历中注明依据。")
路2 = b.存(os.path.join(出目录, "客户提供的设备润滑通则-2025版.pdf"))

for p in (路1, 路2):
    with fitz.open(p) as d:
        print("%-52s %d 页 %d 字节" % (os.path.basename(p), d.page_count, os.path.getsize(p)))
