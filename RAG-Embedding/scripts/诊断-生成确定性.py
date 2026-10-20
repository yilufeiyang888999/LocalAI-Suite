# -*- coding: utf-8 -*-
"""诊断：生成端是不是「确定的函数」——同一份提示词反复送，输出会不会变。

为什么要单独立一个脚本：
  判断「答案波动」时，第一反应总是「温度太高，降到 0 就好了」。**实测不是**：
  本机 Spark-X2.5-4B（CUDA）在 temperature=0.0 下，同一份提示词 5 次里出现 **2 种输出**
  （4:1 分裂）—— 说明非确定性来自引擎层（kernel / KV 复用），不是采样参数能关掉的。
  推论（很重要）：**拿 5 次采样去比两个提示词的好坏是没有判据的**，
  要用 `实验-提示词对照.py` 加大样本 + 多事实计数。

做法：**检索只做一次**，把同一段资料拼成的同一份提示词反复送进生成端，只换 temperature。

用法：
  python scripts\\诊断-生成确定性.py --问 "宁波恒力不良率"
  python scripts\\诊断-生成确定性.py --问 "宁波恒力不良率" --温度 0,0.1,0.7 --次 5
"""

import os
import sys
import json
import argparse
import importlib.util
import urllib.request

脚本目录 = os.path.dirname(os.path.abspath(__file__))
生成服务 = "http://127.0.0.1:8000"


def 载入服务模块():
    规范 = importlib.util.spec_from_file_location("知识库服务_确定性", os.path.join(脚本目录, "知识库服务.py"))
    模块 = importlib.util.module_from_spec(规范)
    规范.loader.exec_module(模块)
    return 模块


def 生成(提示, 温度):
    载荷 = json.dumps({
        "messages": [{"role": "user", "content": 提示}],
        "temperature": 温度, "max_tokens": 400,
        "chat_template_kwargs": {"enable_thinking": False},
    }, ensure_ascii=False).encode("utf-8")
    请求 = urllib.request.Request(生成服务 + "/v1/chat/completions", data=载荷,
                                  headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(请求, timeout=300) as 响应:
        return (json.loads(响应.read().decode("utf-8"))
                ["choices"][0]["message"].get("content") or "").strip()


def 主():
    解析 = argparse.ArgumentParser()
    解析.add_argument("--问", default="宁波恒力不良率")
    解析.add_argument("--温度", default="0,0.1", help="逗号分隔")
    解析.add_argument("--次", type=int, default=5)
    args = 解析.parse_args()

    服 = 载入服务模块()
    print("建基座 …", flush=True)
    服.建基座()

    结果 = 服.检索(args.问, 3)                      # 检索只做一次 → 排掉检索变量
    资料 = 服.拼资料(结果["取"])
    提示 = 服.提示模板.format(警示="", 资料=资料, 问句=args.问)

    print("=" * 92)
    print("同一份提示词（检索只做一次）→ 只换 temperature")
    print("问句：%s ｜ 资料 %d 字 ｜ Top-3：%s"
          % (args.问, len(资料), [服.基座["块列表"][i][0] for i in 结果["取"]]))
    print("=" * 92)

    for 温度 in [float(x) for x in args.温度.split(",") if x]:
        输出集 = []
        print()
        print("── temperature = %s ──" % 温度)
        for i in range(1, args.次 + 1):
            答案 = 生成(提示, 温度)
            输出集.append(答案)
            print("  第 %d 次 | %s" % (i, 答案.replace("\n", " ")))
        print("  → %d 次的去重种数：%d %s"
              % (args.次, len(set(输出集)),
                 "（1 = 确定）" if len(set(输出集)) == 1 else "（**不是确定函数**）"))
    return 0


if __name__ == "__main__":
    sys.exit(主())
