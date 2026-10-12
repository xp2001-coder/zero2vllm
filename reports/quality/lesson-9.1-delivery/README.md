# 9.1 课程交付与插图修订检查

[返回质量导航](../README.md) · [9.1 正文](../../../docs/chapters/part-09-quantization/01-quantization-system-overview/README.md)

本轮完成量化系统全景，并将 zero-point、反量化与重量化明确纳入图、公式、代码和练习。实际检查时间（UTC）：`2026-09-20T09:09:56.649812+00:00`；Windows / Python 3.11.9。

| 检查 | 实际结果 | 记录 |
| --- | --- | --- |
| 全仓库回归 | 413 项，退出 0，跳过 0 项 | [regression.json](regression.json)、[完整日志](tests.txt) |
| 新增针对性检查 | 33 项：Passport 12 项、数值与入口 21 项；纳入上述回归 | [测试代码](../../../tests/reference/test_quantization_overview.py) |
| 课程数值案例 | 17 项通过，含非零 z、RQ 饱和、双重舍入与错误解码反例 | [summary.json](../../../docs/chapters/part-09-quantization/01-quantization-system-overview/results/summary.json) |
| 文档命令 | 9 条匹配预期；含一条预期退出 2 | [commands.json](../../../docs/chapters/part-09-quantization/01-quantization-system-overview/results/commands.json) |
| 固定源码 | vLLM v0.8.5：4 文件、9 函数，静态核验 | [源码定位](../../../docs/chapters/part-09-quantization/01-quantization-system-overview/source-map.md) |
| 文档与分级目录 | 由最终内容运行检查 | [course.json](course.json)、[layout.json](layout.json) |

## 图文交付

六张中文原创图保留可编辑 SVG 和 PNG，均逐张查看过标签、公式、箭头和排版。非零 zero-point 与 RQ 图中的数值已通过固定输入和 CPU 结果核对；对称 Linear 明确是 z=0 特例，repack 与重量化分别描述。

正文采用 15 项教学结构；另有实验手册、14 组含答案练习、20 个术语条目和两份模型 Passport。中英文大纲、篇级目录和学习入口已同步。

## 验证范围

源码摘要：`eeebc8752473d290b198c04f4ea9ec6852710bd59835f80865baa31ac2b50d6c`。测试期间代码/数据摘要未改变，实验与回归使用同一快照。历史章节运行记录保持原样。

本次只运行课程 CPU 程序；vLLM 是静态源码阅读，没有运行 GPU 或真实 Qwen。CI 已增加离线实验和示例入口，但本次没有提交、推送或执行 GitHub 远程 CI。
