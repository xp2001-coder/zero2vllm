# 9.2 课程交付检查

[返回质量导航](../README.md) · [9.2 正文](../../../docs/chapters/part-09-quantization/02-numerical-representations-and-tensor-contracts/README.md)

本轮交付数值表示、分组 Q/DQ、张量轴/尾组/scale 精度、MX/NV 数值分解与 effective bpw。实际回归时间（UTC）：`2026-09-20T09:52:47.844792+00:00`，Windows / Python 3.11.9。

| 检查 | 实际结果 | 记录 |
| --- | --- | --- |
| 全仓库回归 | 452 项，退出 0，跳过 0 项 | [regression.json](regression.json)、[完整日志](tests.txt) |
| 新增针对性测试 | 39 项，已纳入回归；整数22、浮点14、课程入口/对照3 | [测试目录](../../../tests/quantization/) |
| 课程固定案例 | 20 项全部通过 | [summary.json](../../../docs/chapters/part-09-quantization/02-numerical-representations-and-tensor-contracts/results/summary.json) |
| 文档命令 | 12 条符合预期；含一条预期退出2 | [commands.json](../../../docs/chapters/part-09-quantization/02-numerical-representations-and-tensor-contracts/results/commands.json) |
| 独立 FP8 对照 | PyTorch 2.7.1+cpu CPU，共512位模式一致 | 同 summary 的 torch_oracle |
| 文档与目录 | 19篇129节，20节完成；检查实际链接和生成段落 | [course.json](course.json)、[layout.json](layout.json) |

## 教学与视觉检查

正文按15项结构展开，配实验手册、16组含答案练习、两份CPU Passport。7张中文原创插图保留SVG与PNG，已逐张查看标签、位域、分组边界、公式和数值；MX图额外注明E8M0的255为NaN，粒度图明确每个虚线框独立共享参数。

## 验证范围

源码摘要：`48be978fc5725f2d5f716f6d4da611d80634921c23218875a7b73ce887cad23d`；实验、命令记录与回归使用同一代码/数据快照，测试期间摘要未改变。旧章节与历史运行记录保持原样。

本轮只执行CPU数值程序及PyTorch编码解释，没有真实模型生产、量化checkpoint、GPU kernel或运行时服务。已为CI增加入口，但本次没有提交、推送或执行GitHub远程CI。
