# 9.4 课程交付检查

[返回质量导航](../README.md) · [9.4 正文](../../../docs/chapters/part-09-quantization/04-quantization-algorithms-and-error-control/README.md)

本轮交付舍入/裁剪、GPTQ 补偿推导与核心实现、缩放/旋转和 FakeQuant/STE 教学。实际回归时间（UTC）：`2026-09-29T06:54:36.037048+00:00`，Windows / Python 3.11.9。

| 检查 | 实际结果 | 记录 |
| --- | --- | --- |
| 全仓库回归 | 524项通过，跳过0项，退出0 | [regression.json](regression.json)、[完整日志](tests.txt) |
| 新增针对性测试 | 33项，已纳入全量回归 | [test_algorithms.py](../../../tests/quantization/test_algorithms.py) |
| 固定 CPU 案例 | 23项全部通过 | [summary.json](../../../docs/chapters/part-09-quantization/04-quantization-algorithms-and-error-control/results/summary.json) |
| 文档命令 | 10条符合预期，包括两次参数检查退出2 | [commands.json](../../../docs/chapters/part-09-quantization/04-quantization-algorithms-and-error-control/results/commands.json) |
| PyTorch 数学核验 | 2个 GPTQ 约束补偿对照、4个 STE 内部/饱和点，全部通过 | 同 summary 的 torch_oracles |
| 文档与目录 | 19篇129节、22节完成；链接与生成段落检查通过 | [course.json](course.json)、[layout.json](layout.json) |

## 教学与视觉检查

正文按15项结构展开，配实验手册、18组含答案练习、源码阅读定位、CPU Passport和实际运行记录。8张中文原创图保留 SVG/PNG，逐张查看文字、数值与布局；量化台阶按 nearest-even 生成，单个边界点归属由公式与测试明确。

正文区分逆 Hessian 与 Cholesky 因子、计算 block 与量化 group、变换因子与量化步长，以及校准误差与独立质量。正则化目标用迹明确输出列求和；图中缩放收益不单调、留出误差变差等结果与固定案例一致。

## 证据边界

源码摘要：`18960990a50965d3605514de5e516bfb93d5dc73841c7718f551113e777ae3f4`；实验、命令与回归来自同一代码/数据快照，测试期间源码摘要未改变。旧章节正文、实验与历史质量记录保持原样。

核心数值代码仅为 CPU 小矩阵教学。论文与上游源码阅读不等于执行完整算法框架；没有真实模型质量、GPU kernel 或 A100/C500 性能结果。已补充 CI 入口，本轮未提交、推送或执行远程 CI。
