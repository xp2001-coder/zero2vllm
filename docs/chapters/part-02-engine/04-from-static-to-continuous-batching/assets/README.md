# 2.4 原创插图

本节沿用前两节的深蓝、青色与紫色矢量风格。PNG 用于正文阅读，SVG 保留可编辑文字、图形和无障碍描述；图中内容由本节输入与调度记录重新绘制。

| 图 | 可编辑源文件 | 对应内容 |
| --- | --- | --- |
| [补位时间线](batching-timeline.png) | [SVG](batching-timeline.svg) | `mixed_lengths` 的静态/连续逐轮结果 |
| [调度边界](iteration-boundary.png) | [SVG](iteration-boundary.svg) | 边界事件、准入、冻结计划、执行与回收 |
| [三个预算](three-budgets.png) | [SVG](three-budgets.svg) | `token_pressure` 的第 0、1 轮；R_A=6、R_B=4 |
| [长 prefill 的取舍](prefill-tradeoff.png) | [SVG](prefill-tradeoff.svg) | `long_prefill_turn` 的连续轨迹与输出间隔 |

输入见 [workloads.json](../data/workloads.json)，公开记录见 [summary.json](../results/baseline/summary.json)。时间线中的 P/D 标注表示本轮输入阶段与位置数，w/hold 分别表示等待准入和活跃但未选中。所有轮次都是逻辑边界，没有用图宽表示真实耗时。

这些图片为课程原创，不复用第三方文章或论文中的图片。修改负载或策略时，先重新运行实验，再核对图内轨迹与数值。
