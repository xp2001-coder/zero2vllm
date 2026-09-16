# Part 6 · 调度、批处理与重叠执行

[课程总目录](../README.md) · [学习路线](../learning-paths.md)

前置知识：Part 2—3、5。

## 本篇章节

<!-- BEGIN part-index -->

| 节 | 课程 | 验收产出 | 进度 | 正文 |
| --- | --- | --- | --- | --- |
| 6.1 | [连续批处理与调度策略](01-continuous-batching-and-scheduling-policies/README.md) | 模拟长短请求混合并比较吞吐和尾延迟 | 🧩 | 教学骨架 |
| 6.2 | [Chunked Prefill](02-chunked-prefill/README.md) | 扫描 chunk/token 预算，观察 TTFT 与 ITL 权衡 | 🧩 | 教学骨架 |
| 6.3 | [Token 预算与显存预算](03-token-and-gpu-memory-budgets/README.md) | 手算 4096 token 预算中两个 decode 加一个长 prefill，标记理想上限与真实约束 | 🧩 | 教学骨架 |
| 6.4 | [异步调度与输出处理](04-async-scheduling-and-output-processing/README.md) | 阅读时间线并定位 CPU 空洞和同步点 | 📋 | 教学骨架 |
| 6.5 | [DBO 与 TBO 对照](05-comparing-dbo-and-tbo/README.md) | 画两微批次时间线，检查阈值和重叠是否真实发生 | 🧩 | 教学骨架 |
| 6.6 | [调度功能组合与演进](06-scheduling-combinations-and-evolution/README.md) | 制作所选版本组合矩阵，记录拒绝/回退原因 | 📋 | 教学骨架 |

<!-- END part-index -->
