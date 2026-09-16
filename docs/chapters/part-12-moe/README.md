# Part 12 · MoE 与通信优化

[课程总目录](../README.md) · [学习路线](../learning-paths.md)

前置知识：Part 7、8、11。

## 本篇章节

<!-- BEGIN part-index -->

| 节 | 课程 | 验收产出 | 进度 | 正文 |
| --- | --- | --- | --- | --- |
| 12.1 | [MoE 计算链路](01-the-moe-computation-path/README.md) | 手算 token 路由并统计各专家负载 | 🧩 | 教学骨架 |
| 12.2 | [Expert Parallelism](02-expert-parallelism/README.md) | 画出 token 到专家的分发和汇总 | 🧩 | 教学骨架 |
| 12.3 | [Fused MoE Kernel](03-fused-moe-kernels/README.md) | 对照路由、通信和 GEMM 三类耗时 | 📋 | 教学骨架 |
| 12.4 | [All-to-All 后端](04-all-to-all-backends/README.md) | 比较延迟/吞吐模式与拓扑条件，识别已移除后端 | 📋 | 教学骨架 |
| 12.5 | [EPLB 与弹性专家并行](05-eplb-and-elastic-expert-parallelism/README.md) | 模拟热点专家并评估迁移成本和再均衡收益 | 🧩 | 教学骨架 |
| 12.6 | [MoE 重叠与综合调优](06-moe-overlap-and-combined-tuning/README.md) | 建立算子时间线与消融矩阵，解释收益来源 | 📋 | 教学骨架 |

<!-- END part-index -->
