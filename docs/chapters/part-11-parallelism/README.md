# Part 11 · 并行切分与分布式执行

[课程总目录](../README.md) · [学习路线](../learning-paths.md)

前置知识：Part 3、5—8；掌握 collective 基础。

## 本篇章节

<!-- BEGIN part-index -->

| 节 | 课程 | 验收产出 | 进度 | 正文 |
| --- | --- | --- | --- | --- |
| 11.1 | [并行维度与通信基础](01-parallel-dimensions-and-communication-basics/README.md) | 画出进程组与张量切分，计算通信数据量 | 🧩 | 教学骨架 |
| 11.2 | [Tensor Parallelism](02-tensor-parallelism/README.md) | 手算两路 TP 的前向合并与显存分布 | 🧩 | 教学骨架 |
| 11.3 | [Pipeline Parallelism](03-pipeline-parallelism/README.md) | 制作 pipeline 时间线并估算负载不均的影响 | 🧩 | 教学骨架 |
| 11.4 | [Data Parallelism](04-data-parallelism/README.md) | 比较请求路由、排队与副本扩展吞吐 | 🧩 | 教学骨架 |
| 11.5 | [Context Parallelism 概念](05-context-parallelism-concepts/README.md) | 区分上下文并行、序列并行和分块预填充 | 🧩 | 教学骨架 |
| 11.6 | [Prefill Context Parallelism](06-prefill-context-parallelism/README.md) | 推导 prefill 张量分工与通信，核对模型/runner 支持 | 🧩 | 教学骨架 |
| 11.7 | [Decode Context Parallelism](07-decode-context-parallelism/README.md) | 推导每卡 KV 和输出归约，核对 DCP 合法度数 | 🧩 | 教学骨架 |
| 11.8 | [Sequence Parallelism 与 Async TP](08-sequence-parallelism-and-async-tp/README.md) | 说明 SP 与 PCP/DCP 的缓存/通信目标差异 | 📋 | 教学骨架 |
| 11.9 | [组合并行与拓扑规划](09-combining-parallelism-and-planning-topology/README.md) | 交付一份进程布局和容量预算，标明无法运行的组合 | 🧩 | 教学骨架 |
| 11.10 | [分布式启动与排错](10-distributed-launch-and-debugging/README.md) | 设计启动、卡数不足、断连与通信失败的排查流程 | 📋 | 教学骨架 |

<!-- END part-index -->
