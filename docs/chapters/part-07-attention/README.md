# Part 7 · Attention 与高性能算子

[课程总目录](../README.md) · [学习路线](../learning-paths.md)

前置知识：Part 1、3、5；补充 GPU 编程。

## 本篇章节

<!-- BEGIN part-index -->

| 节 | 课程 | 验收产出 | 进度 | 正文 |
| --- | --- | --- | --- | --- |
| 7.1 | [FlashAttention 原理](01-flashattention-principles/README.md) | 分块计算 softmax 并与朴素实现对齐 | 🧩 | 教学骨架 |
| 7.2 | [FlashAttention 版本与接入](02-flashattention-versions-and-integration/README.md) | 区分算法版本、安装包与 vLLM 实际选择的后端 | 📋 | 教学骨架 |
| 7.3 | [注意力后端选择](03-selecting-an-attention-backend/README.md) | 建立 head size、dtype、KV 格式、图执行的能力矩阵 | 🧩 | 教学骨架 |
| 7.4 | [MLA 与稀疏 Attention](04-mla-and-sparse-attention/README.md) | 分析压缩 KV、索引选择和 attention kernel 的分工 | 📋 | 教学骨架 |
| 7.5 | [Decode 与共享前缀注意力](05-decode-and-shared-prefix-attention/README.md) | 比较短/长上下文和高/低 batch 下的算子时间 | 🧩 | 教学骨架 |
| 7.6 | [线性层与融合算子](06-linear-layers-and-fused-kernels/README.md) | 逐级完成数值参考、mask/stride/尾块检查，再选择一个小算子做 microbenchmark | 🧩 | 教学骨架 |
| 7.7 | [Kernel 选择与调优](07-kernel-selection-and-tuning/README.md) | 解释形状、架构和量化格式如何影响调度与回退 | 🧩 | 教学骨架 |

<!-- END part-index -->
