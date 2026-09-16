# Part 9 · 量化算法、算子实现与推理加速

[课程总目录](../README.md) · [学习路线](../learning-paths.md)

前置知识：Part 1、4、7；内核调优结合 Part 8，MoE 量化补读 Part 12.1—12.3。

## 本篇章节

<!-- BEGIN part-index -->

| 节 | 课程 | 验收产出 | 进度 | 正文 |
| --- | --- | --- | --- | --- |
| 9.1 | [低精度表示与计算路径](01-low-precision-representations-and-compute-paths/README.md) | 定义张量、位布局与缩放契约，计算含元数据的内存/带宽预算 | 🧩 | 教学骨架 |
| 9.2 | [量化算法](02-quantization-algorithms/README.md) | 固定格式与计算后端比较算法，输出校准/质量/转换成本分析 | 🧩 | 教学骨架 |
| 9.3 | [量化策略与精度分配](03-quantization-strategies-and-precision-allocation/README.md) | 在质量、显存和延迟预算下设计策略，并形成敏感度与消融表 | 🧩 | 教学骨架 |
| 9.4 | [手写量化与数据重排算子](04-writing-quantization-and-data-reordering-kernels/README.md) | 先运行已交付 CPU INT4 实验并核对黄金字节/半 scale 误差界，再实现 GPU Q/DQ 与重排 | 🧩 | 教学骨架 |
| 9.5 | [Weight-only GEMV 与 GEMM](05-weight-only-gemv-and-gemm/README.md) | 先写正确内核，再比较显式反量化与融合路径的时间和访存 | 🧩 | 教学骨架 |
| 9.6 | [低精度 GEMM 与缩放实现](06-low-precision-gemm-and-scaling/README.md) | 实现受设备支持的低精度路径，把量化、矩阵乘和缩放开销分项测量 | 📋 | 教学骨架 |
| 9.7 | [融合算子与内核调优](07-kernel-fusion-and-tuning/README.md) | 用 profiling 解释每次改动，并在形状扫描中定位性能拐点与回退 | 🧩 | 教学骨架 |
| 9.8 | [KV 量化与 Attention 协同](08-kv-quantization-and-attention-integration/README.md) | 实现或改造一条 KV Q/DQ 路径，分别测容量、Attention 时间与模型质量 | 📋 | 教学骨架 |
| 9.9 | [MoE 专家量化与分组计算](09-moe-expert-quantization-and-grouped-computation/README.md) | 分析专家 token 分布，优化一个子路径并核对 dispatch/compute/combine 开销 | 📋 | 教学骨架 |
| 9.10 | [量化算子接入 vLLM](10-integrating-quantized-kernels-into-vllm/README.md) | 为同一量化权重追踪加载前后 shape/layout/dtype 和 kernel 分支，先层后模型再服务验收 | 🧩 | 教学骨架 |
| 9.11 | [从算子收益到端到端加速](11-from-kernel-gains-to-end-to-end-acceleration/README.md) | 提交 A100/C500 独立报告，解释更低位宽或更快 GEMM 未带来整体收益的场景 | 🧩 | 教学骨架 |

<!-- END part-index -->
