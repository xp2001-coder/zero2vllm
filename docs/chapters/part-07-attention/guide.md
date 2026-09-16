# Attention 与算子：从数学到实际后端

对应 7.1、7.3、7.5—7.7。前置：矩阵乘、因果掩码、[分页缓存](../part-05-kv-cache/guide.md)。状态：**推导与实现路线已提供；本篇 GPU kernel 尚未交付和实测**。

本篇吸收 vLLM 第 8 讲、SGLang 第 10、13 讲。目标是能解释内核的输入、访存和计算，再谈优化参数。

## 1. 分清四类问题

| 名称 | 主要问题 | 和其他机制的关系 |
| --- | --- | --- |
| MHA/GQA/MQA/MLA | 模型用什么注意力结构和 KV 表示 | 决定张量维度与缓存成本 |
| FlashAttention | 怎样减少注意力计算的显存读写 | 在分块处理中完成 softmax 和加权求和 |
| PagedAttention | 非连续 KV 块如何分配、寻址和参与计算 | 不等于前缀复用算法 |
| Prefix Cache | 哪些历史计算可以复用 | 命中后仍需要合适的 Attention kernel |

FlashInfer、Triton、FlashAttention 包和 vLLM 的 backend 枚举还属于不同层次。查看包安装成功只证明依赖存在；本轮实际用了哪个 kernel，需要从选择条件、日志或 trace 确认。

## 2. 为什么可以不存完整注意力矩阵

对一行 query，朴素计算先得到全部分数，再做 softmax，最后乘 V。长上下文时，这张中间分数矩阵的读写成本很大。

分块处理时，可以维护当前最大值 `m`、指数和 `l`、未归一化的加权和 `a`。读到新块分数 `s` 和对应 `V` 后：

```text
m_new = max(m, max(s))
alpha = exp(m - m_new)
p = exp(s - m_new)
l_new = alpha * l + sum(p)
a_new = alpha * a + p @ V
output = a_new / l_new
```

初始化 `m = -inf, l = 0, a = 0`，最后一个块之后才得到最终输出。实现时要单独处理全 mask 行，避免 `-inf - (-inf)` 引入 NaN。

**手算**：分数 `[0, ln(2)]`，V 为 `[1, 3]`，分两块。第一块后 `m=0,l=1,a=1`；第二块 `alpha=1/2,p=1`，得到 `l=1.5,a=3.5`，输出 `7/3`，与整体 softmax 相同。

这是精确注意力的分块计算思路，实际浮点顺序可带来数值差异。“精确”不要求所有内核逐位一致。实现细节和 GPU 示例见 [Triton Fused Attention 教程](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)。

## 3. 手写算子的递进顺序

| 步骤 | 新引入的内容 | 先过哪些正确性检查 |
| --- | --- | --- |
| 向量加法 | program、offset、load/store、mask | 长度小于/大于 tile、非整除尾部 |
| Reduction / Softmax | 归约和稳定计算 | 极大/极小输入、全 mask 行、误差 |
| Matmul | M/N/K 分块、布局和数据复用 | 非对齐 K/N、不同 M、stride |
| Attention | 多轮 softmax、因果/变长、Q/K/V 布局 | 与朴素实现比较，位置偏移与边界 |
| 分页 Decode | block table、长 KV、小 query | 非连续块、尾页、不同序列长度 |
| 融合小算子 | Norm/RoPE/量化等融合边界 | 保存同一数值契约，测中间读写变化 |

每步都保留可读的数学参考和输入生成器。Triton 的 tile、warp、stage 配置具有平台和形状条件；NVIDIA 内核经验迁移到 MACA 时先核对编译器、库和硬件支持。

## 4. Prefill 与 Decode 为什么可能选择不同实现

用线性层的 `X[M,K] × W[K,N]` 类比：decode 常有较小 M，prefill 可有较大 M。实际批次、共享前缀、GQA、KV 长度和图模式会改变访存与并行度，不能只按“prefill 算力受限、decode 带宽受限”两个标签得出结论。

Decode 的长 KV 可以按区间并行计算部分结果，再合并 softmax 统计与加权和；额外归约和临时空间也有成本。split-K/分区越多并非总越快。

## 5. 后端选择表怎样填写

为一个具体模型记录：GPU 型号、软件版本、Q dtype、KV dtype/layout、head dimension、KV heads、prefill/decode、分页/滑窗/MLA 条件、图模式。随后记录候选后端、选择原因、真实 kernel 和不支持时的回退。

官方的[Attention 后端能力表](https://docs.vllm.ai/en/latest/design/attention_backends/)用于查阅；实际实验以所选版本为准。不要把 FA3/FA4 的硬件条件推广到 A100，也不要据 NVIDIA 的表推断 C500。

## 6. 验收

**学生**：推导 `7/3` 的例子，解释为何需重缩放旧的 `l/a`；画出非连续 KV 的读取位置。

**工程师**：选择一组真实层的形状，报告 kernel 时间、次数、有效 token、padding、临时空间和正确性。输入用真实模型形状，性能测量与 profiler 分开。量化路径接着读[量化算子](../part-09-quantization/guide.md)。
