# 量化策略与算子：从误差目标到 vLLM 执行路径

对应 9.1—9.5、9.7、9.10—9.11。前置：矩阵乘、dtype、基本推理链路。状态：**导读与 CPU INT4 参考已交付；GPU kernel、真实模型质量和 vLLM 插件集成待完成**。

吸收 vLLM 第 10、11 讲、SGLang 第 8、9 讲。算法继续统一在 **9.2 量化算法**，以相同数据契约比较方法，再把主要精力投入算子实现和加速。

## 1. 先选一个可验证的问题

给定 `X[M,K] × W[K,N]`，希望在质量损失预算内减少权重读取和存储。先记录：目标负载主要是小 M decode 还是大 M prefill？权重、激活、KV 哪个占用最大？硬件原生支持哪条计算路径？

| 决策 | 可选维度 | 必须补充的约束 |
| --- | --- | --- |
| 量化对象 | W / A / KV / 部分专家 | 敏感层、长上下文和路由行为 |
| 表示 | INT4/INT8/FP8/FP4 等 | 实际乘法、累加、输出精度 |
| 粒度 | tensor/channel/token/group/block | scale 大小、访存和缩放开销 |
| 时间 | 离线/在线，静态/动态 | 校准与运行时统计成本 |
| 精度分配 | 分层、分组、专家混合精度 | 收益、异常层和后端支持 |

A100 的起步候选可研究 W4A16/W8A16 或受实现支持的 INT8 路径。官方兼容表不把 Ampere 列为 llm-compressor FP8 W8A8 支持设备；某种 FP8 权重格式可被读取，也不等于原生 FP8 乘法。C500 的格式和 kernel 能力单独核对。[vLLM 量化兼容表](https://docs.vllm.ai/en/latest/features/quantization/)

## 2. 9.2 量化算法：统一比较目标和代价

最小基线从 RTN（就近舍入）开始。随后逐项研究：量化区间与裁剪、优化舍入、输入相关的重构误差、误差补偿、等价变换以及 PTQ/QAT 的成本。

单看 `||W − Ŵ||` 不一定对应层输出误差。对固定校准输入 X，比较 `||XW − XŴ||`，更接近该层实际使用方式；模型任务质量仍要用独立评测集检查。

| 案例方向 | 研究的问题 | 固定哪些条件才可比较 |
| --- | --- | --- |
| RTN 与裁剪 | 范围、异常值和舍入如何影响误差 | 位宽、group、dtype、数据 |
| GPTQ 类重构/补偿 | 利用输入统计减少层重构误差的代价 | 校准样本、处理顺序、格式、kernel |
| AWQ 类通道缩放 | 保护重要通道如何改变可量化性 | 等价变换落点、校准目标、布局 |
| 平滑/旋转等变换 | 异常分布能否转换为更适合量化的形式 | 额外计算、是否折叠、模型结构 |

这些是同一章内的算法实验。作者实现用于核对定义和转换流程：[GPTQ](https://github.com/IST-DASLab/gptq)、[AWQ / TinyChat](https://github.com/mit-han-lab/llm-awq)。实际使用时保留它们自己的许可和来源。

### 缩放补偿到底在哪里

对非零对角缩放矩阵 D，有 `XW = (X D⁻¹)(D W)`。量化 `D W` 后，补偿可以体现在上游激活、可折叠的相邻层或等价的权重解释中，但必须说明最终执行的等式。

离线校准代码为了估计误差，将反缩放临时乘回反量化权重，并不自动说明部署时“只改权重就完成全部变换”。把原始网络、校准模拟和部署计算分别画出来，逐层核对输出。

## 3. checkpoint 格式和执行格式是两份契约

checkpoint 可能保存 packed weight、scale、zero-point 等。kernel 可能需要另一种 tile、排列、对齐或分片。加载后的 repack/permute 将前者转换为后者；这类一次性成本应与每次 forward 的开销分开。

同一个量化模型有可能选择“完整反量化后 matmul”，也可能走“分块读取并融合反量化的 GEMM”。选择与版本、M/N/K、dtype、硬件、布局等有关。课程不沿用某篇笔记中的固定 M 阈值作为通用规则。

## 4. 当前可运行的 Q0/Q1

从[CPU INT4 Linear 实验](04-writing-quantization-and-data-reordering-kernels/lab.md)开始。它已经实现：

- 沿 K 分组的对称 Q/DQ，量化范围 `[-7,7]`，ties-to-even 舍入。
- 两个有符号 INT4 打包到一个 byte，低半字节优先；奇数尾部补零。
- 全零组、尾组、非法值、显式反量化 Linear 与直接读取 packed 数据的数学参考。
- 分开报告实现误差和量化误差，把 scales 计入逻辑存储量。

这是自定义教学格式，尚不兼容 AWQ/GPTQ/Marlin checkpoint。CPU 使用 Python float 计算，不冒充 W4A16 硬件执行。

## 5. Q2/Q3：从独立 Q/DQ 到融合 GEMM

第一版 GPU 实现只选一种明确格式，先处理正确性：连续布局、非整除 K/N、尾组、负数符号扩展、scale 广播，再增加 stride、重排和更广形状。

对完全相同的量化权重建立三条路径：

| 路径 | 目的 | 怎样比较 |
| --- | --- | --- |
| 原始浮点 Linear | 测量策略带来的误差与质量变化 | 允许量化误差，质量需另评 |
| 显式 DQ → 浮点 matmul | 检查数学和布局参考 | 与相同量化权重的融合路径比较 |
| Packed → 分块 DQ + GEMM | 减少完整 dense 中间权重的读写 | 检查实现误差，再测收益 |

先覆盖真实模型的小 M 和大 M，记录 Q/DQ、GEMM、缩放、epilogue、workspace、预打包的分项开销。每次只调一个 tile/流水线/融合条件，并解释寄存器压力、数据复用和带宽变化。

参考 [Triton Matmul](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)和[Marlin](https://github.com/IST-DASLab/marlin)的思路，当前项目未移植这些内核。小 M 的收益不应推广到所有 prefill 形状。

## 6. Q5：接入 vLLM 时追踪完整链路

按所选版本读取 `QuantizationConfig` 与 `QuantizeMethodBase`。典型职责如下，具体签名和支持的层型以源码为准。[量化基类](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/base_config.py)

```text
模型配置 / 用户量化配置
  → QuantizationConfig
  → get_quant_method：为层选择方法
  → create_weights：定义参数与加载布局
  → load_weights：装入文件中的数据、处理分片
  → process_weights_after_loading：重排/准备执行所需状态
  → apply：选择并调用实际执行路径
```

每个阶段填入 shape、dtype、stride/layout、量化轴和 TP 分片。重点检查：

1. 加载/分片后 group 与 scale 是否仍对齐？N/K 非整除时拒绝还是补齐？
2. repack 是否在每次 forward 重复执行？转换前后数值解释是否一致？
3. 不支持的硬件、dtype、group 或形状是否有明确错误/回退？
4. 自定义算子有没有正确的 shape/meta 描述和编译集成？图回放时 buffer 和 workspace 生命周期是否满足约束？
5. 是否真的走到预期 kernel，而不是仅识别了方法名称？

先单层，再模型加载/生成，最后并发服务。插件注册成功不代表层型、图执行、LoRA、MoE 或不同平台组合全部通过。

## 7. 为什么 GEMM 快了，服务还可能变慢

假设原始执行中 GEMM 占 40%，把这一部分时间减半，理想整体加速最多为 `1 / (0.6 + 0.4/2) = 1.25`。若多出激活统计、DQ、内存分配或格式转换，收益还会下降。

含预打包成本的回本次数，可粗略估为 `T_repack / (T_baseline − T_quant)`；分母非正时，这个负载下不存在靠重复运行摊销得到的回本点。真实服务还要计入排队、缓存、批次和质量。

最终分别提交 A100/C500 的 kernel、层、模型、服务四级报告。9.8 KV 与 9.9 MoE 在完整 Linear 链路之后扩展；不要把 KV 压缩、专家量化和稠密 GEMM 的性能结论互相替代。

## 8. 验收

**学生**：运行 CPU 实验，手算 `[-8,-1,0,7,-7]` 的字节 `f8 70 09`，解释量化范围为何比 INT4 存储范围窄一个值。

**量化工程师**：提交策略预算、张量契约、同格式算法对比、真实 kernel 分支与三基线消融。质量、实现正确性和速度分别作结论。完整后续实验分工见[量化篇设计](design.md)。
