# 量化篇设计：算法、算子与推理加速

本篇以**设计量化策略、实现量化算子、解释实际加速**为核心，覆盖 9.1—9.11。已有[专题导读](chapters/05-quantization-operators.md)和 [CPU INT4 Linear 实验](labs/02-int4-linear.md)；GPU 算子、真实模型评测和 vLLM 接入待交付。课程状态与[详细目录](curriculum.md)一致。

## 1. 教学目标与阅读深度

- **学生**：从数学参考和位布局出发，写出 Q/DQ、打包与基础量化 Linear，理解位宽、质量和性能的关系。
- **量化工程师**：从精度分配、布局、融合边界和执行瓶颈进入，完成内核优化、vLLM 接入与端到端消融。

算法集中到 **9.2 量化算法**。GPTQ、AWQ 等作为误差目标、校准和变换策略的比较案例；工具与格式用于解释模型产物怎样进入执行路径，查阅映射保留在文末。

Part 7 提供通用 GPU/Attention 基础，Part 8 提供编译与图执行；本篇重点是低精度数据如何存、搬、算，以及这些选择怎样影响质量和性能。

## 2. 十一个章节如何串起来

| 阶段 | 章节 | 应交付的能力 |
| --- | --- | --- |
| 表示、算法与策略 | 9.1—9.3 | 明确存储/乘法/累加精度；比较算法；根据质量、显存和延迟预算分配精度 |
| 基础算子 | 9.4 | 编写缩放统计、Q/DQ、低比特打包与重排，验证边界 |
| 量化矩阵乘 | 9.5—9.6 | 实现 weight-only 和设备支持的低精度计算路径，处理 scales、累加及输出缩放 |
| 内核加速 | 9.7 | 优化访存、融合、分块、流水线与资源占用，说明适用的形状范围 |
| 特定执行路径 | 9.8—9.9 | 优化 KV/Attention 与 MoE 量化子路径，核对状态、布局与负载特点 |
| 引擎集成与验收 | 9.10—9.11 | 接入 vLLM 调度/执行，验证质量和完整服务收益 |

## 3. 贯穿实验：从量化张量到一个可用的 Linear

以下是贯穿实验设计。Q0/Q1 已提供一种教学格式及 CPU 参考；stride/重排等扩展、Q2—Q5 仍需实现与检查。

| 阶段 | 具体工作 | 产出与检查 |
| --- | --- | --- |
| Q0 策略与契约 | 定义 shape、位宽、量化轴、group size、scales、舍入与饱和；设计质量/性能预算 | 张量契约和策略选择说明 |
| Q1 数学与布局参考 | 已实现 Q/DQ、INT4 pack/unpack 与 Linear；后续扩展重排与反重排 | [CPU 参考](labs/02-int4-linear.md)已检查黄金字节、尾组、负数、零值和误差界；尚未实现 GPU 重排 |
| Q2 基础 GPU 实现 | 先写独立 Q/DQ，再写小规模 weight-only GEMV/GEMM | 对齐参考的 GPU 实现与误差报告 |
| Q3 融合与调优 | 将分块反量化并入矩阵乘，复用 scales；调整 tile、加载和流水线 | 形状扫描、时间线与逐项消融 |
| Q4 扩展路径 | 学习动态激活量化与 GEMM；进阶选择 KV 写读或专家 grouped GEMM | 完整子路径的时间/质量对照 |
| Q5 接入 vLLM | 注册配置/方法，处理加载后重排、kernel 选择和回退；运行层/模型/服务实验 | 集成记录与端到端报告 |

Triton 矩阵乘教程用于学习可读的分块实现、程序排序与自动调优；Marlin 用于分析 weight-only 内核的布局、反量化与计算协作。它们是阅读材料，当前工程尚未集成其实现。[Triton 教程](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html) · [Marlin](https://github.com/IST-DASLab/marlin)

### 张量契约应写到什么程度

首个教学例子采用 `X[M,K] × W[K,N]`，权重沿 K 分组，至少固定：

- `qW` 的符号表示、低/高半字节顺序、存储 dtype、stride、重排映射和对齐要求。
- `scales[ceil(K/G),N]` 的轴顺序及广播规则，其中 G 为 group size；尾组和全零组的处理。
- 输入、反量化、乘法、累加和输出的 dtype；非有限值的支持或拒绝规则。
- 非对称格式的 zero-point 及补偿方式；分组 scale 随 K 变化时，不能无条件提到完整 K 归约之后。
- 独立反量化参考 `Y_ref = X @ dequant(qW, scales)`；加载阶段与每次前向分别承担的工作。

此契约用于第一个小实现，后续按 kernel 与 checkpoint 增加格式适配。各实现保持明确的数学语义和可追溯的布局转换。

## 4. 策略设计需要回答的问题

| 决策 | 分析依据 | 要比较的成本 |
| --- | --- | --- |
| W/A/KV 哪些量化 | 权重读取、激活计算、KV 容量/带宽分别是否构成瓶颈 | 节省的流量与新增 Q/DQ、统计和元数据 |
| 静态还是动态 | 输入分布、校准覆盖、负载漂移、质量目标 | 校准成本与每次前向的统计/量化成本 |
| tensor/channel/group/block/token/head 粒度 | 离群值、误差分布、kernel 能消费的格式 | scales 大小、广播、矩阵乘缩放与布局约束 |
| 哪些层/专家保留高精度 | 层敏感度、专家负载、路由变化、长上下文质量 | 质量收益、混合路径开销与 kernel 切换 |
| 如何随负载选算子 | M/N/K、并发、prefill/decode、上下文长度 | 切换阈值、带宽/计算瓶颈与尾延迟 |
| 算法如何适配执行格式 | 裁剪、舍入、误差补偿和变换的输出要求 | 离线重排能否消化、是否新增在线计算 |

比较算法时固定位宽、粒度与执行后端；比较 kernel 时固定量化后的逻辑权重与语义。联合优化作为单独实验，避免把策略变化产生的收益记到内核改动上。

## 5. 算子加速的实验次序

1. **定位开销**：分测统计、Q/DQ、重排、GEMM/GEMV 和输出处理，分别观察 prefill/decode。
2. **优化流量与布局**：向量加载、连续访问、权重/scale 排布、L2/共享内存复用与 padding 代价。
3. **选择融合边界**：分块反量化+GEMM、Norm/Act+Quant、缩放/bias/activation epilogue；同时检查寄存器压力。
4. **调整调度与流水线**：tile、warp/threadblock 分工、双缓冲、受支持的异步加载、split-K 与归约。
5. **检查资源和形状范围**：寄存器溢出、共享内存冲突、occupancy、尾块、短 K、小 M 与大 M 的拐点。
6. **回到服务链路**：检查启动、图执行、框架调度和量化开销是否抵消了算子收益。

A100 沿 CUDA/Triton 与对应内核库验证；C500 按 MACA、编译器和算子库的实际交付选择实现。平台专用指令、流水线能力与工具链分别核对。[平台说明](backends.md)

## 6. vLLM 接入的阅读重点

9.10 沿 `QuantizationConfig` / `QuantizeMethodBase` 阅读权重创建、`process_weights_after_loading` 和 `apply` 的职责，再追踪 Linear/MoE 的实际 kernel 选择。正式章节固定源码版本，检查元数据、加载时重排、运行时分派与回退。[量化接口](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/base_config.py)

检查加载后的布局/scales、重复加载或更新、目标 kernel 命中、不支持的 shape 与回退行为；进入模型后再验证图执行、取消和适用的功能组合。[Kernel 配置](https://github.com/vllm-project/vllm/blob/main/vllm/config/kernel.py)

## 7. 正确性和性能分层验收

### 两类误差分别检查

- **实现误差**：GPU kernel 对比相同 `qW/scales` 的反量化参考。位打包、索引和量化整数等离散结果单独精确检查，浮点计算按 dtype/累加长度制定容差。
- **策略损失**：使用已验证的量化路径与原精度模型比较 logits、困惑度或任务质量，并检查长上下文、异常样本和专家路由相关表现。

实现误差合格后才解释模型质量变化。校准数据与质量评测数据分开记录，观察逐层/逐步结果及任务质量。

### 三个必要基线

| 基线 | 用途 |
| --- | --- |
| 原精度的库实现 | 评估量化策略和完整加速收益 |
| 同一量化权重，独立反量化后计算 | 验证语义，测显式 Q/DQ 与中间张量代价 |
| 同一量化权重，融合执行 | 测内核优化效果；可加入固定版本的成熟量化内核对照 |

分别报告纯 kernel、含前后处理的算子、模型前向和服务延迟。加载时预打包单独计时并说明摊销；在线动态量化计入每次请求实际开销。记录 GPU 同步/事件计时、预热、重复次数、缓存状态和测量区间。

形状覆盖 decode 小 M、prefill 大 M、对齐/非对齐 N/K、多种 group size。报告时间、实际存储、有效带宽或计算量口径、质量与 TTFT/TPOT/Goodput。A100/C500 独立记录，不预填加速倍数。[实验规范](experiments.md) · [验证模板](templates/feature-validation.json)

## 8. 原有算法、格式与工具怎样保留

| 查阅对象 | 新章节归属 |
| --- | --- |
| GPTQ、AWQ 等算法 | 9.2 算法比较；9.3 策略权衡 |
| INT4/INT8、FP8 E4M3/E5M2、NVFP4/MXFP4/MXFP8、W4A16/W8A16/W8A8/W4A8 | 9.1 表示/计算路径；9.5—9.6 实现与硬件条件 |
| Marlin、Machete、Humming、CUTLASS、Triton、DeepGEMM、FlashInfer 等内核 | 9.5—9.7、9.9 比较阅读与调优；9.10 分派 |
| AutoAWQ/AutoGPTQ/GPTQModel、LLM Compressor/compressed-tensors、ModelOpt、TorchAO、Quark、INC、BnB/GGUF | 9.10 checkpoint/配置/布局接口与版本兼容索引 |
| FP8/INT8/INT4/NVFP4/TurboQuant 等 KV 格式 | 9.8 KV 与 Attention 协同，按后端验证 |
| experts_int8、moe_wna16、混合精度与其他注册项 | 9.3 策略、9.9 专家算子、9.10 集成 |

全部已盘点名称仍在 [特性索引](vllm-feature-map.md) 和 [注册项清单](feature-inventory.json) 中。查阅名称、上游实现、平台可运行、实际收益分别记录。
