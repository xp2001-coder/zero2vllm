# Part 9 · 量化算法、格式、算子与推理系统

> **Quantization from Algorithms to Systems**

[课程总目录](../README.md) · [学习路线](../learning-paths.md) · [v1.0 总设计](design.md) · [Passport 规范](templates/model-passport.md)

**基线已确定，正文逐节建设。** 四个 Module、14 节。核心命题：Quantization Algorithm ≠ Quantized Checkpoint Format ≠ Compute Kernel。

Qwen3-8B W4A16 G128 贯穿主线；依次扩展 INT8 W8A8、硬件相关低精度路径、KV 与 MoE/并行。CPU 小矩阵 → 少量层 → 完整模型；MobileNetV2/MQBench 为经典 PTQ/QAT 支线。

从 [9.1 完整课程](01-quantization-system-overview/README.md)开始：含 zero-point 的 Q/DQ/RQ、Linear 手算、17 个 CPU 案例、两份 Passport、6 张中文图与 14 组练习。接着学习 [9.2 数值表示与张量契约](02-numerical-representations-and-tensor-contracts/README.md)：分组 Q/DQ、低精度浮点、尾组与存储开销，含 20 个 CPU 案例和 7 张中文图。随后用 [9.3 校准与误差测量](03-calibration-observers-and-error-measurement/README.md)完成 Observer、受控对照、协议记录与重构目标。[9.4 量化算法与误差控制](04-quantization-algorithms-and-error-control/README.md)进一步推导 GPTQ、比较舍入与缩放，配合可手算的 CPU 实验。另有 [9.8 CPU INT4 打包实验](08-bit-packing-serialization-and-repack/lab.md)和[专题导读](guide.md)。真实模型与 GPU 实测继续单独验收。

前置：Part 1—3 与矩阵运算；CPU 从 9.1 开始；GPU 实验补读 Part 4、7—8，MoE/并行补读 Part 11—12。

## 四个 Module

| Module | 教学范围 | 小节 |
| --- | --- | --- |
| A · Quantization Foundations | 数值、契约与校准 | 9.1—9.3 |
| B · Algorithms & Model Production | 算法、策略与模型生产 | 9.4—9.6 |
| C · Format, Packing & Kernel | 格式、packing、repack 与算子 | 9.7—9.10 |
| D · Runtime State & Systems | KV/MoE、vLLM 与端到端验证 | 9.11—9.14 |

**重点区域：9.7—9.10、9.13—9.14。** 格式、repack、kernel、vLLM 与实测收益需要完整代码和证据。

## 本篇章节

<!-- BEGIN part-index -->

| 节 | 课程 | 验收产出 | 进度 | 正文 |
| --- | --- | --- | --- | --- |
| 9.1 | [量化系统全景与术语分层](01-quantization-system-overview/README.md) | 20 个术语、17 个 CPU 案例、两份 Passport、6 张中文图与固定源码核验 | ✅ | 完整课程 |
| 9.2 | [数值表示、量化公式与张量契约](02-numerical-representations-and-tensor-contracts/README.md) | 20 个 CPU 案例、两份参数 Passport、7 张中文图、16 组练习与 512 个 FP8 位模式对照 | ✅ | 完整课程 |
| 9.3 | [Calibration、Observer 与误差测量](03-calibration-observers-and-error-measurement/README.md) | 20 个 CPU 案例、校准协议与 Passport、7 张中文图、16 组练习及 PyTorch Observer 对照 | ✅ | 完整课程 |
| 9.4 | [量化算法与误差控制](04-quantization-algorithms-and-error-control/README.md) | 23 个 CPU 案例、GPTQ 核心参考与受控对照、8 张中文图、18 组练习及 PyTorch 数学核验 | ✅ | 完整课程 |
| 9.5 | [Activation Quantization 与混合精度策略](05-activation-quantization-and-mixed-precision/README.md) | per-token INT8 参考、敏感度图、精度分配表与固定质量/固定内存对照 | 🧩 | 教学骨架 |
| 9.6 | [量化框架、导出流程与生态](06-quantization-frameworks-and-export/README.md) | Framework Matrix，以及 LightCompress 与 LLM Compressor 各一条真实 export 记录 | 📋 | 教学骨架 |
| 9.7 | [Checkpoint Format 与 Quantized Weight ABI](07-checkpoint-formats-and-quantized-weight-abi/README.md) | 真实 checkpoint 检查器与至少两种格式的非方阵布局/元数据分析 | 📋 | 教学骨架 |
| 9.8 | [位打包、序列化与 Repack](08-bit-packing-serialization-and-repack/README.md) | 沿用现有 CPU INT4 实验；扩展 n-bit、序列化与 repack，提交字节级黄金测试 | 🧩 | 教学骨架 |
| 9.9 | [Weight-only GEMV / GEMM 与调优](09-weight-only-gemv-gemm-and-tuning/README.md) | 三基线正确性、形状扫描、逐项 profiling 消融与 W4A16 Kernel Optimization Report | 🧩 | 教学骨架 |
| 9.10 | [激活量化、低精度 GEMM 与融合](10-activation-quantization-low-precision-gemm-and-fusion/README.md) | Low Precision GEMM Report：未融合分项与融合总耗时、正确性、资源与负载边界 | 🧩 | 教学骨架 |
| 9.11 | [KV Cache 量化与 Attention 协同](11-kv-cache-quantization-and-attention/README.md) | KV 容量/并发/写读开销/Attention 延迟/质量报告，prefill 与 decode 分开 | 📋 | 教学骨架 |
| 9.12 | [MoE、并行切分与量化](12-moe-parallel-sharding-and-quantization/README.md) | Distributed Quantization Contract 与 MoE 案例；CPU 切分重组、单卡计算、多卡扩展分开验收 | 📋 | 教学骨架 |
| 9.13 | [量化路径接入 vLLM](13-integrating-quantization-into-vllm/README.md) | 层→模型→离线→在线四级记录，以及 config→loader→repack→backend→kernel 的证据链 | 🧩 | 教学骨架 |
| 9.14 | [从算子收益到端到端推理加速](14-from-kernel-gains-to-end-to-end-acceleration/README.md) | A100/C500 独立端到端报告与 Algorithm→Format→Kernel→Runtime 系统图 | 🧩 | 教学骨架 |

<!-- END part-index -->

## 写作与验收

每节沿用[15 项统一结构](templates/lesson.md)，每个实验维护[模型 Passport](templates/model-passport.md)。课程进度、CPU/源码验证、A100/C500/多卡实测分开标记。
