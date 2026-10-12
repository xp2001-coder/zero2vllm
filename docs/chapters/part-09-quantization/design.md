# Part 9 · 量化算法、格式、算子与推理系统

**课程设计基线 v1.0 · 2026-09-20**。固定 4 个 Module、14 个一级教学单元；后续在节内增加专题与实验页。前三篇保持当前教学版本。

[本篇目录](README.md) · [专题导读](guide.md) · [统一写作模板](templates/lesson.md)

> **Quantization from Algorithms to Systems**

## 0. 本篇定位

本篇不把量化仅理解为“把 FP16 权重变成 INT4”，而是完整研究：

```text
浮点模型
   ↓
量化数值契约
   ↓
校准与算法优化
   ↓
量化模型生产
   ↓
Checkpoint / Metadata
   ↓
Packing / Serialization
   ↓
Loader / Repack
   ↓
Kernel
   ↓
Hardware
   ↓
Runtime
   ↓
Model / Serving Performance
```

全篇核心命题：

> **Quantization Algorithm ≠ Quantized Checkpoint Format ≠ Compute Kernel**

同时区分两条实际流程。

### 模型生产

```text
Floating Model + Calibration Data
        ↓
Quantization Scheme
        ↓
Algorithm / Optimization
        ↓
Quantized Tensor
        ↓
Checkpoint Export
```

### 模型消费

```text
Quantized Checkpoint
        ↓
Runtime Loader
        ↓
Metadata Parsing
        ↓
Repack / Initialization
        ↓
Kernel Dispatch
        ↓
Hardware Execution
        ↓
Quality / Memory / Latency / Throughput
```

两条路径通过明确的 **Quantized Model Contract** 连接。

---

## 1. 全篇固定实验主线

### 1.1 LLM 主线

贯穿模型：

```text
Qwen3-8B
```

第一阶段统一研究：

```text
W4A16
Group Size = 128
```

依次扩展：

```text
W4A16
  ↓
W8A8 / INT8
  ↓
FP8 等硬件相关路径
  ↓
KV Quantization
  ↓
MoE / TP / EP
```

Qwen3-8B 不代表所有实验都直接使用完整模型。

实验分为三层：

```text
Level 1
Small Matrix / Single Linear
        ↓
检查公式、axis、packing、误差

Level 2
Few Layers / Small Model
        ↓
研究 calibration、reconstruction、误差传播

Level 3
Qwen3-8B
        ↓
质量、显存、Runtime、Serving
```

---

### 1.2 CNN 经典量化支线

统一使用一个主要案例：

```text
MobileNetV2
```

主要服务：

```text
Observer
FakeQuant
PTQ
QAT
STE
AdaRound
BRECQ
QDrop
LSQ
BN / Activation Statistics
```

框架：

```text
MQBench        主实验框架
PPQ            框架设计对照
```

CNN 支线不承担整篇主叙事，不要求读者完成完整 CNN 部署链才能继续 LLM 主线。

---

### 1.3 量化工具教学分工

| 工具                            | 本课程主要职责                                               |
| ----------------------------- | ----------------------------------------------------- |
| 自己实现的 CPU / PyTorch reference | 公式、Q/DQ、packing、layout、误差黄金参考                         |
| MQBench                       | 经典 PTQ/QAT、Observer、FakeQuant                         |
| PPQ                           | Graph-based quantization 与 backend abstraction        |
| LightCompress                 | LLM 量化算法、重构、变换、策略研究                                   |
| AutoRound                     | rounding optimization 与 Algorithm / Format 解耦         |
| LLM Compressor                | 可部署量化模型生产、compressed-tensors 与 vLLM 路径                |
| ModelOpt                      | NVIDIA 低精度与 hardware-aware quantization 对照            |
| vLLM                          | Loader、Repack、QuantMethod、Kernel dispatch 与最终 Serving |
| SGLang                        | Runtime 对照                                            |
| TensorRT-LLM                  | NVIDIA Runtime / Hardware-specific quantization 对照    |
| FastLLM / llama.cpp           | 轻量 Runtime 与不同低比特布局对照                                 |

这些是**教学职责划分**，不是对项目能力边界的定义。

---

## 2. 全篇统一产物：Quantized Model Passport

每个实验维护来源与算法、数值契约、存储与执行、验证证据四组信息。完整字段与填写规则见 [Passport v1.0 规范](templates/model-passport.md)。

## 十四节教学单元

| Module | 教学范围 | 小节 |
| --- | --- | --- |
| A · Quantization Foundations | 数值、契约与校准 | 9.1—9.3 |
| B · Algorithms & Model Production | 算法、策略与模型生产 | 9.4—9.6 |
| C · Format, Packing & Kernel | 格式、packing、repack 与算子 | 9.7—9.10 |
| D · Runtime State & Systems | KV/MoE、vLLM 与端到端验证 | 9.11—9.14 |

| 小节 | 教学单元 | 验收产物 |
| --- | --- | --- |
| 9.1 | [量化系统全景与术语分层](01-quantization-system-overview/README.md) | 术语分类、第一版 Passport、无 packing 的最小 Q/DQ 与生产/消费系统图 |
| 9.2 | [数值表示、量化公式与张量契约](02-numerical-representations-and-tensor-contracts/README.md) | 对称/非对称 CPU 参考、非方阵/尾组/数值边界测试与含元数据的 bpw 报告 |
| 9.3 | [Calibration、Observer 与误差测量](03-calibration-observers-and-error-measurement/README.md) | Calibration Protocol、Observer 参考与固定数据/固定 token budget 的分组对照 |
| 9.4 | [量化算法与误差控制](04-quantization-algorithms-and-error-control/README.md) | RTN/裁剪/小矩阵舍入实现，GPTQ/AWQ 简化推导与受控算法对比表 |
| 9.5 | [Activation Quantization 与混合精度策略](05-activation-quantization-and-mixed-precision/README.md) | per-token INT8 参考、敏感度图、精度分配表与固定质量/固定内存对照 |
| 9.6 | [量化框架、导出流程与生态](06-quantization-frameworks-and-export/README.md) | Framework Matrix，以及 LightCompress 与 LLM Compressor 各一条真实 export 记录 |
| 9.7 | [Checkpoint Format 与 Quantized Weight ABI](07-checkpoint-formats-and-quantized-weight-abi/README.md) | 真实 checkpoint 检查器与至少两种格式的非方阵布局/元数据分析 |
| 9.8 | [位打包、序列化与 Repack](08-bit-packing-serialization-and-repack/README.md) | 沿用现有 CPU INT4 实验；扩展 n-bit、序列化与 repack，提交字节级黄金测试 |
| 9.9 | [Weight-only GEMV / GEMM 与调优](09-weight-only-gemv-gemm-and-tuning/README.md) | 三基线正确性、形状扫描、逐项 profiling 消融与 W4A16 Kernel Optimization Report |
| 9.10 | [激活量化、低精度 GEMM 与融合](10-activation-quantization-low-precision-gemm-and-fusion/README.md) | Low Precision GEMM Report：未融合分项与融合总耗时、正确性、资源与负载边界 |
| 9.11 | [KV Cache 量化与 Attention 协同](11-kv-cache-quantization-and-attention/README.md) | KV 容量/并发/写读开销/Attention 延迟/质量报告，prefill 与 decode 分开 |
| 9.12 | [MoE、并行切分与量化](12-moe-parallel-sharding-and-quantization/README.md) | Distributed Quantization Contract 与 MoE 案例；CPU 切分重组、单卡计算、多卡扩展分开验收 |
| 9.13 | [量化路径接入 vLLM](13-integrating-quantization-into-vllm/README.md) | 层→模型→离线→在线四级记录，以及 config→loader→repack→backend→kernel 的证据链 |
| 9.14 | [从算子收益到端到端推理加速](14-from-kernel-gains-to-end-to-end-acceleration/README.md) | A100/C500 独立端到端报告与 Algorithm→Format→Kernel→Runtime 系统图 |

每节入口保留本次确认的能力目标、图解计划、实现、实验与验收项；当前均为教学计划，后续逐节交付正文。

## 3. 三条贯穿式实验轨道

### Track A · Classic PTQ / QAT

```text
MobileNetV2
    ↓
MQBench
    ↓
Observer / FakeQuant
    ↓
PTQ / QAT
    ↓
AdaRound / BRECQ / QDrop / LSQ
```

作用：

> 建立经典量化方法论。

---

### Track B · Main LLM Track

```text
Qwen3-8B BF16/FP16
      ↓
W4A16 G128
      ↓
RTN / GPTQ / AWQ / AutoRound
      ↓
Checkpoint
      ↓
Packing
      ↓
Repack
      ↓
Kernel
      ↓
vLLM
      ↓
TTFT / TPOT / Memory / Quality
```

这是 **Part 9 主轴**。

---

### Track C · Advanced Systems

```text
W8A8
 ↓
FP8 / FP4
 ↓
KV Quant
 ↓
MoE Quant
 ↓
TP / EP
 ↓
A100 / C500
```

作用：

> 从静态 Weight-only 走到动态状态和分布式系统。

---

## 4. 每一节统一模板

以后写 9.1～9.14 时，强制使用同一个结构。

```text
1. 本节为什么存在
2. 本节需要解决的问题
3. 前置知识
4. 核心概念
5. 数学推导
6. 中文图解
7. 最小 CPU / PyTorch 实现
8. 真实框架对应
9. GPU / Runtime 实验
10. 常见误区
11. 工程边界
12. 练习
13. 实验产物
14. 完成标准
15. 源码与论文入口
```

---

## 5. 每节实验状态统一

建议继续使用课程已有状态体系，并进一步给实验增加：

```text
🧮 CPU Reference
🟢 A100 Verified
🟣 C500 Verified
🟠 Multi-GPU Verified
📖 Source Verified
⚪ Planned
```

例如：

```text
9.8 Packing
🧮 ✅
📖 ✅
A100 —
C500 —

9.9 W4A16 GEMM
🧮 ✅
A100 ✅
C500 🧩
```

这样不会出现：

> “章节写完了，所以这个功能一定跑过。”

---

## 6. 开发顺序

虽然课程顺序是：

```text
9.1 → 9.14
```

实际开发不要按这个顺序平铺。

第一阶段先做一条 Vertical Slice：

```text
9.1
 ↓
9.2
 ↓
9.4
 ↓
9.6
 ↓
9.7
 ↓
9.8
 ↓
9.9
 ↓
9.13
 ↓
9.14
```

也就是优先打通：

## Qwen3-8B W4A16 End-to-End

先证明：

```text
Algorithm
→ Format
→ Pack
→ Load
→ Repack
→ Kernel
→ vLLM
→ Serving
```

是真的完整闭环。

随后补：

```text
9.3 Calibration
9.5 Activation / Mixed Precision
9.10 W/A GEMM
9.11 KV
9.12 MoE / Distributed
```

---

## 7. Part 9 的最终毕业标准

Part 9 完成不能以：

> “14 篇 Markdown 都写完了。”

为标准。

至少需要产生：

```text
1 × Quantized Model Passport specification

1 × CPU quantization reference library

1 × bit-level packing/repack implementation

1 × MQBench classic PTQ/QAT track

1 × Qwen3-8B W4A16 complete pipeline

1 × GPTQ/AWQ/AutoRound controlled comparison

1 × real quantized checkpoint format analysis

1 × W4A16 GPU kernel optimization report

1 × W/A low-precision compute report

1 × KV quantization report

1 × vLLM quantization runtime trace

1 × A100 end-to-end report

1 × C500 independent platform report

1 × final Algorithm→Format→Kernel→Runtime system map
```

做到这里，Part 9 才算真正完成。

---

## 8. 本篇最终希望读者达到的水平

不是：

> “我会用 GPTQ 把模型变成 INT4。”

而是看到：

```text
Qwen3-8B-W4A16-G128
```

会继续追问：

```text
哪个算法？

校准集是什么？
sym 还是 asym？
group 沿哪个轴？

scale 什么 dtype？
zero-point 怎样编码？

checkpoint format 是什么？
qweight shape 是什么？
怎样 packing？

加载后有没有 repack？
in-memory layout 是什么？

activation dtype 是什么？
accumulator dtype 是什么？

vLLM 使用哪个 QuantMethod？
实际 Linear backend 是什么？
真正执行哪个 kernel？

是在 A100 还是 C500？
Prefill 还是 Decode？
batch/context 多大？

PPL 损失多少？
显存降多少？
TTFT / TPOT 到底改善多少？
```

到这里，读者才真正从：

> **“会做量化”**

进入：

> **“理解 Quantized Inference System”。**

## 基线实施约定

### 教学重心与当前交付边界

9.7—9.10、9.13—9.14 是本篇最重的区域。9.1—9.6 讲深算法基础，同时为格式、repack、kernel、vLLM 和实测收益服务。PTQ/QAT、Observer、FakeQuant、STE 的基础保留在主线；CNN 支线不构成 LLM 学习的额外前置。

当前 [9.1](01-quantization-system-overview/README.md) 已交付完整课程、CPU Q/DQ/RQ 与 Passport；已有 [CPU INT4 打包实验](08-bit-packing-serialization-and-repack/lab.md)和[专题导读](guide.md)继续保留。[9.2](02-numerical-representations-and-tensor-contracts/README.md) 也已交付数值表示、分组 Q/DQ、尾组和存储开销的 CPU 课程。[9.3](03-calibration-observers-and-error-measurement/README.md) 已交付校准协议、基础 Observer、受控对照与误差测量的 CPU 课程；KL 优化器、CNN/MQBench 模型实验仍在后续支线扩展。[9.4](04-quantization-algorithms-and-error-control/README.md) 已交付 RTN/裁剪/舍入、GPTQ 核心参考、缩放/旋转与 STE 的 CPU 教学；完整算法框架与真实模型仍需后续验收。9.5—9.14 的正式正文、真实量化模型生产与 GPU 性能仍逐项建设。总设计中的“必须”“完成证据”是验收目标，具体交付以各节结果为准。

### 验证状态需要证据

课程继续使用 `planned / partial / complete` 与 `scaffold / lesson`。实验另记 CPU Reference、A100 Verified、C500 Verified、Multi-GPU Verified、Source Verified、Planned 六种范围；只有附命令、环境、结果并通过检查时才标 Verified。源码阅读记录不能替代 GPU 记录；不适用、未运行和执行失败分别说明。

9.1 新增无打包的 Q/DQ/RQ 数值记录与 Passport 程序；旧 CPU INT4 参考记录继续保留。n-bit、序列化与 repack 扩展仍是计划，9.8 的 partial 不代表这些扩展已经实现。所有 GPU/多卡记录仍为未验证。上文状态示例只演示记法，不代表当前项目状态。

### 开发顺序与最小校准前置

阅读按 9.1—9.14；开发先交付 9.1，再逐步贯通 W4A16 的生产、格式、执行与服务。9.3 的完整 Observer 专题可以后补，但首次 GPTQ/AWQ/AutoRound 对照之前，必须已有最小 Calibration Protocol、独立评测集和基线质量记录。

先固定小矩阵，再固定少量层，最后固定 Qwen3-8B 的模型/tokenizer revision。完整模型的量化成本与内存需求实测后记录。LightCompress 与 LLM Compressor 各自建立独立生产路径，格式转换需核验数值与输出。

### 数学、格式与源码口径

- 统一采用 `X[M,K] × W[K,N]`；如引用列样本记法，先给出转置关系。GPTQ 类输入 Gram/Hessian 代理在当前约定下与 `XᵀX` 成正比，常数依损失归一化确定。
- effective bpw 的分子以 bit 计，包含实际 padding；若输入统计为 byte，先乘 8。
- Checkpoint ABI 与执行布局分别记录；本课程的 Quantized Weight ABI 是分析契约的教学术语，不声称各框架存在统一标准 ABI。
- Runtime 图表达职责与追踪入口；方法选择、backend 选择和 repack 的调用时机按所选 tag/commit 核对，不规定所有后端都在 `apply()` 后选择 kernel。
- A100 的 W/A 实践优先验证 INT8 W8A8。FP8/FP4 的存储、模拟、weight-only fallback 与原生低精度计算分别说明；C500 按实际 MACA 软件栈独立确认。
- Qwen3-8B 贯穿稠密 LLM 主线；MoE 使用单独的专家夹具或明确命名的 MoE 模型，不能把稠密模型写成 MoE 实测。
- 每个 API、格式和算子案例固定版本。公开课程只链接可公开查阅的源码、论文与文档；中文图由课程自行绘制。

### 原有材料迁移

旧 9.4 的 INT4 教学实验归入新 9.8；旧 9.7 的访存、流水线、寄存器、split-K、autotune 内容由新 9.9/9.10 承接。算法集中在新 9.4，KV 在 9.11，MoE 在 9.12，vLLM 接入在 9.13，服务验收在 9.14。

篇级写作遵循[统一模板](templates/lesson.md)，每节的代码、实验输入、图和公开结果就近放置；有实际内容后再创建子目录。
