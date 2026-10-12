# 9.9 Weight-only GEMV / GEMM 与调优

[本篇目录](../README.md) · [总设计](../design.md) · [Passport](../templates/model-passport.md) · [上一节 9.8](../08-bit-packing-serialization-and-repack/README.md) · [下一节 9.10](../10-activation-quantization-low-precision-gemm-and-fusion/README.md)

> **v1.0 教学计划，正文待编写。** 以下描述计划覆盖的概念、中文图解、代码和验收产物，不能据此认定已实现或已实测。

所属：Module C · Format, Packing & Kernel。设备：CPU 参考；A100/C500 分开实测。

基础材料可先阅读[量化导读](../guide.md)；已交付的打包实验位于 [9.8](../08-bit-packing-serialization-and-repack/lab.md)。

## 读完能解释什么

理解 W4A16/W8A16 为什么可能加速，以及为什么不一定加速。

重点区分：

### Explicit Dequantization

```text
INT4
 ↓
FP16 Weight Tensor
 ↓
GEMM
```

### Fused

```text
INT4 Load
 ↓
Unpack
 ↓
Scale
 ↓
Compute
```

理解：

```text
INT4 storage
```

不等于：

```text
native INT4 Tensor Core compute
```

## Prefill / Decode

必须分别分析：

```text
Decode:
small M
memory-bound tendency

Prefill:
large M
higher GEMM utilization
```

## 实现递进

```text
CPU golden
    ↓
GPU correctness baseline
    ↓
explicit DQ
    ↓
fused DQ + GEMM/GEMV
    ↓
tiling
    ↓
scale reuse
    ↓
shared memory
    ↓
pipeline
    ↓
register tuning
    ↓
split-K / shape-specific path
```

## 实验

### CPU

Golden reference。

### A100

重点主平台。

Shape Sweep：

```text
M
N
K
batch
group size
```

分别覆盖 Prefill / Decode。

### C500

待 MACA 路径稳定后独立实现与测量。

不直接拿 A100 结论代替。

## Profiling

每次优化都执行：

```text
Profile
↓
Hypothesis
↓
Change One Factor
↓
Correctness
↓
Performance
```

## 完成证据

一份完整：

### W4A16 Kernel Optimization Report

## 编排摘要

<!-- BEGIN lesson-plan -->

1. W4A16/W8A16：补充概念、推导、例子及实现边界。
2. 显式/融合 DQ：补充概念、推导、例子及实现边界。
3. 实际乘法类型：补充概念、推导、例子及实现边界。
4. prefill/decode shape sweep：补充概念、推导、例子及实现边界。
5. tiling/scale/L2/共享内存/流水线/寄存器/split-K/autotune：补充概念、推导、例子及实现边界。

<!-- END lesson-plan -->

## 验收清单

<!-- BEGIN lesson-exercise -->

目标产出：**三基线正确性、形状扫描、逐项 profiling 消融与 W4A16 Kernel Optimization Report**。

- [ ] 提供输入、命令、结果与解释。
- [ ] 补充参考答案和至少一个边界案例。

<!-- END lesson-exercise -->

完整正文按[统一写作模板](../templates/lesson.md)逐节交付；验收前保持教学骨架状态。
