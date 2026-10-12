# 9.10 激活量化、低精度 GEMM 与融合

[本篇目录](../README.md) · [总设计](../design.md) · [Passport](../templates/model-passport.md) · [上一节 9.9](../09-weight-only-gemv-gemm-and-tuning/README.md) · [下一节 9.11](../11-kv-cache-quantization-and-attention/README.md)

> **v1.0 教学计划，正文待编写。** 以下描述计划覆盖的概念、中文图解、代码和验收产物，不能据此认定已实现或已实测。

所属：Module C · Format, Packing & Kernel。设备：GPU；CPU 数值参考。

基础材料可先阅读[量化导读](../guide.md)；已交付的打包实验位于 [9.8](../08-bit-packing-serialization-and-repack/lab.md)。

## 读完能解释什么

理解 W/A quantization 比 W-only 多了什么。

典型路径：

```text
FP16 Activation
      ↓
statistics
      ↓
scale
      ↓
quant
      ↓
low-precision GEMM
      ↓
epilogue
```

## 第一实践路径

A100 优先：

```text
INT8 W8A8
```

讲清：

```text
static
dynamic
per-token
```

然后理论与条件性实验：

```text
FP8
W4A8
FP4
```

## 实现

分项实现：

```text
amax/statistics
quant
GEMM
scale
dequant
epilogue
```

然后 fusion：

```text
Norm + Quant
Act + Quant
Quant + GEMM
GEMM + DQ
```

## 实验

必须分项计时，不能只有：

```text
fused kernel total time
```

否则不知道收益来源。

## 完成证据

Low Precision GEMM Report：

```text
Q
GEMM
DQ
epilogue
total
```

以及 fusion ablation。

### 融合测量边界

未融合基线分项计时；融合路径测整体并结合 profiler 定位工作。不能把互相重叠的阶段简单相加，也不能凭空分摊一个融合 kernel 内不可单独测得的耗时。

## 编排摘要

<!-- BEGIN lesson-plan -->

1. INT8 W8A8 优先：补充概念、推导、例子及实现边界。
2. FP8/W4A8/FP4 条件化路径：补充概念、推导、例子及实现边界。
3. statistics/Q/GEMM/scale/DQ/epilogue 分项计时：补充概念、推导、例子及实现边界。
4. Norm/Act+Quant 等融合：补充概念、推导、例子及实现边界。

<!-- END lesson-plan -->

## 验收清单

<!-- BEGIN lesson-exercise -->

目标产出：**Low Precision GEMM Report：未融合分项与融合总耗时、正确性、资源与负载边界**。

- [ ] 提供输入、命令、结果与解释。
- [ ] 补充参考答案和至少一个边界案例。

<!-- END lesson-exercise -->

完整正文按[统一写作模板](../templates/lesson.md)逐节交付；验收前保持教学骨架状态。
