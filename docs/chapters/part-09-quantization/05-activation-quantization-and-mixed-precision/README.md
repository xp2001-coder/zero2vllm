# 9.5 Activation Quantization 与混合精度策略

[本篇目录](../README.md) · [总设计](../design.md) · [Passport](../templates/model-passport.md) · [上一节 9.4](../04-quantization-algorithms-and-error-control/README.md) · [下一节 9.6](../06-quantization-frameworks-and-export/README.md)

> **v1.0 教学计划，正文待编写。** 以下描述计划覆盖的概念、中文图解、代码和验收产物，不能据此认定已实现或已实测。

所属：Module B · Algorithms & Model Production。设备：CPU/PyTorch；A100/C500 独立验证。

基础材料可先阅读[量化导读](../guide.md)；已交付的打包实验位于 [9.8](../08-bit-packing-serialization-and-repack/lab.md)。

## 读完能解释什么

理解三个不同问题：

```text
Weight-only
Activation Quantization
Precision Allocation
```

严格区分：

```text
static activation scale
dynamic activation scale
```

以及：

```text
per-tensor
per-token
per-block
```

理解：

```text
Activation Outlier
```

为什么会推动：

```text
SmoothQuant
Rotation
Channel Transformation
```

## 主线顺序

第一阶段：

```text
W4A16
```

第二阶段优先：

```text
INT8 W8A8
```

作为 A100 主要 W/A 实验。

然后讲清但不强制 A100 原生执行：

```text
FP8
W4A8
FP4
```

## Precision Allocation

研究：

```text
layer sensitivity
channel sensitivity
activation outlier
parameter importance
```

并形成：

```text
layer → precision
```

策略。

SandwichQuant / NAP 可以在这里作为：

> Parameter-subspace adaptation case study

但不能变成整节唯一方法。

## 图

1. static vs dynamic activation quantization。
2. outlier channel。
3. mixed-precision layer map。
4. Weight → Activation → KV 三阶段演进。

## 实现

PyTorch：

```text
per-token activation quant
dynamic INT8 activation quant
mixed precision policy
```

## 完成证据

* sensitivity map。
* precision allocation table。
* fixed-quality memory comparison。
* fixed-memory quality comparison。

## 编排摘要

<!-- BEGIN lesson-plan -->

1. 静态/动态激活 scale：补充概念、推导、例子及实现边界。
2. per-token/per-block：补充概念、推导、例子及实现边界。
3. outlier 与变换：补充概念、推导、例子及实现边界。
4. 层/通道/专家敏感度：补充概念、推导、例子及实现边界。
5. W4A16→INT8 W8A8→条件化 FP8/W4A8/FP4：补充概念、推导、例子及实现边界。

<!-- END lesson-plan -->

## 验收清单

<!-- BEGIN lesson-exercise -->

目标产出：**per-token INT8 参考、敏感度图、精度分配表与固定质量/固定内存对照**。

- [ ] 提供输入、命令、结果与解释。
- [ ] 补充参考答案和至少一个边界案例。

<!-- END lesson-exercise -->

完整正文按[统一写作模板](../templates/lesson.md)逐节交付；验收前保持教学骨架状态。
