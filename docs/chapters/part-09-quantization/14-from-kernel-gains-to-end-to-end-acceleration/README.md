# 9.14 从算子收益到端到端推理加速

[本篇目录](../README.md) · [总设计](../design.md) · [Passport](../templates/model-passport.md) · [上一节 9.13](../13-integrating-quantization-into-vllm/README.md) · [下一篇 10.1](../../part-10-speculation/01-drafting-verification-and-rejection-sampling/README.md)

> **v1.0 教学计划，正文待编写。** 以下描述计划覆盖的概念、中文图解、代码和验收产物，不能据此认定已实现或已实测。

所属：Module D · Runtime State & Systems。设备：双平台；CPU 分析。

基础材料可先阅读[量化导读](../guide.md)；已交付的打包实验位于 [9.8](../08-bit-packing-serialization-and-repack/lab.md)。

## 唯一核心问题

> **Quantization actually helps what?**

不再介绍新算法。

只做系统闭环。

---

## 五层验收

### Level 1 · Numerical Correctness

```text
CPU Golden
vs
GPU Result
```

### Level 2 · Quantization Quality

```text
PPL
task accuracy
zero-shot
```

### Level 3 · Kernel

```text
latency
bandwidth
utilization
```

### Level 4 · Model

```text
prefill latency
decode latency
model memory
KV memory
```

### Level 5 · Serving

```text
TTFT
TPOT
ITL
Throughput
Goodput
P50
P95
P99
```

---

## 必须回答的反例

例如：

```text
Model size ↓ 4×
```

为什么可能只有：

```text
TPOT ↓ 15%
```

分析：

```text
unpack
scale load
dequant
layout conversion
kernel launch
memory traffic
occupancy
unsupported hardware instruction
non-quantized operators
scheduler
communication
```

---

## 双平台报告

### A100

完整报告：

```text
software
driver
model
scheme
format
kernel
workload
quality
memory
performance
profile
```

### C500

使用完全独立的：

```text
MACA
vLLM-MetaX
operator library
kernel
runtime version
```

记录。

不能写：

> “理论上应该和 A100 一样。”

未实测写：

```text
未验证
```

## 编排摘要

<!-- BEGIN lesson-plan -->

1. 数值/模型质量/kernel/model/serving 五层验收：补充概念、推导、例子及实现边界。
2. PPL/显存/TTFT/TPOT/ITL/吞吐/Goodput/尾延迟：补充概念、推导、例子及实现边界。
3. 预处理摊销与双平台报告：补充概念、推导、例子及实现边界。
4. 量化如何改变P/D曲线与容量配比，避免只按压缩倍率估速：补充概念、推导、例子及实现边界。

<!-- END lesson-plan -->

## 验收清单

<!-- BEGIN lesson-exercise -->

目标产出：**A100/C500 独立端到端报告与 Algorithm→Format→Kernel→Runtime 系统图**。

- [ ] 提供输入、命令、结果与解释。
- [ ] 补充参考答案和至少一个边界案例。

<!-- END lesson-exercise -->

完整正文按[统一写作模板](../templates/lesson.md)逐节交付；验收前保持教学骨架状态。


## 与分离式服务的衔接

本节的相关结论将接入 [Part 13：分布式推理工程](../../part-13-disaggregation/README.md)。量化如何改变P/D曲线与容量配比，避免只按压缩倍率估速。先明确计量单位和验证范围，再用于集群推算。
