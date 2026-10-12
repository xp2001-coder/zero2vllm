# 9.6 量化框架、导出流程与生态

[本篇目录](../README.md) · [总设计](../design.md) · [Passport](../templates/model-passport.md) · [上一节 9.5](../05-activation-quantization-and-mixed-precision/README.md) · [下一节 9.7](../07-checkpoint-formats-and-quantized-weight-abi/README.md)

> **v1.0 教学计划，正文待编写。** 以下描述计划覆盖的概念、中文图解、代码和验收产物，不能据此认定已实现或已实测。

所属：Module B · Algorithms & Model Production。设备：CPU 检查；GPU 生产/加载。

基础材料可先阅读[量化导读](../guide.md)；已交付的打包实验位于 [9.8](../08-bit-packing-serialization-and-repack/lab.md)。

## 读完能解释什么

知道不同工具**为什么存在**，而不是只会敲命令。

统一从六个问题分析每个框架：

1. 解决什么问题？
2. 核心 abstraction 是什么？
3. 支持什么算法族？
4. 生成什么产物？
5. 谁来消费？
6. 最大限制是什么？

---

## 重点框架

### MQBench

```text
Observer
FakeQuant
Quantizer
Backend
```

### PPQ

```text
Graph
Dispatcher
Quantizer
Optimization Pass
Backend
```

### LightCompress

重点：

```text
algorithm implementation
block reconstruction
transformation
quantization
export
```

### AutoRound

重点：

```text
Algorithm
≠
Export Format
```

### LLM Compressor

重点：

```text
Recipe / Modifier
→ compressed-tensors
→ vLLM
```

### ModelOpt

重点：

```text
hardware-aware low precision
FP8 / FP4 / NVIDIA path
```

---

## 必做两条真实生产路径

### Path A

```text
LightCompress
    ↓
Quantized Checkpoint
    ↓
vLLM
```

### Path B

```text
LLM Compressor
    ↓
compressed-tensors
    ↓
vLLM
```

禁止默认：

```text
LightCompress
→ LLM Compressor
→ 再量一次
```

若发生格式转换，必须验证：

```text
q values
scale
zero-point
model output
```

是否保持。

## 完成证据

Framework Matrix：

```text
framework
algorithm
scheme
format
runtime
hardware
```

加至少两条真实 export 记录。

## 编排摘要

<!-- BEGIN lesson-plan -->

1. MQBench/PPQ：补充概念、推导、例子及实现边界。
2. LightCompress：补充概念、推导、例子及实现边界。
3. AutoRound：补充概念、推导、例子及实现边界。
4. LLM Compressor：补充概念、推导、例子及实现边界。
5. ModelOpt：补充概念、推导、例子及实现边界。
6. 算法/导出格式解耦：补充概念、推导、例子及实现边界。
7. 两条独立生产路径：补充概念、推导、例子及实现边界。

<!-- END lesson-plan -->

## 验收清单

<!-- BEGIN lesson-exercise -->

目标产出：**Framework Matrix，以及 LightCompress 与 LLM Compressor 各一条真实 export 记录**。

- [ ] 提供输入、命令、结果与解释。
- [ ] 补充参考答案和至少一个边界案例。

<!-- END lesson-exercise -->

完整正文按[统一写作模板](../templates/lesson.md)逐节交付；验收前保持教学骨架状态。
