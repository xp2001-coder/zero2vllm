# 9.13 量化路径接入 vLLM

[本篇目录](../README.md) · [总设计](../design.md) · [Passport](../templates/model-passport.md) · [上一节 9.12](../12-moe-parallel-sharding-and-quantization/README.md) · [下一节 9.14](../14-from-kernel-gains-to-end-to-end-acceleration/README.md)

> **v1.0 教学计划，正文待编写。** 以下描述计划覆盖的概念、中文图解、代码和验收产物，不能据此认定已实现或已实测。

所属：Module D · Runtime State & Systems。设备：GPU/双平台；CPU 静态检查。

基础材料可先阅读[量化导读](../guide.md)；已交付的打包实验位于 [9.8](../08-bit-packing-serialization-and-repack/lab.md)。

## 读完能解释什么

能够从 checkpoint 一直追踪到实际 Kernel。

计划追踪的职责主线（具体调用时机按固定版本核验）：

```text
HF Model / Config
      ↓
quantization_config
      ↓
QuantizationConfig
      ↓
get_quant_method() → QuantMethod
      ↓
create_weights()
      ↓
load_weights()
      ↓
process_weights_after_loading()
  └─ repack / 初始化执行状态（按后端）
      ↓
LinearMethod.apply()
      ↓
调用实际 backend / kernel
```

Backend selection 单独追踪：可能在方法构造、加载后处理或执行阶段发生。它与权重生命周期有关，但不能统一画成每次 `apply()` 之后才开始选择。

## 核心要求

不能只回答：

> vLLM 支持 GPTQ。

必须回答：

```text
支持哪种 checkpoint？
怎样解析？
是否 repack？
使用什么 QuantLinear？
最终哪个 backend？
fallback 是什么？
```

## Runtime 对照

主线：

```text
vLLM
```

对照：

```text
SGLang
TensorRT-LLM
FastLLM / llama.cpp
```

但其他 Runtime 不要求和 vLLM 同等源码深度。

## 实验

完成：

```text
Layer
↓
Model
↓
Offline inference
↓
Online serving
```

四级验收。

## 完成证据

Runtime Trace：

```text
config
→ loader
→ tensor
→ repack
→ QuantMethod
→ backend
→ kernel
```

并附日志 / 源码 / profiler 证据。

### 源码追踪边界

上图是职责索引，不是所有版本都遵循的严格调用时序。需补查 `get_quant_method()`；QuantMethod 通常参与权重创建，backend 也可能在初始化或加载后确定。实际顺序以固定版本的源码与日志为准。

## 编排摘要

<!-- BEGIN lesson-plan -->

1. quantization_config/QuantizationConfig/get_quant_method/create_weights/load/process_weights_after_loading/apply：补充概念、推导、例子及实现边界。
2. repack/dispatch/回退：补充概念、推导、例子及实现边界。
3. 固定源码追踪：补充概念、推导、例子及实现边界。

<!-- END lesson-plan -->

## 验收清单

<!-- BEGIN lesson-exercise -->

目标产出：**层→模型→离线→在线四级记录，以及 config→loader→repack→backend→kernel 的证据链**。

- [ ] 提供输入、命令、结果与解释。
- [ ] 补充参考答案和至少一个边界案例。

<!-- END lesson-exercise -->

完整正文按[统一写作模板](../templates/lesson.md)逐节交付；验收前保持教学骨架状态。
