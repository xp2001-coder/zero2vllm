# 9.12 MoE、并行切分与量化

[本篇目录](../README.md) · [总设计](../design.md) · [Passport](../templates/model-passport.md) · [上一节 9.11](../11-kv-cache-quantization-and-attention/README.md) · [下一节 9.13](../13-integrating-quantization-into-vllm/README.md)

> **v1.0 教学计划，正文待编写。** 以下描述计划覆盖的概念、中文图解、代码和验收产物，不能据此认定已实现或已实测。

所属：Module D · Runtime State & Systems。设备：CPU；单卡；多卡扩展。

基础材料可先阅读[量化导读](../guide.md)；已交付的打包实验位于 [9.8](../08-bit-packing-serialization-and-repack/lab.md)。

## 读完能解释什么

理解量化在 distributed runtime 中如何变化。

### TP

研究：

```text
qweight shard
scale shard
zero-point shard
group boundary
g_idx
```

### Fused QKV

研究：

> 三个 projection 能否使用不同精度和不同 quantization metadata？

### MoE

链路：

```text
Router
 ↓
Top-k
 ↓
Dispatch
 ↓
Expert
 ↓
Quantized Grouped GEMM
 ↓
Combine
```

同时研究：

```text
hot expert
cold expert
expert sensitivity
token imbalance
mixed precision
```

### EP

进一步研究：

```text
metadata placement
communication
expert shard
grouped compute
```

## 实验分层

### CPU

切分 / 重组 / metadata consistency。

### 单卡

Grouped GEMM / expert simulation。

### 多卡

真正 TP/EP 通信性能列为扩展验收。

## 完成证据

Distributed Quantization Contract + MoE Case Report。

### 模型边界

Qwen3-8B 是稠密主线案例；专家实验另用明确的 MoE 夹具或模型。CPU/单卡切分模拟不能标记为多卡通信实测。

## 编排摘要

<!-- BEGIN lesson-plan -->

1. TP qweight/scale/zp/g_idx 切分与 group boundary：补充概念、推导、例子及实现边界。
2. fused QKV：补充概念、推导、例子及实现边界。
3. 专家敏感度/负载：补充概念、推导、例子及实现边界。
4. Router/Dispatch/Grouped GEMM/Combine：补充概念、推导、例子及实现边界。
5. EP：补充概念、推导、例子及实现边界。

<!-- END lesson-plan -->

## 验收清单

<!-- BEGIN lesson-exercise -->

目标产出：**Distributed Quantization Contract 与 MoE 案例；CPU 切分重组、单卡计算、多卡扩展分开验收**。

- [ ] 提供输入、命令、结果与解释。
- [ ] 补充参考答案和至少一个边界案例。

<!-- END lesson-exercise -->

完整正文按[统一写作模板](../templates/lesson.md)逐节交付；验收前保持教学骨架状态。
