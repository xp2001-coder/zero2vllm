# 9.11 KV Cache 量化与 Attention 协同

[本篇目录](../README.md) · [总设计](../design.md) · [Passport](../templates/model-passport.md) · [上一节 9.10](../10-activation-quantization-low-precision-gemm-and-fusion/README.md) · [下一节 9.12](../12-moe-parallel-sharding-and-quantization/README.md)

> **v1.0 教学计划，正文待编写。** 以下描述计划覆盖的概念、中文图解、代码和验收产物，不能据此认定已实现或已实测。

所属：Module D · Runtime State & Systems。设备：CPU 参考；GPU/专项。

基础材料可先阅读[量化导读](../guide.md)；已交付的打包实验位于 [9.8](../08-bit-packing-serialization-and-repack/lab.md)。

## 读完能解释什么

理解：

```text
Weight = static parameter

KV = dynamic runtime state
```

KV 每一步不断增加：

$$
K_t,V_t
$$

因此量化问题包括：

```text
write Q
scale management
cache storage
Attention read
DQ
long-context error
```

## 内容

比较：

```text
per-tensor
per-head
per-token
```

以及：

```text
FP8
INT8
INT4 / FP4
```

算法案例：

```text
KIVI
KVQuant
MiKV
```

## 实验

测：

```text
KV memory
max context
max concurrency
quantization overhead
Attention latency
quality
long-context error
```

Prefill / Decode 分开。

## 完成证据

Quantized KV Cache Report。

### 状态边界

逐步增长先采用无淘汰的自回归缓存例子；滑动窗口、前缀共享、回收与压缩带来的容量变化需单独说明。

## 编排摘要

<!-- BEGIN lesson-plan -->

1. 动态状态：补充概念、推导、例子及实现边界。
2. 写入 Q/读取 DQ：补充概念、推导、例子及实现边界。
3. scale 生命周期：补充概念、推导、例子及实现边界。
4. per-tensor/head/token：补充概念、推导、例子及实现边界。
5. FP8/INT8/INT4/FP4：补充概念、推导、例子及实现边界。
6. KIVI/KVQuant/MiKV：补充概念、推导、例子及实现边界。
7. 长上下文质量：补充概念、推导、例子及实现边界。
8. KV量化后的线上字节/QDQ成本与PD布局契约，衔接13.5/13.9：补充概念、推导、例子及实现边界。

<!-- END lesson-plan -->

## 验收清单

<!-- BEGIN lesson-exercise -->

目标产出：**KV 容量/并发/写读开销/Attention 延迟/质量报告，prefill 与 decode 分开**。

- [ ] 提供输入、命令、结果与解释。
- [ ] 补充参考答案和至少一个边界案例。

<!-- END lesson-exercise -->

完整正文按[统一写作模板](../templates/lesson.md)逐节交付；验收前保持教学骨架状态。


## 与分离式服务的衔接

本节的相关结论将接入 [Part 13：分布式推理工程](../../part-13-disaggregation/README.md)。KV量化后的线上字节/QDQ成本与PD布局契约，衔接13.5/13.9。先明确计量单位和验证范围，再用于集群推算。
