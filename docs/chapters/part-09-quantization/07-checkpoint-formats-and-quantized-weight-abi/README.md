# 9.7 Checkpoint Format 与 Quantized Weight ABI

[本篇目录](../README.md) · [总设计](../design.md) · [Passport](../templates/model-passport.md) · [上一节 9.6](../06-quantization-frameworks-and-export/README.md) · [下一节 9.8](../08-bit-packing-serialization-and-repack/README.md)

> **v1.0 教学计划，正文待编写。** 以下描述计划覆盖的概念、中文图解、代码和验收产物，不能据此认定已实现或已实测。

所属：Module C · Format, Packing & Kernel。设备：CPU；真实 checkpoint。

基础材料可先阅读[量化导读](../guide.md)；已交付的打包实验位于 [9.8](../08-bit-packing-serialization-and-repack/lab.md)。

## 读完能解释什么

彻底理解：

> 数学上都是 INT4，不代表 checkpoint binary-compatible。

定义：

$$
\boxed{
\text{Quantized Weight ABI}
=
\text{packing layout}
+
\text{metadata convention}
+
\text{runtime/kernel contract}
}
$$

重点研究：

```text
qweight
qzeros
scales
g_idx
bias
```

分别：

```text
shape
dtype
axis
stride
meaning
```

## 重点案例

使用非方阵：

```text
W[K,N]
K ≠ N
```

比较 GPTQ-compatible 和 AWQ-compatible representation。

重点讲：

```text
pack axis
bit order
zero-point convention
scale shape
group mapping
g_idx
```

## 格式

重点：

```text
GPTQ
AWQ
compressed-tensors
GGUF
```

但明确：

```text
safetensors / bin
```

只是容器层，不等于 quantized weight ABI。

## 图

1. 同一 logical INT4 matrix 的两种物理 layout。
2. qweight / scales / qzeros / g_idx 关系。
3. Algorithm → same/different format。
4. Format → multiple kernels。

## 实现

读取真实 checkpoint：

```python
inspect_quantized_checkpoint.py
```

输出：

```text
tensor name
shape
dtype
stride
metadata
```

## 完成证据

至少比较两类真实量化 checkpoint。

## 编排摘要

<!-- BEGIN lesson-plan -->

1. GPTQ/AWQ/compressed-tensors/GGUF：补充概念、推导、例子及实现边界。
2. qweight/qzeros/scales/g_idx：补充概念、推导、例子及实现边界。
3. shape/dtype/axis/stride：补充概念、推导、例子及实现边界。
4. 容器：补充概念、推导、例子及实现边界。
5. 磁盘契约与执行契约：补充概念、推导、例子及实现边界。

<!-- END lesson-plan -->

## 验收清单

<!-- BEGIN lesson-exercise -->

目标产出：**真实 checkpoint 检查器与至少两种格式的非方阵布局/元数据分析**。

- [ ] 提供输入、命令、结果与解释。
- [ ] 补充参考答案和至少一个边界案例。

<!-- END lesson-exercise -->

完整正文按[统一写作模板](../templates/lesson.md)逐节交付；验收前保持教学骨架状态。
