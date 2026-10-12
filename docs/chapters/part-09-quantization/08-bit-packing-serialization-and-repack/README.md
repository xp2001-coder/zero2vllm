# 9.8 位打包、序列化与 Repack

[本篇目录](../README.md) · [总设计](../design.md) · [Passport](../templates/model-passport.md) · [上一节 9.7](../07-checkpoint-formats-and-quantized-weight-abi/README.md) · [下一节 9.9](../09-weight-only-gemv-gemm-and-tuning/README.md)

> **v1.0 教学计划，正文待编写。** 以下描述计划覆盖的概念、中文图解、代码和验收产物，不能据此认定已实现或已实测。

所属：Module C · Format, Packing & Kernel。设备：CPU 必做；GPU 在 9.9 扩展。

已有[CPU INT4 打包与 Linear 实验](lab.md)及[原始结果](results/README.md)。现有实现不含下面计划扩展的通用 n-bit、序列化与 repack。

## 读完能解释什么

能够真正回答：

> 这些 INT4 数字怎样进入文件和显存？

## 内容

### INT4

```text
2 × INT4 → uint8
8 × INT4 → int32
```

### 非标准位宽

```text
INT3
INT5
INT6
INT7
```

比较：

```text
bit-stream
padding
block-wise word
```

不同选择。

禁止写成：

> “INT3 一定只能用某一种 bit-stream。”

---

## 关键概念

```text
Logical Quantized Tensor
        ↓
On-disk Packing
        ↓
Serialization
        ↓
Loader
        ↓
Repack
        ↓
Kernel-friendly Layout
```

必须区分：

```text
on-disk layout
≠
in-memory compute layout
```

## 实现

自己实现：

```python
pack_int4()
unpack_int4()

pack_generic_nbit()
unpack_generic_nbit()

serialize()
load()

repack()
inverse_repack()
```

覆盖：

```text
odd length
tail
non-aligned K
non-square matrix
endianness
```

## 实验

CPU 必做。

GPU 后续在 9.9 使用。

## 完成证据

Byte-level Golden Tests。

## 编排摘要

<!-- BEGIN lesson-plan -->

1. INT4 byte/word packing：补充概念、推导、例子及实现边界。
2. INT3/5/6/7 bit-stream/padding/block-word：补充概念、推导、例子及实现边界。
3. 序列化/读取：补充概念、推导、例子及实现边界。
4. repack/逆变换：补充概念、推导、例子及实现边界。
5. 字节序：补充概念、推导、例子及实现边界。
6. 奇数尾部与非对齐：补充概念、推导、例子及实现边界。

<!-- END lesson-plan -->

## 验收清单

<!-- BEGIN lesson-exercise -->

目标产出：**沿用现有 CPU INT4 实验；扩展 n-bit、序列化与 repack，提交字节级黄金测试**。

- [ ] 提供输入、命令、结果与解释。
- [ ] 补充参考答案和至少一个边界案例。

<!-- END lesson-exercise -->

完整正文按[统一写作模板](../templates/lesson.md)逐节交付；验收前保持教学骨架状态。
