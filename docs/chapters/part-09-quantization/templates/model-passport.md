# Quantized Model Passport · v1.0 字段规范

## A. 来源与算法

```text
Model:
Model revision:
Tokenizer revision:

Quantization framework:
Framework revision:

Algorithm:
Calibration dataset:
Calibration samples:
Token budget:
Sequence length:
Seed:
```

## B. 数值契约

```text
Weight representation:
Activation representation:
KV representation:

Sym / Asym:
Group size:
Group axis:

Scale dtype:
Scale granularity:

Zero-point:
Rounding:
Clipping:
Saturation:
```

## C. 存储与执行

```text
Checkpoint format:
Format version:

qweight shape:
qweight dtype:
scale shape:
zero-point shape:
g_idx:

Packing:
On-disk layout:

Repack:
In-memory layout:

Activation dtype:
Multiply dtype:
Accumulator dtype:
Output dtype:

Runtime:
QuantMethod:
Linear backend:
Actual kernel:
```

## D. 验证证据

```text
Hardware:
Driver:
Runtime:
Framework:

Correctness:
PPL / Accuracy:
Model memory:
KV memory:

Prefill:
Decode:
TTFT:
TPOT:
Throughput:
Goodput:

Evidence:
- config
- checkpoint inspection
- runtime log
- source trace
- profiler
```

允许：

```text
Unknown / 未验证
```

但禁止通过模型名称猜测真实 Kernel、accumulator 或硬件路径。

## 填写与证据规则

- 模板版本：`1.0`；每份记录补充实验 ID、记录日期、对应章节与验证范围。
- A—D 每个字段分别填写“配置/预期”“实际观察”“证据位置”；未确定填 `Unknown / 未验证`，不适用填“不适用 + 原因”。
- `Runtime` 在 C 表示推理引擎及版本；D 另外记录 CUDA/MACA、驱动、编译器与算子库版本，避免同名字段混淆。
- 存储和执行字段对每一种张量/层分别记录。`qzeros`、`g_idx` 等是格式相关字段，不要求所有 checkpoint 都有这些名称。
- 记录 checkpoint 容器与格式版本、文件摘要；内存 stride 是加载后观察值，不能仅从磁盘形状猜测。
- 补充预处理、BOS/EOS、截断/padding 与独立评测集；图变换、高精度保留层和 calibration 成本也属于算法对照条件。
- 存储报告计入 scales、zero-points、group 映射、padding 和其他元数据；区分逻辑 payload、文件大小和实际显存。
- 性能证据记录 M/N/K、并发、输入/输出长度、预热、计时区间、统计口径、SLO；推测的 kernel 不填作实测。
- 同一配置在 A100 与 C500 上分别生成记录，不把一个平台的证据复制成另一个平台的结论。

| 字段 | 配置 / 预期 | 实际观察 | 证据位置 |
| --- | --- | --- | --- |
| Actual kernel | Unknown | 未验证 | 未提供 |
| Accumulator dtype | Unknown | 未验证 | 未提供 |
| TTFT / TPOT | 待定义负载 | 未验证 | 未提供 |

本页是字段规范与填写模板；[9.1](../01-quantization-system-overview/README.md) 已提供 [QuantizedModelPassport](../../../../zero2vllm/quantization/passport.py) 的第一版程序实现与两份示例记录。它校验字段结构和证据引用是否提供，不自动证明引用内容真实。新增实验要逐项核查观察值。

[总设计](../design.md) · [本篇目录](../README.md)
