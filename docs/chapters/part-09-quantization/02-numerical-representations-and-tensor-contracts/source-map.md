# 9.2 源码与规范定位

[正文](README.md) · [实验](lab.md)

核验范围：2026-09-20 阅读公开规范，并检查本地安装的 PyTorch **2.7.1+cpu** 对应函数与 CPU 位模式解码。以下链接可由 GitHub 读者独立访问。没有执行 vLLM、CUDA、MACA 或模型导出；本节不查询浮动版本的后端支持表。

## 一、课程实现

| 文件 / 入口 | 输入到输出 | 应先读什么 |
| --- | --- | --- |
| [tensor_contract.py](../../../../zero2vllm/quantization/tensor_contract.py) `_geometry` | 逻辑 shape/axis/G → 物理与参数 shape | None、换轴、ceil 与尾组 |
| 同文件 `_quantize` | FP32 数值 → q/s/z | 真实元素统计、scale cast、z、Q、q_pad=z |
| 同文件 `dequantize` | 契约 → float64 矩阵 | 参数索引与是否返回 padding |
| 同文件 `calculate_effective_bpw` | 契约+存储假设 → 位数账单 | 分开的 q/s/z 字节取整与最终对齐 |
| [minifloat.py](../../../../zero2vllm/quantization/minifloat.py) | code↔有限浮点格点；块解码 | bias、subnormal、special、scale 格式 |
| [lesson0902.py](../../../../scripts/experiments/lesson0902.py) | 固定输入 → 结果与 Passport | golden、独立 oracle、验证范围 |

## 二、参数生成的真实实现

入口：[PyTorch v2.7.1 observer.py](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/ao/quantization/observer.py)，类 `UniformQuantizationObserverBase` 的 `_calculate_qparams()`。

本轮阅读安装版本的函数体确认：整数 affine 分支把区间扩展至包含零，scale 有 epsilon 下限，zero-point 通过舍入和 clamp 生成；对称分支依据 `quant_max-quant_min` 计算分母，qscheme/dtype 会影响 zero-point。`per_channel_affine_float_qparams` 是另一个分支，不能与本课整数 z 混用。

对应差异：本课对称采用明确的窄范围，全零 scale=1，过小存储 scale 报错，且在 Q 前做声明 dtype 的 scale 舍入。因此我们只比较职责和约定，没有声称逐位复制 PyTorch 的全部 Observer 行为。

## 三、Q/DQ 和广播

- [QuantizeLinear 21](https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html#quantizelinear-21)：舍入、饱和、scale/zero-point。
- [DequantizeLinear 21](https://onnx.ai/onnx/operators/onnx__DequantizeLinear.html#dequantizelinear-21)：per-tensor、per-axis、blocked 参数维度。

重点读 axis 的定义，不只看公式。ONNX per-axis 的一维参数随指定维度变化；本课 axis 表示对二维矩阵的哪一轴分组。它们需要按广播契约转换，而非按同名参数直接复制。

## 四、元素格式与独立对照

- [ONNX Float8](https://onnx.ai/onnx/technical/float8.html)：E4M3FN 与 E5M2 的偏置、有限范围、特殊编码；不要忽略 FN/FNUZ 后缀。
- [ONNX Float4](https://onnx.ai/onnx/technical/float4.html)：E2M1 的完整值表。

`check_torch_float8()` 使用安装版本的 `torch.float8_e4m3fn` 与 `torch.float8_e5m2`，逐个比较 256 个 UINT8 位模式的解释。NaN 按分类比较，Inf 按符号与值比较，零额外比较符号。它既不是低精度矩阵乘法，也不是编码器全部舍入边界的外部认证；本课编码器另用有限格点 round-trip、中点与饱和测试验证。

## 五、块缩放规范

- [OCP Microscaling Formats v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)：§5 的数据类型组合，§6.2 的元素，§6.3 的 E8M0 scale。
- [NVIDIA NVFP4 格式介绍](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)：16 元素块与两级缩放。

课程图和 `decode_scaled_block()` 表达数值乘数关系，不是二进制文件布局。OCP 格式定义不等于完整量化算法；NV 的直接 DQ scale 与某些 API 的 inverse scale 也要明确区分。后续学习格式与 kernel 时，继续补参数布局和实际调用证据。
