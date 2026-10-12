# 9.1 公开源码与文档入口

[返回正文](README.md) · [实验](lab.md)

本节使用 vLLM v0.8.5 的固定提交理解接口职责，与第三篇的入门源码版本衔接。这个历史切片用于教学，不是当前版本功能或 Qwen3-8B 支持矩阵。后续真实部署会重新锁定兼容的软件组合。

## vLLM 的九个阅读点

固定提交：`ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d`。可选静态核验覆盖 4 个文件（含 LICENSE）、9 个函数，检查文件摘要、函数存在性与指定调用；不导入或执行 vLLM。

| 阅读点 | 问题 |
| --- | --- |
| [QuantizationConfig.get_quant_method](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/model_executor/layers/quantization/base_config.py#L131) | 配置对象要为具体层返回量化方法 |
| [QuantizeMethodBase.process_weights_after_loading](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/model_executor/layers/quantization/base_config.py#L37) | 加载后处理是接口职责，不是每次前向都必须重排 |
| [LinearBase.__init__](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/model_executor/layers/linear.py#L218) | 层构造时获得 quant_method |
| [ReplicatedLinear.__init__](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/model_executor/layers/linear.py#L266) | 按量化方法创建参数；ReplicatedLinear 作为最小接口示例 |
| [ReplicatedLinear.forward](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/model_executor/layers/linear.py#L327) | 前向把输入与 bias 交给量化方法 |
| [GPTQMarlinConfig.get_quant_method](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/model_executor/layers/quantization/gptq_marlin.py#L153) | 配置、层类型与方法选择衔接 |
| [GPTQMarlinLinearMethod.create_weights](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/model_executor/layers/quantization/gptq_marlin.py#L210) | 创建 qweight/scales 等参数，并选择执行 kernel 类 |
| [GPTQMarlinLinearMethod.process_weights_after_loading](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/model_executor/layers/quantization/gptq_marlin.py#L335) | 把加载后处理委托给已选 kernel |
| [GPTQMarlinLinearMethod.apply](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/model_executor/layers/quantization/gptq_marlin.py#L338) | 调用已选实现，不从类名猜测实际执行路径 |

注意：这里沿接口把职责接起来，没有声称一次真实请求必然调用所有示例类。完整 loader 及具体模型走哪种 Linear，留到 9.13 结合运行证据追踪。

在这个固定版本中，`GPTQMarlinLinearMethod.create_weights()` 中存在 `choose_mp_linear_kernel()` 调用；随后保存 `self.kernel`，加载后处理与前向再委托给它。类名、配置名与最终执行实现必须分开记录。

## AutoRound：算法和导出格式分开记录

阅读 [AutoRound 固定 README](https://github.com/intel/auto-round/blob/2b5a3c5506a27838d63b1a8d82be5d8b5f7d61cc/README.md)，提交 `2b5a3c5506a27838d63b1a8d82be5d8b5f7d61cc`。该文档的导出选项列出了 `auto_gptq`、`auto_awq` 等名称；这说明工具中的算法选择与格式选择应分开记录。

本节只核对文档声明，没有安装此版本或生成 checkpoint；不同 scheme、格式和运行时组合仍须逐项验证。生产实验在 9.6 展开，不把此处的文档阅读记为模型导出成功。

## 数学与接口背景

- [PyTorch 2.7 Linear](https://docs.pytorch.org/docs/2.7/generated/torch.nn.Linear.html)：官方权重存放为 `[out_features,in_features]`，前向使用其转置；本课手算采用 `W[K,N]`。
- [GPTQ 原论文](https://arxiv.org/abs/2210.17323)：算法层的阅读入口，详细推导安排在 9.4。
- [AWQ 原论文](https://arxiv.org/abs/2306.00978)：算法和激活感知变换的阅读入口，同样归入 9.4。
- [vLLM FP8 文档](https://docs.vllm.ai/en/latest/features/quantization/llm_compressor/fp8/)：辅助检查 weight-only fallback 与原生 W8A8 的区别；这是动态页面，硬件实验须保存所选版本记录。

固定源码清单见 [data/source-manifest.json](data/source-manifest.json)。公开仓库只提供摘要和定位；显式获取的上游源文件保留在本地缓存，不重新打包分发。
