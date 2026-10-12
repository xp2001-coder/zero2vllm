# 3.3 固定版本源码定位表

[正文](README.md) · [实验](lab.md) · [源码清单](data/source-manifest.json)

版本：vLLM v0.8.5 / V1；提交 `ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d`。

本表来自固定官方源码的语法树定位，覆盖 15 个文件、27 个函数和 2 类结构。源码中的调用存在，不代表设备上已经成功运行该分支；请结合条件、布局和配置阅读。

| 顺序 | 关注问题 | 函数与固定位置 |
| --- | --- | --- |
| 1 | Runner 发起模型加载 | [`GPUModelRunner.load_model`](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/v1/worker/gpu_model_runner.py#L1328) |
| 2 | 选择加载器并加载模型 | [`get_model`](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/model_executor/model_loader/__init__.py#L12) |
| 3 | 初始化、写入权重、后处理与 eval | [`DefaultModelLoader.load_model`](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/model_executor/model_loader/loader.py#L446) |
| 4 | 解析并实例化模型类 | [`_initialize_model`](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/model_executor/model_loader/loader.py#L114) |
| 5 | 配置中的架构名映射到注册类 | [`get_model_architecture`](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/model_executor/model_loader/utils.py#L87) |
| 6 | 按架构列表解析实现 | [`_ModelRegistry.resolve_model_cls`](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/model_executor/models/registry.py#L467) |
| 7 | 模型参数名到融合目标参数的映射 | [`LlamaModel.load_weights`](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/model_executor/models/llama.py#L381) |
| 8 | 根据 shard_id 写入 Q/K/V 对应区域 | [`QKVParallelLinear.weight_loader`](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/model_executor/layers/linear.py#L947) |
| 9 | 量化与注意力的加载后处理 | [`_process_weights_after_loading`](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/model_executor/model_loader/loader.py#L164) |
| 10 | Linear 的运行入口 | [`ColumnParallelLinear.forward`](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/model_executor/layers/linear.py#L478) |
| 11 | 未量化 Linear 的 GEMM 分派 | [`UnquantizedLinearMethod.apply`](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/model_executor/layers/linear.py#L197) |
| 12 | 非 ROCm 路径返回 PyTorch linear | [`dispatch_unquantized_gemm`](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/model_executor/layers/utils.py#L96) |
| 13 | 持久请求状态与本轮执行排列 | [`GPUModelRunner._update_states`](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/v1/worker/gpu_model_runner.py#L285) |
| 14 | 填补批次中的空位 | [`InputBatch.condense`](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/v1/worker/gpu_input_batch.py#L451) |
| 15 | token、位置、槽位与注意力元数据 | [`GPUModelRunner._prepare_inputs`](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/v1/worker/gpu_model_runner.py#L488) |
| 16 | 前向、选择 logits、采样和部分预填充过滤 | [`GPUModelRunner.execute_model`](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/v1/worker/gpu_model_runner.py#L1003) |
| 17 | 融合 QKV、RoPE、Attention 和输出投影 | [`LlamaAttention.forward`](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/model_executor/models/llama.py#L198) |
| 18 | 最终隐状态到词表 logits | [`LlamaForCausalLM.compute_logits`](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/model_executor/models/llama.py#L560) |
| 19 | 传入形状和 V1 条件 | [`get_attn_backend`](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/attention/selector.py#L81) |
| 20 | 全局覆盖、环境变量与平台选择 | [`_cached_get_attn_backend`](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/attention/selector.py#L108) |
| 21 | CUDA 平台的 V1 后端分支 | [`CudaPlatformBase.get_attn_backend_cls`](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/platforms/cuda.py#L176) |
| 22 | 按后端布局创建并绑定 KV 张量 | [`GPUModelRunner.initialize_kv_cache`](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/v1/worker/gpu_model_runner.py#L1689) |
| 23 | 本后端块大小约束与 KV 布局 | [`FlashAttentionBackend.get_kv_cache_shape`](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/v1/attention/backends/flash_attn.py#L57) |
| 24 | 构造本后端执行元数据 | [`FlashAttentionMetadataBuilder.build`](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/v1/attention/backends/flash_attn.py#L297) |
| 25 | 访问 forward context 并调用注意力入口 | [`Attention.forward`](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/attention/layer.py#L167) |
| 26 | 通过层名与上下文到达具体实现 | [`unified_attention_with_output`](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/attention/layer.py#L404) |
| 27 | 先写 KV，再调用变长分页 Attention | [`FlashAttentionImpl.forward`](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/v1/attention/backends/flash_attn.py#L481) |

## 元数据与返回值

- [`FlashAttentionMetadata`](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/v1/attention/backends/flash_attn.py#L69)：核对 `num_actual_tokens`、`query_start_loc`、`seq_lens`、`block_table`、`slot_mapping`、`num_input_tokens`。
- [`ModelRunnerOutput`](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/v1/outputs.py#L76)：核对 `req_ids`、`req_id_to_index`、`sampled_token_ids`。

## 按问题阅读，而不是记住文件名

1. **加载**：先跟随 `load_model → get_model → loader.load_model`，再分开看初始化、参数写入、后处理；模型注册成功只是第一步。
2. **布局**：从 Llama 的名称映射追到 `QKVParallelLinear.weight_loader`，画出 TP=1 的 Q/K/V 目标区域。再沿未量化 Linear 找到 PyTorch 的入口。
3. **打包**：先看 `_update_states` 如何保留请求身份、更新批次，再逐行核对 `_prepare_inputs` 的 token、位置、写入槽、Query 边界和 logits 行号。
4. **注意力**：选择器回答“用哪个后端”，shape 方法回答“缓存怎样解释”，builder 回答“这轮有哪些元数据”，impl 回答“先写什么、再读什么”。它们不是一个函数。
5. **输出**：在 `execute_model` 中找到真实 token 裁剪、隐藏状态选择、词表投影、采样、部分 Prefill 过滤及请求映射，解释两条请求为什么只有一条保留候选。

实验工具只下载清单列出的官方文件，不下载完整项目或模型，不导入和执行上游 Python。全文哈希按 LF 换行归一化，避免 Windows 换行造成假失败。调用与字段检查的范围以清单为准，不构成上游全部行为的测试。
