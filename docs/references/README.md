# 参考资料与出处

查阅日期：2026-09-15。课程组织参考 zero-to-sglang；技术条目优先来自 vLLM 官方文档和源码。

## 来源与版本口径

初次特性盘点查阅了可变化的 `stable` 文档和 GitHub `main`，当时 API 返回的 HEAD 为 `00972dfd72988942138a7a6089eaee08580210b8`，仅作为查阅线索；没有逐文件证明网页缓存均对应这一提交。专题导读还查阅了 `latest` 设计文档和相关源码，具体来源随导读列出，不把它们统一声称为上述 SHA 的快照。

学习入口见[七篇专题导读](topic-guides.md)。各章参考资料使用读者可访问的论文、官方文档和项目源码；动态 API、硬件支持和性能数字需分别核对，GPU 运行版本仍待锁定。

因此本轮输出是**有日期和来源的课程覆盖清单**。正式源码导读须固定 tag/commit，实机记录还须固定镜像摘要、插件、模型和算子库。文档、注册表与实际运行可能处于不同更新阶段；冲突需明确记在章节里。

## 来源总表

<!-- BEGIN course-sources -->

| 索引 | 官方来源 / 项目 |
| --- | --- |
| `quickstart` | [vLLM Quickstart](https://docs.vllm.ai/en/stable/getting_started/quickstart/) |
| `gpu` | [GPU 安装与设备条件](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/) |
| `features` | [vLLM Features 与兼容性入口](https://docs.vllm.ai/en/stable/features/) |
| `models` | [模型支持与任务](https://docs.vllm.ai/en/stable/models/supported_models/) |
| `generation` | [Generative Models](https://docs.vllm.ai/en/stable/models/generative_models/) |
| `pooling` | [Pooling Models](https://docs.vllm.ai/en/stable/models/pooling_models/) |
| `architecture` | [架构设计](https://docs.vllm.ai/en/stable/design/arch_overview/) |
| `runner` | [Model Runner V2](https://docs.vllm.ai/en/stable/design/model_runner_v2/) |
| `hf` | [Hugging Face 集成](https://docs.vllm.ai/en/stable/design/huggingface_integration/) |
| `serve` | [OpenAI-compatible Server](https://docs.vllm.ai/en/stable/serving/online_serving/openai_compatible_server/) |
| `apc` | [Automatic Prefix Caching](https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/) |
| `prefix_design` | [Prefix Caching 设计](https://docs.vllm.ai/en/stable/design/prefix_caching/) |
| `hybrid` | [Hybrid KV Cache Manager](https://docs.vllm.ai/en/stable/design/hybrid_kv_cache_manager/) |
| `kv_offload` | [KV Offloading](https://docs.vllm.ai/en/stable/features/kv_offloading_usage/) |
| `context` | [Context Extension](https://docs.vllm.ai/en/stable/features/context_extension/) |
| `index` | [IndexCache](https://docs.vllm.ai/en/stable/features/index_cache/) |
| `optimization` | [Optimization and Tuning](https://docs.vllm.ai/en/stable/configuration/optimization/) |
| `dbo` | [Dual Batch Overlap](https://docs.vllm.ai/en/stable/design/dbo/) |
| `attention_support` | [Attention Backend 支持矩阵](https://docs.vllm.ai/en/stable/design/attention_backends/) |
| `customop` | [CustomOp](https://docs.vllm.ai/en/stable/design/custom_op/) |
| `compile` | [torch.compile 集成](https://docs.vllm.ai/en/stable/design/torch_compile/) |
| `graphs` | [CUDA Graphs](https://docs.vllm.ai/en/stable/design/cuda_graphs/) |
| `optimization_levels` | [Optimization Levels](https://docs.vllm.ai/en/stable/design/optimization_levels/) |
| `ir` | [vLLM IR](https://docs.vllm.ai/en/stable/design/vllm_ir/) |
| `vit_graphs` | [ViT CUDA Graphs](https://docs.vllm.ai/en/stable/design/cuda_graphs_multimodal/) |
| `quant` | [量化与硬件矩阵](https://docs.vllm.ai/en/stable/features/quantization/) |
| `online_quant` | [Online Quantization](https://docs.vllm.ai/en/stable/features/quantization/online/) |
| `compressor` | [LLM Compressor](https://docs.vllm.ai/en/stable/features/quantization/llm_compressor/) |
| `kv_quant` | [Quantized KV Cache](https://docs.vllm.ai/en/stable/features/quantization/quantized_kvcache/) |
| `spec` | [Speculative Decoding](https://docs.vllm.ai/en/stable/features/speculative_decoding/) |
| `eagle` | [EAGLE](https://docs.vllm.ai/en/stable/features/speculative_decoding/eagle/) |
| `mtp` | [MTP](https://docs.vllm.ai/en/stable/features/speculative_decoding/mtp/) |
| `spec_dynamic` | [Dynamic Speculative Decoding](https://docs.vllm.ai/en/stable/features/speculative_decoding/dynamic_speculative_decoding/) |
| `spec_adaptive` | [Adaptive Verification](https://docs.vllm.ai/en/stable/features/speculative_decoding/adaptive_verification/) |
| `parallel` | [Parallelism and Scaling](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/) |
| `cp` | [Context Parallel Deployment](https://docs.vllm.ai/en/stable/serving/context_parallel_deployment/) |
| `dp` | [Data Parallel Deployment](https://docs.vllm.ai/en/stable/serving/data_parallel_deployment/) |
| `ep` | [Expert Parallel Deployment](https://docs.vllm.ai/en/stable/serving/expert_parallel_deployment/) |
| `moe_design` | [Fused MoE Modular Kernel](https://docs.vllm.ai/en/stable/design/fused_moe_modular_kernel/) |
| `moe_features` | [MoE Kernel Features](https://docs.vllm.ai/en/stable/design/moe_kernel_features/) |
| `disagg` | [Disaggregated Prefilling](https://docs.vllm.ai/en/stable/features/disagg_prefill/) |
| `encoder_disagg` | [Disaggregated Encoder](https://docs.vllm.ai/en/stable/features/disagg_encoder/) |
| `nixl` | [NIXL 兼容性矩阵](https://docs.vllm.ai/en/stable/features/nixl_connector_compatibility/) |
| `multimodal` | [Multimodal Inputs](https://docs.vllm.ai/en/stable/features/multimodal_inputs/) |
| `lora` | [LoRA Adapters](https://docs.vllm.ai/en/stable/features/lora/) |
| `prompt_embed` | [Prompt Embedding Inputs](https://docs.vllm.ai/en/stable/features/prompt_embeds/) |
| `structured` | [Structured Outputs](https://docs.vllm.ai/en/stable/features/structured_outputs/) |
| `tools` | [Tool Calling](https://docs.vllm.ai/en/stable/features/tool_calling/) |
| `reasoning` | [Reasoning Outputs](https://docs.vllm.ai/en/stable/features/reasoning_outputs/) |
| `thinking` | [Interleaved Thinking](https://docs.vllm.ai/en/stable/features/interleaved_thinking/) |
| `custom_logits` | [Custom Logits Processors](https://docs.vllm.ai/en/stable/features/custom_logitsprocs/) |
| `load` | [Model Resolution](https://docs.vllm.ai/en/stable/configuration/model_resolution/) |
| `memory` | [Conserving Memory](https://docs.vllm.ai/en/stable/configuration/conserving_memory/) |
| `sleep` | [Sleep Mode](https://docs.vllm.ai/en/stable/features/sleep_mode/) |
| `weight_transfer` | [Weight Transfer](https://docs.vllm.ai/en/stable/training/weight_transfer/) |
| `async_rl` | [Async Reinforcement Learning](https://docs.vllm.ai/en/stable/training/async_rl/) |
| `layerwise` | [Layerwise Reloading](https://docs.vllm.ai/en/stable/training/layerwise/) |
| `bench` | [Benchmark CLI](https://docs.vllm.ai/en/stable/benchmarking/cli/) |
| `sweep` | [Parameter Sweeps](https://docs.vllm.ai/en/stable/benchmarking/sweeps/) |
| `profiling` | [Profiling vLLM](https://docs.vllm.ai/en/stable/contributing/profiling/) |
| `metrics` | [Production Metrics](https://docs.vllm.ai/en/stable/usage/metrics/) |
| `per_request` | [Per-Request Metrics](https://docs.vllm.ai/en/stable/features/per_request_metrics/) |
| `batch_invariance` | [Batch Invariance](https://docs.vllm.ai/en/stable/features/batch_invariance/) |
| `plugin` | [Plugin System](https://docs.vllm.ai/en/stable/design/plugin_system/) |
| `deployment` | [Kubernetes 部署](https://docs.vllm.ai/en/stable/deployment/k8s/) |
| `cache_config` | [缓存配置与 dtype 清单](https://github.com/vllm-project/vllm/blob/main/vllm/config/cache.py) |
| `scheduler` | [调度配置](https://github.com/vllm-project/vllm/blob/main/vllm/config/scheduler.py) |
| `parallel_config` | [并行配置与度数校验](https://github.com/vllm-project/vllm/blob/main/vllm/config/parallel.py) |
| `compile_config` | [编译模式与融合 Pass](https://github.com/vllm-project/vllm/blob/main/vllm/config/compilation.py) |
| `kernel` | [Linear/MoE Kernel 配置](https://github.com/vllm-project/vllm/blob/main/vllm/config/kernel.py) |
| `spec_config` | [推测方法与模型类型](https://github.com/vllm-project/vllm/blob/main/vllm/config/speculative.py) |
| `quant_registry` | [量化方法注册表](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/__init__.py) |
| `attention_registry` | [Attention 后端注册表](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/registry.py) |
| `kv_factory` | [KV Connector 注册表](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/kv_transfer/kv_connector/factory.py) |
| `course` | [Datawhale zero-to-sglang](https://github.com/datawhalechina/zero-to-sglang) |
| `transformers` | [Transformers](https://github.com/huggingface/transformers) |
| `nano` | [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) |
| `mini` | [mini-sglang](https://github.com/sgl-project/mini-sglang) |
| `flashattention` | [FlashAttention 官方实现与版本](https://github.com/Dao-AILab/flash-attention) |
| `sglang_tbo` | [SGLang EP：TBO/SBO](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/expert_parallelism.mdx) |
| `metax` | [vLLM-MetaX](https://github.com/MetaX-MACA/vLLM-metax) |
| `metax_install` | [MetaX 安装](https://vllm-metax.readthedocs.io/en/stable/getting_started/installation/maca.html) |
| `metax_quick` | [MetaX Quickstart](https://vllm-metax.readthedocs.io/en/stable/getting_started/quickstart.html) |
| `metax_models` | [MetaX 模型列表](https://vllm-metax.readthedocs.io/en/stable/models/supported_models.html) |
| `diffusion` | [离散扩散语言模型配置](https://github.com/vllm-project/vllm/blob/main/vllm/config/diffusion.py) |
| `quant_base` | [vLLM 量化方法与配置接口](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/base_config.py) |
| `triton_matmul` | [Triton 矩阵乘教程](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html) |
| `marlin_kernel` | [Marlin 量化内核实现](https://github.com/IST-DASLab/marlin) |
| `triton_attention` | [Triton Fused Attention 教程](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html) |
| `sglang_tuning` | [SGLang Hyperparameter Tuning](https://docs.sglang.io/docs/advanced_features/hyperparameter_tuning) |

<!-- END course-sources -->

## 使用方式与致谢

目录、练习和说明独立编写；注册项名称用于定位代码，未复制上游实现。框架比较参考 [SGLang](https://github.com/sgl-project/sglang)、[llama.cpp](https://github.com/ggml-org/llama.cpp)、[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)。

zero-to-sglang 的 README 声明 CC BY-NC-SA 4.0。本项目原创内容同样采用 [CC BY-NC-SA 4.0](../../LICENSE)。第三方文字、图片和代码保留各自的来源与许可。

注册表核对范围与更新步骤见 [特性与优化地图](feature-map.md)；本地代码和 GPU 验证情况见 [验证记录](../../reports/history/validation-before-restructure.md)。
