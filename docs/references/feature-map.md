# vLLM 特性、优化与并行策略地图

本表用于检查课程覆盖范围、查找对应章节和设计实验。查阅日期：**2026-09-15**。

**阅读范围覆盖官方公开功能与设计，实机主线是单卡 A100、单卡 C500。** 正式章节共 19 篇、129 章；当前 GPU 功能均为待验证。先通过下表定位章节，再从 [课程目录](../chapters/README.md) 查看练习，从 [来源总表](README.md) 进入官方资料。

## 1. 覆盖口径与状态

- **课程收录**：正文应该讲什么，包含基础、进阶、实验和源码入口。
- **上游证据**：官方功能文档、设计文档、源码注册项、历史接口或外部扩展。这些证据的成熟度不同。
- **实机状态**：A100 和 C500 各自记录 `not_run / passed / failed / unsupported`，附版本、模型、算子与运行证据。
- **组合状态**：单个功能可用，不代表两个功能同时开启可用；组合需要独立验证。

本次来源为动态 `stable` 文档与 `main` 源码。查阅时上游 HEAD 的线索及限制保存在 [课程目录 JSON](../../metadata/course/catalog.json)。本文件没有将所有网页内容认定为同一 commit，也没有把 main 的新功能认定为已发布稳定功能。

## 2. 全域覆盖索引

| 技术域 | 必须讲到的内容 | 章节 |
| --- | --- | --- |
| 模型与生成基础 | Tokenizer、模板、RoPE/ALiBi、MHA/MQA/GQA/MLA、采样、停止、数值检查 | 1.1—1.6、2.1—2.2 |
| 引擎与生命周期 | 请求状态、EngineCore、Scheduler、Executor、Worker、ModelRunner/V2、V1 演进 | 2.3—2.7、3.1—3.5 |
| KV 与 APC | 分页/块表/布局、Prefix Cache、哈希/盐/隔离、逐出、reset、KV events、亲和路由 | 5.1—5.4 |
| 混合缓存与长上下文 | SWA/SSM/HMA、Mamba/GDN、状态检查点、KV sharing、RoPE scaling/YaRN、IndexCache | 5.5—5.9 |
| 调度与批处理 | 连续批处理、FCFS/priority、Chunked Prefill、token/显存预算、抢占、异步调度 | 6.1—6.4 |
| 重叠执行 | DBO、ubatch、TBO/SBO 对照、shared expert overlap、Async TP、通信融合 | 6.5—6.6、8.5、12.6 |
| Attention | FlashAttention 1/2/3/4、FlashInfer、Triton、SDPA/Flex、paged/varlen、cascade、MLA/稀疏 | 7.1—7.5 |
| Kernel | GEMM/GEMV、RMSNorm/RoPE/激活融合、Marlin/Machete/CUTLASS/DeepGEMM/Humming/CuTeDSL | 7.6—7.7、9.8—9.10、9.12—9.13、12.3 |
| 编译与图 | torch.compile、动态形状/特化、CUDA Graph 全模式、编译缓存、JIT、融合 Pass、vLLM IR | 8.1—8.7 |
| 量化 | 算法统一比较、策略/混合精度、Q/DQ/打包、量化 GEMV/GEMM、融合/访存/流水线、KV/MoE 算子、vLLM 接入、端到端验证 | 9.1—9.14 |
| 推测解码 | N-gram/Suffix/Draft/EAGLE/MTP/MLP/Medusa/PARD/DFlash/DSpark、动态推测、拒绝采样 | 10.1—10.9 |
| 并行与拓扑 | TP/PP/DP/EP/CP/PCP/DCP/SP、进程组、collectives、组合度数、mp/Ray/多节点 | 11.1—11.10 |
| MoE | 路由/专家/分发/汇总、Fused MoE、DeepEP/MoRI/NIXL、EPLB、弹性扩缩、负载均衡 | 12.1—12.6 |
| 分离式服务 | PD、KV Connector、NIXL/Mooncake/LMCache/FlexKV/HF3FS、异构并行、EC/Encoder 分离 | 13.1—13.6 |
| 模型能力 | 生成/评分、Pooling、Embedding、token 任务、分类/reward/rerank、encoder-decoder、离散扩散 dLLM | 14.1—14.3 |
| 多模态与适配 | 图像/视频/音频、processor/encoder cache、ViT、LoRA、多租户、prompt embeds、I/O | 14.4—14.7 |
| API 与约束生成 | 离线/在线/Batch/Responses/音频、Structured Outputs、Tool Calling、Reasoning、Thinking | 15.1—15.7 |
| 加载与内存 | 模型解析、分片权重、流式加载、weight offload、KV offload、Sleep/Wake、Pause/Resume | 16.1—16.4 |
| RL 推理生命周期 | 权重传输/逐层 reload、IPC/NCCL/RDT、异步 rollout、distribution replay、训练器集成 | 16.5—16.6 |
| 评测与运维 | bench/sweep、SLO/Goodput、trace replay、profiling、Prometheus/OTel、Batch Invariance | 17.1—17.6 |
| 平台与扩展 | CUDA/MACA、CPU/ROCm/XPU/TPU、模型/算子/量化插件、部署生态、废弃迁移与贡献 | 18.1—18.5 |

量化篇固定四个 Module、14 节：9.4 统一比较算法，9.6 讲框架与生产，9.7/9.8 讲格式与重排，9.9/9.10 讲算子，9.13/9.14 讲接入与服务验收。格式/工具/内核到新章节的映射、贯穿实验与双重误差检查见 [量化篇设计](../chapters/part-09-quantization/design.md)。

## 3. 先讲清楚几个容易混淆的概念

### 3.1 APC、PagedAttention、FlashAttention 与 IndexCache

| 名称 | 优化的对象 | 要观察的结果 |
| --- | --- | --- |
| PagedAttention / 分页 KV | KV 存储组织与注意力访问 | 内存浪费、块管理和 kernel 寻址 |
| APC / Prefix Cache | 已计算的相同前缀 | prefill 工作量与首 token 延迟；热缓存必须单独测 |
| FlashAttention | Attention 的计算与数据搬运 | HBM 访问、中间矩阵、kernel 时间；不是跨请求缓存策略 |
| IndexCache | 特定 DSA 模型的 top-k 索引计算 | 索引复用频率、执行开销和模型质量 |
| KV Offloading | KV 存储层级 | CPU/GPU 传输与重算代价 |
| Weight Offloading | 模型权重驻留位置 | 每轮搬运、加载与执行开销；不等同于 KV 卸载 |

APC 的匹配要考虑 token、模型相关输入、LoRA/多模态标识和隔离需求。物理页大小与前缀匹配粒度也可能不同，不能把“未命中”都归因于文本不同。[APC 设计](https://docs.vllm.ai/en/stable/design/prefix_caching/) · [缓存配置](https://github.com/vllm-project/vllm/blob/main/vllm/config/cache.py)

FlashAttention 与分页 KV 可以在同一执行路径协作；具体组合看后端支持。IndexCache 的本次文档针对 DeepSeek-V3.2/兼容 DSA 模型，收录它时需要单独做质量检查。[Attention 支持矩阵](https://docs.vllm.ai/en/stable/design/attention_backends/) · [IndexCache](https://docs.vllm.ai/en/stable/features/index_cache/)

### 3.2 CP、PCP、DCP 与 SP

**本文 CP 专指 Context Parallelism（上下文并行）。** 部分 vLLM 兼容性表中的 CP 指 **Chunked Prefill（分块预填充）**；课程写出全称，防止把单卡分块误认为多卡切分。

| 策略 | 切分什么 | 主要解决什么 | 实验与配置入口 |
| --- | --- | --- | --- |
| TP | 单层权重/张量维度、Attention heads、词表等 | 模型容量与单层计算分摊 | `tensor_parallel_size`；多卡 |
| PP | 模型层/阶段 | 容量与流水线分工 | `pipeline_parallel_size`；多卡 |
| DP | 请求/副本工作量 | 服务吞吐扩展 | `data_parallel_size`；多卡；MoE 组关系另看 EP |
| EP | MoE 专家 | 专家权重分布与 token 路由 | `enable_expert_parallel`；复用相关 ranks |
| CP | 上下文序列维度 | 长序列计算与缓存分布 | 概念总类，按 prefill/decode 继续区分 |
| PCP | prefill 的序列计算 | 分摊预填充工作 | `prefill_context_parallel_size`；增加进程维度 |
| DCP | decode 使用的上下文 KV | 减少相关 KV 复制、分摊上下文访问 | `decode_context_parallel_size`；复用进程组 |
| SP | 部分中间激活的序列维度 | 减少复制、配合 TP/通信优化 | 编译 `enable_sp` 和相关 MoE 路径；不是通用独立扩容维度 |
| Chunked Prefill | 一次 prefill 的调度工作量 | 平衡 TTFT、ITL 和批次预算 | 单卡也能使用；不增加 rank |

来源：[并行部署](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/) · [上下文并行](https://docs.vllm.ai/en/stable/serving/context_parallel_deployment/) · [编译与 SP](https://github.com/vllm-project/vllm/blob/main/vllm/config/compilation.py)

在本次读取的 `ParallelConfig` 中，普通布局下一个 DP 副本的 worker 数按 **TP × PCP × PP** 计算，跨 DP 再乘 DP；DCP 不再作为独立乘数，EP 也不能机械地再乘一次。PCP=1 时要求 TP 能被 DCP 整除；PCP>1 时当前校验允许 DCP 为 1、PCP 或 TP×PCP。**这些是本次配置源码的规则，不能推广到所有历史版本、launcher 或模型。** [并行配置源码](https://github.com/vllm-project/vllm/blob/main/vllm/config/parallel.py)

用于课堂的两个**布局推导例子**：

- TP=4、PCP=1、PP=1、DP=1、DCP=2：4 个 workers，DCP 复用其中的分组。
- TP=2、PCP=2、PP=1、DP=1、DCP=4：4 个 workers，DCP 可跨 TP×PCP。配置校验通过只是第一步，还需模型/Attention/runner 支持。

PCP/DCP 还要讲 Q/KV 分布、KV interleave、输出与 LSE 归约、`ag_rs`/`a2a`、query replication 和 P/D 传输布局。文档可能落后于新配置，正式实验以选定版本的代码、报错和测试证据共同判断。[ParallelConfig](https://github.com/vllm-project/vllm/blob/main/vllm/config/parallel.py)

### 3.3 用户提到的 TBO 在哪里

vLLM 的对应专题使用官方名 **DBO（Dual Batch Overlap）**，入口为 `--enable-dbo`：把批次分成两个微批次，协调计算与 MoE 通信。官方设计文档描述的是 DP+EP、DeepEP 和至少两张 GPU 的场景，并有 prefill/decode token 阈值。[vLLM DBO](https://docs.vllm.ai/en/stable/design/dbo/)

**TBO（Two-Batch Overlap）和 SBO（Single-Batch Overlap）** 放入 SGLang 比较阅读：比较隐藏通信的方法与时间线，不把其参数名搬到 vLLM。DBO、异步 CPU 调度、Async TP、shared-expert overlap 是相关但不同的优化。[SGLang EP](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/expert_parallelism.mdx)

本课程不提供虚构的 `--enable-tbo` vLLM 启动参数，也不会把单卡请求并发作为 DBO 实测。

## 4. A100 与 C500 怎样落地

| 特性组 | A100 单卡教学安排 | C500 单卡教学安排 | 额外条件 |
| --- | --- | --- | --- |
| 基础生成、连续批处理、APC、Chunked Prefill | 首批实机任务，待跑 | 首批实机任务，待跑 | 固定模型、栈版本与正确性基线 |
| Attention、图执行、编译和融合 | 逐 backend 验证 | 按 MetaX 插件/算子交付逐项验证 | 不能只凭同名 API 判定兼容 |
| Weight-only / 低精度计算算子 | 以 Ampere 和实际 kernel 条件筛选 | 以 MACA/mcoplib/插件条件筛选 | 模型格式、dtype、质量与 kernel 都要记录 |
| FP8/FP4/MX 格式与 KV 量化 | 分清原生计算、存储与转换路径 | 同样分层检查 | 不把低精度格式名称当硬件支持证明 |
| Spec、LoRA、多模态、Pooling | 选适配模型后验证 | 选适配模型后验证 | 单项通过后再测组合 |
| TP/PP/DP/EP/PCP/DCP/SP/DBO | CPU 推导、模拟与源码阅读 | 同样提供学习路径 | 实际并行和通信需要额外、受支持的多卡环境 |
| PD/Encoder 分离、网络 KV 传输 | 协议分析与资源预算 | 同样提供学习路径 | 多进程演示不等同于多机性能验证 |

A100 上要区分 **FP8 权重存储/反量化** 与 **原生 FP8 W8A8 计算**：本次官方量化矩阵没有把 Ampere 列为 LLM Compressor FP8 W8A8 的支持架构，不能把相关实验默认写成 A100 可运行。AWQ/GPTQ/Marlin 也仍需核对具体模型和软件组合。[量化硬件矩阵](https://docs.vllm.ai/en/stable/features/quantization/)

FA2、FA3、FA4 的目标架构和功能条件分别讲解，A100 先核对适合的执行路径；C500 按 MACA 对应实现核对。FA3/FA4 的官方目标与限制见 [FlashAttention](https://github.com/Dao-AILab/flash-attention)，不从“有 FlashAttention”推断所有代际都能用。

## 5. 不能漏掉的组合实验

| 组合 | 核心问题 | 最小对照与证据 |
| --- | --- | --- |
| APC × Chunked Prefill | 命中后还要调度多少 token？ | 冷/热/无共享前缀 × 不同 chunk；命中量、TTFT、ITL |
| APC × LoRA / 多模态 | 哪些输入能共享？ | 相同 token 但不同 adapter/媒体；匹配键和结果 |
| APC × prompt logprobs | 复用是否影响需要输出的中间分数？ | 固定版本、是否拒绝/回退、分数与缓存行为 |
| Hybrid/SWA/SSM × APC/Spec | 状态怎样保存、匹配和回滚？ | 多组缓存容量、检查点、取消/拒绝后数值对齐 |
| Graph × compile × shape | 是计算收益还是少了启动开销？ | eager/编译/图的分步对照；冷启动、稳态、图显存 |
| Quant × Attention × KV dtype | 格式与算子组合是否支持？ | 检查实际后端、误差、质量、workspace 与吞吐 |
| Spec × Graph × async scheduling | 草稿/验证/回滚是否兼容？ | 逐项与组合开关、接受率、回退/错误及尾延迟 |
| Spec × Quant / LoRA / 多模态 | draft 与 target 的能力是否匹配？ | 固定模型配对、adapter、输入和采样；不能只测加载 |
| TP × PCP × DCP | KV 分片和进程组是否一致？ | 张量布局、度数校验、通信与数值；先做 CPU 推导 |
| EP × DBO × EPLB | 通信是否被隐藏，重排是否抵消收益？ | All-to-All/计算时间线、专家负载、迁移成本 |
| PD × Connector × 不同并行度 | KV 能否正确互传？ | block/layout/dtype/model、TP 映射、超时回收 |
| Sleep / reload × 缓存 | 旧状态是否失效？ | 恢复/更新前后权重标识、缓存 reset、输出一致性 |
| Structured / Tool / Reasoning | 增量输出如何组合？ | schema、解析、模板、流式字段与取消行为 |
| 多模态 × encoder graph/cache | 编码输入形状是否适配？ | 编码正确性、processor/encoder 分段时间、cache 命中 |

上表是**待执行的课程实验设计**，并未宣称所有组合均受支持。若版本明确拒绝，记录为有依据的限制案例；不通过绕过校验制造“成功”。

性能报告至少包含：环境、模型 revision、输入输出 token 分布、采样、实际生效配置、预热与缓存状态、负载方式、失败率、质量、指标和原始记录。详细口径见 [实验规范](../engineering/experiments.md)。

## 6. 源码注册项附录

<!-- BEGIN feature-inventory -->

下面逐项列出本次核对的 **14 组、233 个命名条目**。计数包含别名、默认选项、占位项和模型映射，**不是 233 个独立算法**。机器可读清单见 [features.json](../../metadata/upstream/features.json)。

清单用途是避免遗漏后端家族和新配置；不作为复制粘贴的启动菜单。模型专用项、平台专用项、弃用项、示例连接器均保留说明。

### 量化方法与快捷配置

章节：9.1、9.2、9.4、9.5、9.6、9.7、9.13。来源：[官方源码](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/__init__.py)。

别名、算法、框架和在线量化快捷方式并存；fbgemm_fp8、fp_quant 在本次源码中标为 deprecated。BnB/GGUF 在文档有条目，但不在本次 QuantizationMethods 枚举中。

- `awq`、`auto_awq`、`fp8`、`fbgemm_fp8`、`fp_quant`、`modelopt`。
- `modelopt_fp4`、`modelopt_mxfp8`、`modelopt_mixed`、`auto_gptq`、`gptq`、`gptq_marlin`。
- `awq_marlin`、`humming`、`compressed-tensors`、`experts_int8`、`quark`、`moe_wna16`。
- `torchao`、`inc`、`mxfp4`、`gpt_oss_mxfp4`、`deepseek_v4_fp8`、`online`。
- `fp8_per_tensor`、`fp8_per_block`、`fp8_per_channel`、`int8_per_channel_weight_only`、`nvfp4_per_token`、`mxfp8`。

### AttentionBackendEnum

章节：7.2、7.3、7.4、7.5。来源：[官方源码](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/registry.py)。

含平台专用、模型专用和扩展占位项；TORCH_SDPA 是 ViT 使用的标记，CUSTOM 需注册实现，不能把全部名称当作任意模型可切换的后端。

- `FLASH_ATTN`、`FLASH_ATTN_DIFFKV`、`TRITON_ATTN`、`TRITON_ATTN_DIFFKV`、`TRITON_FLASH_ATTN`、`ROCM_ATTN`。
- `ROCM_AITER_MLA`、`ROCM_AITER_TRITON_MLA`、`ROCM_AITER_FA`、`ROCM_AITER_MLA_SPARSE`、`XPU_MLA_SPARSE`、`TORCH_SDPA`。
- `FLASHINFER`、`TRITON_FLASHINFER`、`FLASHINFER_MLA`、`TOKENSPEED_MLA`、`FLASHINFER_MLA_SPARSE`、`FLASHINFER_MLA_SPARSE_SM120`。
- `FLASHINFER_MLA_SPARSE_SM90`、`TRITON_MLA`、`CUTLASS_MLA`、`FLASHMLA`、`FLASHMLA_SPARSE`、`FLASHMLA_SPARSE_DSV4`。
- `FLASHINFER_MLA_SPARSE_DSV4`、`ROCM_FLASHMLA_SPARSE_DSV4`、`FLASHMLA_SPARSE_DSV41`、`FLASHINFER_MLA_SPARSE_DSV41`、`B12X`、`FLASH_ATTN_MLA`。
- `FLASH_ATTN_MLA_SPARSE`、`MINIMAX_M3_SPARSE`、`CUTLASS_MSA`、`TRITON_MSA`、`NO_ATTENTION`、`FLEX_ATTENTION`。
- `HPC_ATTN`、`ROCM_AITER_UNIFIED_ATTN`、`CPU_ATTN`、`CPU_MLA`、`AMX_MLA`、`TURBOQUANT`。
- `CUSTOM`。

### MambaAttentionBackendEnum

章节：5.6、14.7。来源：[官方源码](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/registry.py)。

递推状态模型的实现标识；其缓存和回滚语义单独讲解。

- `MAMBA1`、`MAMBA2`、`SHORT_CONV`、`LINEAR`、`GDN_ATTN`、`CUSTOM`。

### CacheDType

章节：5.5、9.11。来源：[官方源码](https://github.com/vllm-project/vllm/blob/main/vllm/config/cache.py)。

含 dtype、模型专用布局和低比特方案；通过配置解析不能证明内核支持。低精度存储不等于同精度计算。

- `auto`、`float16`、`bfloat16`、`fp8`、`fp8_e4m3`、`fp8_e5m2`。
- `fp8_inc`、`fp8_ds_mla`、`nvfp4_ds_mla`、`turboquant_k8v4`、`turboquant_4bit_nc`、`turboquant_k3v4_nc`。
- `turboquant_3bit_nc`、`int4_per_token_head`、`int8_per_token_head`、`fp8_per_token_head`、`nvfp4`、`nvfp4_4over6`。

### 推测方法展开后的通用标识

章节：10.2、10.3、10.4、10.5、10.6。来源：[官方源码](https://github.com/vllm-project/vllm/blob/main/vllm/config/speculative.py)。

SpeculativeMethod 中的 MTPModelTypes 另列；PARD、动态推测、自适应验证属文档专题，不应据名称发明 method 值。源码有标识仍须验证 runner 支持。

- `ngram`、`medusa`、`mlp_speculator`、`draft_model`、`suffix`、`custom_class`。
- `eagle`、`eagle3`、`extract_hidden_states`、`ngram_gpu`、`dflash`、`dspark`。

### MTPModelTypes

章节：10.5。来源：[官方源码](https://github.com/vllm-project/vllm/blob/main/vllm/config/speculative.py)。

模型类型/自动识别映射，不是 27 种通用独立算法；课程讲映射与检查方法，逐模型能力以实际选定版本为准。

- `deepseek_mtp`、`dots3_note_mtp`、`mimo_mtp`、`mimo_v2_mtp`、`glm4_moe_mtp`、`glm4_moe_lite_mtp`。
- `glm_ocr_mtp`、`ernie_mtp`、`nemotron_h_mtp`、`exaone_moe_mtp`、`exaone4_5_mtp`、`qwen3_next_mtp`。
- `qwen4_exp_mtp`、`qwen3_5_mtp`、`longcat_flash_mtp`、`bailing_hybrid_v3_mtp`、`minimax_m3_mtp`、`bailing_hybrid_mtp`。
- `mtp`、`kimi_k3_mtp`、`pangu_ultra_moe_mtp`、`step3p5_mtp`、`hy_v3_mtp`、`hy_v4_mtp`。
- `gemma4_mtp`、`inkling_mtp`、`glm5_next_mtp`。

### LinearBackend

章节：7.6、7.7、9.7、9.8、9.9、9.10、9.13。来源：[官方源码](https://github.com/vllm-project/vllm/blob/main/vllm/config/kernel.py)。

按量化格式、shape、设备与优先级选择；支持 per-quant 覆盖。auto/emulation 等与原生优化实现要区分。

- `auto`、`cutlass`、`flashinfer_cutlass`、`flashinfer_cutedsl`、`flashinfer_trtllm`、`flashinfer_cudnn`。
- `flashinfer_b12x`、`b12x`、`marlin`、`humming`、`triton`、`deep_gemm`。
- `torch`、`aiter`、`machete`、`fbgemm`、`conch`、`exllama`。
- `emulation`、`xpu`、`xpu_woq`。

### MoEBackend

章节：12.3、9.12、9.13。来源：[官方源码](https://github.com/vllm-project/vllm/blob/main/vllm/config/kernel.py)。

不同路由/量化/模型与平台能力；mega-MoE 有额外模型接线条件，不能当通用开关。

- `auto`、`triton`、`batched_triton`、`deep_gemm`、`deep_gemm_mega_moe`、`cutlass`。
- `flashinfer_trtllm`、`flashinfer_cutlass`、`flashinfer_cutedsl`、`flashinfer_b12x`、`b12x`、`flashinfer_moe_ep_mega_deep_gemm`。
- `flashinfer_moe_ep_mega_cutedsl`、`marlin`、`humming`、`triton_unfused`、`aiter`、`aiter_triton_mxfp4_bf16`。
- `flydsl`、`hpc`、`emulation`、`rdna3`。

### SparseIndexerTopkBackend

章节：5.9、7.4。来源：[官方源码](https://github.com/vllm-project/vllm/blob/main/vllm/config/kernel.py)。

稀疏索引 top-k 内核选项，与 AttentionBackendEnum 分层处理。

- `auto`、`deep_select`、`cooperative`、`persistent`、`per_row`、`flashinfer`。
- `torch`。

### All-to-All 配置选项

章节：12.4。来源：[官方源码](https://github.com/vllm-project/vllm/blob/main/vllm/config/parallel.py)。

pplx/naive 在本次配置校验中走已移除/回退处理，不列为当前可用选项；DBO 设计文档的限制另行核对。

- `allgather_reducescatter`、`deepep_high_throughput`、`deepep_low_latency`、`mori_high_throughput`、`mori_low_latency`、`nixl_ep`。
- `flashinfer_nvlink_one_sided`、`flashinfer_nvlink_two_sided`。

### KVConnectorFactory 内置注册项

章节：13.6、13.8、13.7。来源：[官方源码](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/kv_transfer/kv_connector/factory.py)。

包含示例和 benchmark 专用连接器，不把它们都当生产传输方案；外部 module path 还可扩展；HMA 兼容性需检查。

- `ExampleConnector`、`ExampleHiddenStatesConnector`、`LMCacheConnectorV1`、`LMCacheMPConnector`、`NixlConnector`、`NixlPullConnector`。
- `NixlPushConnector`、`MultiConnector`、`HiSparseConnector`、`MoRIIOConnector`、`OffloadingConnector`、`DecodeBenchConnector`。
- `MooncakeConnector`、`MooncakeStoreConnector`、`FlexKVConnectorV1`、`SimpleCPUOffloadConnector`、`HF3FSKVConnector`。

### CompilationMode

章节：8.1。来源：[官方源码](https://github.com/vllm-project/vllm/blob/main/vllm/config/compilation.py)。

编译方式枚举；不等同于 O0/O1 等整体优化预设。

- `NONE`、`STOCK_TORCH_COMPILE`、`DYNAMO_TRACE_ONCE`、`VLLM_COMPILE`。

### CUDAGraphMode

章节：8.2。来源：[官方源码](https://github.com/vllm-project/vllm/blob/main/vllm/config/compilation.py)。

图捕获/重放模式；编译与 graph 可以分别配置，需检查组合。

- `NONE`、`PIECEWISE`、`FULL`、`FULL_DECODE_ONLY`、`FULL_AND_PIECEWISE`。

### PassConfig 中的布尔优化项

章节：8.4、8.5、8.6。来源：[官方源码](https://github.com/vllm-project/vllm/blob/main/vllm/config/compilation.py)。

仅枚举布尔项，阈值和其他 CompilationConfig 字段从源码索引继续查阅；部分融合仅有 ROCm/AITER 等平台路径。

- `fuse_norm_quant`、`fuse_act_quant`、`fuse_attn_quant`、`eliminate_noops`、`enable_sp`、`fuse_gemm_comms`。
- `fuse_allreduce_rms`、`enable_qk_norm_rope_fusion`、`fuse_rope_kvcache_cat_mla`、`fuse_act_padding`、`fuse_mla_dual_rms_norm`、`fuse_rope_kvcache`。
- `fuse_qk_norm_rope_kvcache`。

<!-- END feature-inventory -->

## 7. 继续追踪配置、模型与扩展

上述枚举覆盖容易漏掉的后端家族；以下入口用于更新所有公开配置和新模型，避免静态目录逐渐失真：

| 入口 | 更新时检查什么 | 归属 |
| --- | --- | --- |
| [Engine Arguments](https://docs.vllm.ai/en/stable/configuration/engine_args/) 与 [serve](https://docs.vllm.ai/en/stable/cli/serve/) | 新参数、默认值、优先级、已废弃接口与 CLI 子命令 | 3.5、18.5 |
| [完整配置源码目录](https://github.com/vllm-project/vllm/tree/main/vllm/config) | Model/Diffusion/Cache/Scheduler/Attention/Parallel/Compilation/Kernel/Load/Offload/Speculative/LoRA/Multimodal/Mamba/Pooler/Observability/Profiler/KV/EC/Weight Transfer/Reasoning/Structured Outputs/Fault Tolerance | 对应 Part 3—18 |
| [环境变量](https://docs.vllm.ai/en/stable/configuration/env_vars/) | 实验开关、排错、缓存目录、设备相关配置 | 3.5、17.3、18.2 |
| [模型支持表](https://docs.vllm.ai/en/stable/models/supported_models/) 与 [模型注册源码](https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/models) | 新架构、任务、专用注意力/量化/推测能力与 Transformers backend | 1.5、7.4、10.5、14 |
| [Features 兼容性入口](https://docs.vllm.ai/en/stable/features/) | 新功能、已移除/旧引擎差异、组合限制 | 全部专题 |
| [插件设计](https://docs.vllm.ai/en/stable/design/plugin_system/) 与 [MetaX](https://github.com/MetaX-MACA/vLLM-metax) | 上游与 out-of-tree 功能、版本绑定、MACA 实际覆盖 | 4、18 |
| [官方例子](https://docs.vllm.ai/en/stable/examples/) | Renderer/Derenderer、Trace Replay、音频、RL、扩展端点等新使用路径 | 13—17 |

还需要显式讲解的配置细节已归入相应专题：`kv_cache_memory_bytes` 与利用率预算、`prefix_match_unit` 与物理 block size、前缀检查点保留间隔、Mamba cache mode/dtype、KV 量化跳过层、graph capture sizes、编译缓存键、load format、NUMA 绑定、请求级观测与失败恢复。

**历史与外部项目单独标注：** V0 multi-step、best-of、旧 guided decoding 字段、旧量化入口等放进迁移课；SGLang TBO/SBO、vLLM-Omni、Dynamo/llm-d 和其他硬件插件放进对照/集成课。不把其全部能力归为上游 vLLM 或 MACA 当前可用能力。

上游还有离散扩散语言模型（dLLM）配置：canvas block 与迭代去噪复用部分推测调度数据通路；已归入 14.7。它与外部 Omni 项目的范围需要分开说明，存在配置也不等于任意扩散模型均可运行。[DiffusionConfig](https://github.com/vllm-project/vllm/blob/main/vllm/config/diffusion.py)

## 8. 更新与验收流程

1. 选择上游 release/tag/commit，同时记录 MACA 插件对应版本；保留上一版目录便于比较。
2. 核对官方 Features、Design、Serving、Models、配置类型和注册表，记录新增、重命名、废弃、移除。
3. 在课程目录为每个新增技术域指定章节、练习与设备条件；更新本文及 JSON 清单。
4. 对容易混淆的术语、默认值和组合限制回查源码/文档，冲突写入版本说明。
5. A100/C500 按功能与组合各自补充实机证据；没有运行保持 `not_run`。
6. 运行 `python -m scripts.docs.course --write` 同步生成区段，再运行 `python -m scripts.docs.course --check` 检查一致性与本地链接；结合人工源码核对和实验记录验收。

本轮交付的是目录与教学设计的扩充，CPU 参考代码及其已运行结果仍按 [验证记录](../../reports/history/validation-before-restructure.md) 记录；新增 GPU 专题不沿用 CPU 的“通过”状态。
