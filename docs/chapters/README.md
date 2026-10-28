# 课程目录与学习路线

<!-- BEGIN course-summary -->

本版：19 篇、135 个正式章节规划。与 [README](../../README.md#课程大纲) 和 [结构化目录](../../metadata/course/catalog.json) 同步。

<!-- END course-summary -->

## 按路线进入目录

- **学生路线**：0.1/0.3 → Part 1 与 CPU 实验 → Part 2 的模型/请求/缓存；有 GPU 后选 4.1 或 4.2，再做真实模型与 Part 3。0.2 可后置。
- **工程师路线**：0.3、1.4、4 与 17.1—17.3 建立可测量基线 → Part 3 定位调用链 → 按瓶颈进入 Part 5—10 → 功能组合、平台和分布式扩展。
- **量化方向**：基线 → 9.1—9.3 契约与校准 → 9.4—9.6 算法与生产 → 9.7—9.10 格式与算子 → 9.13/9.14 接入及服务；9.11/9.12 按 KV/MoE 需求扩展。
- **没有 GPU**：先做 CPU 引导实验、缓存/调度模拟、切分推导与源码阅读；后续公开实机数据后完成分析题。

所有专题均在正式课程覆盖中。阅读顺序可以不同，章节中的基础解释、进阶内容和验证状态分别呈现。分阶段产出、条件和章节分工见[学习路线](learning-paths.md)。

## 实验设备标记

| 标记 | 练习条件 |
| --- | --- |
| CPU | 数学、状态机、切分/通信模拟、源码与数据分析 |
| GPU | 单卡实验候选；须核对模型显存、版本和实际后端 |
| 双平台 | 维护者分别验证单卡 A100/C500；学生选择一条路线 |
| 多卡 / 多机 | 实际通信或分离式服务需要额外设备；CPU 练习不证明性能 |
| 专项 | 限定模型、GPU 架构或额外依赖；A100/C500 不自动满足 |
| 多进程 | 本地进程通信可先学习，完整训练/推理场景另核设备 |

## 章节规划

🧩 表示已有部分材料；📋 表示已列出内容和练习、正文待编写。表中的练习是交付目标，尚未交付的实验不会标为可运行。GPU 均待实测。

<!-- BEGIN course-chapters -->

### Part 0 · 开始之前

[进入本篇目录](part-00-orientation/README.md)

前置：基础 Python。

| 编号 | 章节 | 验收练习或产出 | 设备 | 状态 |
| --- | --- | --- | --- | --- |
| 0.1 | [读者路线与环境准备](part-00-orientation/01-learning-paths-and-environment-setup/README.md) | 运行 CPU 实验并填写环境清单，区分已验证与待验证 | CPU/GPU | 🧩 |
| 0.2 | [第一次模型生成](part-00-orientation/02-your-first-model-generation/README.md) | 画出输入消息到生成文本的数据流，保存一次服务记录 | GPU | 📋 |
| 0.3 | [实验与源码工作方法](part-00-orientation/03-working-with-experiments-and-source-code/README.md) | 建立个人实验目录和源码阅读笔记模板 | CPU | 📋 |

源码与资料：[Datawhale zero-to-sglang](https://github.com/datawhalechina/zero-to-sglang) · [vLLM Quickstart](https://docs.vllm.ai/en/stable/getting_started/quickstart/) · [GPU 安装与设备条件](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)。

### Part 1 · 理解大模型推理

[进入本篇目录](part-01-inference/README.md)

前置：基础 Python；1.1 补充矩阵乘法；本篇 CPU 实验无需 PyTorch 或 GPU。

| 编号 | 章节 | 验收练习或产出 | 设备 | 状态 |
| --- | --- | --- | --- | --- |
| 1.1 | [从 token 到下一步预测](part-01-inference/01-next-token/README.md) | 手算小矩阵的 Attention 并标出张量维度 | CPU | ✅ |
| 1.2 | [Prefill 与 Decode](part-01-inference/02-prefill-decode/README.md) | 估算两阶段的 FLOPs、访存量与主要瓶颈 | CPU 参考与推导；GPU 性能另行验证 | ✅ |
| 1.3 | [KV Cache 的作用](part-01-inference/03-kv-cache/README.md) | 复现已有 logits 对齐实验并推导 KV 容量 | CPU 参考与推导；GPU 性能另行验证 | ✅ |
| 1.4 | [如何读性能结果](part-01-inference/04-performance-metrics/README.md) | 用原始时间序列重算指标并识别错误口径 | CPU | ✅ |
| 1.5 | [模型结构与长上下文基础](part-01-inference/05-model-structures/README.md) | 比较不同结构的缓存对象和复杂度 | CPU | ✅ |
| 1.6 | [采样与数值正确性](part-01-inference/06-sampling-correctness/README.md) | 实现小词表采样并检查边界、分布和停止规则 | CPU | ✅ |

源码与资料：[模型支持与任务](https://docs.vllm.ai/en/stable/models/supported_models/) · [Generative Models](https://docs.vllm.ai/en/stable/models/generative_models/) · [Benchmark CLI](https://docs.vllm.ai/en/stable/benchmarking/cli/)。

### Part 2 · 从零构建教学推理引擎

[进入本篇目录](part-02-engine/README.md)

前置：Part 1。

| 编号 | 章节 | 验收练习或产出 | 设备 | 状态 |
| --- | --- | --- | --- | --- |
| 2.1 | [构建最小 PyTorch 模型](part-02-engine/01-pytorch-model/README.md) | 用同一组权重逐层比较 logits | CPU | ✅ |
| 2.2 | [加载真实模型权重](part-02-engine/02-loading-real-model-weights/README.md) | 固定 SmolLM2-135M revision，映射权重并逐层对齐参考输出 | CPU | ✅ |
| 2.3 | [请求与缓存生命周期](part-02-engine/03-request-and-cache-lifecycles/README.md) | 回放 11 个生命周期案例，验证取消隔离与资源回收 | CPU | ✅ |
| 2.4 | [从静态批处理到连续批处理](part-02-engine/04-from-static-to-continuous-batching/README.md) | 回放六组双策略负载，核对输出、预算与调度边界 | CPU | ✅ |
| 2.5 | [分页缓存](part-02-engine/05-paged-caching/README.md) | 连接物理块与 KV 张量，回放 11 个案例并对齐完整重算 logits | CPU | ✅ |
| 2.6 | [前缀缓存](part-02-engine/06-prefix-caching/README.md) | 运行 12 个 CPU 案例，核对冷暖 logits、共享块和取消后的引用回收 | CPU | ✅ |
| 2.7 | [HTTP 与流式输出](part-02-engine/07-http-and-streaming-output/README.md) | 运行 14 个服务案例及可选前缀后端，核对事件顺序、客户端断连与资源回收 | CPU | ✅ |

源码与资料：[Transformers](https://github.com/huggingface/transformers) · [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) · [mini-sglang](https://github.com/sgl-project/mini-sglang) · [架构设计](https://docs.vllm.ai/en/stable/design/arch_overview/)。

### Part 3 · 走进真实 vLLM

[进入本篇目录](part-03-vllm/README.md)

前置：Part 2；工程师可从本篇进入。

沿请求、调度/KV、执行张量、Runner 演进和配置解析五条线阅读固定源码；每节用 CPU 例子核对语义，再进入双平台实测。

| 编号 | 章节 | 验收练习或产出 | 设备 | 状态 |
| --- | --- | --- | --- | --- |
| 3.1 | [vLLM 中一次请求的旅程](part-03-vllm/01-a-request-s-journey-through-vllm/README.md) | 核对固定提交的 25 个源文件、38 个函数与 4 类消息，绘制请求路径并解释取消回收 | CPU（源码阅读）；GPU 实测另验收 | ✅ |
| 3.2 | [vLLM 调度与 KV 管理](part-03-vllm/02-scheduling-and-kv-management-in-vllm/README.md) | 手算逐轮 token/块账本；运行 10 个计数案例并核验固定源码分支 | CPU（计数实验与源码阅读）；GPU 实测另验收 | ✅ |
| 3.3 | [模型执行与注意力后端](part-03-vllm/03-model-execution-and-attention-backends/README.md) | 手算双请求张量与缓存地址；运行 12 个 CPU 案例并核验 27 个固定源码函数 | CPU（张量实验与源码阅读）；GPU 实测另验收 | ✅ |
| 3.4 | [V1 与 ModelRunner 演进](part-03-vllm/04-v1-and-modelrunner-evolution/README.md) | 手算状态映射与增量载荷；运行 10 个 CPU 案例，核验两版本 26 个函数并设计受控对照 | CPU（语义实验与源码阅读）；GPU 实测另验收 | ✅ |
| 3.5 | [配置与能力的解析流程](part-03-vllm/05-resolving-configuration-and-capabilities/README.md) | 运行 12 个配置案例，追踪字段覆盖与改写；核验 17 个固定源码函数并列出待验证能力 | CPU（语义实验与源码阅读）；GPU 实测另验收 | ✅ |

源码与资料：[架构设计](https://docs.vllm.ai/en/stable/design/arch_overview/) · [Model Runner V2](https://docs.vllm.ai/en/stable/design/model_runner_v2/) · [Hugging Face 集成](https://docs.vllm.ai/en/stable/design/huggingface_integration/) · [vLLM Features 与兼容性入口](https://docs.vllm.ai/en/stable/features/) · [OpenAI-compatible Server](https://docs.vllm.ai/en/stable/serving/online_serving/openai_compatible_server/)。

### Part 4 · CUDA 与 MACA 单卡基线

[进入本篇目录](part-04-backends/README.md)

前置：Part 0、1、3。

| 编号 | 章节 | 验收练习或产出 | 设备 | 状态 |
| --- | --- | --- | --- | --- |
| 4.1 | [CUDA 路径与 A100 基线](part-04-backends/01-the-cuda-path-and-a100-baseline/README.md) | 固定环境并保存单卡 A100 的正确性与性能基线 | GPU | 🧩 |
| 4.2 | [MACA 路径与 C500 基线](part-04-backends/02-the-maca-path-and-c500-baseline/README.md) | 固定配套环境并保存单卡 C500 的独立验证记录 | GPU | 🧩 |
| 4.3 | [两个平台上的同一个实验](part-04-backends/03-the-same-experiment-on-two-platforms/README.md) | 形成完整推理栈对照报告与失败记录 | 双平台 | 📋 |

源码与资料：[GPU 安装与设备条件](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/) · [vLLM-MetaX](https://github.com/MetaX-MACA/vLLM-metax) · [MetaX 安装](https://vllm-metax.readthedocs.io/en/stable/getting_started/installation/maca.html) · [MetaX Quickstart](https://vllm-metax.readthedocs.io/en/stable/getting_started/quickstart.html)。

### Part 5 · KV Cache、Prefix Cache 与长上下文

[进入本篇目录](part-05-kv-cache/README.md)

前置：Part 1—3。

| 编号 | 章节 | 验收练习或产出 | 设备 | 状态 |
| --- | --- | --- | --- | --- |
| 5.1 | [PagedAttention 与缓存布局](part-05-kv-cache/01-pagedattention-and-cache-layouts/README.md) | 计算不同块大小的浪费并解释寻址过程 | CPU/GPU | 🧩 |
| 5.2 | [Automatic Prefix Caching](part-05-kv-cache/02-automatic-prefix-caching/README.md) | 设计整块命中、尾块未命中、无共享、salt 隔离和容量逐出用例 | CPU/GPU | 🧩 |
| 5.3 | [缓存键与共享隔离](part-05-kv-cache/03-cache-keys-and-sharing-isolation/README.md) | 设计共享与不应共享的请求用例，检查命中正确性 | CPU/GPU | 🧩 |
| 5.4 | [缓存生命周期与路由](part-05-kv-cache/04-cache-lifecycle-and-routing/README.md) | 回放容量不足轨迹并解释命中率和排队变化 | CPU/GPU | 🧩 |
| 5.5 | [混合模型缓存管理](part-05-kv-cache/05-cache-management-for-hybrid-models/README.md) | 绘制混合层的内存分配与有效缓存范围 | CPU/GPU | 📋 |
| 5.6 | [Mamba 与线性模型状态缓存](part-05-kv-cache/06-state-caching-for-mamba-and-linear-models/README.md) | 用状态递推模拟说明缓存与回滚为何不同于普通 KV | CPU/GPU | 📋 |
| 5.7 | [KV Offloading 与分层缓存](part-05-kv-cache/07-kv-offloading-and-tiered-caching/README.md) | 比较重算和传输成本；核对命中、传输字节与显存 | GPU/多机 | 📋 |
| 5.8 | [上下文扩展与 KV Sharing](part-05-kv-cache/08-context-extension-and-kv-sharing/README.md) | 设计长上下文质量评估并区分容量与有效理解长度 | CPU/GPU | 📋 |
| 5.9 | [IndexCache 与稀疏索引缓存](part-05-kv-cache/09-indexcache-and-sparse-index-caching/README.md) | 分析索引复用成本与质量变化；不把它当 APC | 专项 | 📋 |

源码与资料：[Automatic Prefix Caching](https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/) · [Prefix Caching 设计](https://docs.vllm.ai/en/stable/design/prefix_caching/) · [缓存配置与 dtype 清单](https://github.com/vllm-project/vllm/blob/main/vllm/config/cache.py) · [Hybrid KV Cache Manager](https://docs.vllm.ai/en/stable/design/hybrid_kv_cache_manager/) · [KV Offloading](https://docs.vllm.ai/en/stable/features/kv_offloading_usage/) · [Context Extension](https://docs.vllm.ai/en/stable/features/context_extension/) · [IndexCache](https://docs.vllm.ai/en/stable/features/index_cache/)。

### Part 6 · 调度、批处理与重叠执行

[进入本篇目录](part-06-scheduling/README.md)

前置：Part 2—3、5。

| 编号 | 章节 | 验收练习或产出 | 设备 | 状态 |
| --- | --- | --- | --- | --- |
| 6.1 | [连续批处理与调度策略](part-06-scheduling/01-continuous-batching-and-scheduling-policies/README.md) | 模拟长短请求混合并比较吞吐和尾延迟 | CPU/GPU | 🧩 |
| 6.2 | [Chunked Prefill](part-06-scheduling/02-chunked-prefill/README.md) | 扫描 chunk/token 预算，观察 TTFT 与 ITL 权衡 | CPU/GPU | 🧩 |
| 6.3 | [Token 预算与显存预算](part-06-scheduling/03-token-and-gpu-memory-budgets/README.md) | 手算 4096 token 预算中两个 decode 加一个长 prefill，标记理想上限与真实约束 | CPU/GPU | 🧩 |
| 6.4 | [异步调度与输出处理](part-06-scheduling/04-async-scheduling-and-output-processing/README.md) | 阅读时间线并定位 CPU 空洞和同步点 | GPU | 📋 |
| 6.5 | [DBO 与 TBO 对照](part-06-scheduling/05-comparing-dbo-and-tbo/README.md) | 画两微批次时间线，检查阈值和重叠是否真实发生 | 多卡 | 🧩 |
| 6.6 | [调度功能组合与演进](part-06-scheduling/06-scheduling-combinations-and-evolution/README.md) | 制作所选版本组合矩阵，记录拒绝/回退原因 | CPU/GPU/多卡 | 📋 |

源码与资料：[Optimization and Tuning](https://docs.vllm.ai/en/stable/configuration/optimization/) · [调度配置](https://github.com/vllm-project/vllm/blob/main/vllm/config/scheduler.py) · [Dual Batch Overlap](https://docs.vllm.ai/en/stable/design/dbo/) · [SGLang EP：TBO/SBO](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/expert_parallelism.mdx)。

### Part 7 · Attention 与高性能算子

[进入本篇目录](part-07-attention/README.md)

前置：Part 1、3、5；补充 GPU 编程。

| 编号 | 章节 | 验收练习或产出 | 设备 | 状态 |
| --- | --- | --- | --- | --- |
| 7.1 | [FlashAttention 原理](part-07-attention/01-flashattention-principles/README.md) | 分块计算 softmax 并与朴素实现对齐 | CPU/GPU | 🧩 |
| 7.2 | [FlashAttention 版本与接入](part-07-attention/02-flashattention-versions-and-integration/README.md) | 区分算法版本、安装包与 vLLM 实际选择的后端 | GPU/专项 | 📋 |
| 7.3 | [注意力后端选择](part-07-attention/03-selecting-an-attention-backend/README.md) | 建立 head size、dtype、KV 格式、图执行的能力矩阵 | GPU/专项 | 🧩 |
| 7.4 | [MLA 与稀疏 Attention](part-07-attention/04-mla-and-sparse-attention/README.md) | 分析压缩 KV、索引选择和 attention kernel 的分工 | 专项 | 📋 |
| 7.5 | [Decode 与共享前缀注意力](part-07-attention/05-decode-and-shared-prefix-attention/README.md) | 比较短/长上下文和高/低 batch 下的算子时间 | GPU | 🧩 |
| 7.6 | [线性层与融合算子](part-07-attention/06-linear-layers-and-fused-kernels/README.md) | 逐级完成数值参考、mask/stride/尾块检查，再选择一个小算子做 microbenchmark | GPU | 🧩 |
| 7.7 | [Kernel 选择与调优](part-07-attention/07-kernel-selection-and-tuning/README.md) | 解释形状、架构和量化格式如何影响调度与回退 | GPU/专项 | 🧩 |

源码与资料：[Attention 后端注册表](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/registry.py) · [Attention Backend 支持矩阵](https://docs.vllm.ai/en/stable/design/attention_backends/) · [FlashAttention 官方实现与版本](https://github.com/Dao-AILab/flash-attention) · [Linear/MoE Kernel 配置](https://github.com/vllm-project/vllm/blob/main/vllm/config/kernel.py) · [CustomOp](https://docs.vllm.ai/en/stable/design/custom_op/) · [Triton Fused Attention 教程](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)。

### Part 8 · 编译、CUDA Graph 与执行优化

[进入本篇目录](part-08-compilation/README.md)

前置：Part 3、6、7。

| 编号 | 章节 | 验收练习或产出 | 设备 | 状态 |
| --- | --- | --- | --- | --- |
| 8.1 | [torch.compile 集成](part-08-compilation/01-torch-compile-integration/README.md) | 区分两层编译链，追踪冷/热编译、specialization 和一个自定义算子的 lowering | GPU | 🧩 |
| 8.2 | [CUDA Graph 执行模式](part-08-compilation/02-cuda-graph-execution-modes/README.md) | 分别记录 eager、分段图、完整图的启动开销与显存 | GPU | 🧩 |
| 8.3 | [形状特化与编译缓存](part-08-compilation/03-shape-specialization-and-compilation-caches/README.md) | 比较冷启动与稳态，定位重复编译 | GPU | 🧩 |
| 8.4 | [计算融合 Pass](part-08-compilation/04-compute-fusion-passes/README.md) | 对照融合前后 kernel 数和误差，不只看开关值 | GPU | 🧩 |
| 8.5 | [通信融合与 Async TP](part-08-compilation/05-communication-fusion-and-async-tp/README.md) | 画出依赖关系，测量通信隐藏比例 | 多卡 | 📋 |
| 8.6 | [优化级别、CustomOp 与 vLLM IR](part-08-compilation/06-optimization-levels-customop-and-vllm-ir/README.md) | 追踪一个算子从模型代码到后端实现的过程 | CPU/GPU | 📋 |
| 8.7 | [多模态图与编译排错](part-08-compilation/07-multimodal-graphs-and-compilation-debugging/README.md) | 对齐图执行结果并保存最小失败场景 | GPU/专项 | 📋 |

源码与资料：[torch.compile 集成](https://docs.vllm.ai/en/stable/design/torch_compile/) · [编译模式与融合 Pass](https://github.com/vllm-project/vllm/blob/main/vllm/config/compilation.py) · [CUDA Graphs](https://docs.vllm.ai/en/stable/design/cuda_graphs/) · [Optimization Levels](https://docs.vllm.ai/en/stable/design/optimization_levels/) · [vLLM IR](https://docs.vllm.ai/en/stable/design/vllm_ir/) · [ViT CUDA Graphs](https://docs.vllm.ai/en/stable/design/cuda_graphs_multimodal/)。

### Part 9 · 量化算法、格式、算子与推理系统

[进入本篇目录](part-09-quantization/README.md)

前置：Part 1—3 与矩阵运算；CPU 从 9.1 开始；GPU 实验补读 Part 4、7—8，MoE/并行补读 Part 11—12。

v1.0 基线固定四个 Module、14 节。以 Qwen3-8B W4A16 G128 贯穿，依次扩展 INT8 W8A8、硬件相关路径与 KV；9.7—9.10、9.13—9.14 为教学重点。

| 编号 | 章节 | 验收练习或产出 | 设备 | 状态 |
| --- | --- | --- | --- | --- |
| 9.1 | [量化系统全景与术语分层](part-09-quantization/01-quantization-system-overview/README.md) | 20 个术语、17 个 CPU 案例、两份 Passport、6 张中文图与固定源码核验 | CPU（数值实验与源码阅读）；GPU 实测另验收 | ✅ |
| 9.2 | [数值表示、量化公式与张量契约](part-09-quantization/02-numerical-representations-and-tensor-contracts/README.md) | 20 个 CPU 案例、两份参数 Passport、7 张中文图、16 组练习与 512 个 FP8 位模式对照 | CPU；可选 PyTorch 位模式核验，GPU 实测另验收 | ✅ |
| 9.3 | [Calibration、Observer 与误差测量](part-09-quantization/03-calibration-observers-and-error-measurement/README.md) | 20 个 CPU 案例、校准协议与 Passport、7 张中文图、16 组练习及 PyTorch Observer 对照 | CPU；可选 PyTorch Observer 核验，真实模型与 GPU 另验收 | ✅ |
| 9.4 | [量化算法与误差控制](part-09-quantization/04-quantization-algorithms-and-error-control/README.md) | 23 个 CPU 案例、GPTQ 核心参考与受控对照、8 张中文图、18 组练习及 PyTorch 数学核验 | CPU；可选 PyTorch 数学核验，完整算法框架/模型与 GPU 另验收 | ✅ |
| 9.5 | [Activation Quantization 与混合精度策略](part-09-quantization/05-activation-quantization-and-mixed-precision/README.md) | per-token INT8 参考、敏感度图、精度分配表与固定质量/固定内存对照 | CPU/PyTorch；A100/C500 独立验证 | 🧩 |
| 9.6 | [量化框架、导出流程与生态](part-09-quantization/06-quantization-frameworks-and-export/README.md) | Framework Matrix，以及 LightCompress 与 LLM Compressor 各一条真实 export 记录 | CPU 检查；GPU 生产/加载 | 📋 |
| 9.7 | [Checkpoint Format 与 Quantized Weight ABI](part-09-quantization/07-checkpoint-formats-and-quantized-weight-abi/README.md) | 真实 checkpoint 检查器与至少两种格式的非方阵布局/元数据分析 | CPU；真实 checkpoint | 📋 |
| 9.8 | [位打包、序列化与 Repack](part-09-quantization/08-bit-packing-serialization-and-repack/README.md) | 沿用现有 CPU INT4 实验；扩展 n-bit、序列化与 repack，提交字节级黄金测试 | CPU 必做；GPU 在 9.9 扩展 | 🧩 |
| 9.9 | [Weight-only GEMV / GEMM 与调优](part-09-quantization/09-weight-only-gemv-gemm-and-tuning/README.md) | 三基线正确性、形状扫描、逐项 profiling 消融与 W4A16 Kernel Optimization Report | CPU 参考；A100/C500 分开实测 | 🧩 |
| 9.10 | [激活量化、低精度 GEMM 与融合](part-09-quantization/10-activation-quantization-low-precision-gemm-and-fusion/README.md) | Low Precision GEMM Report：未融合分项与融合总耗时、正确性、资源与负载边界 | GPU；CPU 数值参考 | 🧩 |
| 9.11 | [KV Cache 量化与 Attention 协同](part-09-quantization/11-kv-cache-quantization-and-attention/README.md) | KV 容量/并发/写读开销/Attention 延迟/质量报告，prefill 与 decode 分开 | CPU 参考；GPU/专项 | 📋 |
| 9.12 | [MoE、并行切分与量化](part-09-quantization/12-moe-parallel-sharding-and-quantization/README.md) | Distributed Quantization Contract 与 MoE 案例；CPU 切分重组、单卡计算、多卡扩展分开验收 | CPU；单卡；多卡扩展 | 📋 |
| 9.13 | [量化路径接入 vLLM](part-09-quantization/13-integrating-quantization-into-vllm/README.md) | 层→模型→离线→在线四级记录，以及 config→loader→repack→backend→kernel 的证据链 | GPU/双平台；CPU 静态检查 | 🧩 |
| 9.14 | [从算子收益到端到端推理加速](part-09-quantization/14-from-kernel-gains-to-end-to-end-acceleration/README.md) | A100/C500 独立端到端报告与 Algorithm→Format→Kernel→Runtime 系统图 | 双平台；CPU 分析 | 🧩 |

四个 Module：A 9.1—9.3 基础；B 9.4—9.6 算法与模型生产；C 9.7—9.10 格式/打包/算子；D 9.11—9.14 动态状态与系统。

9.1 已交付术语分层、含 zero-point 的 Q/DQ/RQ、Passport 与固定源码实验。9.2 已交付数值表示、非方阵张量契约、尾组与 effective bpw 的 CPU 课程。9.3 已交付校准协议、Observer、受控对照与误差测量的 CPU 课程。9.4 已交付误差控制推导、GPTQ 核心参考和缩放/舍入的 CPU 对照。现有 CPU INT4 打包实验归入 9.8，其通过不代表 9.8 整节或 GPU 路径完成。见[总设计](part-09-quantization/design.md)。

源码与资料：[量化与硬件矩阵](https://docs.vllm.ai/en/stable/features/quantization/) · [vLLM 量化方法与配置接口](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/base_config.py) · [Linear/MoE Kernel 配置](https://github.com/vllm-project/vllm/blob/main/vllm/config/kernel.py) · [Triton 矩阵乘教程](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html) · [Marlin 量化内核实现](https://github.com/IST-DASLab/marlin) · [Quantized KV Cache](https://docs.vllm.ai/en/stable/features/quantization/quantized_kvcache/) · [量化方法注册表](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/__init__.py) · [Online Quantization](https://docs.vllm.ai/en/stable/features/quantization/online/) · [LLM Compressor](https://docs.vllm.ai/en/stable/features/quantization/llm_compressor/)。

### Part 10 · Speculative Decoding 推测解码

[进入本篇目录](part-10-speculation/README.md)

前置：Part 1、3、5、6。

| 编号 | 章节 | 验收练习或产出 | 设备 | 状态 |
| --- | --- | --- | --- | --- |
| 10.1 | [Draft、Verify 与拒绝采样](part-10-speculation/01-drafting-verification-and-rejection-sampling/README.md) | 用小词表验证拒绝采样分布与边界 | CPU | 🧩 |
| 10.2 | [N-gram 与 Suffix](part-10-speculation/02-n-gram-and-suffix-methods/README.md) | 比较重复文本和开放生成的命中率 | CPU/GPU | 📋 |
| 10.3 | [独立 Draft Model](part-10-speculation/03-separate-draft-models/README.md) | 扫描草稿长度并记录接受率与端到端收益 | GPU | 📋 |
| 10.4 | [EAGLE 与 EAGLE3](part-10-speculation/04-eagle-and-eagle3/README.md) | 画出特征/草稿/验证调用关系并核对 checkpoint | GPU/专项 | 📋 |
| 10.5 | [MTP 与模型原生多 token 预测](part-10-speculation/05-mtp-and-native-multi-token-prediction/README.md) | 验证预测模块适配并观察逐位置接受率 | GPU/专项 | 📋 |
| 10.6 | [其他草稿与并行提案](part-10-speculation/06-other-draft-methods-and-parallel-proposals/README.md) | 区分文档方案、注册项与所选 runner 的可运行实现 | GPU/专项 | 📋 |
| 10.7 | [动态推测与自适应验证](part-10-speculation/07-dynamic-speculation-and-adaptive-verification/README.md) | 改变 QPS 并检查草稿长度、验证量与尾延迟 | GPU/专项 | 📋 |
| 10.8 | [推测的缓存与执行组合](part-10-speculation/08-speculation-caching-and-execution-combinations/README.md) | 为支持的组合设计错误恢复和正确性检查 | GPU/多卡 | 🧩 |
| 10.9 | [接受率与真实收益](part-10-speculation/09-acceptance-rates-and-actual-gains/README.md) | 解释高接受率但没有加速的场景并交付消融报告 | CPU/GPU | 🧩 |

源码与资料：[Speculative Decoding](https://docs.vllm.ai/en/stable/features/speculative_decoding/) · [推测方法与模型类型](https://github.com/vllm-project/vllm/blob/main/vllm/config/speculative.py) · [EAGLE](https://docs.vllm.ai/en/stable/features/speculative_decoding/eagle/) · [MTP](https://docs.vllm.ai/en/stable/features/speculative_decoding/mtp/) · [Dynamic Speculative Decoding](https://docs.vllm.ai/en/stable/features/speculative_decoding/dynamic_speculative_decoding/) · [Adaptive Verification](https://docs.vllm.ai/en/stable/features/speculative_decoding/adaptive_verification/)。

### Part 11 · 并行切分与分布式执行

[进入本篇目录](part-11-parallelism/README.md)

前置：Part 3、5—8；掌握 collective 基础。

| 编号 | 章节 | 验收练习或产出 | 设备 | 状态 |
| --- | --- | --- | --- | --- |
| 11.1 | [并行维度与通信基础](part-11-parallelism/01-parallel-dimensions-and-communication-basics/README.md) | 画出进程组与张量切分，计算通信数据量 | CPU/多卡 | 🧩 |
| 11.2 | [Tensor Parallelism](part-11-parallelism/02-tensor-parallelism/README.md) | 手算两路 TP 的前向合并与显存分布 | CPU/多卡 | 🧩 |
| 11.3 | [Pipeline Parallelism](part-11-parallelism/03-pipeline-parallelism/README.md) | 制作 pipeline 时间线并估算负载不均的影响 | CPU/多卡 | 🧩 |
| 11.4 | [Data Parallelism](part-11-parallelism/04-data-parallelism/README.md) | 比较请求路由、排队与副本扩展吞吐 | CPU/多卡 | 🧩 |
| 11.5 | [Context Parallelism 概念](part-11-parallelism/05-context-parallelism-concepts/README.md) | 区分上下文并行、序列并行和分块预填充 | CPU/多卡 | 🧩 |
| 11.6 | [Prefill Context Parallelism](part-11-parallelism/06-prefill-context-parallelism/README.md) | 推导 prefill 张量分工与通信，核对模型/runner 支持 | CPU/多卡 | 🧩 |
| 11.7 | [Decode Context Parallelism](part-11-parallelism/07-decode-context-parallelism/README.md) | 推导每卡 KV 和输出归约，核对 DCP 合法度数 | CPU/多卡 | 🧩 |
| 11.8 | [Sequence Parallelism 与 Async TP](part-11-parallelism/08-sequence-parallelism-and-async-tp/README.md) | 说明 SP 与 PCP/DCP 的缓存/通信目标差异 | CPU/多卡 | 📋 |
| 11.9 | [组合并行与拓扑规划](part-11-parallelism/09-combining-parallelism-and-planning-topology/README.md) | 交付一份进程布局和容量预算，标明无法运行的组合 | CPU/多卡 | 🧩 |
| 11.10 | [分布式启动与排错](part-11-parallelism/10-distributed-launch-and-debugging/README.md) | 设计启动、卡数不足、断连与通信失败的排查流程 | 多卡/多机 | 📋 |

源码与资料：[Parallelism and Scaling](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/) · [并行配置与度数校验](https://github.com/vllm-project/vllm/blob/main/vllm/config/parallel.py) · [Context Parallel Deployment](https://docs.vllm.ai/en/stable/serving/context_parallel_deployment/) · [Data Parallel Deployment](https://docs.vllm.ai/en/stable/serving/data_parallel_deployment/) · [Expert Parallel Deployment](https://docs.vllm.ai/en/stable/serving/expert_parallel_deployment/) · [编译模式与融合 Pass](https://github.com/vllm-project/vllm/blob/main/vllm/config/compilation.py)。

### Part 12 · MoE 与通信优化

[进入本篇目录](part-12-moe/README.md)

前置：Part 7、8、11。

| 编号 | 章节 | 验收练习或产出 | 设备 | 状态 |
| --- | --- | --- | --- | --- |
| 12.1 | [MoE 计算链路](part-12-moe/01-the-moe-computation-path/README.md) | 手算 token 路由并统计各专家负载 | CPU/GPU | 🧩 |
| 12.2 | [Expert Parallelism](part-12-moe/02-expert-parallelism/README.md) | 画出 token 到专家的分发和汇总 | CPU/多卡 | 🧩 |
| 12.3 | [Fused MoE Kernel](part-12-moe/03-fused-moe-kernels/README.md) | 对照路由、通信和 GEMM 三类耗时 | GPU/多卡 | 📋 |
| 12.4 | [All-to-All 后端](part-12-moe/04-all-to-all-backends/README.md) | 比较延迟/吞吐模式与拓扑条件，识别已移除后端 | 多卡/专项 | 📋 |
| 12.5 | [EPLB 与弹性专家并行](part-12-moe/05-eplb-and-elastic-expert-parallelism/README.md) | 模拟热点专家并评估迁移成本和再均衡收益 | CPU/多卡 | 🧩 |
| 12.6 | [MoE 重叠与综合调优](part-12-moe/06-moe-overlap-and-combined-tuning/README.md) | 建立算子时间线与消融矩阵，解释收益来源 | 多卡 | 📋 |

源码与资料：[Expert Parallel Deployment](https://docs.vllm.ai/en/stable/serving/expert_parallel_deployment/) · [并行配置与度数校验](https://github.com/vllm-project/vllm/blob/main/vllm/config/parallel.py) · [Fused MoE Modular Kernel](https://docs.vllm.ai/en/stable/design/fused_moe_modular_kernel/) · [MoE Kernel Features](https://docs.vllm.ai/en/stable/design/moe_kernel_features/) · [Linear/MoE Kernel 配置](https://github.com/vllm-project/vllm/blob/main/vllm/config/kernel.py) · [Dual Batch Overlap](https://docs.vllm.ai/en/stable/design/dbo/)。

### Part 13 · 分布式推理工程：PD 分离、KV 系统与集群调度

[进入本篇目录](part-13-disaggregation/README.md)

前置：已有推理服务部署与压测经验；Part 5、6、11 和 17.1—17.2；平台案例需 Linux/容器/网络与 Kubernetes 基础。

面向推理、平台与性能工程师：从实测容量与 KV 成本到 Dynamo/llm-d 控制面，再到扩缩容、观测、故障恢复与生产验收。离线建模是设计工具，真实集群验收独立完成。

| 编号 | 章节 | 验收练习或产出 | 设备 | 状态 |
| --- | --- | --- | --- | --- |
| 13.1 | [PD 分离的收益、代价与适用边界](part-13-disaggregation/01-prefill-decode-disaggregation/README.md) | 运行离线比较器、复算请求指标并完成9组练习，交付同预算方案评审；GPU/集群收益另行实测 | Python 3.11+标准库；离线分析已验证，GPU与真实KV传输待后续实测 | ✅ |
| 13.2 | [P-only / D-only 测量与负载协议](part-13-disaggregation/02-isolated-measurement-and-workload-protocol/README.md) | 定义可复现的单阶段测量协议，区分配置速率、实际到达率、完成吞吐和 fake 路径边界 | 单卡 A100/C500 独立测量；并行实例曲线需对应多卡环境 | 📋 |
| 13.3 | [性能曲线、SLO 查表与不确定性](part-13-disaggregation/03-performance-surfaces-and-slo-lookup/README.md) | 规划可追溯性能表和 SLO 查询器，保留失败/超时/拒绝与所有复测，不把 q_ref 当合格容量 | 离线分析机与可追溯实测数据；平台 Profiler 另需其支持环境 | 📋 |
| 13.4 | [P/D 配比、卡数与集群容量规划](part-13-disaggregation/04-pd-ratios-and-capacity-planning/README.md) | 设计容量规划器：输出 P/D 实例、GPU/节点数、预算与不可行原因，标明忽略传输的初估边界 | 离线规划与目标集群资源清单；配置需真实集群复核 | 📋 |
| 13.5 | [KV 字节预算、传输成本与关键路径](part-13-disaggregation/05-kv-volume-and-transfer-critical-path/README.md) | 手算 KV 流量和链路上限，设计测量表区分传输总时长与暴露在关键路径上的等待 | 双端设备和实际链路；单卡仅做字节预算与本地开销测量 | 📋 |
| 13.6 | [vLLM KV Connector 协议与生命周期](part-13-disaggregation/06-the-kv-connector-protocol/README.md) | 追踪固定版本的生产/传输/消费时序，定位可复用时刻并设计假连接器故障案例 | 固定 vLLM/Connector 版本与双端部署；模拟仅验证协议 | 🧩 |
| 13.7 | [Mooncake：Transfer Engine、Store 与 KV 调度](part-13-disaggregation/07-mooncake-transfer-engine-and-store/README.md) | 给出固定版本组件图、传输微基准方案与重算/远端读取决策，独立记录硬件兼容性 | 受所选 Mooncake 版本支持的双端环境与实际网络 | 📋 |
| 13.8 | [连接器生态与分层 KV 存储](part-13-disaggregation/08-connectors-and-tiered-storage/README.md) | 按传输库/存储服务/运行时连接器比较能力，记录命中率与真实延迟收益的差异 | 目标缓存层与实际链路；按组件和硬件分别验证 | 📋 |
| 13.9 | [异构并行、KV 重分片与拓扑放置](part-13-disaggregation/09-heterogeneous-parallelism-and-transfer-compatibility/README.md) | 建立逐版本兼容矩阵，计算每 rank 字节与拓扑流量，不假定 A100/C500 或任意并行度可互传 | 满足所选并行布局的多卡/多机与对应拓扑 | 🧩 |
| 13.10 | [集群控制面：Dynamo、llm-d 与弹性调度](part-13-disaggregation/10-routing-queues-and-elasticity/README.md) | 交付控制面职责图、固定版本能力矩阵与路由/扩缩容消融方案；用同负载比较缓存亲和、排队和扩容恢复 | 平台支持的 GPU 集群；Operator/EPP 案例需 Kubernetes 与监控 | 📋 |
| 13.11 | [Encoder 分离与 E/P/D 资源规划](part-13-disaggregation/11-encoder-disaggregation-and-caching/README.md) | 把 P/D 规划扩展为 E/P/D，分别预算 embedding 与 KV 传输，明确计时边界 | 模型支持的 E/P/D 执行资源和实际传输链路 | 📋 |
| 13.12 | [生产运维、故障恢复与端到端验收](part-13-disaggregation/12-distributed-serving-lifecycles/README.md) | 交付 SLO 看板设计、故障演练/排空回滚手册及同资源容量验收报告；报告恢复时间、联合 goodput 与预测误差 | 隔离的多机验收环境；容量/故障测试不在生产流量上直接试验 | 🧩 |

13.1已交付正文、离线请求轨迹分析与练习；13.2—13.12仍为工程规划。单卡A100/C500及真实集群实验须独立验证，不能用合成数据作为性能结论。

源码与资料：[Disaggregated Prefilling](https://docs.vllm.ai/en/stable/features/disagg_prefill/) · [Disaggregated Encoder](https://docs.vllm.ai/en/stable/features/disagg_encoder/) · [KV Connector 注册表](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/kv_transfer/kv_connector/factory.py) · [NIXL 兼容性矩阵](https://docs.vllm.ai/en/stable/features/nixl_connector_compatibility/) · [KV Offloading](https://docs.vllm.ai/en/stable/features/kv_offloading_usage/) · [DistServe — SLO-aware PD resource allocation](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin) · [Mooncake — KVCache-centric serving architecture](https://www.usenix.org/conference/fast25/presentation/qin) · [Mooncake Transfer Engine](https://kvcache-ai.github.io/Mooncake/design/transfer-engine/index.html) · [vLLM MooncakeConnector](https://docs.vllm.ai/en/latest/features/mooncake_connector_usage/) · [NVIDIA Dynamo 架构](https://docs.nvidia.com/dynamo/dev/knowledge-base/concepts/architecture) · [NVIDIA Dynamo Profiler](https://docs.nvidia.com/dynamo/dev/knowledge-base/modular-components/profiler/overview) · [NVIDIA Dynamo Planner 设计](https://docs.nvidia.com/dynamo/dev/knowledge-base/modular-components/planner/planner-design) · [NVIDIA Dynamo KVBM 配置与存储层级](https://docs.nvidia.com/dynamo/dev/reference/components/kvbm-configuration) · [llm-d 架构](https://llm-d.ai/docs/architecture) · [vLLM Production Stack](https://github.com/vllm-project/production-stack)。

### Part 14 · 模型类型、多模态与适配器

[进入本篇目录](part-14-model-types/README.md)

前置：Part 1、3、5。

| 编号 | 章节 | 验收练习或产出 | 设备 | 状态 |
| --- | --- | --- | --- | --- |
| 14.1 | [模型注册与执行任务](part-14-model-types/01-model-registration-and-execution-tasks/README.md) | 识别架构与任务匹配，制作能力清单 | CPU/GPU | 🧩 |
| 14.2 | [文本生成与评分](part-14-model-types/02-text-generation-and-scoring/README.md) | 区分生成结果与评分结果并验证 token 对齐 | GPU | 📋 |
| 14.3 | [Pooling 全任务](part-14-model-types/03-pooling-tasks/README.md) | 设计表示、分类和排序的正确性与性能评测 | GPU | 📋 |
| 14.4 | [多模态输入管线](part-14-model-types/04-the-multimodal-input-pipeline/README.md) | 追踪媒体到张量/token 的转换并记录开销 | GPU | 📋 |
| 14.5 | [多模态执行优化](part-14-model-types/05-multimodal-execution-optimization/README.md) | 拆开编码与文本生成耗时并验证组合条件 | GPU/多卡/专项 | 📋 |
| 14.6 | [LoRA 与多租户适配](part-14-model-types/06-lora-and-multi-tenant-adaptation/README.md) | 核对 adapter 隔离、内存预算和多请求正确性 | GPU | 📋 |
| 14.7 | [Prompt Embeddings 与特殊架构](part-14-model-types/07-prompt-embeddings-and-specialized-architectures/README.md) | 追踪输入表示和 dLLM 分块去噪，区分上游引擎与外部项目职责 | GPU/专项 | 📋 |

源码与资料：[模型支持与任务](https://docs.vllm.ai/en/stable/models/supported_models/) · [Pooling Models](https://docs.vllm.ai/en/stable/models/pooling_models/) · [Multimodal Inputs](https://docs.vllm.ai/en/stable/features/multimodal_inputs/) · [LoRA Adapters](https://docs.vllm.ai/en/stable/features/lora/) · [Prompt Embedding Inputs](https://docs.vllm.ai/en/stable/features/prompt_embeds/) · [ViT CUDA Graphs](https://docs.vllm.ai/en/stable/design/cuda_graphs_multimodal/) · [离散扩散语言模型配置](https://github.com/vllm-project/vllm/blob/main/vllm/config/diffusion.py)。

### Part 15 · 服务接口与生成能力

[进入本篇目录](part-15-serving/README.md)

前置：Part 3、14。

| 编号 | 章节 | 验收练习或产出 | 设备 | 状态 |
| --- | --- | --- | --- | --- |
| 15.1 | [离线与在线入口](part-15-serving/01-offline-and-online-entry-points/README.md) | 保存同负载下不同入口的输入输出契约 | CPU/GPU | 🧩 |
| 15.2 | [API 能力地图](part-15-serving/02-api-feature-map/README.md) | 按版本列出 endpoint 与模型能力条件 | GPU | 📋 |
| 15.3 | [Structured Outputs](part-15-serving/03-structured-outputs/README.md) | 验证约束满足率并测量语法编译和逐 token mask 成本 | CPU/GPU | 📋 |
| 15.4 | [Tool Calling](part-15-serving/04-tool-calling/README.md) | 构建工具调用解析用例与流式增量组装检查 | CPU/GPU | 📋 |
| 15.5 | [Reasoning 与 Interleaved Thinking](part-15-serving/05-reasoning-and-interleaved-thinking/README.md) | 核对流式 reasoning/content/tool 三类输出 | CPU/GPU | 📋 |
| 15.6 | [自定义采样与处理器](part-15-serving/06-custom-sampling-and-processors/README.md) | 实现有约束的小处理器并检查批处理语义 | CPU/GPU | 📋 |
| 15.7 | [服务治理与扩展](part-15-serving/07-serving-operations-and-scaling/README.md) | 设计连接超时与高负载场景并保存完整请求结果 | GPU/多机 | 🧩 |

源码与资料：[OpenAI-compatible Server](https://docs.vllm.ai/en/stable/serving/online_serving/openai_compatible_server/) · [Structured Outputs](https://docs.vllm.ai/en/stable/features/structured_outputs/) · [Tool Calling](https://docs.vllm.ai/en/stable/features/tool_calling/) · [Reasoning Outputs](https://docs.vllm.ai/en/stable/features/reasoning_outputs/) · [Interleaved Thinking](https://docs.vllm.ai/en/stable/features/interleaved_thinking/) · [Custom Logits Processors](https://docs.vllm.ai/en/stable/features/custom_logitsprocs/)。

### Part 16 · 权重加载、内存与 RL 推理生命周期

[进入本篇目录](part-16-weight-loading/README.md)

前置：Part 3—5、14。

| 编号 | 章节 | 验收练习或产出 | 设备 | 状态 |
| --- | --- | --- | --- | --- |
| 16.1 | [模型解析与权重加载](part-16-weight-loading/01-model-resolution-and-weight-loading/README.md) | 分解读取、解码、拷贝和首次生成的时间 | GPU | 🧩 |
| 16.2 | [高效加载与启动](part-16-weight-loading/02-efficient-loading-and-startup/README.md) | 比较冷/热文件缓存下启动时间与主机峰值内存 | GPU/专项 | 📋 |
| 16.3 | [权重 Offload 与内存预算](part-16-weight-loading/03-weight-offloading-and-memory-budgets/README.md) | 建立权重/激活/KV/workspace/graph 的显存账本 | CPU/GPU | 📋 |
| 16.4 | [Sleep、Wake 与 Pause/Resume](part-16-weight-loading/04-sleep-wake-and-pause-resume/README.md) | 检查释放后的占用与恢复后输出一致性 | GPU | 📋 |
| 16.5 | [权重更新与传输](part-16-weight-loading/05-weight-updates-and-transfer/README.md) | 画训练器到推理器的更新流程和同步点 | 多进程/多卡 | 📋 |
| 16.6 | [强化学习推理集成](part-16-weight-loading/06-reinforcement-learning-inference-integration/README.md) | 分析权重滞后、采样分布和吞吐；明确 vLLM 的推理角色 | GPU/多卡 | 📋 |

源码与资料：[Model Resolution](https://docs.vllm.ai/en/stable/configuration/model_resolution/) · [Conserving Memory](https://docs.vllm.ai/en/stable/configuration/conserving_memory/) · [Sleep Mode](https://docs.vllm.ai/en/stable/features/sleep_mode/) · [Weight Transfer](https://docs.vllm.ai/en/stable/training/weight_transfer/) · [Async Reinforcement Learning](https://docs.vllm.ai/en/stable/training/async_rl/) · [Layerwise Reloading](https://docs.vllm.ai/en/stable/training/layerwise/)。

### Part 17 · 评测、性能分析与优化方法

[进入本篇目录](part-17-profiling/README.md)

前置：Part 4；按实验选择 Part 5—16。

| 编号 | 章节 | 验收练习或产出 | 设备 | 状态 |
| --- | --- | --- | --- | --- |
| 17.1 | [基准测试工具与负载](part-17-profiling/01-benchmark-tools-and-workloads/README.md) | 生成可重复输入并区分开环 QPS 与闭环并发 | CPU/GPU | 🧩 |
| 17.2 | [延迟目标与参数搜索](part-17-profiling/02-latency-targets-and-parameter-searches/README.md) | 寻找满足尾延迟目标的配置，计入失败与超时 | CPU/GPU | 🧩 |
| 17.3 | [Profiling 全链路](part-17-profiling/03-profiling-across-the-stack/README.md) | 无 profiler 测性能，另取匹配负载抓 trace；区分累计 kernel 时间、重叠与墙钟时间 | GPU | 🧩 |
| 17.4 | [观测与请求级指标](part-17-profiling/04-observability-and-per-request-metrics/README.md) | 把服务时间线与单请求延迟对应起来 | CPU/GPU | 📋 |
| 17.5 | [质量、复现与 Batch Invariance](part-17-profiling/05-quality-reproducibility-and-batch-invariance/README.md) | 比较 logits/质量/性能三种结果并解释差异 | CPU/GPU | 📋 |
| 17.6 | [系统消融与综合项目](part-17-profiling/06-system-ablations-and-capstone-projects/README.md) | 交付可复现优化报告、原始数据及不生效的配置 | 双平台 | 🧩 |

源码与资料：[Benchmark CLI](https://docs.vllm.ai/en/stable/benchmarking/cli/) · [Parameter Sweeps](https://docs.vllm.ai/en/stable/benchmarking/sweeps/) · [Profiling vLLM](https://docs.vllm.ai/en/stable/contributing/profiling/) · [Production Metrics](https://docs.vllm.ai/en/stable/usage/metrics/) · [Per-Request Metrics](https://docs.vllm.ai/en/stable/features/per_request_metrics/) · [Batch Invariance](https://docs.vllm.ai/en/stable/features/batch_invariance/)。

### Part 18 · 平台扩展、部署生态与开源贡献

[进入本篇目录](part-18-deployment/README.md)

前置：Part 3、4、17。

| 编号 | 章节 | 验收练习或产出 | 设备 | 状态 |
| --- | --- | --- | --- | --- |
| 18.1 | [平台与插件机制](part-18-deployment/01-platforms-and-plugins/README.md) | 追踪 MetaX 插件如何覆盖平台行为与算子选择 | CPU/GPU | 📋 |
| 18.2 | [C500 适配专题](part-18-deployment/02-adapting-to-c500/README.md) | 逐项补齐 C500 功能证据，与 A100 结果独立记录 | 双平台 | 📋 |
| 18.3 | [其他硬件与框架对照](part-18-deployment/03-other-hardware-and-framework-comparisons/README.md) | 比较同一概念的接口和实现，标注第三方/硬件专用内容 | CPU/专项 | 📋 |
| 18.4 | [部署与周边生态](part-18-deployment/04-deployment-and-the-surrounding-ecosystem/README.md) | 设计服务、缓存、路由和监控的部署图 | CPU/多机 | 📋 |
| 18.5 | [从问题到贡献与版本维护](part-18-deployment/05-from-an-issue-to-a-contribution-and-version-maintenance/README.md) | 提交有证据的改动，并同步正文、目录、兼容矩阵与来源 | CPU/GPU | 📋 |

源码与资料：[Plugin System](https://docs.vllm.ai/en/stable/design/plugin_system/) · [CustomOp](https://docs.vllm.ai/en/stable/design/custom_op/) · [vLLM-MetaX](https://github.com/MetaX-MACA/vLLM-metax) · [MetaX 模型列表](https://vllm-metax.readthedocs.io/en/stable/models/supported_models.html) · [Kubernetes 部署](https://docs.vllm.ai/en/stable/deployment/k8s/) · [vLLM Features 与兼容性入口](https://docs.vllm.ai/en/stable/features/) · [SGLang Hyperparameter Tuning](https://docs.sglang.io/docs/advanced_features/hyperparameter_tuning)。

<!-- END course-chapters -->

## 跨章节综合作业

1. **学生结课**：完成小型生成引擎，能解释一次缓存命中、一次抢占及一次流式取消。
2. **单卡优化**：固定模型和负载，对 APC、分块预填充、图执行、量化、推测逐项消融，给出符合延迟目标的配置。
3. **国产平台适配**：为 C500 提交可复现环境及逐功能验证，解释与 A100 的完整推理栈差异。
4. **分布式设计**：为给定模型/负载推导 TP/PP/DP/PCP/DCP/EP 布局、KV 容量和通信量；有设备时执行并公开记录。
5. **服务集成**：结合 LoRA、结构化输出、多模态或工具调用，验证功能组合和尾延迟。

## 每章完成标准

每章包含：问题、目标、前置、原理、最小实验/模拟、源码入口、正确性、性能口径、平台条件、组合限制、练习、出处。采用 [章节模板](../maintenance/templates/lesson.md)。

只读/模拟章节明确其可完成的验证，不编造 GPU 结果；涉及真实算子的章节在有实机数据后才完成对应平台验收。

[已完成的 CPU 引导实验](part-01-inference/03-kv-cache/lab.md) · [特性与优化地图](../references/feature-map.md) · [开发顺序](../engineering/roadmap.md) · [来源总表](../references/README.md)
