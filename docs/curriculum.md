# 课程目录与学习路线

<!-- BEGIN course-summary -->

本版：19 篇、126 个正式章节规划。与 [README](../README.md#课程大纲) 和 [结构化目录](curriculum-catalog.json) 同步。

<!-- END course-summary -->

## 按路线进入目录

- **学生路线**：0.1/0.3 → Part 1 与 CPU 实验 → Part 2 的模型/请求/缓存；有 GPU 后选 4.1 或 4.2，再做真实模型与 Part 3。0.2 可后置。
- **工程师路线**：0.3、1.4、4 与 17.1—17.3 建立可测量基线 → Part 3 定位调用链 → 按瓶颈进入 Part 5—10 → 功能组合、平台和分布式扩展。
- **量化方向**：基线 → 9.3 策略与 9.1/9.2 契约/算法 → 9.4—9.7 算子与调优 → 9.10/9.11 接入及服务；9.8/9.9 按 KV/MoE 需求扩展。
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

前置：基础 Python。

| 编号 | 章节 | 验收练习或产出 | 设备 | 状态 |
| --- | --- | --- | --- | --- |
| 0.1 | 读者路线与环境准备 | 运行 CPU 实验并填写环境清单，区分已验证与待验证 | CPU/GPU | 🧩 |
| 0.2 | 第一次模型生成 | 画出输入消息到生成文本的数据流，保存一次服务记录 | GPU | 📋 |
| 0.3 | 实验与源码工作方法 | 建立个人实验目录和源码阅读笔记模板 | CPU | 📋 |

源码与资料：[Datawhale zero-to-sglang](https://github.com/datawhalechina/zero-to-sglang) · [vLLM Quickstart](https://docs.vllm.ai/en/stable/getting_started/quickstart/) · [GPU 安装与设备条件](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)。

### Part 1 · 理解大模型推理

前置：Part 0；矩阵乘法与基础 PyTorch。

| 编号 | 章节 | 验收练习或产出 | 设备 | 状态 |
| --- | --- | --- | --- | --- |
| 1.1 | [从 token 到下一步预测](../docs/chapters/01-next-token.md) | 手算小矩阵的 Attention 并标出张量维度 | CPU | ✅ |
| 1.2 | Prefill 与 Decode | 估算两阶段的 FLOPs、访存量与主要瓶颈 | CPU/GPU | 📋 |
| 1.3 | KV Cache 的作用 | 复现已有 logits 对齐实验并推导 KV 容量 | CPU/GPU | 🧩 |
| 1.4 | 如何读性能结果 | 用原始时间序列重算指标并识别错误口径 | CPU | 🧩 |
| 1.5 | 模型结构与长上下文基础 | 比较不同结构的缓存对象和复杂度 | CPU | 📋 |
| 1.6 | 采样与数值正确性 | 实现小词表采样并检查边界、分布和停止规则 | CPU | 📋 |

源码与资料：[模型支持与任务](https://docs.vllm.ai/en/stable/models/supported_models/) · [Generative Models](https://docs.vllm.ai/en/stable/models/generative_models/) · [Benchmark CLI](https://docs.vllm.ai/en/stable/benchmarking/cli/)。

### Part 2 · 从零构建教学推理引擎

前置：Part 1。

| 编号 | 章节 | 验收练习或产出 | 设备 | 状态 |
| --- | --- | --- | --- | --- |
| 2.1 | 构建最小 PyTorch 模型 | 用同一组权重逐层比较 logits | CPU/GPU | 📋 |
| 2.2 | 加载真实模型权重 | 加载候选小模型并对齐参考输出 | GPU | 📋 |
| 2.3 | 请求与缓存生命周期 | 画出状态机并验证取消后资源回收 | CPU/GPU | 📋 |
| 2.4 | 从静态批处理到连续批处理 | 回放长短请求混合的调度轨迹 | CPU/GPU | 📋 |
| 2.5 | 分页缓存 | 把已有逻辑块模拟与 Attention 张量存储连接 | CPU/GPU | 🧩 |
| 2.6 | 前缀缓存 | 构建共享前缀树形请求并检查复用与回收 | CPU/GPU | 📋 |
| 2.7 | HTTP 与流式输出 | 让断连请求释放资源并保存流式事件记录 | GPU | 📋 |

源码与资料：[Transformers](https://github.com/huggingface/transformers) · [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) · [mini-sglang](https://github.com/sgl-project/mini-sglang) · [架构设计](https://docs.vllm.ai/en/stable/design/arch_overview/)。

### Part 3 · 走进真实 vLLM

前置：Part 2；工程师可从本篇进入。

先追踪请求、状态和张量三条线，再进入具体类。先读 3.1/3.3 理解执行，再回到 3.2 深入调度；3.4 单独处理版本演进。

| 编号 | 章节 | 验收练习或产出 | 设备 | 状态 |
| --- | --- | --- | --- | --- |
| 3.1 | [vLLM 中一次请求的旅程](../docs/chapters/01-request-lifecycle.md) | 为固定版本标出请求 ID、进程/线程边界、消息内容及取消后的资源回收 | CPU/GPU | 🧩 |
| 3.2 | [vLLM 调度与 KV 管理](../docs/chapters/02-scheduling-cache.md) | 把教学状态机映射到正式实现 | CPU/GPU | 🧩 |
| 3.3 | [模型执行与注意力后端](../docs/chapters/01-request-lifecycle.md) | 追踪 [K,N] 权重加载及一次调度到张量化、实际 Attention/Linear 内核的完整链路 | GPU | 🧩 |
| 3.4 | [V1 与 ModelRunner 演进](../docs/chapters/01-request-lifecycle.md) | 制作 runner×图模式对照；按固定源码版本核对能力，避免把设计目标当已支持 | CPU | 🧩 |
| 3.5 | [配置与能力的解析流程](../docs/chapters/01-request-lifecycle.md) | 追踪用户参数如何变成实际运行配置 | CPU/GPU | 🧩 |

源码与资料：[架构设计](https://docs.vllm.ai/en/stable/design/arch_overview/) · [Model Runner V2](https://docs.vllm.ai/en/stable/design/model_runner_v2/) · [Hugging Face 集成](https://docs.vllm.ai/en/stable/design/huggingface_integration/) · [vLLM Features 与兼容性入口](https://docs.vllm.ai/en/stable/features/) · [OpenAI-compatible Server](https://docs.vllm.ai/en/stable/serving/online_serving/openai_compatible_server/)。

### Part 4 · CUDA 与 MACA 单卡基线

前置：Part 0、1、3。

| 编号 | 章节 | 验收练习或产出 | 设备 | 状态 |
| --- | --- | --- | --- | --- |
| 4.1 | CUDA 路径与 A100 基线 | 固定环境并保存单卡 A100 的正确性与性能基线 | GPU | 🧩 |
| 4.2 | MACA 路径与 C500 基线 | 固定配套环境并保存单卡 C500 的独立验证记录 | GPU | 🧩 |
| 4.3 | 两个平台上的同一个实验 | 形成完整推理栈对照报告与失败记录 | 双平台 | 📋 |

源码与资料：[GPU 安装与设备条件](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/) · [vLLM-MetaX](https://github.com/MetaX-MACA/vLLM-metax) · [MetaX 安装](https://vllm-metax.readthedocs.io/en/stable/getting_started/installation/maca.html) · [MetaX Quickstart](https://vllm-metax.readthedocs.io/en/stable/getting_started/quickstart.html)。

### Part 5 · KV Cache、Prefix Cache 与长上下文

前置：Part 1—3。

| 编号 | 章节 | 验收练习或产出 | 设备 | 状态 |
| --- | --- | --- | --- | --- |
| 5.1 | [PagedAttention 与缓存布局](../docs/chapters/02-scheduling-cache.md) | 计算不同块大小的浪费并解释寻址过程 | CPU/GPU | 🧩 |
| 5.2 | [Automatic Prefix Caching](../docs/chapters/02-scheduling-cache.md) | 设计整块命中、尾块未命中、无共享、salt 隔离和容量逐出用例 | CPU/GPU | 🧩 |
| 5.3 | [缓存键与共享隔离](../docs/chapters/02-scheduling-cache.md) | 设计共享与不应共享的请求用例，检查命中正确性 | CPU/GPU | 🧩 |
| 5.4 | [缓存生命周期与路由](../docs/chapters/02-scheduling-cache.md) | 回放容量不足轨迹并解释命中率和排队变化 | CPU/GPU | 🧩 |
| 5.5 | 混合模型缓存管理 | 绘制混合层的内存分配与有效缓存范围 | CPU/GPU | 📋 |
| 5.6 | Mamba 与线性模型状态缓存 | 用状态递推模拟说明缓存与回滚为何不同于普通 KV | CPU/GPU | 📋 |
| 5.7 | KV Offloading 与分层缓存 | 比较重算和传输成本；核对命中、传输字节与显存 | GPU/多机 | 📋 |
| 5.8 | 上下文扩展与 KV Sharing | 设计长上下文质量评估并区分容量与有效理解长度 | CPU/GPU | 📋 |
| 5.9 | IndexCache 与稀疏索引缓存 | 分析索引复用成本与质量变化；不把它当 APC | 专项 | 📋 |

源码与资料：[Automatic Prefix Caching](https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/) · [Prefix Caching 设计](https://docs.vllm.ai/en/stable/design/prefix_caching/) · [缓存配置与 dtype 清单](https://github.com/vllm-project/vllm/blob/main/vllm/config/cache.py) · [Hybrid KV Cache Manager](https://docs.vllm.ai/en/stable/design/hybrid_kv_cache_manager/) · [KV Offloading](https://docs.vllm.ai/en/stable/features/kv_offloading_usage/) · [Context Extension](https://docs.vllm.ai/en/stable/features/context_extension/) · [IndexCache](https://docs.vllm.ai/en/stable/features/index_cache/)。

### Part 6 · 调度、批处理与重叠执行

前置：Part 2—3、5。

| 编号 | 章节 | 验收练习或产出 | 设备 | 状态 |
| --- | --- | --- | --- | --- |
| 6.1 | [连续批处理与调度策略](../docs/chapters/02-scheduling-cache.md) | 模拟长短请求混合并比较吞吐和尾延迟 | CPU/GPU | 🧩 |
| 6.2 | [Chunked Prefill](../docs/chapters/02-scheduling-cache.md) | 扫描 chunk/token 预算，观察 TTFT 与 ITL 权衡 | CPU/GPU | 🧩 |
| 6.3 | [Token 预算与显存预算](../docs/chapters/02-scheduling-cache.md) | 手算 4096 token 预算中两个 decode 加一个长 prefill，标记理想上限与真实约束 | CPU/GPU | 🧩 |
| 6.4 | 异步调度与输出处理 | 阅读时间线并定位 CPU 空洞和同步点 | GPU | 📋 |
| 6.5 | [DBO 与 TBO 对照](../docs/chapters/06-parallel-moe-disagg.md) | 画两微批次时间线，检查阈值和重叠是否真实发生 | 多卡 | 🧩 |
| 6.6 | 调度功能组合与演进 | 制作所选版本组合矩阵，记录拒绝/回退原因 | CPU/GPU/多卡 | 📋 |

源码与资料：[Optimization and Tuning](https://docs.vllm.ai/en/stable/configuration/optimization/) · [调度配置](https://github.com/vllm-project/vllm/blob/main/vllm/config/scheduler.py) · [Dual Batch Overlap](https://docs.vllm.ai/en/stable/design/dbo/) · [SGLang EP：TBO/SBO](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/expert_parallelism.mdx)。

### Part 7 · Attention 与高性能算子

前置：Part 1、3、5；补充 GPU 编程。

| 编号 | 章节 | 验收练习或产出 | 设备 | 状态 |
| --- | --- | --- | --- | --- |
| 7.1 | [FlashAttention 原理](../docs/chapters/03-attention-kernels.md) | 分块计算 softmax 并与朴素实现对齐 | CPU/GPU | 🧩 |
| 7.2 | FlashAttention 版本与接入 | 区分算法版本、安装包与 vLLM 实际选择的后端 | GPU/专项 | 📋 |
| 7.3 | [注意力后端选择](../docs/chapters/03-attention-kernels.md) | 建立 head size、dtype、KV 格式、图执行的能力矩阵 | GPU/专项 | 🧩 |
| 7.4 | MLA 与稀疏 Attention | 分析压缩 KV、索引选择和 attention kernel 的分工 | 专项 | 📋 |
| 7.5 | [Decode 与共享前缀注意力](../docs/chapters/03-attention-kernels.md) | 比较短/长上下文和高/低 batch 下的算子时间 | GPU | 🧩 |
| 7.6 | [线性层与融合算子](../docs/chapters/03-attention-kernels.md) | 逐级完成数值参考、mask/stride/尾块检查，再选择一个小算子做 microbenchmark | GPU | 🧩 |
| 7.7 | [Kernel 选择与调优](../docs/chapters/03-attention-kernels.md) | 解释形状、架构和量化格式如何影响调度与回退 | GPU/专项 | 🧩 |

源码与资料：[Attention 后端注册表](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/registry.py) · [Attention Backend 支持矩阵](https://docs.vllm.ai/en/stable/design/attention_backends/) · [FlashAttention 官方实现与版本](https://github.com/Dao-AILab/flash-attention) · [Linear/MoE Kernel 配置](https://github.com/vllm-project/vllm/blob/main/vllm/config/kernel.py) · [CustomOp](https://docs.vllm.ai/en/stable/design/custom_op/) · [Triton Fused Attention 教程](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)。

### Part 8 · 编译、CUDA Graph 与执行优化

前置：Part 3、6、7。

| 编号 | 章节 | 验收练习或产出 | 设备 | 状态 |
| --- | --- | --- | --- | --- |
| 8.1 | [torch.compile 集成](../docs/chapters/04-compile-profiling.md) | 区分两层编译链，追踪冷/热编译、specialization 和一个自定义算子的 lowering | GPU | 🧩 |
| 8.2 | [CUDA Graph 执行模式](../docs/chapters/04-compile-profiling.md) | 分别记录 eager、分段图、完整图的启动开销与显存 | GPU | 🧩 |
| 8.3 | [形状特化与编译缓存](../docs/chapters/04-compile-profiling.md) | 比较冷启动与稳态，定位重复编译 | GPU | 🧩 |
| 8.4 | [计算融合 Pass](../docs/chapters/04-compile-profiling.md) | 对照融合前后 kernel 数和误差，不只看开关值 | GPU | 🧩 |
| 8.5 | 通信融合与 Async TP | 画出依赖关系，测量通信隐藏比例 | 多卡 | 📋 |
| 8.6 | 优化级别、CustomOp 与 vLLM IR | 追踪一个算子从模型代码到后端实现的过程 | CPU/GPU | 📋 |
| 8.7 | 多模态图与编译排错 | 对齐图执行结果并保存最小失败场景 | GPU/专项 | 📋 |

源码与资料：[torch.compile 集成](https://docs.vllm.ai/en/stable/design/torch_compile/) · [编译模式与融合 Pass](https://github.com/vllm-project/vllm/blob/main/vllm/config/compilation.py) · [CUDA Graphs](https://docs.vllm.ai/en/stable/design/cuda_graphs/) · [Optimization Levels](https://docs.vllm.ai/en/stable/design/optimization_levels/) · [vLLM IR](https://docs.vllm.ai/en/stable/design/vllm_ir/) · [ViT CUDA Graphs](https://docs.vllm.ai/en/stable/design/cuda_graphs_multimodal/)。

### Part 9 · 量化算法、算子实现与推理加速

前置：Part 1、4、7；内核调优结合 Part 8，MoE 量化补读 Part 12.1—12.3。

算法集中在 9.2；9.3 设计精度策略；9.4—9.9 围绕算子实现与加速；9.10 接入 vLLM；9.11 验证完整推理链路。各格式与工具保留为案例和兼容索引。

| 编号 | 章节 | 验收练习或产出 | 设备 | 状态 |
| --- | --- | --- | --- | --- |
| 9.1 | [低精度表示与计算路径](../docs/chapters/05-quantization-operators.md) | 定义张量、位布局与缩放契约，计算含元数据的内存/带宽预算 | CPU/GPU | 🧩 |
| 9.2 | [量化算法](../docs/chapters/05-quantization-operators.md) | 固定格式与计算后端比较算法，输出校准/质量/转换成本分析 | CPU/GPU | 🧩 |
| 9.3 | [量化策略与精度分配](../docs/chapters/05-quantization-operators.md) | 在质量、显存和延迟预算下设计策略，并形成敏感度与消融表 | CPU/GPU | 🧩 |
| 9.4 | [手写量化与数据重排算子](../docs/chapters/05-quantization-operators.md) | 先运行已交付 CPU INT4 实验并核对黄金字节/半 scale 误差界，再实现 GPU Q/DQ 与重排 | CPU/GPU | 🧩 |
| 9.5 | [Weight-only GEMV 与 GEMM](../docs/chapters/05-quantization-operators.md) | 先写正确内核，再比较显式反量化与融合路径的时间和访存 | GPU | 🧩 |
| 9.6 | 低精度 GEMM 与缩放实现 | 实现受设备支持的低精度路径，把量化、矩阵乘和缩放开销分项测量 | GPU/专项 | 📋 |
| 9.7 | [融合算子与内核调优](../docs/chapters/05-quantization-operators.md) | 用 profiling 解释每次改动，并在形状扫描中定位性能拐点与回退 | GPU | 🧩 |
| 9.8 | KV 量化与 Attention 协同 | 实现或改造一条 KV Q/DQ 路径，分别测容量、Attention 时间与模型质量 | GPU/专项 | 📋 |
| 9.9 | MoE 专家量化与分组计算 | 分析专家 token 分布，优化一个子路径并核对 dispatch/compute/combine 开销 | GPU/多卡 | 📋 |
| 9.10 | [量化算子接入 vLLM](../docs/chapters/05-quantization-operators.md) | 为同一量化权重追踪加载前后 shape/layout/dtype 和 kernel 分支，先层后模型再服务验收 | GPU/双平台 | 🧩 |
| 9.11 | [从算子收益到端到端加速](../docs/chapters/05-quantization-operators.md) | 提交 A100/C500 独立报告，解释更低位宽或更快 GEMM 未带来整体收益的场景 | 双平台 | 🧩 |

学生先学习 9.1—9.4 的数学参考与打包，9.5 的内核实践需要 GPU；已有量化经验的工程师可先完成 9.3 的策略设计，再进入 9.4—9.11。

贯穿实验：定义量化张量契约 → 参考 Q/DQ 与打包 → 融合量化 Linear → 调优与特定算子扩展 → vLLM 服务消融。见[量化篇设计与验收](quantization-design.md)。

源码与资料：[量化与硬件矩阵](https://docs.vllm.ai/en/stable/features/quantization/) · [vLLM 量化方法与配置接口](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/base_config.py) · [Linear/MoE Kernel 配置](https://github.com/vllm-project/vllm/blob/main/vllm/config/kernel.py) · [Triton 矩阵乘教程](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html) · [Marlin 量化内核实现](https://github.com/IST-DASLab/marlin) · [Quantized KV Cache](https://docs.vllm.ai/en/stable/features/quantization/quantized_kvcache/) · [量化方法注册表](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/__init__.py) · [Online Quantization](https://docs.vllm.ai/en/stable/features/quantization/online/) · [LLM Compressor](https://docs.vllm.ai/en/stable/features/quantization/llm_compressor/)。

### Part 10 · Speculative Decoding 推测解码

前置：Part 1、3、5、6。

| 编号 | 章节 | 验收练习或产出 | 设备 | 状态 |
| --- | --- | --- | --- | --- |
| 10.1 | [Draft、Verify 与拒绝采样](../docs/chapters/07-speculation-serving.md) | 用小词表验证拒绝采样分布与边界 | CPU | 🧩 |
| 10.2 | N-gram 与 Suffix | 比较重复文本和开放生成的命中率 | CPU/GPU | 📋 |
| 10.3 | 独立 Draft Model | 扫描草稿长度并记录接受率与端到端收益 | GPU | 📋 |
| 10.4 | EAGLE 与 EAGLE3 | 画出特征/草稿/验证调用关系并核对 checkpoint | GPU/专项 | 📋 |
| 10.5 | MTP 与模型原生多 token 预测 | 验证预测模块适配并观察逐位置接受率 | GPU/专项 | 📋 |
| 10.6 | 其他草稿与并行提案 | 区分文档方案、注册项与所选 runner 的可运行实现 | GPU/专项 | 📋 |
| 10.7 | 动态推测与自适应验证 | 改变 QPS 并检查草稿长度、验证量与尾延迟 | GPU/专项 | 📋 |
| 10.8 | [推测的缓存与执行组合](../docs/chapters/07-speculation-serving.md) | 为支持的组合设计错误恢复和正确性检查 | GPU/多卡 | 🧩 |
| 10.9 | [接受率与真实收益](../docs/chapters/07-speculation-serving.md) | 解释高接受率但没有加速的场景并交付消融报告 | CPU/GPU | 🧩 |

源码与资料：[Speculative Decoding](https://docs.vllm.ai/en/stable/features/speculative_decoding/) · [推测方法与模型类型](https://github.com/vllm-project/vllm/blob/main/vllm/config/speculative.py) · [EAGLE](https://docs.vllm.ai/en/stable/features/speculative_decoding/eagle/) · [MTP](https://docs.vllm.ai/en/stable/features/speculative_decoding/mtp/) · [Dynamic Speculative Decoding](https://docs.vllm.ai/en/stable/features/speculative_decoding/dynamic_speculative_decoding/) · [Adaptive Verification](https://docs.vllm.ai/en/stable/features/speculative_decoding/adaptive_verification/)。

### Part 11 · 并行切分与分布式执行

前置：Part 3、5—8；掌握 collective 基础。

| 编号 | 章节 | 验收练习或产出 | 设备 | 状态 |
| --- | --- | --- | --- | --- |
| 11.1 | [并行维度与通信基础](../docs/chapters/06-parallel-moe-disagg.md) | 画出进程组与张量切分，计算通信数据量 | CPU/多卡 | 🧩 |
| 11.2 | [Tensor Parallelism](../docs/chapters/06-parallel-moe-disagg.md) | 手算两路 TP 的前向合并与显存分布 | CPU/多卡 | 🧩 |
| 11.3 | [Pipeline Parallelism](../docs/chapters/06-parallel-moe-disagg.md) | 制作 pipeline 时间线并估算负载不均的影响 | CPU/多卡 | 🧩 |
| 11.4 | [Data Parallelism](../docs/chapters/06-parallel-moe-disagg.md) | 比较请求路由、排队与副本扩展吞吐 | CPU/多卡 | 🧩 |
| 11.5 | [Context Parallelism 概念](../docs/chapters/06-parallel-moe-disagg.md) | 区分上下文并行、序列并行和分块预填充 | CPU/多卡 | 🧩 |
| 11.6 | [Prefill Context Parallelism](../docs/chapters/06-parallel-moe-disagg.md) | 推导 prefill 张量分工与通信，核对模型/runner 支持 | CPU/多卡 | 🧩 |
| 11.7 | [Decode Context Parallelism](../docs/chapters/06-parallel-moe-disagg.md) | 推导每卡 KV 和输出归约，核对 DCP 合法度数 | CPU/多卡 | 🧩 |
| 11.8 | Sequence Parallelism 与 Async TP | 说明 SP 与 PCP/DCP 的缓存/通信目标差异 | CPU/多卡 | 📋 |
| 11.9 | [组合并行与拓扑规划](../docs/chapters/06-parallel-moe-disagg.md) | 交付一份进程布局和容量预算，标明无法运行的组合 | CPU/多卡 | 🧩 |
| 11.10 | 分布式启动与排错 | 设计启动、卡数不足、断连与通信失败的排查流程 | 多卡/多机 | 📋 |

源码与资料：[Parallelism and Scaling](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/) · [并行配置与度数校验](https://github.com/vllm-project/vllm/blob/main/vllm/config/parallel.py) · [Context Parallel Deployment](https://docs.vllm.ai/en/stable/serving/context_parallel_deployment/) · [Data Parallel Deployment](https://docs.vllm.ai/en/stable/serving/data_parallel_deployment/) · [Expert Parallel Deployment](https://docs.vllm.ai/en/stable/serving/expert_parallel_deployment/) · [编译模式与融合 Pass](https://github.com/vllm-project/vllm/blob/main/vllm/config/compilation.py)。

### Part 12 · MoE 与通信优化

前置：Part 7、8、11。

| 编号 | 章节 | 验收练习或产出 | 设备 | 状态 |
| --- | --- | --- | --- | --- |
| 12.1 | [MoE 计算链路](../docs/chapters/06-parallel-moe-disagg.md) | 手算 token 路由并统计各专家负载 | CPU/GPU | 🧩 |
| 12.2 | [Expert Parallelism](../docs/chapters/06-parallel-moe-disagg.md) | 画出 token 到专家的分发和汇总 | CPU/多卡 | 🧩 |
| 12.3 | Fused MoE Kernel | 对照路由、通信和 GEMM 三类耗时 | GPU/多卡 | 📋 |
| 12.4 | All-to-All 后端 | 比较延迟/吞吐模式与拓扑条件，识别已移除后端 | 多卡/专项 | 📋 |
| 12.5 | [EPLB 与弹性专家并行](../docs/chapters/06-parallel-moe-disagg.md) | 模拟热点专家并评估迁移成本和再均衡收益 | CPU/多卡 | 🧩 |
| 12.6 | MoE 重叠与综合调优 | 建立算子时间线与消融矩阵，解释收益来源 | 多卡 | 📋 |

源码与资料：[Expert Parallel Deployment](https://docs.vllm.ai/en/stable/serving/expert_parallel_deployment/) · [并行配置与度数校验](https://github.com/vllm-project/vllm/blob/main/vllm/config/parallel.py) · [Fused MoE Modular Kernel](https://docs.vllm.ai/en/stable/design/fused_moe_modular_kernel/) · [MoE Kernel Features](https://docs.vllm.ai/en/stable/design/moe_kernel_features/) · [Linear/MoE Kernel 配置](https://github.com/vllm-project/vllm/blob/main/vllm/config/kernel.py) · [Dual Batch Overlap](https://docs.vllm.ai/en/stable/design/dbo/)。

### Part 13 · 分离式部署与 KV 传输

前置：Part 5、6、11。

| 编号 | 章节 | 验收练习或产出 | 设备 | 状态 |
| --- | --- | --- | --- | --- |
| 13.1 | [Prefill-Decode 分离](../docs/chapters/06-parallel-moe-disagg.md) | 推导传输成本与分离收益的临界条件 | CPU/多机 | 🧩 |
| 13.2 | [KV Connector 协议](../docs/chapters/06-parallel-moe-disagg.md) | 绘制一次远端 KV 的生产/传输/消费流程 | CPU/多机 | 🧩 |
| 13.3 | 连接器与分层存储地图 | 逐项核对传输协议、存储层与外部依赖 | 多机/专项 | 📋 |
| 13.4 | [异构并行与传输兼容](../docs/chapters/06-parallel-moe-disagg.md) | 做兼容性表与流量预算，不能假定任意配置可互传 | CPU/多机 | 🧩 |
| 13.5 | Encoder 分离与缓存 | 画出图像/音频编码结果的跨进程流转 | 多机/专项 | 📋 |
| 13.6 | [分布式服务生命周期](../docs/chapters/06-parallel-moe-disagg.md) | 分析故障注入轨迹与请求完成/回收语义 | 多机 | 🧩 |

源码与资料：[Disaggregated Prefilling](https://docs.vllm.ai/en/stable/features/disagg_prefill/) · [Disaggregated Encoder](https://docs.vllm.ai/en/stable/features/disagg_encoder/) · [KV Connector 注册表](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/kv_transfer/kv_connector/factory.py) · [NIXL 兼容性矩阵](https://docs.vllm.ai/en/stable/features/nixl_connector_compatibility/) · [KV Offloading](https://docs.vllm.ai/en/stable/features/kv_offloading_usage/)。

### Part 14 · 模型类型、多模态与适配器

前置：Part 1、3、5。

| 编号 | 章节 | 验收练习或产出 | 设备 | 状态 |
| --- | --- | --- | --- | --- |
| 14.1 | [模型注册与执行任务](../docs/chapters/01-request-lifecycle.md) | 识别架构与任务匹配，制作能力清单 | CPU/GPU | 🧩 |
| 14.2 | 文本生成与评分 | 区分生成结果与评分结果并验证 token 对齐 | GPU | 📋 |
| 14.3 | Pooling 全任务 | 设计表示、分类和排序的正确性与性能评测 | GPU | 📋 |
| 14.4 | 多模态输入管线 | 追踪媒体到张量/token 的转换并记录开销 | GPU | 📋 |
| 14.5 | 多模态执行优化 | 拆开编码与文本生成耗时并验证组合条件 | GPU/多卡/专项 | 📋 |
| 14.6 | LoRA 与多租户适配 | 核对 adapter 隔离、内存预算和多请求正确性 | GPU | 📋 |
| 14.7 | Prompt Embeddings 与特殊架构 | 追踪输入表示和 dLLM 分块去噪，区分上游引擎与外部项目职责 | GPU/专项 | 📋 |

源码与资料：[模型支持与任务](https://docs.vllm.ai/en/stable/models/supported_models/) · [Pooling Models](https://docs.vllm.ai/en/stable/models/pooling_models/) · [Multimodal Inputs](https://docs.vllm.ai/en/stable/features/multimodal_inputs/) · [LoRA Adapters](https://docs.vllm.ai/en/stable/features/lora/) · [Prompt Embedding Inputs](https://docs.vllm.ai/en/stable/features/prompt_embeds/) · [ViT CUDA Graphs](https://docs.vllm.ai/en/stable/design/cuda_graphs_multimodal/) · [离散扩散语言模型配置](https://github.com/vllm-project/vllm/blob/main/vllm/config/diffusion.py)。

### Part 15 · 服务接口与生成能力

前置：Part 3、14。

| 编号 | 章节 | 验收练习或产出 | 设备 | 状态 |
| --- | --- | --- | --- | --- |
| 15.1 | [离线与在线入口](../docs/chapters/07-speculation-serving.md) | 保存同负载下不同入口的输入输出契约 | CPU/GPU | 🧩 |
| 15.2 | API 能力地图 | 按版本列出 endpoint 与模型能力条件 | GPU | 📋 |
| 15.3 | Structured Outputs | 验证约束满足率并测量语法编译和逐 token mask 成本 | CPU/GPU | 📋 |
| 15.4 | Tool Calling | 构建工具调用解析用例与流式增量组装检查 | CPU/GPU | 📋 |
| 15.5 | Reasoning 与 Interleaved Thinking | 核对流式 reasoning/content/tool 三类输出 | CPU/GPU | 📋 |
| 15.6 | 自定义采样与处理器 | 实现有约束的小处理器并检查批处理语义 | CPU/GPU | 📋 |
| 15.7 | [服务治理与扩展](../docs/chapters/07-speculation-serving.md) | 设计连接超时与高负载场景并保存完整请求结果 | GPU/多机 | 🧩 |

源码与资料：[OpenAI-compatible Server](https://docs.vllm.ai/en/stable/serving/online_serving/openai_compatible_server/) · [Structured Outputs](https://docs.vllm.ai/en/stable/features/structured_outputs/) · [Tool Calling](https://docs.vllm.ai/en/stable/features/tool_calling/) · [Reasoning Outputs](https://docs.vllm.ai/en/stable/features/reasoning_outputs/) · [Interleaved Thinking](https://docs.vllm.ai/en/stable/features/interleaved_thinking/) · [Custom Logits Processors](https://docs.vllm.ai/en/stable/features/custom_logitsprocs/)。

### Part 16 · 权重加载、内存与 RL 推理生命周期

前置：Part 3—5、14。

| 编号 | 章节 | 验收练习或产出 | 设备 | 状态 |
| --- | --- | --- | --- | --- |
| 16.1 | [模型解析与权重加载](../docs/chapters/01-request-lifecycle.md) | 分解读取、解码、拷贝和首次生成的时间 | GPU | 🧩 |
| 16.2 | 高效加载与启动 | 比较冷/热文件缓存下启动时间与主机峰值内存 | GPU/专项 | 📋 |
| 16.3 | 权重 Offload 与内存预算 | 建立权重/激活/KV/workspace/graph 的显存账本 | CPU/GPU | 📋 |
| 16.4 | Sleep、Wake 与 Pause/Resume | 检查释放后的占用与恢复后输出一致性 | GPU | 📋 |
| 16.5 | 权重更新与传输 | 画训练器到推理器的更新流程和同步点 | 多进程/多卡 | 📋 |
| 16.6 | 强化学习推理集成 | 分析权重滞后、采样分布和吞吐；明确 vLLM 的推理角色 | GPU/多卡 | 📋 |

源码与资料：[Model Resolution](https://docs.vllm.ai/en/stable/configuration/model_resolution/) · [Conserving Memory](https://docs.vllm.ai/en/stable/configuration/conserving_memory/) · [Sleep Mode](https://docs.vllm.ai/en/stable/features/sleep_mode/) · [Weight Transfer](https://docs.vllm.ai/en/stable/training/weight_transfer/) · [Async Reinforcement Learning](https://docs.vllm.ai/en/stable/training/async_rl/) · [Layerwise Reloading](https://docs.vllm.ai/en/stable/training/layerwise/)。

### Part 17 · 评测、性能分析与优化方法

前置：Part 4；按实验选择 Part 5—16。

| 编号 | 章节 | 验收练习或产出 | 设备 | 状态 |
| --- | --- | --- | --- | --- |
| 17.1 | [基准测试工具与负载](../docs/chapters/04-compile-profiling.md) | 生成可重复输入并区分开环 QPS 与闭环并发 | CPU/GPU | 🧩 |
| 17.2 | [延迟目标与参数搜索](../docs/chapters/04-compile-profiling.md) | 寻找满足尾延迟目标的配置，计入失败与超时 | CPU/GPU | 🧩 |
| 17.3 | [Profiling 全链路](../docs/chapters/04-compile-profiling.md) | 无 profiler 测性能，另取匹配负载抓 trace；区分累计 kernel 时间、重叠与墙钟时间 | GPU | 🧩 |
| 17.4 | 观测与请求级指标 | 把服务时间线与单请求延迟对应起来 | CPU/GPU | 📋 |
| 17.5 | 质量、复现与 Batch Invariance | 比较 logits/质量/性能三种结果并解释差异 | CPU/GPU | 📋 |
| 17.6 | [系统消融与综合项目](../docs/chapters/04-compile-profiling.md) | 交付可复现优化报告、原始数据及不生效的配置 | 双平台 | 🧩 |

源码与资料：[Benchmark CLI](https://docs.vllm.ai/en/stable/benchmarking/cli/) · [Parameter Sweeps](https://docs.vllm.ai/en/stable/benchmarking/sweeps/) · [Profiling vLLM](https://docs.vllm.ai/en/stable/contributing/profiling/) · [Production Metrics](https://docs.vllm.ai/en/stable/usage/metrics/) · [Per-Request Metrics](https://docs.vllm.ai/en/stable/features/per_request_metrics/) · [Batch Invariance](https://docs.vllm.ai/en/stable/features/batch_invariance/)。

### Part 18 · 平台扩展、部署生态与开源贡献

前置：Part 3、4、17。

| 编号 | 章节 | 验收练习或产出 | 设备 | 状态 |
| --- | --- | --- | --- | --- |
| 18.1 | 平台与插件机制 | 追踪 MetaX 插件如何覆盖平台行为与算子选择 | CPU/GPU | 📋 |
| 18.2 | C500 适配专题 | 逐项补齐 C500 功能证据，与 A100 结果独立记录 | 双平台 | 📋 |
| 18.3 | 其他硬件与框架对照 | 比较同一概念的接口和实现，标注第三方/硬件专用内容 | CPU/专项 | 📋 |
| 18.4 | 部署与周边生态 | 设计服务、缓存、路由和监控的部署图 | CPU/多机 | 📋 |
| 18.5 | 从问题到贡献与版本维护 | 提交有证据的改动，并同步正文、目录、兼容矩阵与来源 | CPU/GPU | 📋 |

源码与资料：[Plugin System](https://docs.vllm.ai/en/stable/design/plugin_system/) · [CustomOp](https://docs.vllm.ai/en/stable/design/custom_op/) · [vLLM-MetaX](https://github.com/MetaX-MACA/vLLM-metax) · [MetaX 模型列表](https://vllm-metax.readthedocs.io/en/stable/models/supported_models.html) · [Kubernetes 部署](https://docs.vllm.ai/en/stable/deployment/k8s/) · [vLLM Features 与兼容性入口](https://docs.vllm.ai/en/stable/features/) · [SGLang Hyperparameter Tuning](https://docs.sglang.io/docs/advanced_features/hyperparameter_tuning)。

<!-- END course-chapters -->

## 跨章节综合作业

1. **学生结课**：完成小型生成引擎，能解释一次缓存命中、一次抢占及一次流式取消。
2. **单卡优化**：固定模型和负载，对 APC、分块预填充、图执行、量化、推测逐项消融，给出符合延迟目标的配置。
3. **国产平台适配**：为 C500 提交可复现环境及逐功能验证，解释与 A100 的完整推理栈差异。
4. **分布式设计**：为给定模型/负载推导 TP/PP/DP/PCP/DCP/EP 布局、KV 容量和通信量；有设备时执行并公开记录。
5. **服务集成**：结合 LoRA、结构化输出、多模态或工具调用，验证功能组合和尾延迟。

## 每章完成标准

每章包含：问题、目标、前置、原理、最小实验/模拟、源码入口、正确性、性能口径、平台条件、组合限制、练习、出处。采用 [章节模板](templates/chapter.md)。

只读/模拟章节明确其可完成的验证，不编造 GPU 结果；涉及真实算子的章节在有实机数据后才完成对应平台验收。

[已完成的 CPU 引导实验](labs/01-kv-cache.md) · [特性与优化地图](vllm-feature-map.md) · [开发顺序](roadmap.md) · [来源总表](references.md)
