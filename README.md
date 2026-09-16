<div align="center">
  <img src="./docs/assets/zero2vllm-banner.webp" alt="zero2vllm：理解推理，手写算子" width="100%">
  <h1>zero2vllm</h1>
  <p>
    <a href="./README.md"><img src="https://img.shields.io/badge/简体中文-2563eb?style=for-the-badge" alt="简体中文"></a>
    <a href="./README_en.md"><img src="https://img.shields.io/badge/English-6b7280?style=for-the-badge" alt="English"></a>
  </p>
  <h3>📚《从零手搓 vLLM》</h3>
  <p><em>从理解大模型如何生成，理解其推理过程中的缓存与调度，到手写算子以及对应的优化手段，让我们更能彻底地理解vLLM。</em></p>
</div>

<div align="center">
  <a href="https://github.com/xpxpxp2001xpxpxp/zero2vllm/stargazers"><img src="https://img.shields.io/github/stars/xpxpxp2001xpxpxp/zero2vllm?style=flat&amp;logo=github" alt="GitHub stars"></a>
  <a href="https://github.com/xpxpxp2001xpxpxp/zero2vllm/forks"><img src="https://img.shields.io/github/forks/xpxpxp2001xpxpxp/zero2vllm?style=flat&amp;logo=github" alt="GitHub forks"></a>
  <a href="https://github.com/xpxpxp2001xpxpxp/zero2vllm"><img src="https://img.shields.io/badge/GitHub-Project-2563eb?style=flat&amp;logo=github" alt="GitHub 项目"></a>
  <img src="https://img.shields.io/badge/README-中文%20%7C%20English-16a34a?style=flat" alt="中英文 README">
  <img src="https://img.shields.io/badge/Python-3.11%2B-3776ab?style=flat&amp;logo=python&amp;logoColor=white" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/GPU_targets-A100%20%7C%20C500-0f766e?style=flat" alt="GPU 实验目标：A100 与 C500">
  <a href="./docs/README.md"><img src="https://img.shields.io/badge/课程导航-开始阅读-2563eb?style=flat&amp;logo=gitbook&amp;logoColor=white" alt="课程导航"></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey?style=flat" alt="CC BY-NC-SA 4.0"></a>
</div>

<div align="center">
  <p>推理原理 · 量化算子 · CPU 入门 · NVIDIA CUDA / 沐曦 MACA</p>
  <p>从最小实验出发，逐步构建教学推理引擎，再深入真实 vLLM 源码。</p>
</div>

---

[第一次来](#第一次来从这里开始) · [快速开始](#快速开始) · [课程大纲](#课程大纲) · [特性与优化地图](#特性与优化地图) · [学习路线](#学习路线) · [框架阅读地图](#框架阅读地图) · [当前进度](#当前进度) · [参与贡献](#参与贡献)

[文档导航](docs/README.md) · [开发路线](docs/roadmap.md) · [更新记录](CHANGELOG.md) · [使用许可](LICENSE)

> **课程建设中。** Part 1 的 1.1—1.6 已提供完整正文、CPU 代码、操作命令、练习答案和统一检查入口；另有 INT4 Linear 实验、7 篇专题导读和双后端配置模板。完整教学引擎与后续章节继续建设，A100/C500 的 GPU 实验尚待实机验证。

## 第一次来，从这里开始

根据已有基础选一个入口，完成后再沿[学习路线](docs/learning-paths.md)深入。课程的完整范围保留在后面的 **19 篇、126 章**大纲中。

| 你的起点 | 第一份材料 | 做完后得到什么 |
| --- | --- | --- |
| 会 Python，想理解推理 | [第一课：从 token 到下一步预测](docs/chapters/01-next-token.md) | 手算一次 Attention，解释最后一行 logits 怎样选出下一个 token |
| 已用过模型部署，想读源码 | [请求、状态与张量](docs/chapters/01-request-lifecycle.md) | 一张请求调用链，分清调度决策与模型执行 |
| 做量化或想写算子 | [CPU INT4 Linear 实验](docs/labs/02-int4-linear.md) → [量化导读](docs/chapters/05-quantization-operators.md) | 一份位布局、误差和存储报告，再推进到 GPU 内核 |
| 已有可用 GPU 环境，想做优化 | [实验规范](docs/experiments.md) → [Profiling 导读](docs/chapters/04-compile-profiling.md) | 一份固定负载的实验设计，区分正确性、质量与性能 |

首次学习可按[现有材料的短路线](docs/learning-paths.md#先用现有材料走通一条短路线)推进。算法统一在 **9.2 量化算法**，量化主线重点是策略、数据布局、算子实现、内核调优与 vLLM 接入。

**完整学习第一篇：**[Part 1 学习与命令总表](docs/chapters/part1-guide.md)，从生成原理走到缓存、性能指标、模型结构和采样正确性。

其余专题与 38 份资料的对应关系见[导读与知识库映射](docs/knowledge-base.md)。导读表示已有材料，目录中的 🧩 不等于完整章节或 GPU 验收完成。

## 为什么做这门课

调用模型生成一句话，是理解推理系统的起点。继续往下看，还会遇到许多具体问题：

- 为什么读入一段长文本，与逐个生成 token，表现出不同的性能瓶颈？
- 历史 token 的 K/V 应该怎样保存？并发请求增加时，显存又该如何分配？
- 一个请求提前结束后，新的请求怎样进入正在运行的批次？
- 同一份模型切换到另一种 GPU，哪些逻辑可以共用，哪些算子和依赖需要适配？
- 怎样用正确性检查和性能数据判断一次修改是否有效？

**zero2vllm 围绕这些问题组织课程，以 vLLM 为主要源码学习对象。** 我们从可手算的小例子开始，逐步实现模型前向、生成循环、KV Cache、调度和服务，再把教学实现与真实推理框架对应起来。

课程采用“**问题 → 原理 → 最小实现 → 实验 → 源码对应**”的学习方式。各章围绕可以运行、观察或检查的结果组织，进阶部分再讨论性能与平台差异；已交付的内容以各章状态为准。

## 你将学到什么

以下是完整课程的建设目标，已完成内容见[当前进度](#当前进度)。

- **理解推理过程**：串起 Tokenizer、Attention、Prefill、Decode 与逐 token 生成。
- **构建教学引擎**：从简单循环走到缓存管理、连续批处理、分页缓存和流式服务。
- **阅读 vLLM 源码**：追踪请求入口、调度器、KV 管理、模型执行与注意力后端。
- **系统学习优化策略**：APC、FlashAttention、编译与图执行、权重/激活/KV 量化、推测解码和 DBO。
- **掌握分布式设计**：TP、PP、DP、EP、CP、PCP、DCP、SP，以及 MoE、KV 传输与分离式部署。
- **认识完整服务能力**：多模态、Pooling、LoRA、结构化输出、工具调用、推理输出、权重更新与运维。
- **设计可信的实验**：比较输出正确性，测量首响应、生成速度、吞吐与尾延迟。
- **理解跨平台适配**：在 CUDA 与 MACA 两条路径中辨认公共逻辑、运行环境和算子差异。
- **学习工程协作**：固定实验版本，整理最小复现，阅读补丁并验证改动。

## 适合谁

**学生与初学者**：会写基础 Python，希望理解推理系统如何工作。可以从 CPU 实验开始，沿基础路线学习；矩阵乘法、PyTorch 与 Transformer 的必要知识会在课程中补充。

**有基础的工程师**：已了解 PyTorch 或模型部署，希望进一步掌握调度、缓存、源码、性能分析和国产 GPU 适配。可以按问题进入相应章节的工程进阶部分。

前置知识：

- 必需：Python 的函数、类、列表和基本调试。
- 建议：了解矩阵运算、神经网络前向计算；进入真实模型章节前熟悉基础 PyTorch。
- 按需学习：CUDA/GPU 编程、C++ 和底层算子，在后端与优化专题中逐步引入。

## 学习路线

| 路线 | 从哪里进入 | 主线与阶段产出 |
| --- | --- | --- |
| 学生 | CPU 引导实验，0.1/0.3 与 Part 1 | 数学参考 → 请求/缓存模拟 → 真实小模型 → vLLM 映射；逐步交付可解释的教学引擎 |
| 工程师 | 固定一种平台基线，提前使用 Part 17 的测量方法 | 调用链与瓶颈 → 单项优化 → 组合回归 → 平台与分布式扩展 |
| 量化方向 | 建立基线后，从 9.3 策略设计进入 | 张量契约 → Q/DQ 与打包 → 量化 GEMV/GEMM → 调优 → vLLM 接入与服务消融 |

具体章节顺序、阶段交付物、GPU 条件与交叉章节分工见[学习路线](docs/learning-paths.md)。每章共用基础正文，附工程进阶与平台说明；完整技术范围见[课程目录](docs/curriculum.md)。

**现在就能做：**完成 Part 1 六节正文与 CPU 实验，继续运行 INT4 Linear 实验，阅读专题导读、检查逻辑块分配和配置模板。没有 GPU 时先完成 CPU 阶段，0.2 的真实模型生成体验可后置。

### 根据手头硬件选择实验

| 手头的环境 | 可以从哪里开始 | 后续实践 |
| --- | --- | --- |
| 普通电脑 CPU | 当前的因果 Attention、KV Cache 实验与逻辑块模拟 | 学习基础原理，分析已有实验记录 |
| 可用的 NVIDIA GPU | 在所选版本与显存允许时运行小模型 | 沿 CUDA 路径完成 GPU 章节 |
| 沐曦 C500 | 使用对应的官方软件栈 | 沿 MACA 路径完成 GPU 章节 |
| 暂时没有 GPU | 完成 CPU 代码练习 | 随课程发布，使用维护者的真实 GPU 记录完成分析练习 |

**A100 和 C500 是我们首批计划验证的设备。** 学生可以只选择一种 GPU 路径；最低显存要求会随具体模型和实验的实测结果公布。

## 快速开始

### 1. 完成第一课：从 token 到下一步预测

需要 **Python 3.11+**。在仓库根目录执行，不需要第三方包、模型下载或 GPU：

```bash
python -m zero2vllm next-token-demo --output runs/lesson-01/default.json
```

从五个教学词项和二维向量开始，逐项观察 Embedding、Q/K/V、因果 Attention、logits 和生成循环。默认输入“我 喜欢”，第一步选中 ID `2`；`verification.passed` 应为 `true`。权重是为手算设置的，没有训练语言能力。

阅读[第一课正文与五道练习](docs/chapters/01-next-token.md)，对照[实际运行结果](reports/lesson-01-next-token.json)。预计 60—90 分钟完成手算、实验和自查。

整篇六节的命令、参数说明和练习见 [Part 1 总表](docs/chapters/part1-guide.md)。一次运行 28 个案例、全套测试与文档检查：

```bash
python scripts/run_part1.py
```

结果保存在 `runs/part1/summary.json` 及各案例文件中。性能指标使用标明来源的模拟时间线，容量/Roofline 为假设估算，均不代表 GPU 实测。

### 2. 继续观察 KV Cache

下载并解压仓库，或克隆到本地后，在包含 `pyproject.toml` 的根目录执行。需要 **Python 3.11+**，无需下载模型或安装第三方包。

```bash
python -m zero2vllm demo
```

这个实验使用固定随机权重的单头 Attention 模型，分别运行“完整重算”和“使用 KV Cache”的生成过程，比较每一步整个词表上的输出分数（logits）。

默认输入 4 个 token，生成 8 个新 token。检查结果：

| 检查项 | 预期 |
| --- | --- |
| 生成的 token ID | 两条路径一致 |
| 最大 logits 误差 | 不超过 `1e-12` |
| Q/K/V 投影处理的位置数 | 完整重算 60，缓存路径 11 |

这是用于学习计算过程的小模型，token ID 没有自然语言含义；投影位置计数也不能直接换算为 GPU 加速倍数。

继续阅读[引导实验：KV Cache 前后是否计算一致](docs/labs/01-kv-cache.md)，或查看[已运行的结果](reports/cpu-kv-cache.json)。

### 3. 保存结果与运行检查

```bash
python -m zero2vllm demo --output runs/cpu-kv-cache.json
python -m unittest discover -s tests -v
```

试着改变 prompt 或输出长度，先预测重复计算量，再检查结果。

### 4. 量化方向：运行 INT4 Linear

```bash
python -m zero2vllm quant-demo --output runs/cpu-int4-linear.json
```

实验实现沿 K 分组的 Q/DQ、有符号 INT4 打包及两条量化 Linear 参考路径。默认包含尾组和奇数打包，分别报告实现误差与量化误差，并把 scale 计入存储。详见[实验教程](docs/labs/02-int4-linear.md)；它是 CPU 数学参考，GPU 融合内核与 vLLM 集成待完成。

### 5. 准备 GPU 路径

```bash
python -m zero2vllm doctor --output runs/environment.json
python -m zero2vllm profile configs/cuda-a100.toml
python -m zero2vllm profile configs/maca-c500.toml
```

这一步用于整理环境信息和检查配置。模板会列出尚未填写的实际版本；配置完整后可预览容器内的启动参数。正式部署步骤随 A100、C500 实机验证补齐，见[双后端环境与配置](docs/backends.md)。

## 课程大纲

<!-- BEGIN course-outline -->

本版共 **19 篇、126 个正式章节规划**，另有两个独立的 CPU 实验。可点击章节名称进入已有正文或专题导读，完成程度以状态标记为准；前置知识、练习产出、设备条件见[详细课程目录](docs/curriculum.md)。

**状态：✅ 已完成并有本地验证；🧩 有部分材料或代码；📋 已规划、正文待编写。** 功能已被上游收录也不等于本项目已经实测。

[✅ CPU 引导实验：KV Cache 前后是否计算一致](docs/labs/01-kv-cache.md)

[✅ CPU 量化实验：INT4 打包、分组缩放与 Linear](docs/labs/02-int4-linear.md)

| 篇 | 专题范围 | 章节数 |
| --- | --- | --- |
| Part 0 | 开始之前 | 3 |
| Part 1 | 理解大模型推理 | 6 |
| Part 2 | 从零构建教学推理引擎 | 7 |
| Part 3 | 走进真实 vLLM | 5 |
| Part 4 | CUDA 与 MACA 单卡基线 | 3 |
| Part 5 | KV Cache、Prefix Cache 与长上下文 | 9 |
| Part 6 | 调度、批处理与重叠执行 | 6 |
| Part 7 | Attention 与高性能算子 | 7 |
| Part 8 | 编译、CUDA Graph 与执行优化 | 7 |
| Part 9 | 量化算法、算子实现与推理加速 | 11 |
| Part 10 | Speculative Decoding 推测解码 | 9 |
| Part 11 | 并行切分与分布式执行 | 10 |
| Part 12 | MoE 与通信优化 | 6 |
| Part 13 | 分离式部署与 KV 传输 | 6 |
| Part 14 | 模型类型、多模态与适配器 | 7 |
| Part 15 | 服务接口与生成能力 | 7 |
| Part 16 | 权重加载、内存与 RL 推理生命周期 | 6 |
| Part 17 | 评测、性能分析与优化方法 | 6 |
| Part 18 | 平台扩展、部署生态与开源贡献 | 5 |

<details>
<summary>Part 0 · 开始之前（3 章）</summary>

| 章节 | 必须覆盖的知识点 | 状态 |
| --- | --- | --- |
| 0.1 读者路线与环境准备 | Python、Linux、容器、驱动/运行时/框架版本、CPU 与双后端路线 | 🧩 |
| 0.2 第一次模型生成 | 离线 LLM、在线服务、流式请求、Tokenizer 与 chat template | 📋 |
| 0.3 实验与源码工作方法 | 固定 revision、阅读配置/日志/调用链、最小复现与原始数据 | 📋 |

来源与阅读入口：[Datawhale zero-to-sglang](https://github.com/datawhalechina/zero-to-sglang) · [vLLM Quickstart](https://docs.vllm.ai/en/stable/getting_started/quickstart/) · [GPU 安装与设备条件](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)。

</details>

<details>
<summary>Part 1 · 理解大模型推理（6 章）</summary>

| 章节 | 必须覆盖的知识点 | 状态 |
| --- | --- | --- |
| 1.1 [从 token 到下一步预测](docs/chapters/01-next-token.md) | Tokenizer、Embedding、Transformer、因果 Attention、LM Head | ✅ |
| 1.2 [Prefill 与 Decode](docs/chapters/01-prefill-decode.md) | 计算密集/带宽密集、HBM/SRAM、Roofline、算术强度 | ✅ |
| 1.3 [KV Cache 的作用](docs/chapters/01-kv-cache.md) | 缓存更新、位置偏移、MHA/MQA/GQA、每 token 显存 | ✅ |
| 1.4 [如何读性能结果](docs/chapters/01-performance-metrics.md) | TTFT、TPOT、ITL、吞吐、Goodput、P50/P95/P99、排队 | ✅ |
| 1.5 [模型结构与长上下文基础](docs/chapters/01-model-structures.md) | RoPE、ALiBi、Sliding Window、MLA、稀疏/线性 Attention、SSM | ✅ |
| 1.6 [采样与数值正确性](docs/chapters/01-sampling-correctness.md) | Greedy、temperature、top-k/top-p/min-p、penalty、seed、EOS/stop | ✅ |

来源与阅读入口：[模型支持与任务](https://docs.vllm.ai/en/stable/models/supported_models/) · [Generative Models](https://docs.vllm.ai/en/stable/models/generative_models/) · [Benchmark CLI](https://docs.vllm.ai/en/stable/benchmarking/cli/)。

</details>

<details>
<summary>Part 2 · 从零构建教学推理引擎（7 章）</summary>

| 章节 | 必须覆盖的知识点 | 状态 |
| --- | --- | --- |
| 2.1 构建最小 PyTorch 模型 | 模型前向、生成循环、RMSNorm/RoPE/MLP、与 CPU 参考对齐 | 📋 |
| 2.2 加载真实模型权重 | 配置、safetensors、权重映射、Tokenizer、模型 revision | 📋 |
| 2.3 请求与缓存生命周期 | 创建、运行、完成、取消、异常、缓存归属 | 📋 |
| 2.4 从静态批处理到连续批处理 | 迭代级调度、活跃队列、token 预算、请求插入退出 | 📋 |
| 2.5 分页缓存 | 逻辑块、物理块、block table、slot mapping、设备 KV 存储 | 🧩 |
| 2.6 前缀缓存 | 跨请求共享、引用计数、哈希匹配、逐出、缓存失效 | 📋 |
| 2.7 HTTP 与流式输出 | 异步迭代、背压、断连、取消传播、请求错误 | 📋 |

来源与阅读入口：[Transformers](https://github.com/huggingface/transformers) · [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) · [mini-sglang](https://github.com/sgl-project/mini-sglang) · [架构设计](https://docs.vllm.ai/en/stable/design/arch_overview/)。

</details>

<details>
<summary>Part 3 · 走进真实 vLLM（5 章）</summary>

| 章节 | 必须覆盖的知识点 | 状态 |
| --- | --- | --- |
| 3.1 [vLLM 中一次请求的旅程](docs/chapters/01-request-lifecycle.md) | 输入处理→EngineCoreClient→EngineCore→Executor/Worker/ModelRunner；请求 ID、进程通信、流式队列与取消 | 🧩 |
| 3.2 [vLLM 调度与 KV 管理](docs/chapters/02-scheduling-cache.md) | 输入处理、调度输出、KVCacheManager、BlockPool、执行反馈 | 🧩 |
| 3.3 [模型执行与注意力后端](docs/chapters/01-request-lifecycle.md) | 模型注册/初始化/权重加载/后处理；input_ids、positions、slot_mapping、Attention metadata、采样 | 🧩 |
| 3.4 [V1 与 ModelRunner 演进](docs/chapters/01-request-lifecycle.md) | V0/V1 与 MRV1/MRV2 两个维度；持久状态/执行排列、StagedWriteTensor、GPU 元数据、采样与图管理 | 🧩 |
| 3.5 [配置与能力的解析流程](docs/chapters/01-request-lifecycle.md) | EngineArgs、VllmConfig、环境变量、平台默认值、优先级与校验 | 🧩 |

来源与阅读入口：[架构设计](https://docs.vllm.ai/en/stable/design/arch_overview/) · [Model Runner V2](https://docs.vllm.ai/en/stable/design/model_runner_v2/) · [Hugging Face 集成](https://docs.vllm.ai/en/stable/design/huggingface_integration/) · [vLLM Features 与兼容性入口](https://docs.vllm.ai/en/stable/features/) · [OpenAI-compatible Server](https://docs.vllm.ai/en/stable/serving/online_serving/openai_compatible_server/)。

</details>

<details>
<summary>Part 4 · CUDA 与 MACA 单卡基线（3 章）</summary>

| 章节 | 必须覆盖的知识点 | 状态 |
| --- | --- | --- |
| 4.1 CUDA 路径与 A100 基线 | 上游 vLLM、CUDA/PyTorch、显存规格、eager 起步、离线/在线 | 🧩 |
| 4.2 MACA 路径与 C500 基线 | vLLM-MetaX、MACA、mcoplib、官方镜像、版本关系 | 🧩 |
| 4.3 两个平台上的同一个实验 | 同权重/输入/采样/负载、dtype、后端差异、误差容限 | 📋 |

来源与阅读入口：[GPU 安装与设备条件](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/) · [vLLM-MetaX](https://github.com/MetaX-MACA/vLLM-metax) · [MetaX 安装](https://vllm-metax.readthedocs.io/en/stable/getting_started/installation/maca.html) · [MetaX Quickstart](https://vllm-metax.readthedocs.io/en/stable/getting_started/quickstart.html)。

</details>

<details>
<summary>Part 5 · KV Cache、Prefix Cache 与长上下文（9 章）</summary>

| 章节 | 必须覆盖的知识点 | 状态 |
| --- | --- | --- |
| 5.1 [PagedAttention 与缓存布局](docs/chapters/02-scheduling-cache.md) | 页大小、block table、slot mapping、NHD/HND 与新布局、碎片 | 🧩 |
| 5.2 [Automatic Prefix Caching](docs/chapters/02-scheduling-cache.md) | APC 完整块匹配、未命中后缀计算、prefill 节省、最后 token/logits 边界；Radix 对照 | 🧩 |
| 5.3 [缓存键与共享隔离](docs/chapters/02-scheduling-cache.md) | 链式哈希、token/多模态/LoRA 标识、cache salt、哈希算法 | 🧩 |
| 5.4 [缓存生命周期与路由](docs/chapters/02-scheduling-cache.md) | 引用计数、LRU、逐出、抢占重算、reset、KV events、亲和路由 | 🧩 |
| 5.5 混合模型缓存管理 | Full/SWA/SSM 分组、Hybrid KV Cache Manager、页对齐、共享层 | 📋 |
| 5.6 Mamba 与线性模型状态缓存 | Mamba1/2、GDN、卷积/SSM 状态、检查点、ReplaySSM/RecoverSSM | 📋 |
| 5.7 KV Offloading 与分层缓存 | 原生/LMCache、GPU↔CPU、异步换入换出、存储与连接器 | 📋 |
| 5.8 上下文扩展与 KV Sharing | RoPE scaling/YaRN、模型长度、滑窗、跨层 KV sharing/fast prefill | 📋 |
| 5.9 IndexCache 与稀疏索引缓存 | DSA top-k 索引跨层复用、复用频率、稀疏注意力精度 | 📋 |

来源与阅读入口：[Automatic Prefix Caching](https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/) · [Prefix Caching 设计](https://docs.vllm.ai/en/stable/design/prefix_caching/) · [缓存配置与 dtype 清单](https://github.com/vllm-project/vllm/blob/main/vllm/config/cache.py) · [Hybrid KV Cache Manager](https://docs.vllm.ai/en/stable/design/hybrid_kv_cache_manager/) · [KV Offloading](https://docs.vllm.ai/en/stable/features/kv_offloading_usage/) · [Context Extension](https://docs.vllm.ai/en/stable/features/context_extension/) · [IndexCache](https://docs.vllm.ai/en/stable/features/index_cache/)。

</details>

<details>
<summary>Part 6 · 调度、批处理与重叠执行（6 章）</summary>

| 章节 | 必须覆盖的知识点 | 状态 |
| --- | --- | --- |
| 6.1 [连续批处理与调度策略](docs/chapters/02-scheduling-cache.md) | FCFS、priority、等待/运行队列、公平性、抢占与饥饿 | 🧩 |
| 6.2 [Chunked Prefill](docs/chapters/02-scheduling-cache.md) | 分块预填充、混合 prefill/decode、长请求阈值、部分 prefill 预算 | 🧩 |
| 6.3 [Token 预算与显存预算](docs/chapters/02-scheduling-cache.md) | max_num_seqs、max_num_batched_tokens、版本中的 scheduled-token 上限、max_model_len、KV/encoder/推测预算 | 🧩 |
| 6.4 异步调度与输出处理 | CPU/GPU overlap、输入准备、采样反馈、异步输出、取消边界 | 📋 |
| 6.5 [DBO 与 TBO 对照](docs/chapters/06-parallel-moe-disagg.md) | vLLM Dual Batch Overlap、ubatch、计算/通信重叠；SGLang TBO/SBO 对照 | 🧩 |
| 6.6 调度功能组合与演进 | async scheduling × spec decode × graphs × PP/LoRA；旧 multi-step/best-of | 📋 |

来源与阅读入口：[Optimization and Tuning](https://docs.vllm.ai/en/stable/configuration/optimization/) · [调度配置](https://github.com/vllm-project/vllm/blob/main/vllm/config/scheduler.py) · [Dual Batch Overlap](https://docs.vllm.ai/en/stable/design/dbo/) · [SGLang EP：TBO/SBO](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/expert_parallelism.mdx)。

</details>

<details>
<summary>Part 7 · Attention 与高性能算子（7 章）</summary>

| 章节 | 必须覆盖的知识点 | 状态 |
| --- | --- | --- |
| 7.1 [FlashAttention 原理](docs/chapters/03-attention-kernels.md) | IO-aware tiling、online softmax、SRAM 复用、精确注意力 | 🧩 |
| 7.2 FlashAttention 版本与接入 | FA1/FA2/FA3/FA4、varlen、prefill/decode、设备与 dtype 条件 | 📋 |
| 7.3 [注意力后端选择](docs/chapters/03-attention-kernels.md) | FlashInfer、Triton、SDPA/FlexAttention、CPU/ROCm/XPU、自动选择 | 🧩 |
| 7.4 MLA 与稀疏 Attention | FlashMLA、CUTLASS MLA、FlashInfer MLA、DSA/MSA、模型专用后端 | 📋 |
| 7.5 [Decode 与共享前缀注意力](docs/chapters/03-attention-kernels.md) | 分区归约、split-K、变长/分页、cascade attention、共享前缀 | 🧩 |
| 7.6 [线性层与融合算子](docs/chapters/03-attention-kernels.md) | Triton vector add→reduction/Softmax→Matmul→Attention；GEMM/GEMV、RMSNorm、RoPE、SwiGLU、CUDA/CUTLASS/CuTeDSL | 🧩 |
| 7.7 [Kernel 选择与调优](docs/chapters/03-attention-kernels.md) | Marlin、Machete、DeepGEMM、Humming、FlashInfer、b12x、autotune/JIT | 🧩 |

来源与阅读入口：[Attention 后端注册表](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/registry.py) · [Attention Backend 支持矩阵](https://docs.vllm.ai/en/stable/design/attention_backends/) · [FlashAttention 官方实现与版本](https://github.com/Dao-AILab/flash-attention) · [Linear/MoE Kernel 配置](https://github.com/vllm-project/vllm/blob/main/vllm/config/kernel.py) · [CustomOp](https://docs.vllm.ai/en/stable/design/custom_op/) · [Triton Fused Attention 教程](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)。

</details>

<details>
<summary>Part 8 · 编译、CUDA Graph 与执行优化（7 章）</summary>

| 章节 | 必须覆盖的知识点 | 状态 |
| --- | --- | --- |
| 8.1 [torch.compile 集成](docs/chapters/04-compile-profiling.md) | Dynamo/FX/Inductor 与 Triton TTIR/TTGIR/目标代码；guards、断图、动态 shape、piecewise compile | 🧩 |
| 8.2 [CUDA Graph 执行模式](docs/chapters/04-compile-profiling.md) | NONE/PIECEWISE/FULL/FULL_DECODE_ONLY/FULL_AND_PIECEWISE | 🧩 |
| 8.3 [形状特化与编译缓存](docs/chapters/04-compile-profiling.md) | capture sizes、padding、warmup、缓存键、recompile、启动成本 | 🧩 |
| 8.4 [计算融合 Pass](docs/chapters/04-compile-profiling.md) | Norm+Quant、Act+Quant、Attention+Quant、QK Norm+RoPE、RoPE+KV | 🧩 |
| 8.5 通信融合与 Async TP | GEMM/collective overlap、all-reduce+RMSNorm、SP、对称内存 | 📋 |
| 8.6 优化级别、CustomOp 与 vLLM IR | 优化预设、pass 优先级、算子 dispatch/lowering、平台覆盖 | 📋 |
| 8.7 多模态图与编译排错 | ViT CUDA Graph、encoder compile、动态输入、回退、DBO/Spec 组合 | 📋 |

来源与阅读入口：[torch.compile 集成](https://docs.vllm.ai/en/stable/design/torch_compile/) · [编译模式与融合 Pass](https://github.com/vllm-project/vllm/blob/main/vllm/config/compilation.py) · [CUDA Graphs](https://docs.vllm.ai/en/stable/design/cuda_graphs/) · [Optimization Levels](https://docs.vllm.ai/en/stable/design/optimization_levels/) · [vLLM IR](https://docs.vllm.ai/en/stable/design/vllm_ir/) · [ViT CUDA Graphs](https://docs.vllm.ai/en/stable/design/cuda_graphs_multimodal/)。

</details>

<details>
<summary>Part 9 · 量化算法、算子实现与推理加速（11 章）</summary>

| 章节 | 必须覆盖的知识点 | 状态 |
| --- | --- | --- |
| 9.1 [低精度表示与计算路径](docs/chapters/05-quantization-operators.md) | INT4/INT8、FP8/FP4/MX 格式；scale/zero-point、舍入/饱和、粒度；存储/乘法/累加精度 | 🧩 |
| 9.2 [量化算法](docs/chapters/05-quantization-operators.md) | 统一比较误差目标、校准、裁剪、舍入、误差补偿与等价变换；以 GPTQ/AWQ 等为案例，区分 PTQ/QAT | 🧩 |
| 9.3 [量化策略与精度分配](docs/chapters/05-quantization-operators.md) | W/A/KV、weight-only/低精度计算、静态/动态、在线/离线、分层/分组/专家混合精度、负载感知 | 🧩 |
| 9.4 [手写量化与数据重排算子](docs/chapters/05-quantization-operators.md) | amax/reduction、Q/DQ、per-token/per-group scales、INT4 半字节打包、符号扩展、重排与尾块 | 🧩 |
| 9.5 [Weight-only GEMV 与 GEMM](docs/chapters/05-quantization-operators.md) | W4A16/W8A16、packed layout、分块反量化融合、scale 复用、decode 小 M 与 prefill 大 M | 🧩 |
| 9.6 低精度 GEMM 与缩放实现 | W8A8/W4A8、INT8/FP8/FP4 条件化路径、动态激活量化、分组缩放、累加精度、zero-point 补偿、epilogue | 📋 |
| 9.7 [融合算子与内核调优](docs/chapters/05-quantization-operators.md) | Norm/Act+Quant、GEMM epilogue、tiling、L2/共享内存复用、流水线/双缓冲、寄存器、split-K、autotune | 🧩 |
| 9.8 KV 量化与 Attention 协同 | KV 格式/布局、per-head/per-token scales、量化写入/融合读取、低比特反量化、异常层与长上下文误差 | 📋 |
| 9.9 MoE 专家量化与分组计算 | 按专家精度/scale、路由后的激活量化、grouped GEMM、打包布局、负载不均与通信协同 | 📋 |
| 9.10 [量化算子接入 vLLM](docs/chapters/05-quantization-operators.md) | QuantizationConfig→get_quant_method→create_weights→load→process_weights_after_loading→apply；布局/分片/dispatch/回退 | 🧩 |
| 9.11 [从算子收益到端到端加速](docs/chapters/05-quantization-operators.md) | microbenchmark→层→模型→服务、prefill/decode、精度/质量双重误差、TTFT/TPOT/Goodput、预处理摊销 | 🧩 |

学生先完成数学参考、打包与基础 GEMM；量化工程师可从策略/算子设计进入。贯穿实验、算子契约与验收方法见[量化篇设计](docs/quantization-design.md)。

来源与阅读入口：[量化与硬件矩阵](https://docs.vllm.ai/en/stable/features/quantization/) · [vLLM 量化方法与配置接口](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/base_config.py) · [Linear/MoE Kernel 配置](https://github.com/vllm-project/vllm/blob/main/vllm/config/kernel.py) · [Triton 矩阵乘教程](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html) · [Marlin 量化内核实现](https://github.com/IST-DASLab/marlin) · [Quantized KV Cache](https://docs.vllm.ai/en/stable/features/quantization/quantized_kvcache/) · [量化方法注册表](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/__init__.py) · [Online Quantization](https://docs.vllm.ai/en/stable/features/quantization/online/) · [LLM Compressor](https://docs.vllm.ai/en/stable/features/quantization/llm_compressor/)。

</details>

<details>
<summary>Part 10 · Speculative Decoding 推测解码（9 章）</summary>

| 章节 | 必须覆盖的知识点 | 状态 |
| --- | --- | --- |
| 10.1 [Draft、Verify 与拒绝采样](docs/chapters/07-speculation-serving.md) | 目标/草稿分布、接受率、回退、额外 token、分布保持条件 | 🧩 |
| 10.2 N-gram 与 Suffix | ngram、ngram_gpu、prompt lookup、suffix matching、自适应深度 | 📋 |
| 10.3 独立 Draft Model | 草稿模型、词表对齐、draft TP、显存预算、投机 token 数 | 📋 |
| 10.4 EAGLE 与 EAGLE3 | 特征预测、draft tree/验证、hidden state extraction、模型配套 | 📋 |
| 10.5 MTP 与模型原生多 token 预测 | MTP 模块、权重、共享状态、多模型家族、混合模型回滚 | 📋 |
| 10.6 其他草稿与并行提案 | MLP speculator、Medusa、PARD、DFlash/DSpark、speculators、custom proposer | 📋 |
| 10.7 动态推测与自适应验证 | Dynamic Speculative Decoding、负载感知、置信度、Adaptive Verification | 📋 |
| 10.8 [推测的缓存与执行组合](docs/chapters/07-speculation-serving.md) | KV 分配/回滚、CUDA Graph、异步调度、LoRA/量化/多模态/CP | 🧩 |
| 10.9 [接受率与真实收益](docs/chapters/07-speculation-serving.md) | 每请求/每位置接受率、accepted tokens、草稿成本、有效吞吐 | 🧩 |

来源与阅读入口：[Speculative Decoding](https://docs.vllm.ai/en/stable/features/speculative_decoding/) · [推测方法与模型类型](https://github.com/vllm-project/vllm/blob/main/vllm/config/speculative.py) · [EAGLE](https://docs.vllm.ai/en/stable/features/speculative_decoding/eagle/) · [MTP](https://docs.vllm.ai/en/stable/features/speculative_decoding/mtp/) · [Dynamic Speculative Decoding](https://docs.vllm.ai/en/stable/features/speculative_decoding/dynamic_speculative_decoding/) · [Adaptive Verification](https://docs.vllm.ai/en/stable/features/speculative_decoding/adaptive_verification/)。

</details>

<details>
<summary>Part 11 · 并行切分与分布式执行（10 章）</summary>

| 章节 | 必须覆盖的知识点 | 状态 |
| --- | --- | --- |
| 11.1 [并行维度与通信基础](docs/chapters/06-parallel-moe-disagg.md) | TP/PP/DP/EP/CP/PCP/DCP/SP、rank/group、all-reduce/all-gather/reduce-scatter/all-to-all | 🧩 |
| 11.2 [Tensor Parallelism](docs/chapters/06-parallel-moe-disagg.md) | TP、column/row parallel、vocab parallel、Attention heads、KV 复制 | 🧩 |
| 11.3 [Pipeline Parallelism](docs/chapters/06-parallel-moe-disagg.md) | PP、层切分、不均匀分层、microbatch、流水线气泡 | 🧩 |
| 11.4 [Data Parallelism](docs/chapters/06-parallel-moe-disagg.md) | DP、副本、内部/外部/混合负载均衡、dense 与 MoE 差异 | 🧩 |
| 11.5 [Context Parallelism 概念](docs/chapters/06-parallel-moe-disagg.md) | CP 沿上下文切分、长序列、Q/KV 分布、ring/all-gather 算法 | 🧩 |
| 11.6 [Prefill Context Parallelism](docs/chapters/06-parallel-moe-disagg.md) | PCP、prefill token 切分、rank 扩展、TP/PCP 组合、缓存布局 | 🧩 |
| 11.7 [Decode Context Parallelism](docs/chapters/06-parallel-moe-disagg.md) | DCP、decode KV 分片、复用进程组、MLA/GQA、ag_rs/a2a、Q replicate | 🧩 |
| 11.8 Sequence Parallelism 与 Async TP | SP 激活切分、TP 配合、融合 pass、序列并行 MoE 路径 | 📋 |
| 11.9 [组合并行与拓扑规划](docs/chapters/06-parallel-moe-disagg.md) | TP×PP×DP×PCP、DCP/EP 组复用、NVLink/PCIe/RDMA/NUMA | 🧩 |
| 11.10 分布式启动与排错 | mp/Ray/external launcher、节点发现、通信超时、NCCL/P2P、故障 | 📋 |

来源与阅读入口：[Parallelism and Scaling](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/) · [并行配置与度数校验](https://github.com/vllm-project/vllm/blob/main/vllm/config/parallel.py) · [Context Parallel Deployment](https://docs.vllm.ai/en/stable/serving/context_parallel_deployment/) · [Data Parallel Deployment](https://docs.vllm.ai/en/stable/serving/data_parallel_deployment/) · [Expert Parallel Deployment](https://docs.vllm.ai/en/stable/serving/expert_parallel_deployment/) · [编译模式与融合 Pass](https://github.com/vllm-project/vllm/blob/main/vllm/config/compilation.py)。

</details>

<details>
<summary>Part 12 · MoE 与通信优化（6 章）</summary>

| 章节 | 必须覆盖的知识点 | 状态 |
| --- | --- | --- |
| 12.1 [MoE 计算链路](docs/chapters/06-parallel-moe-disagg.md) | Router、top-k、shared/routed experts、dispatch、专家计算、combine | 🧩 |
| 12.2 [Expert Parallelism](docs/chapters/06-parallel-moe-disagg.md) | EP、专家切分、Attention DP/TP 与专家组、placement、冗余专家 | 🧩 |
| 12.3 Fused MoE Kernel | Modular Kernel、prepare/finalize、量化布局、batched/mega MoE | 📋 |
| 12.4 All-to-All 后端 | allgather/reducescatter、DeepEP、MoRI、NIXL EP、FlashInfer NVLink | 📋 |
| 12.5 [EPLB 与弹性专家并行](docs/chapters/06-parallel-moe-disagg.md) | EPLB、重排/冗余、异步均衡、Elastic EP、扩缩容与故障恢复 | 🧩 |
| 12.6 MoE 重叠与综合调优 | DBO、shared expert overlap、通信融合、量化、分离式部署组合 | 📋 |

来源与阅读入口：[Expert Parallel Deployment](https://docs.vllm.ai/en/stable/serving/expert_parallel_deployment/) · [并行配置与度数校验](https://github.com/vllm-project/vllm/blob/main/vllm/config/parallel.py) · [Fused MoE Modular Kernel](https://docs.vllm.ai/en/stable/design/fused_moe_modular_kernel/) · [MoE Kernel Features](https://docs.vllm.ai/en/stable/design/moe_kernel_features/) · [Linear/MoE Kernel 配置](https://github.com/vllm-project/vllm/blob/main/vllm/config/kernel.py) · [Dual Batch Overlap](https://docs.vllm.ai/en/stable/design/dbo/)。

</details>

<details>
<summary>Part 13 · 分离式部署与 KV 传输（6 章）</summary>

| 章节 | 必须覆盖的知识点 | 状态 |
| --- | --- | --- |
| 13.1 [Prefill-Decode 分离](docs/chapters/06-parallel-moe-disagg.md) | PD disaggregation、阶段独立扩展、TTFT/TPOT 目标、路由 | 🧩 |
| 13.2 [KV Connector 协议](docs/chapters/06-parallel-moe-disagg.md) | scheduler/worker 角色、异步存取、生命周期、失败回收、KV events | 🧩 |
| 13.3 连接器与分层存储地图 | NIXL、Mooncake/Store、LMCache、MoRIIO、FlexKV、HF3FS、原生 offload | 📋 |
| 13.4 [异构并行与传输兼容](docs/chapters/06-parallel-moe-disagg.md) | P/D 不同 TP/PCP/DCP、KV layout、block size、dtype、模型一致性 | 🧩 |
| 13.5 Encoder 分离与缓存 | 多模态 encoder、EC transfer、embedding cache、renderer 拆分 | 📋 |
| 13.6 [分布式服务生命周期](docs/chapters/06-parallel-moe-disagg.md) | push/pull、租约/通知、超时、重试、load failure recovery、缓存亲和 | 🧩 |

来源与阅读入口：[Disaggregated Prefilling](https://docs.vllm.ai/en/stable/features/disagg_prefill/) · [Disaggregated Encoder](https://docs.vllm.ai/en/stable/features/disagg_encoder/) · [KV Connector 注册表](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/kv_transfer/kv_connector/factory.py) · [NIXL 兼容性矩阵](https://docs.vllm.ai/en/stable/features/nixl_connector_compatibility/) · [KV Offloading](https://docs.vllm.ai/en/stable/features/kv_offloading_usage/)。

</details>

<details>
<summary>Part 14 · 模型类型、多模态与适配器（7 章）</summary>

| 章节 | 必须覆盖的知识点 | 状态 |
| --- | --- | --- |
| 14.1 [模型注册与执行任务](docs/chapters/01-request-lifecycle.md) | generative/pooling、runner/task、原生实现/Transformers backend、encoder-decoder | 🧩 |
| 14.2 文本生成与评分 | 生成、logprobs/prompt logprobs、score、generative scoring、beam search 版本差异 | 📋 |
| 14.3 Pooling 全任务 | Embedding、token embedding、分类/token classification、reward、rerank/score | 📋 |
| 14.4 多模态输入管线 | 图像/视频/音频、processor cache、输入预算、placeholder、MM embeddings | 📋 |
| 14.5 多模态执行优化 | encoder batching/DP、ViT attention/FP8、encoder compile/graphs、EC cache | 📋 |
| 14.6 LoRA 与多租户适配 | Multi-LoRA、混合 batch、动态装卸、CPU 缓存、resolver、默认多模态 LoRA | 📋 |
| 14.7 Prompt Embeddings 与特殊架构 | prompt embeds、custom I/O、MLA/混合 SSM、离散扩散 dLLM canvas/denoising、Omni 扩展边界 | 📋 |

来源与阅读入口：[模型支持与任务](https://docs.vllm.ai/en/stable/models/supported_models/) · [Pooling Models](https://docs.vllm.ai/en/stable/models/pooling_models/) · [Multimodal Inputs](https://docs.vllm.ai/en/stable/features/multimodal_inputs/) · [LoRA Adapters](https://docs.vllm.ai/en/stable/features/lora/) · [Prompt Embedding Inputs](https://docs.vllm.ai/en/stable/features/prompt_embeds/) · [ViT CUDA Graphs](https://docs.vllm.ai/en/stable/design/cuda_graphs_multimodal/) · [离散扩散语言模型配置](https://github.com/vllm-project/vllm/blob/main/vllm/config/diffusion.py)。

</details>

<details>
<summary>Part 15 · 服务接口与生成能力（7 章）</summary>

| 章节 | 必须覆盖的知识点 | 状态 |
| --- | --- | --- |
| 15.1 [离线与在线入口](docs/chapters/07-speculation-serving.md) | LLM/AsyncLLM、chat/complete/serve/run-batch、OpenAI Batch、流式 SSE | 🧩 |
| 15.2 API 能力地图 | Completions/Chat/Responses、Embeddings/Score/Rerank、音频/实时接口、tokenize/detokenize | 📋 |
| 15.3 Structured Outputs | JSON Schema、regex、choice、grammar、structural tag、XGrammar/Guidance | 📋 |
| 15.4 Tool Calling | tool_choice、自动/指定/required、parallel tools、parser、chat template | 📋 |
| 15.5 Reasoning 与 Interleaved Thinking | reasoning parser、内容分离、思考预算、工具轮次与交错推理 | 📋 |
| 15.6 自定义采样与处理器 | Custom Logits Processors、logit bias、允许/禁止 token、custom arguments/I/O | 📋 |
| 15.7 [服务治理与扩展](docs/chapters/07-speculation-serving.md) | 前端进程扩展、限流/背压、取消、健康检查、指标、集成框架 | 🧩 |

来源与阅读入口：[OpenAI-compatible Server](https://docs.vllm.ai/en/stable/serving/online_serving/openai_compatible_server/) · [Structured Outputs](https://docs.vllm.ai/en/stable/features/structured_outputs/) · [Tool Calling](https://docs.vllm.ai/en/stable/features/tool_calling/) · [Reasoning Outputs](https://docs.vllm.ai/en/stable/features/reasoning_outputs/) · [Interleaved Thinking](https://docs.vllm.ai/en/stable/features/interleaved_thinking/) · [Custom Logits Processors](https://docs.vllm.ai/en/stable/features/custom_logitsprocs/)。

</details>

<details>
<summary>Part 16 · 权重加载、内存与 RL 推理生命周期（6 章）</summary>

| 章节 | 必须覆盖的知识点 | 状态 |
| --- | --- | --- |
| 16.1 [模型解析与权重加载](docs/chapters/01-request-lifecycle.md) | 本地/Hugging Face/ModelScope、revision、safetensors、sharded state、lazy loading | 🧩 |
| 16.2 高效加载与启动 | fastsafetensors、Run:ai Streamer、Tensorizer、InstantTensor、加载并发、JIT warmup | 📋 |
| 16.3 权重 Offload 与内存预算 | CPU weight offload、prefetch/UVA 路径、KV offload、pinned memory、图内存 | 📋 |
| 16.4 Sleep、Wake 与 Pause/Resume | sleep levels、权重/KV 释放、wake tags、暂停恢复、缓存重置 | 📋 |
| 16.5 权重更新与传输 | Weight Transfer、IPC/NCCL/Sharded RDT、逐层 reload、版本一致性 | 📋 |
| 16.6 强化学习推理集成 | RLHF/async rollout、训练-推理交替、distribution replay、routed experts、TRL | 📋 |

来源与阅读入口：[Model Resolution](https://docs.vllm.ai/en/stable/configuration/model_resolution/) · [Conserving Memory](https://docs.vllm.ai/en/stable/configuration/conserving_memory/) · [Sleep Mode](https://docs.vllm.ai/en/stable/features/sleep_mode/) · [Weight Transfer](https://docs.vllm.ai/en/stable/training/weight_transfer/) · [Async Reinforcement Learning](https://docs.vllm.ai/en/stable/training/async_rl/) · [Layerwise Reloading](https://docs.vllm.ai/en/stable/training/layerwise/)。

</details>

<details>
<summary>Part 17 · 评测、性能分析与优化方法（6 章）</summary>

| 章节 | 必须覆盖的知识点 | 状态 |
| --- | --- | --- |
| 17.1 [基准测试工具与负载](docs/chapters/04-compile-profiling.md) | bench latency/throughput/serve/startup/mm-processor、trace replay、合成/真实数据 | 🧩 |
| 17.2 [延迟目标与参数搜索](docs/chapters/04-compile-profiling.md) | SLO/Goodput、并发、输入输出分布、sweep、Pareto、冷/热缓存 | 🧩 |
| 17.3 [Profiling 全链路](docs/chapters/04-compile-profiling.md) | PyTorch Profiler、Nsight/厂商工具、CPU/GPU timeline；语义/shape/dtype 对齐、次数与单次耗时、关键路径 | 🧩 |
| 17.4 观测与请求级指标 | Prometheus/Grafana、OpenTelemetry、KV/调度/Spec/MoE 指标、per-request metrics | 📋 |
| 17.5 质量、复现与 Batch Invariance | 数值容差、模型质量、长上下文、seed、批次不变性、确定性代价 | 📋 |
| 17.6 [系统消融与综合项目](docs/chapters/04-compile-profiling.md) | APC×Chunked Prefill×Graphs×Quant×Spec、双平台成本、容量规划 | 🧩 |

来源与阅读入口：[Benchmark CLI](https://docs.vllm.ai/en/stable/benchmarking/cli/) · [Parameter Sweeps](https://docs.vllm.ai/en/stable/benchmarking/sweeps/) · [Profiling vLLM](https://docs.vllm.ai/en/stable/contributing/profiling/) · [Production Metrics](https://docs.vllm.ai/en/stable/usage/metrics/) · [Per-Request Metrics](https://docs.vllm.ai/en/stable/features/per_request_metrics/) · [Batch Invariance](https://docs.vllm.ai/en/stable/features/batch_invariance/)。

</details>

<details>
<summary>Part 18 · 平台扩展、部署生态与开源贡献（5 章）</summary>

| 章节 | 必须覆盖的知识点 | 状态 |
| --- | --- | --- |
| 18.1 平台与插件机制 | out-of-tree platform、模型/算子/量化/LoRA/I/O 插件、能力检测 | 📋 |
| 18.2 C500 适配专题 | MACA/mcoplib、Attention/量化/图执行能力、配置迁移、数值和性能排查 | 📋 |
| 18.3 其他硬件与框架对照 | ROCm/XPU/CPU/TPU/其他插件，SGLang、TensorRT-LLM、llama.cpp、vLLM-Omni 边界 | 📋 |
| 18.4 部署与周边生态 | Docker/Kubernetes/Helm/Ray/Nginx、路由、llm-d/Dynamo/Production Stack、应用集成 | 📋 |
| 18.5 从问题到贡献与版本维护 | 单元/集成/性能检查、最小复现、PR、废弃迁移、功能台账更新 | 📋 |

来源与阅读入口：[Plugin System](https://docs.vllm.ai/en/stable/design/plugin_system/) · [CustomOp](https://docs.vllm.ai/en/stable/design/custom_op/) · [vLLM-MetaX](https://github.com/MetaX-MACA/vLLM-metax) · [MetaX 模型列表](https://vllm-metax.readthedocs.io/en/stable/models/supported_models.html) · [Kubernetes 部署](https://docs.vllm.ai/en/stable/deployment/k8s/) · [vLLM Features 与兼容性入口](https://docs.vllm.ai/en/stable/features/) · [SGLang Hyperparameter Tuning](https://docs.sglang.io/docs/advanced_features/hyperparameter_tuning)。

</details>

<!-- END course-outline -->

## 特性与优化地图

课程按 **2026-09-15 查阅的 vLLM 官方功能文档、设计说明与源码注册表**扩展。详细的术语、设备条件、组合实验和注册项见[完整特性地图](docs/vllm-feature-map.md)；[14 组、233 个注册/配置命名条目](docs/feature-inventory.json)用于后续检查遗漏，其中包含别名、模型映射与占位项。

### 重点覆盖

| 专题 | 已纳入的内容 |
| --- | --- |
| 量化 | 算法统一比较、W/A/KV 策略与混合精度；手写 Q/DQ、INT4 打包、weight-only GEMV/GEMM、低精度矩阵乘、融合/流水线/访存优化、KV/专家量化、vLLM 算子接入与端到端消融 |
| 缓存 | Paged KV、APC/Prefix Cache、哈希/盐/隔离、逐出/reset、KV events、prefix match 粒度、Hybrid/SWA/Mamba/GDN、检查点、KV sharing、KV offload、IndexCache |
| Attention 与算子 | FlashAttention 1/2/3/4、FlashInfer、Triton、SDPA/FlexAttention、FlashMLA/CUTLASS MLA、稀疏/分页/变长/cascade attention、GEMM/GEMV、融合算子与 autotune |
| 调度与重叠 | Continuous Batching、FCFS/priority、Chunked Prefill、token/显存预算、抢占、async scheduling、DBO、TBO/SBO 对照、shared expert overlap、Async TP |
| 编译与图执行 | torch.compile、动态 shape、特化、编译缓存、JIT warmup、所有 CUDA Graph 模式、计算/通信融合 Pass、优化级别、CustomOp、vLLM IR、ViT graphs |
| 推测解码 | Draft Model、N-gram/GPU N-gram、Suffix、EAGLE/EAGLE3、MTP、MLP/Medusa/PARD、DFlash/DSpark、自定义 proposer、动态推测、自适应验证、接受率与回滚 |
| 并行切分 | **TP、PP、DP、EP、CP、PCP、DCP、SP**；进程组、张量布局、组合度数、collectives、拓扑、mp/Ray/多机排错 |
| MoE | Router/top-k、专家 placement、dispatch/combine、Fused/Modular/Mega MoE、DeepEP/MoRI/NIXL、EPLB、Elastic EP、负载均衡与通信重叠 |
| 分离式部署 | PD、KV Connector、NIXL/Mooncake/LMCache/FlexKV/HF3FS、异构并行映射、分层缓存、Encoder/EC 分离、租约/通知/失败恢复 |
| 模型与服务 | 离散扩散 dLLM、多模态、Pooling/Embedding/分类/Reward/Rerank、LoRA、Prompt Embeddings、Structured Outputs、Tool Calling、Reasoning/Interleaved Thinking、Batch/流式/音频接口 |
| 生命周期与运维 | 高效权重加载、weight offload、Sleep/Wake、Pause/Resume、权重传输/reload、RL rollout、bench/sweep、SLO/Goodput、profiling、监控、Batch Invariance、平台插件 |

### 并行术语速查

| 名称 | 切分对象 | 需要记住的区别 |
| --- | --- | --- |
| TP | 一层内的张量/权重 | 与 head、MLP、词表和 collective 对应 |
| PP | 模型层/阶段 | 要分析流水线气泡与负载 |
| DP | 请求/副本 | dense 与 MoE 的组关系不同 |
| EP | MoE 专家 | token 路由与 All-to-All；不机械增加一个独立乘数 |
| CP | 上下文维度 | 本课程写全称 Context Parallelism |
| PCP | prefill 序列计算 | 本次源码会扩展进程维度；是否能运行仍看模型和后端 |
| DCP | decode 上下文 KV | 复用进程组，不额外乘一次 GPU 数 |
| SP | 部分中间激活的序列维度 | 配合 TP/编译及 MoE 路径，不等同于 PCP/DCP |

**Chunked Prefill 也曾被缩写为 CP，但它属于调度策略，单卡即可学习。** PCP/DCP 的合法度数、Q/KV 分布与模型限制见[并行策略详解](docs/vllm-feature-map.md)和[上游并行配置](https://github.com/vllm-project/vllm/blob/main/vllm/config/parallel.py)。

**TBO 已纳入对照专题，vLLM 侧使用官方名称 DBO（Dual Batch Overlap）。** 课程分别讲 DBO、异步调度、Async TP，以及 SGLang 的 TBO/SBO，避免混用参数。[vLLM DBO 设计](https://docs.vllm.ai/en/stable/design/dbo/)

### 学习范围与实验范围

- 单卡 A100/C500：先做模型基线，再逐项验证缓存、调度、Attention、编译、量化、推测和服务能力。
- 多卡/多机：TP/PP/DP/EP/PCP/DCP/SP/DBO 与分离式部署都有正式章节、推导和实验设计；真实通信性能需要额外设备。
- 硬件/模型专用功能：FP8/FP4、特定 FlashAttention 代际、稀疏模型与专用 kernel 全部收录，并注明支持条件。
- 功能组合：单独设计 APC×Chunked Prefill、Quant×KV dtype、Graph×Spec、EP×DBO、PD×并行映射等消融实验。

当前新增章节为**教学规划**，A100/C500 的 GPU 实测状态仍为待验证。完整支持矩阵随固定版本和实机记录逐项建立。


## 框架阅读地图

学习 vLLM 时，也可以借助其他项目理解不同层次的设计。下表给出课程计划采用的阅读视角；具体代码以各项目的固定版本为准。

| 项目 | 在课程中的角色 | 带着什么问题阅读 |
| --- | --- | --- |
| [Transformers](https://github.com/huggingface/transformers) | 模型定义、权重加载与生成的参考 | 模型怎样定义和加载？生成循环怎样调用前向计算？ |
| [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) | 精简 vLLM 风格实现的阅读材料 | 缓存、调度、模型执行怎样组合成一个小型引擎？ |
| [mini-sglang](https://github.com/sgl-project/mini-sglang) | 模块化教学实现的阅读材料 | 分块 prefill、前缀缓存和调度如何协作？ |
| [vLLM](https://github.com/vllm-project/vllm) | 核心源码与正式服务实验 | PagedAttention、连续批处理与模型执行怎样落地？ |
| [SGLang](https://github.com/sgl-project/sglang) | 缓存与运行时设计的比较阅读 | RadixAttention 如何组织前缀复用？与 vLLM 的缓存管理如何对应？ |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | 本地推理与资源受限场景的扩展阅读 | C/C++ 执行、量化与 CPU/GPU 协同怎样影响部署选择？ |
| [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | NVIDIA 平台优化的扩展阅读 | 运行时与专用算子如何配合？性能结论依赖哪些硬件条件？ |
| [vLLM-MetaX](https://github.com/MetaX-MACA/vLLM-metax) | MACA 路径的正式插件与源码 | 平台插件、适配算子和软件版本如何协同？ |

学生先按课程完成自己的小实现，再按章节阅读对应项目；工程师可以围绕同一个问题比较实现。**框架介绍和阅读材料的覆盖范围，与本工程已经实测的平台范围分别记录。**

## 双后端实践如何组织

公共课程和实验口径共用，运行环境按后端分别管理：

| 路径 | 正式运行栈 | 首批验证目标 | 当前状态 |
| --- | --- | --- | --- |
| NVIDIA CUDA | 上游 vLLM 与对应 CUDA/PyTorch 环境 | 单卡 A100 | 配置模板已有，待实测 |
| 沐曦 MACA | vLLM + vLLM-MetaX + 配套 MACA/算子库 | 单卡 C500 | 配置模板已有，待实测 |

模型、输入、dtype、采样参数和负载方法尽量统一；驱动、镜像、注意力后端与实际版本分别记录。先在每个平台建立正确性和性能基线，再分析两套系统的差异。[环境说明](docs/backends.md) · [实验记录规范](docs/experiments.md)

教学引擎先完成可检查的参考实现，再逐步接入优化算子。当前的 CPU 数学实验与逻辑块模拟，是后续 PyTorch 和 GPU 实现的起点。

## 当前进度

当前代码版本：**v0.1.0.dev0**。

| 已有成果 | 验证或阅读入口 |
| --- | --- |
| 工程设计、课程分层和迭代验收标准 | [工程设计](docs/engineering.md)、[开发路线](docs/roadmap.md) |
| 19 篇、126 章大纲与特性/优化索引 | [详细目录](docs/curriculum.md)、[特性地图](docs/vllm-feature-map.md)、[一致性检查](reports/curriculum-checks.json) |
| Part 1 全六节：正文、CPU 代码、参数命令、练习答案与统一运行 | [整篇学习入口](docs/chapters/part1-guide.md)、[运行器](scripts/run_part1.py)、[验证记录](reports/validation.md) |
| 因果 Attention、KV Cache 与贪心生成 | [参考代码](zero2vllm/reference/attention.py)、[引导实验](docs/labs/01-kv-cache.md) |
| 独立的逻辑块分配与回收模拟 | [块管理代码](zero2vllm/reference/blocks.py)；尚未接入 Attention |
| 环境采集、A100/C500 配置检查 | [环境工具与配置说明](docs/backends.md) |
| CPU 数学参考与本地检查结果 | [验证记录](reports/validation.md)、[KV 实验](reports/cpu-kv-cache.json)、[INT4 实验](reports/cpu-int4-linear.json) |
| 38 份资料到课程的映射、7 篇专题导读 | [知识库入口](docs/knowledge-base.md)、[本轮更新报告](reports/knowledge-update.md) |
| 三条学习路线、章节分工与目录同步检查 | [学习路线](docs/learning-paths.md)、[维护方法](CONTRIBUTING.md)、[框架审查记录](reports/framework-review.md) |

**接下来：**试学 Part 1 并完成综合练习，再进入 Part 2 的 PyTorch 教学引擎。A100/C500 环境与小模型基线可同时准备；量化 GPU 算子和 vLLM 接入按[开发路线](docs/roadmap.md)推进。每个阶段以代码、正文和验证记录共同验收。

## 项目结构

```text
zero2vllm/
├── docs/                    # 课程目录、功能台账、引导实验、设计与平台说明
│   ├── curriculum-catalog.json  # 章节、练习、设备和来源的维护入口
│   ├── feature-inventory.json   # 本次核对的后端与配置枚举
│   ├── knowledge-sources.json   # 本地资料来源、哈希与章节映射
│   ├── knowledge-base.md        # 7 篇导读与资料使用说明
│   ├── chapters/               # Part 1 六节正文、整篇导航与各专题导读
│   ├── learning-paths.md        # 三条学习路线与交叉章节分工
│   ├── vllm-feature-map.md      # 功能地图、并行术语和组合约束
│   ├── labs/                # 已编写的实验教程
│   └── templates/           # 章节与实验记录模板
├── zero2vllm/
│   ├── reference/           # CPU 数学参考与逻辑块模拟
│   ├── profiles.py          # 双后端配置检查
│   ├── environment.py       # 环境信息采集
│   └── cli.py               # Part 1 六节实验 / demo / quant-demo / doctor / profile
├── vllm/                    # 本地参考知识库，可选，默认不提交
├── configs/                 # A100、C500 环境配置模板
├── scripts/                 # 环境采集、目录生成与本地文档检查
├── examples/part1/           # 明确标记为模拟的请求时间线
├── tests/                   # 数学正确性、缓存行为与配置检查
├── reports/                 # 实际结果与验证记录
├── .github/                 # CPU 自动检查、Issue 表单与 PR 模板
├── CONTRIBUTING.md          # 贡献流程与目录维护
├── CHANGELOG.md             # 更新记录
├── LICENSE                  # CC BY-NC-SA 4.0 许可正文
├── .editorconfig            # 文本格式约定
├── .gitattributes           # Git 换行约定
├── .gitignore               # 排除本地配置、模型权重和运行缓存
└── pyproject.toml
```

代码沿单一源码树演进，后续通过阶段标签对应课程进度。具体模块职责见[工程设计](docs/engineering.md)。

## 常见问题

**没有 GPU，能学到哪一步？**

现在即可完成 CPU KV Cache、INT4 打包和 Linear 实验，阅读 7 篇导读，修改逻辑块模拟，并完成调度、切分、通信预算的手算。GPU 内核与实际服务性能需要对应设备验证。

**需要先掌握 CUDA，或同时拥有 A100 和 C500 吗？**

可以先从 Python 和基础实验开始。GPU 实践选择一条路径即可，底层编程知识在后端和优化章节中按需补充。

**“手写 mini-vLLM”与当前代码是什么关系？**

它是课程的阶段目标。目前已完成数学参考和部分缓存逻辑，完整的 PyTorch 模型、调度器和服务仍按大纲推进。

**多卡专题需要我现在就有多张卡吗？**

可以先完成张量切分、进程组和通信成本的 CPU 推导/模拟，再阅读源码。真正的通信、DBO、EP 与多机性能需要相应设备；单卡结果不能代替这部分验证。

**这里的“完整”如何维护？**

以查阅日期对应的官方公开功能和配置为覆盖目标，保留注册表清单和章节映射。模型、后端、插件和组合持续变化，更新时同时记录新增、废弃及平台条件，不把一个大纲描述成永久完整的支持矩阵。

**课程会跟随所有框架的最新版本吗？**

当前导读注明查阅来源与适用范围。进入完整源码实验和 GPU 验收时再锁定 tag/commit、插件和环境版本；新版本通过独立验证后更新。其他框架也保留各自的版本和实验条件。

## 参与贡献

可以从一个小问题开始：

- 改进概念解释、公式推导或练习。
- 补充正确性测试，复现并修复缓存和调度问题。
- 提交 A100/C500 的实际环境与实验记录。
- 整理框架源码阅读笔记、性能分析或环境排查过程。

章节采用统一的“基础内容 + 动手实验 + 工程进阶”结构。提交内容应注明运行条件和真实状态；代码改动附相关检查，性能结论附原始数据。

[贡献说明](CONTRIBUTING.md) · [章节模板](docs/templates/chapter.md) · [实验记录规范](docs/experiments.md) · [维护者上传说明](docs/github-publishing.md)

## 参考与致谢

课程组织方式受到 [Datawhale zero-to-sglang](https://github.com/datawhalechina/zero-to-sglang) 的启发：让基础理解、逐步实现和真实源码阅读形成连续的学习过程。感谢 vLLM、SGLang、nano-vllm、mini-sglang、Transformers、llama.cpp、TensorRT-LLM 与 MetaX 社区提供的公开源码和技术资料。

本项目的设计与当前代码独立编写。各参考项目的角色与来源见上方阅读地图及[参考资料](docs/references.md)。

---

If this project helps you, please give it a Star ⭐!

## 📄 License

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

This work is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/).

---

<div align="center">

帮助更多人系统学习大模型推理引擎

Made with ❤️ by zero2vllm

</div>
