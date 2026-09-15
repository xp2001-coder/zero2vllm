# 知识库如何进入课程

本轮读取本地 `vllm/` 中的 **33 个 ZIP 内的 Markdown 和 5 个独立 Markdown，共 38 份资料**：vLLM 20 讲、SGLang 15 讲和 3 份总结/测试手册。盘点日期：2026-09-15。

课程保留 19 篇、126 章的范围。本轮把资料转化为 **7 篇专题导读、一个新的 CPU 量化实验、章节练习和测量规范**。导读是若干章节的共同起点，目录标为“部分材料”，不表示整篇课程或 GPU 实验已经完成。

## 从这里开始读

| 材料 | 解决的问题 | 当前可交付的练习 |
| --- | --- | --- |
| [请求、状态与张量](chapters/01-request-lifecycle.md) | 一次请求经过哪些组件，谁准备张量、谁持有状态？ | 请求链路、加载流程和版本对照表 |
| [调度与缓存](chapters/02-scheduling-cache.md) | token 预算、KV 容量和前缀命中如何共同影响调度？ | 手算调度轮次、块寻址与缓存实验设计 |
| [Attention 与算子](chapters/03-attention-kernels.md) | FlashAttention、分页存储和后端选择分别解决什么？ | Online Softmax 推导、Triton 实现顺序 |
| [编译、图执行与 Profiling](chapters/04-compile-profiling.md) | 如何确认某项优化真的生效？ | 编译/图模式消融、同 shape 时间线分析 |
| [量化策略与算子](chapters/05-quantization-operators.md) | 从算法选择走到布局、内核和 vLLM 集成 | [CPU INT4 Linear 实验](labs/02-int4-linear.md)与 GPU 后续任务 |
| [并行、MoE 与分离式部署](chapters/06-parallel-moe-disagg.md) | 切什么、怎么通信、需要几张卡？ | TP/CP 推导、专家路由和 KV 传输账本 |
| [推测解码与服务闭环](chapters/07-speculation-serving.md) | 接受率、缓存提交和流式结果怎样对应？ | 接受/拒绝手算、停止与取消场景设计 |

学生先读前两篇并运行 CPU 实验；工程师先用第四篇定义测量方法；量化方向可以直接从第五篇和 INT4 实验进入。

## 资料、课程和实测的关系

1. **本地资料**提供问题、案例和阅读线索。原文包含不同版本、平台与作者的解释，保存原样。
2. **课程正文**围绕同一教学目标重新组织；涉及当前 API、默认值和支持条件时核对官方设计或源码，注明范围。
3. **实验记录**必须来自实际执行。CPU 数学验证、模拟和 GPU 测量分开登记，原资料里的性能图不计作我们的实测。

`vllm/` 是可选的本地参考库，`work/` 是整理过程的临时文件，两者默认被 Git 忽略。公开课程正文不依赖读者持有这些 ZIP；目录中的本地文件名用于维护者定位资料。原资料不会因放进本项目而自动改变许可。

[结构化来源台账](knowledge-sources.json)记录来源 ID、文件、ZIP 内成员、原始 Markdown 字节的 SHA-256、吸收重点和章节映射。哈希用于识别文本变化，不能证明文字结论正确。ZIP 图片与外链图片未逐张审查，未执行笔记中的部署命令。

## 本轮处理的版本与概念问题

| 资料中需要区分的内容 | 课程采用的处理方式 |
| --- | --- |
| V0/V1 与 ModelRunner V1/V2 | 引擎架构代际和执行器实现代际分别记录；源码正文不沿用旧教程行号 |
| MRV2 × CUDA Graph 的性能比较 | runner 与图模式分别控制变量；无原始测量记录的百分比不写入结果 |
| SchedulerOutput 与 GPU 输入 | 前者表达调度决策；runner 将其转换为张量、位置和 Attention metadata |
| Prefill、Decode 与采样 | 部分 prefill 不必返回新 token；投机验证时每请求可处理多个 token |
| APC 与 RadixAttention | 对照前缀复用机制，分别查阅 vLLM 与 SGLang 的实现；不互换配置或类名 |
| Chunked Prefill 与 CP | 前者分轮安排计算，后者跨设备分上下文；单卡分块实验不算 CP 实测 |
| DBO 与 TBO/SBO | vLLM DBO 按 DP+EP 等条件查验；SGLang 名称留在框架对照中 |
| 算法、checkpoint 格式、kernel 格式 | 量化篇分别登记；文件写着 INT4 不足以证明运行的是融合低比特 GEMM |
| AWQ 变换及 TinyChat 引用 | 解释缩放补偿的实际落点，回到算法作者的实现；不复用资料中的错误项目链接或加速倍数 |
| 旧版 PD 补丁 | 保留生命周期问题，回到当前 connector 协议与示例核对，不将旧补丁作为通用修复 |
| C600/多卡部署记录 | 用于构造排查问题；A100/C500 单卡仍需各自的设备、版本、模型和原始数据 |

本轮重点核对了 [vLLM 架构](https://docs.vllm.ai/en/latest/design/arch_overview/)、[MRV2 设计](https://docs.vllm.ai/en/latest/design/model_runner_v2/)、[APC](https://docs.vllm.ai/en/latest/design/prefix_caching/)、[CUDA Graph](https://docs.vllm.ai/en/latest/design/cuda_graphs/)、[DBO](https://docs.vllm.ai/en/latest/design/dbo/)、[Context Parallel](https://docs.vllm.ai/en/latest/serving/context_parallel_deployment/)和[量化插件接口](https://docs.vllm.ai/en/latest/features/quantization/)。这些动态页面是阅读来源，尚未锁定为 A100/C500 的运行版本。

## 38 份资料到章节的映射

下表由来源台账生成。章节内容和状态见[详细课程目录](curriculum.md)，逐份资料的使用边界见 JSON 中的 `review_note`。

<!-- BEGIN knowledge-map -->

| 来源 ID | 本地资料文件 | 对应章节 | 吸收重点 |
| --- | --- | --- | --- |
| `vllm-01` | `vllm/第1课-vLLM 快速上手和环境搭建.zip` | 0.1、0.2、4.1、15.1 | 离线/在线入口、调试与流式生成 |
| `vllm-02` | `vllm/第2课-vLLM 核心组件-引擎模块和流式执行.zip` | 3.1、6.4、15.1、15.7 | EngineCoreClient、消息边界、请求到流式输出 |
| `vllm-03` | `vllm/第3课-vLLM 核心组件-Worker_Executor 详解及协作.zip` | 3.1、3.3、11.1、11.10 | Executor/Worker 分工、初始化与 RPC |
| `vllm-04` | `vllm/第4课-vLLM 核心组件-调度器如何实现.zip` | 3.2、6.1、6.2、6.3 | 等待/运行队列、token 预算、执行反馈 |
| `vllm-05` | `vllm/第5课-vLLM核心组件-显存管理之PagedAttention  副本.zip` | 3.2、5.1、5.2、5.3、5.4 | 块分配、哈希、引用计数、逐出与重算 |
| `vllm-06` | `vllm/第6课-vLLM 核心组件-ModelRunner之模型加载详解.zip` | 3.3、14.1、16.1 | 模型注册、实例化、权重加载与自定义模型 |
| `vllm-07` | `vllm/第7课-vLLM 架构-总览及模型推理过程详解.zip` | 3.1、3.2、3.3 | 端到端调用链、输入张量化与状态更新 |
| `vllm-08` | `vllm/第8课-OpenAI Triton 实现大模型的新算子 副本.zip` | 7.1、7.6、7.7、9.4 | Triton 向量/Softmax/Matmul/Attention 的递进练习 |
| `vllm-09` | `vllm/第9课-vLLM 高级特性-cuda graph 详解.zip` | 8.2、8.3 | 固定地址、捕获、回放、分桶与调度 |
| `vllm-10` | `vllm/第10课-vLLM 量化-基础知识总结 副本.zip` | 9.1、9.2、9.3 | 量化算术、误差目标、校准及通道缩放案例 |
| `vllm-11` | `vllm/第11课-vLLM 量化-模块解析与实践 副本.zip` | 9.4、9.5、9.6、9.10、9.11 | 配置到量化方法、权重后处理和真实 kernel 分支 |
| `vllm-12` | `vllm/第12课-vLLM 分布式推理-数据并行详解 副本.zip` | 11.1、11.4 | 请求副本、DP 路由及 Dense/MoE 区别 |
| `vllm-13` | `vllm/第13课-vLLM 分布式推理-张量_流水并行详解 副本.zip` | 11.1、11.2、11.3、11.9 | 列/行切分、PP 气泡与通信原语 |
| `vllm-14` | `vllm/第14课-vLLM 分布式推理-专家并行 副本.zip` | 9.9、12.1、12.2、12.3、12.4 | 专家 dispatch/compute/combine、显式/隐式重排 |
| `vllm-15` | `vllm/第15课-vLLM 分布式推理-EP 负载均衡 副本.zip` | 12.5、12.6 | 逻辑/物理专家、滑动窗口、重排与迁移成本 |
| `vllm-16` | `vllm/第16课-vLLM 高级特性-采样和投机解码 副本.zip` | 10.1、10.3、10.4、10.5、10.6、10.8、10.9 | 提案/验证/接受、草稿成本与 KV 回滚 |
| `vllm-17` | `vllm/第17课-vLLM 高级特性-模型编译 副本.zip` | 8.1、8.3、8.4、8.6 | Dynamo/FX/Inductor、分段编译、缓存与融合 |
| `vllm-18` | `vllm/第18课-vLLM 性能分析和瓶颈定位 副本.zip` | 17.1、17.2、17.3、17.4、17.6 | Profiler/Nsight、关键路径与负载匹配 |
| `vllm-19` | `vllm/第19课-vLLM高级特性之PD分离技术详解-旧版但详细 副本.zip` | 13.1、13.2、13.4、13.6 | PD 协议、传输、接收完成与失败回收 |
| `vllm-20` | `vllm/第20课-vLLM 核心组件-ModelRunnerV2 详解 副本.zip` | 3.4、6.4、8.2、10.8 | 持久状态与执行排列、异步依赖、增量写和采样 |
| `vllm-manual` | `vllm/vLLM原理特性与逐项测试手册.md` | 0.3、4.3、6.3、9.11、13.4、17.1、17.2、17.3、17.6、18.2 | W0—W7 负载、A/B、指标、trace 分析和逐项验证 |
| `vllm-notes` | `vllm/vLLM.md` | 3.1、3.2、5.1、6.3、17.3、18.2 | CPU 元数据与设备张量；平台排查的问题清单 |
| `sglang-01` | `vllm/SGLang/第1课：sglang介绍，环境搭建以及快速上手 副本.zip` | 0.3、18.3 | 框架定位、环境与跨框架比较方法 |
| `sglang-02` | `vllm/SGLang/第2课：sglang双层体系架构全景 副本.zip` | 3.1、15.3、18.3 | 前端语言层与运行时分工 |
| `sglang-03` | `vllm/SGLang/第3课-SGLang 请求端到端生命周期与后端运行时 副本.zip` | 3.1、6.4、15.1、18.3 | TokenizerManager/Scheduler/DetokenizerManager 对照 |
| `sglang-04` | `vllm/SGLang/第4课-SGLang RadixAttention 副本.zip` | 5.2、5.3、5.4、18.3 | Radix 树前缀复用、锁与逐出 |
| `sglang-05` | `vllm/SGLang/第5课-SGLang模型加载过程端到端详解 副本.md` | 14.1、16.1、18.3 | 模型构建与权重加载对照 |
| `sglang-06` | `vllm/SGLang/第6课： Sglang中Schedule核心流程深度解读 副本.zip` | 6.1、6.4、6.5、18.3 | 调度循环、重叠与批次更新 |
| `sglang-07` | `vllm/SGLang/第7课： Sglang chunked prefill深度技术解读 副本.zip` | 6.2、6.3、18.3 | chunk/token 预算和抢占条件 |
| `sglang-08` | `vllm/SGLang/第8课_ Sglang模型量化基础详细介绍 副本.zip` | 9.1、9.2、9.3 | 量化对象、粒度、静态/动态与误差 |
| `sglang-09` | `vllm/SGLang/第9课： Sglang AWQ量化深度解读 副本.zip` | 9.2、9.5、9.7、9.10 | 算法/文件格式/执行格式解耦，加载后重排 |
| `sglang-10` | `vllm/SGLang/第10课：LLM attention架构及其变体创新详细介绍 副本.zip` | 1.5、7.1、7.4、11.5、11.6、11.7 | MHA/GQA/MQA/MLA 与 KV 容量 |
| `sglang-11` | `vllm/SGLang/第11课：SGLang CUDA Graph  副本.md` | 8.2、8.3、18.3 | CUDA Graph runner/backend 分层对照 |
| `sglang-12` | `vllm/SGLang/第12课：SGlang torch.compile技术深度解读 副本.zip` | 8.1、8.3、8.4、18.3 | 编译阶段、guards、断图与缓存 |
| `sglang-13` | `vllm/SGLang/第13课：Triton编译全流程深度解读 副本.zip` | 7.6、7.7、8.1、18.1 | Triton IR/lowering、编译缓存和后端差异 |
| `sglang-14` | `vllm/SGLang/第14课：SGLang 推测解码 副本.zip` | 10.1、10.4、10.8、10.9、18.3 | 树状验证、接受路径、bonus token 与缓存提交 |
| `sglang-15` | `vllm/SGLang/第15课：Sglang分布式原理详细介绍.zip` | 11.1、11.2、11.3、11.4、11.5、11.6、11.7、12.2、13.1 | 并行维度、通信、长上下文与分离部署对照 |
| `sglang-notes` | `vllm/SGLang/sglang.md` | 5.2、6.5、10.9、11.9、17.3、18.3 | 术语对照、容量排查、调参方法；跨框架案例 |

<!-- END knowledge-map -->

## 下一次知识库更新

1. 比较文件名、ZIP 内 Markdown 成员与文本哈希；新增资料先分清框架、版本、设备和证据类型。
2. 更新台账的吸收重点和章节映射；同一主题合并到已有章，算法案例放入 9.2。
3. 在正文补充解释、可检验的问题和来源；已存在的具体实现以所选源码版本为准。
4. 修改课程 JSON 的主题、练习、`reading` 和真实状态，再运行文档同步与检查。
5. 有新代码时运行相应测试；GPU 成功、失败和未运行分别记录。

```bash
python scripts/course_docs.py --write
python scripts/course_docs.py --check
python -m unittest discover -s tests -v
```

文档检查验证映射和本地一致性，不自动判断新笔记是否正确，也不会自动从 ZIP 发布原文。
