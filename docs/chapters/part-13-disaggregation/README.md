# Part 13 · 分布式推理工程：PD 分离、KV 系统与集群调度

[课程总目录](../README.md) · [学习路线](../learning-paths.md)

> **13.1已交付正文、离线分析代码、命令和练习答案。** 其余11节为工程规划；性能查询器、调度模拟器和真实多机实验尚待交付。[13.1](01-prefill-decode-disaggregation/README.md)使用明确标注的合成轨迹验证分析口径，不代表真实PD性能已经验证。

**主要读者：推理服务工程师、GPU 性能工程师和平台工程师。** 目标是能够为真实业务做容量决策、部署分布式推理服务并承担运行维护。建议已有单实例部署、压测和日志分析经验，补读 Part 5、6、11 与17.1—17.2；Kubernetes 案例另需容器、网络和编排基础。

本篇属于工程进阶路线，学生可在完成单机路线后选读。A100/C500 单卡仍用于各自平台的阶段测量；集群验收需要额外设备与真实网络。CPU 建模是工程设计工具，不构成本篇的完整验收。

## 1. 为什么扩充本篇

本篇要回答一个完整问题：**给定模型、硬件、业务流量和延迟要求，怎样决定P/D实例数量、每实例的并行布局、KV传输和存储方案，并证明这套配置真的满足要求？**

原有内容偏重Connector、布局与生命周期。现在补上前半段的测量和配比，以及后半段的容量校准，按下面顺序组织：

**负载协议 → P/D性能表 → SLO查询 → 实例/卡数初估 → KV与队列约束 → Dynamo等控制面 → 生产服务验收。**

PD分离、分布式KV存储、缓存感知调度可以组合，但各有收益和成本。不能把“采用PD”直接写成“吞吐一定提高”。[DistServe](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin)把SLO、P/D资源与并行放置一起优化；[vLLM的功能说明](https://docs.vllm.ai/en/latest/features/disagg_prefill/)强调阶段延迟隔离，并明确提示不能把该功能当作吞吐提升保证。课程必须保留同资源下的混部与Chunked Prefill基线。

## 2. 十二节与四个学习模块

| 模块 | 小节 | 主要问题 |
| --- | --- | --- |
| A · 测量与配比 | 13.1—13.4 | 怎样测P/D，怎样按SLO查表，需要多少实例与GPU？ |
| B · KV数据路径 | 13.5—13.8 | 搬多少字节、怎样搬、怎样存，何时重算更划算？ |
| C · 集群控制面 | 13.9—13.11 | 怎样放置实例，用 Dynamo/llm-d 路由和扩缩容，多模态如何加入？ |
| D · 生产运维与验收 | 13.12 | 怎样观测、排空、回滚，故障和突发下还能否满足 SLO？ |

<!-- BEGIN part-index -->

| 节 | 课程 | 验收产出 | 进度 | 正文 |
| --- | --- | --- | --- | --- |
| 13.1 | [PD 分离的收益、代价与适用边界](01-prefill-decode-disaggregation/README.md) | 运行离线比较器、复算请求指标并完成9组练习，交付同预算方案评审；GPU/集群收益另行实测 | ✅ | 完整课程 |
| 13.2 | [P-only / D-only 测量与负载协议](02-isolated-measurement-and-workload-protocol/README.md) | 定义可复现的单阶段测量协议，区分配置速率、实际到达率、完成吞吐和 fake 路径边界 | 📋 | 教学骨架 |
| 13.3 | [性能曲线、SLO 查表与不确定性](03-performance-surfaces-and-slo-lookup/README.md) | 规划可追溯性能表和 SLO 查询器，保留失败/超时/拒绝与所有复测，不把 q_ref 当合格容量 | 📋 | 教学骨架 |
| 13.4 | [P/D 配比、卡数与集群容量规划](04-pd-ratios-and-capacity-planning/README.md) | 设计容量规划器：输出 P/D 实例、GPU/节点数、预算与不可行原因，标明忽略传输的初估边界 | 📋 | 教学骨架 |
| 13.5 | [KV 字节预算、传输成本与关键路径](05-kv-volume-and-transfer-critical-path/README.md) | 手算 KV 流量和链路上限，设计测量表区分传输总时长与暴露在关键路径上的等待 | 📋 | 教学骨架 |
| 13.6 | [vLLM KV Connector 协议与生命周期](06-the-kv-connector-protocol/README.md) | 追踪固定版本的生产/传输/消费时序，定位可复用时刻并设计假连接器故障案例 | 🧩 | 教学骨架 |
| 13.7 | [Mooncake：Transfer Engine、Store 与 KV 调度](07-mooncake-transfer-engine-and-store/README.md) | 给出固定版本组件图、传输微基准方案与重算/远端读取决策，独立记录硬件兼容性 | 📋 | 教学骨架 |
| 13.8 | [连接器生态与分层 KV 存储](08-connectors-and-tiered-storage/README.md) | 按传输库/存储服务/运行时连接器比较能力，记录命中率与真实延迟收益的差异 | 📋 | 教学骨架 |
| 13.9 | [异构并行、KV 重分片与拓扑放置](09-heterogeneous-parallelism-and-transfer-compatibility/README.md) | 建立逐版本兼容矩阵，计算每 rank 字节与拓扑流量，不假定 A100/C500 或任意并行度可互传 | 🧩 | 教学骨架 |
| 13.10 | [集群控制面：Dynamo、llm-d 与弹性调度](10-routing-queues-and-elasticity/README.md) | 交付控制面职责图、固定版本能力矩阵与路由/扩缩容消融方案；用同负载比较缓存亲和、排队和扩容恢复 | 📋 | 教学骨架 |
| 13.11 | [Encoder 分离与 E/P/D 资源规划](11-encoder-disaggregation-and-caching/README.md) | 把 P/D 规划扩展为 E/P/D，分别预算 embedding 与 KV 传输，明确计时边界 | 📋 | 教学骨架 |
| 13.12 | [生产运维、故障恢复与端到端验收](12-distributed-serving-lifecycles/README.md) | 交付 SLO 看板设计、故障演练/排空回滚手册及同资源容量验收报告；报告恢复时间、联合 goodput 与预测误差 | 🧩 | 教学骨架 |

<!-- END part-index -->

## 3. 贯穿项目：从实测性能表得到配置建议

### 3.1 先定义测量对象

以Qwen3-8B作为候选贯穿模型，不把模型名当完整身份；固定模型/tokenizer revision、精度、后端、硬件、并行组、缓存和Graph配置。先从TP1/PP1开始，再扩展并行配置。

| 维度 | 必须记录的字段 |
| --- | --- |
| 业务场景 | 输入/输出长度联合分布、到达过程、突发、前缀复用、取消/超时和租户 |
| 测点配置 | 目标request_rate、request_num、最大在途并发；独立注明服务端实际batch |
| 环境 | GPU型号/显存、驱动/runtime、框架commit、模型身份、并行rank拓扑 |
| 观测 | 实际到达率、完成req/s、输入/输出token/s、TTFT、逐请求TPOT与ITL分布 |
| 有效性 | 预热/正式窗口、运行时长、样本数、重复轮次、失败/拒绝/未完成数、原始日志 |
| 资源 | GPU/CPU内存、队列、网络字节、链路利用率、缓存命中及模型启动成本 |

输入/输出长度定义工作负载，SLO定义延迟要求，两者不能混为一列。可以用同一份性能表查询多组SLO，但只能覆盖已经测到并满足测量充分性要求的范围。

P侧采用“普通服务、输出1 token”时，应标为**包含首token生成的Prefill基线**，不能称作纯prefill内核耗时。D侧使用fake transfer/fake-prefill时，要验证实际注意力上下文、KV分配、首token和输出计数；fake D的TTFT不是完整业务TTFT。单阶段容量初估不能据此报告语义正确性或真实网络性能。

C600/SGLang可作为补充方法案例，特别适合讲测量流程、环境诊断和续跑；其结果不能换个设备名后用于A100/C500，SGLang的开关也不能直接当作vLLM命令。课程公开命令使用参数化目录，不依赖维护者服务器路径。

### 3.2 scan、sweep与合格容量

- **smoke**检查服务、token长度、结果采集和路径是否成立。
- **scan是可选的范围探索**：无限供压配合并发限制，可产生一个吞吐参考q_ref。它不是SLO合格容量，也未必是稳态极限。
- **sweep才是容量查询的主要数据**：直接配置绝对请求速率，或使用冻结的q_ref乘若干系数；在延迟拐点附近加密。
- 已有工具若依赖scan生成q_ref，需要先完成其前置步骤；支持独立sweep是后续实现要求，不是当前工具已支持的事实。
- request_rate控制供压强度，request_num控制样本量/持续时间，max_concurrency限制在途请求。三者不能互相替代；设定速率也不能冒充实际完成吞吐。
- 并发限制、客户端自身瓶颈和闭环等待会降低实际到达率。分别记录计划/实际发送时间，检查负载发生器是否掩盖排队与尾延迟。

性能表要保留成功但SLO不合格、OOM、超时、失败和测量不足等状态。未知点不填零，失败点不删掉；取容量时不只挑最好的一轮。插值先做留出测点验证，超出长度/并发/拓扑范围则要求补测。

### 3.3 先算实例，再算卡和机器

令qP、qD分别表示同一业务场景下、满足各自阶段延迟预算的**单实例实测请求容量**，单位均为req/s。忽略传输与其他共享资源瓶颈，连续值的平衡关系为：

$$
N_Pq_P\approx N_Dq_D,\qquad \frac{N_P}{N_D}\approx\frac{q_D}{q_P}.
$$

给定目标到达率λ和容量余量系数h≥1，初始整数配置为：

$$
N_P=\left\lceil\frac{h\lambda}{q_P}\right\rceil,\qquad
N_D=\left\lceil\frac{h\lambda}{q_D}\right\rceil.
$$

例如qP=10、qD=2 req/s，连续比值是1:5；若λ=12、h=1.2，向上取整后是2P+8D，即1:4。**比值不是固定部署模板**，余量与取整影响绝对数量。这里的数字只用于手算，不是硬件实测。

若使用输入/输出token吞吐TP、TD，在固定长度场景下必须先对齐每请求工作量：

$$
q_P=\frac{T_P}{L_{P,work}},\qquad q_D=\frac{T_D}{L_{D,work}},\qquad
\frac{N_P}{N_D}\approx\frac{T_D L_{P,work}}{T_P L_{D,work}}.
$$

L_P,work是P端实际计入的输入工作量；有prefix命中时不一定是完整提示长度。L_D,work是D端实际生成并计入吞吐的token数，是否减首token需看实现，不能一律减1。混合长度、命中率和动态batch下，优先实测相同业务分布的req/s，不用平均长度套固定场景曲线。

若每个P/D实例分别占gP/gD张GPU，则总GPU数为N_P gP+N_D gD。g来自实际rank拓扑，不能盲乘TP、PP、EP、CP的所有数字；节点数还受每节点GPU数、实例能否跨节点、故障域和网络放置约束。

阶段服务曲线只提供初估。共享NIC、争用、排队、显存和路由变化后，qP/qD也会变化；应枚举可行配置并通过trace replay和端到端实测校准，而非把公式当作SLO保证。

### 3.4 性能模型必须接受实测校准

Prefill服务时间记为tP(长度集合、token预算、并行配置、命中块)，Decode每步时间记为tD(活跃请求数、各请求上下文长度、并行配置)。先用FLOPs、权重/KV读取字节和通信估算解释瓶颈，再用实测曲线拟合；不能把所有Prefill都认定为纯计算受限，也不能把所有Decode都认定为纯带宽受限。

Roofline形式max(FLOPs/有效算力, bytes/有效带宽)只能提供粗略下界解释，调度、kernel启动和通信等需另外建模。固定batch的每步耗时不等于请求的排队后延迟；混合长度也不能总用平均长度代入非线性成本。校准点与验证点分开，报告拟合残差和不适用区域。

Little定律中的平均在途数N=λW仅在同一稳定系统边界、单位和观测口径下使用。它不能直接给出P95延迟。队列模型要说明到达过程与服务分布，不能默认所有LLM请求都满足M/M/c假设；突发、相关长度和连续批处理优先使用可核验的trace replay。

D端显存还需容纳驻留请求不断增长的KV：按实际活跃请求长度求和，加权重、workspace、图缓存和预留。共享prefix按物理块去重；排队请求是否已预分配KV由实现决定。可用容量、驻留时长和完成率会相互影响，因此算出卡数后还要重新检查内存与稳态。

## 4. 从KV字节到端到端延迟

### 4.1 用清楚的模型假设算字节

对各层采用相同KV头结构、K/V同dtype的标准全注意力MHA/GQA模型，一个请求S个token的逻辑KV大小为：

$$
B_{KV}=2\,L\,S\,H_{KV}\,d_h\,b.
$$

L为层数，H_KV为KV头数，d_h为head维度，b为每元素字节数；2对应K和V。真实预算还要加上块尾padding、量化scale/zero-point、元数据和复制；MLA、滑窗、混合注意力/Mamba分别按状态定义计算。某端只缺一部分prefix时按缺失块计算；“缓存命中”不等于“不需要传输”。

要分别报告：全模型逻辑字节、每rank持有字节、发送到每个rank的字节、每条链路的线上字节。GQA复制、重分片和中转可能使网络总量不同于逻辑KV量，不能总按TP整除。

### 4.2 传输总耗时与暴露等待

串行粗估可先用固定开销加B/BW_eff；再拆成注册、打包、元数据协商、队列、网络、接收重排与可读同步。BW_eff必须按实际消息大小、并发、NUMA/NIC路径测量，不能直接采用网卡标称速率。

如果逐层生产与传输重叠，端到端延迟由依赖图的关键路径决定。不能把计算时间和传输时间全相加，也不能直接假定全部隐藏。分别给出无重叠上界模型、理想重叠模型和实测时间线。

若首token必须在D接管后返回，TTFT需要计入P排队/计算、暴露的KV等待、D排队/首次执行与前端开销。若首token由P返回，handoff可能主要体现在首个token之后的间隔；此时不能只看TTFT而忽略ITL。高分位延迟不能靠直接相加各阶段P95得到，应基于同请求轨迹统计。

还需检查每条共享链路的平均需求λE[B_wire]是否低于有效可用带宽，以及突发时的队列、缓存高水位和超时。平均带宽充足只是必要筛选条件，不能保证尾延迟。

## 5. Mooncake与生态怎样讲

**13.7单独讲Mooncake，13.8负责横向比较。** [Mooncake论文](https://www.usenix.org/conference/fast25/presentation/qin)讨论以KV为中心的存储与调度架构；[开源Transfer Engine](https://kvcache-ai.github.io/Mooncake/design/transfer-engine/index.html)提供Segment/Buffer和批量异步传输抽象。这些层次要分别阅读、分别验证。

| 层次 | 需要讲清的问题 |
| --- | --- |
| Transfer Engine | 注册哪些内存、怎样定位远端、提交任务与检查完成、拓扑/链路选择、资源何时可释放 |
| Store | KV对象怎样命名和定位、副本/容量/驱逐怎样管理、读失败与回退；具体语义按固定版本核对 |
| Runtime Connector | vLLM的调度器/worker怎样准备metadata、映射block、等待KV可读、通知完成和释放 |
| 路由与调度 | 去有prefix的节点、去较空节点、读取远端KV或重算，分别付出多少时间与资源 |

NIXL、Mooncake、LMCache和其他方案不画成完全等价的替换按钮：传输库、缓存服务、运行时连接器承担不同职责，也可能互相组合。协议、设备内存访问、分层存储和故障语义逐项列证据；[vLLM的Mooncake入口](https://docs.vllm.ai/en/latest/features/mooncake_connector_usage/)与[NIXL兼容矩阵](https://docs.vllm.ai/en/latest/features/nixl_connector_compatibility/)用于核对具体集成范围。

A100/C500并不自动构成可直接互传GPU KV的异构集群。厂商runtime、RDMA/设备内存支持、dtype、layout、block、模型/adapter身份都需验证。跨厂商路径是专项课题；单卡不能伪装成多机带宽实验。

## 5.1 分布式推理平台：以Dynamo串起工程链路

**Dynamo是本篇的重点系统案例，llm-d承担架构对照，vLLM Production Stack提供较小的部署与观测参照。** 深入范围和验收方案见[13.10](10-routing-queues-and-elasticity/README.md)。这里的NVIDIA Dynamo与Part 8中的PyTorch `torch._dynamo`职责不同，后者属于编译链路。

| 层次 | 对应对象 | 本篇要做的工程判断 |
| --- | --- | --- |
| 模型执行 | vLLM；SGLang/TRT-LLM作为对照 | 单实例能处理多少工作，实际使用什么并行布局与KV格式 |
| 请求路由和资源池控制 | Dynamo、llm-d | 请求发给谁，何时扩P或扩D，怎样避免缓存热点和排队 |
| KV数据系统 | Mooncake Transfer Engine/Store、NIXL、LMCache、Dynamo KVBM | 传输、缓存管理和存储分别由谁负责，接口与成本是什么 |
| 基础设施编排 | Kubernetes及所选平台的控制器 | GPU/NIC怎样放置，期望副本何时变成真正可用容量 |
| 运维反馈 | 指标、日志、请求trace与告警 | 需求变化、缓存失效、传输瓶颈和实例异常如何区分 |

这些对象可以位于同一部署的不同层，也有替代或集成关系；不能把同一行的项目视为等价组件，更不能据此推断任意组合都可用。[Dynamo架构](https://docs.nvidia.com/dynamo/dev/knowledge-base/concepts/architecture)用于追踪请求、发现与事件路径，以及Router、Planner和Operator的职责；[llm-d架构](https://llm-d.ai/docs/architecture)用于对照Gateway/EPP、InferencePool和模型服务的关系。

贯穿案例按四次设计评审推进：

1. **13.3—13.4，容量评审**：自建实测表与[Dynamo Profiler](https://docs.nvidia.com/dynamo/dev/knowledge-base/modular-components/profiler/overview)的输入、输出对齐。区分实机测量和离线估算，记录外推边界，再得到固定资源预算下的初始P/D布局。
2. **13.5—13.9，数据路径评审**：确认KV经过哪些设备和链路，缓存层次由谁管理。将[Dynamo KVBM](https://docs.nvidia.com/dynamo/dev/reference/components/kvbm-configuration)与Mooncake Store、LMCache按职责比较，将NIXL与传输层分开验证。
3. **13.10，控制策略评审**：研究[Dynamo Planner](https://docs.nvidia.com/dynamo/dev/knowledge-base/modular-components/planner/planner-design)如何使用性能数据和运行指标。自行设计静态配比与弹性配比对照，考虑预算、预热时间、缓存冷启动、滞回和缩容排空。不能把本课程提出的策略当作平台已实现能力。
4. **13.12，上线评审**：同一流量轨迹下交付SLO报告、扩缩容时间线、故障恢复和回滚手册。平台启动成功只算部署检查，最终看合格请求吞吐、尾延迟及成本。

CUDA集群案例先选择官方文档覆盖的版本/引擎/硬件组合；是否覆盖A100须再检查模型精度、kernel与显存。C500路线单独核对MACA、vLLM-MetaX、连接器和设备内存传输条件，不承诺Dynamo/NIXL可直接迁移。可复用的是测量协议、模型和判断方法，具体运行栈需另验。

## 6. 与整套课程的分工

| 位置 | 负责的内容 | 怎样接入Part 13 |
| --- | --- | --- |
| Part 1 | token、P/D阶段、TTFT/TPOT/ITL、带宽与缓存基础 | 作为计量单位与时间线前置，不要求初学者先学集群 |
| Part 5/6 | prefix、分层缓存、调度与token/显存预算 | 引出命中率、缓存驻留、批处理与排队对性能表的影响 |
| 9.11/9.14 | KV量化的质量、Q/DQ与端到端收益 | 比较字节节省、额外量化/传输转换成本和Attention后端兼容 |
| Part 11/12 | TP/PP/CP/EP与MoE拓扑 | 区分rank组与副本数；检查通信和KV分片约束 |
| Part 13 | 角色容量、KV路径、P/D资源配比与集群可行性 | 维护本篇主项目与模型假设 |
| Part 17 | 通用测量、SLO/goodput、统计与profiling | 提供方法；本篇提供分离部署专用数据和验收 |
| Part 18 | 容器/驱动/设备插件、Helm与平台适配、应用接入 | 提供部署基础；Dynamo/llm-d的路由决策、容量控制与集群验收由Part 13主讲 |

## 7. 工程验证阶段与建设顺序

| 层级 | 可用设备 | 后续计划产物 | 能说明什么 |
| --- | --- | --- | --- |
| A | 普通CPU电脑 | 合成性能表、单位检查、整数配比、队列/带宽trace replay | 公式、策略与边界；不能说明真实GPU性能 |
| B | 单卡A100或C500 | 分阶段顺序测量、曲线拟合/留出校验、内存记录 | 该卡/版本的阶段能力；不能冒充两池同时运行 |
| C | 同构双卡/多机，再扩到集群 | 真实KV connector、传输微基准、混部/PD/远端缓存消融与故障注入 | 验证端到端收益、配比预测与扩展瓶颈 |

先写13.1—13.4，以真实业务协议和可追溯测量表完成容量设计；13.5—13.9补足真实KV成本和拓扑条件；13.10以Dynamo为重点实现案例、llm-d为架构对照，13.12完成生产验收。各阶段都保留离线分析入口，但模拟通过、单卡通过和集群通过分别记录。原始CSV/JSON、配置、attempt、失败原因和重复轮次按本节data/results组织，不为未实现实验建立空目录。

最终报告同时保存各阶段SLO与**同一请求联合达标率**。goodput明确采用合格请求数/测量时间，失败、拒绝、超时不能当作合格结果；逐token ITL、逐请求平均TPOT和它们的分位数分别统计。到达窗口和完成窗口、截尾请求及停止供压后的排空处理写入协议。

验收至少比较同GPU数或同成本的混部、PD、PD+远端缓存，覆盖短/长输入、短/长输出、冷/热缓存、突发与一项故障。输出预测误差、失效条件、资源余量和成本，不只给一个“最佳P:D比例”。

公开资料核对日期：2026-10-12。论文作为设计依据，latest/main只作为阅读入口；正式实现需固定commit和平台版本。课程中的手算示例不引用私有实验路径，也不把尚未完成的外部压测写成已交付结果。
