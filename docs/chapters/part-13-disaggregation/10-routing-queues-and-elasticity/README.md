# 13.10 集群控制面：Dynamo、llm-d 与弹性调度

[本篇规划](../README.md) · [全部课程](../../README.md) · [上一节 13.9](../09-heterogeneous-parallelism-and-transfer-compatibility/README.md) · [下一节 13.11](../11-encoder-disaggregation-and-caching/README.md)

> 教学规划已扩充；正文、可运行代码与测量结果待交付。本页不代表已完成实验。

## 学习目标

请求级路由与资源池控制的时间尺度；Dynamo Frontend/Router/Profiler/Planner/Operator；llm-d Gateway/EPP/InferencePool；Production Stack 对照；缓存收益与传输/排队成本；准入/背压/公平；扩缩容预热/滞回/排空；Little定律边界与trace replay。

## 前置知识与设备

面向已有服务部署与压测经验的工程师；缓存、调度、并行与测量基础分别见 Part 5、6、11、17。

平台支持的 GPU 集群；Operator/EPP 案例需 Kubernetes 与监控。离线推导用于筛选方案，不能代替真实集群验收；A100/C500 分别记录支持证据。

## 正文安排

<!-- BEGIN lesson-plan -->

1. 请求级路由与资源池控制的时间尺度：补充概念、推导、例子及实现边界。
2. Dynamo Frontend/Router/Profiler/Planner/Operator：补充概念、推导、例子及实现边界。
3. llm-d Gateway/EPP/InferencePool：补充概念、推导、例子及实现边界。
4. Production Stack 对照：补充概念、推导、例子及实现边界。
5. 缓存收益与传输/排队成本：补充概念、推导、例子及实现边界。
6. 准入/背压/公平：补充概念、推导、例子及实现边界。
7. 扩缩容预热/滞回/排空：补充概念、推导、例子及实现边界。
8. Little定律边界与trace replay：补充概念、推导、例子及实现边界。

<!-- END lesson-plan -->

## 代码与实验设计

### 1. 本节的工程问题

业务负载从短问答切换到长文档问答后，P端排队增加、D端仍有余量。扩P能否解决问题？如果请求已经命中远端KV，但链路拥塞，又该读缓存还是重算？本节要求工程师用前面几节的容量表、KV成本和请求轨迹解释决策，并检查决策落地后的结果。

**重点实现案例为NVIDIA Dynamo；llm-d用于对照控制面结构，vLLM Production Stack用于较小规模部署与可观测性对照。** 正式实验固定一个主平台，另两个采用源码/配置走读，不要求同时维护三套完整集群。Dynamo与Part 8中的`torch._dynamo`分别属于分布式服务系统和编译系统。

### 2. 分清四种决策

| 决策 | 决策对象 | 输入与约束 | 要记录的结果 |
| --- | --- | --- | --- |
| 请求级路由 | 选P/D实例、缓存来源 | 预计工作量、命中块、队列、传输等待、租户配额 | 为什么选该实例，预计/实际等待，路由自身耗时 |
| 引擎迭代调度 | 本轮执行哪些token | token预算、KV空间、抢占、批处理 | 进入执行的时刻与实际batch；详见Part 6 |
| 资源池容量控制 | P/D期望副本数 | 性能曲线、负载趋势、SLO、GPU预算、冷启动 | 建议数量、限额裁剪、待启动数量及达标恢复时间 |
| 基础设施放置 | Pod/rank组落在哪些节点 | 可用GPU、拓扑、网络、资源配额和故障域 | 期望副本与Ready副本差异，放置失败及原因 |

这四种决策使用不同时间尺度。扩容请求发出后，模型加载、通信初始化和预热还需要时间；正在启动的副本不能提前计入可用吞吐。缩容也会带走缓存与在途请求，需要先撤出路由、排空或按实现支持的策略处理请求，再释放资源。

### 3. Dynamo走读路线

以[官方架构](https://docs.nvidia.com/dynamo/dev/knowledge-base/concepts/architecture)为入口，只走读所选版本实际存在的组件与路径：

1. **请求路径**：Frontend接收请求，Router选择执行端；追踪Prefill结果、KV协调与Decode响应的关联。把请求ID、阶段时间戳和所选worker落到同一条时间线。
2. **容量数据**：阅读[Profiler](https://docs.nvidia.com/dynamo/dev/knowledge-base/modular-components/profiler/overview)的输入、曲线与配置输出，区分离线估算和在线测量。映射13.3中的单位、长度分布、并行配置和有效区域；格式不同先写转换规范，不能声称自建表可以直接导入。
3. **扩缩容决策**：阅读[Planner设计](https://docs.nvidia.com/dynamo/dev/knowledge-base/modular-components/planner/planner-design)，辨认依赖性能数据/流量预测的控制与依赖运行指标的控制。逐版本记录可用模式和引擎限制，再比较静态配比与平台策略。
4. **执行与反馈**：追踪Planner期望数量到Operator协调Kubernetes资源，再到worker可接收请求的过程。检查资源不足、启动失败和指标缺失时的实际行为。
5. **缓存路径**：13.8负责KVBM等存储管理组件，13.5—13.7负责传输成本。本节关注缓存状态怎样影响路由；状态事件滞后、丢失或实例移除后，路由依据是否仍可信。

前述是课程走读和测量要求，不是声明所有Dynamo版本、后端、GPU都具备同一组能力。官方dev文档只提供研究入口；实现时冻结release/commit、镜像digest、引擎版本、连接器与实际部署参数。

### 4. 横向比较什么

| 案例 | 本节阅读重点 | 对照产物 |
| --- | --- | --- |
| Dynamo | 请求路由、Profiler/Planner、Operator之间的数据和决策流 | 容量表到期望副本再到可用容量的时间线 |
| llm-d | Gateway/Proxy、Endpoint Picker（EPP）、InferencePool与模型服务；KV亲和与负载如何影响选择 | 对同一负载列出路由信号、选择过程与资源池边界 |
| vLLM Production Stack | vLLM服务、请求router、Helm和监控组件 | 一份最小服务基线的部署与观测清单 |

[llm-d官方架构](https://llm-d.ai/docs/architecture)把EPP选择与模型执行分开；[Production Stack仓库](https://github.com/vllm-project/production-stack)提供部署、路由和观测参考。课程按接口、指标、状态一致性、扩容责任和运维成本比较。项目README中的roadmap只记为计划，不能记为已支持PD或已交付自动扩缩容；同样不能把某平台的参数名照搬到另一平台。

### 5. 实验矩阵与预期观察

固定模型、精度、硬件池、GPU上限和同一流量轨迹；先做单变量对照，再做组合。不要求所有平台实现所有策略，缺失项记为不支持。

| 对照 | 改变项 | 主要观察 |
| --- | --- | --- |
| 路由 | 最短队列、缓存亲和、预计排队/计算/传输成本；平台支持项逐一列出 | 命中率提高是否以热点、跨节点字节和尾延迟为代价 |
| 配比 | 固定P/D数量、在线弹性 | 增长/突发/回落时的SLO恢复、启动耗时、伸缩次数和空闲成本 |
| 缓存状态 | 冷缓存、热缓存、事件滞后、实例重启 | 预测命中与实际命中差异、回退路径及队列变化 |
| 约束 | GPU预算紧张、共享链路拥塞、长短请求混合 | 扩容是否受阻，准入和背压怎样影响合格吞吐与公平性 |

路由分数必须说明各项单位和权重，不能直接把“队列长度+命中率+字节数”相加。输出长度不可预知时记录预测误差。Little定律只检查稳定边界内的均值关系；尾延迟用同请求轨迹或实测统计。

扩缩容对照至少记录：负载变化、决策、资源获配、模型加载、Ready、首次接流、SLO恢复等时刻。区分冷启动成本与稳态性能。若只做trace replay，结果标签必须为模拟；实际平台的故障语义在13.12验收。

### 6. 计划交付的工程材料

- 控制面职责图与请求/缓存/控制事件三条时序，全部自绘。
- 版本与能力矩阵：平台、引擎、GPU/runtime、连接器、所测模式、源码依据、实测状态。
- 容量表到部署配置的映射说明，固定预算与不可行原因。
- 一套主平台部署参数、启动/停止/清理步骤和回退方法，补齐后才能标为可运行实验。
- 路由与弹性消融报告，保留原始请求记录、队列/缓存/网络指标和所有失败测点。

以下通用步骤用于后续实现，当前没有交付上述部署清单或实机性能结果。

1. 先定义输入数据、计数单位、阶段/窗口边界和参考结果。
2. 用实测数据校准本节模型；合成输入只做边界检查，单独标记数据来源与适用范围。
3. 接入真实数据时记录模型、版本、硬件、并行/缓存配置及原始证据。
4. 保留失败与不生效的条件，提供复现命令、边界案例和结果解释。

版本与平台支持逐项核验；先分析代价，再比较收益，不把待实现设计标为实机验证。

## 练习与验收

<!-- BEGIN lesson-exercise -->

目标产出：**交付控制面职责图、固定版本能力矩阵与路由/扩缩容消融方案；用同负载比较缓存亲和、排队和扩容恢复**。

- [ ] 提供输入、命令、结果与解释。
- [ ] 补充参考答案和至少一个边界案例。

<!-- END lesson-exercise -->

完成正文、代码、命令和验证后，才能更改为完整课程。内容结构遵循[章节模板](../../../maintenance/templates/lesson.md)。

## 公开资料

- [DistServe（OSDI 2024）](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin)
- [Mooncake（FAST 2025）](https://www.usenix.org/conference/fast25/presentation/qin)
- [vLLM Disaggregated Prefilling](https://docs.vllm.ai/en/latest/features/disagg_prefill/)
- [Mooncake Transfer Engine](https://kvcache-ai.github.io/Mooncake/design/transfer-engine/index.html)
- [vLLM MooncakeConnector](https://docs.vllm.ai/en/latest/features/mooncake_connector_usage/)
- [NIXL 兼容矩阵](https://docs.vllm.ai/en/latest/features/nixl_connector_compatibility/)
- [Disaggregated Encoder](https://docs.vllm.ai/en/stable/features/disagg_encoder/)
- [KV Connector 注册表](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/kv_transfer/kv_connector/factory.py)
- [KV Offloading](https://docs.vllm.ai/en/stable/features/kv_offloading_usage/)

资料核对日期：2026-10-12。latest/main页面可能变化；正式实验需固定版本、commit与能力矩阵。
