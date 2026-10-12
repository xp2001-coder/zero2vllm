# 13.12 生产运维、故障恢复与端到端验收

[本篇规划](../README.md) · [全部课程](../../README.md) · [上一节 13.11](../11-encoder-disaggregation-and-caching/README.md) · [下一节 14.1](../../part-14-model-types/01-model-registration-and-execution-tasks/README.md)

> 教学规划已扩充；正文、可运行代码与测量结果待交付。本页不代表已完成实验。

## 学习目标

push/pull 生命周期与失败回收；同请求跨组件 trace 与 SLO 告警；超时/重试/取消与输出去重边界；缓存事件滞后与网络故障；滚动升级/排空/回滚；联合 SLO/goodput 与单位合格请求成本；预测残差与 N-1 容量。

## 前置知识与设备

面向已有服务部署与压测经验的工程师；缓存、调度、并行与测量基础分别见 Part 5、6、11、17。

隔离的多机验收环境；容量/故障测试不在生产流量上直接试验。离线推导用于筛选方案，不能代替真实集群验收；A100/C500 分别记录支持证据。

## 正文安排

<!-- BEGIN lesson-plan -->

1. push/pull 生命周期与失败回收：补充概念、推导、例子及实现边界。
2. 同请求跨组件 trace 与 SLO 告警：补充概念、推导、例子及实现边界。
3. 超时/重试/取消与输出去重边界：补充概念、推导、例子及实现边界。
4. 缓存事件滞后与网络故障：补充概念、推导、例子及实现边界。
5. 滚动升级/排空/回滚：补充概念、推导、例子及实现边界。
6. 联合 SLO/goodput 与单位合格请求成本：补充概念、推导、例子及实现边界。
7. 预测残差与 N-1 容量：补充概念、推导、例子及实现边界。

<!-- END lesson-plan -->

## 代码与实验设计

### 工程验收场景

本节承接13.10的主平台部署。目标是交付可供上线评审的证据和操作手册；以下是待实现的验收设计，未宣称已经通过实机验证。

| 场景 | 注入或操作 | 需要回答的问题 |
| --- | --- | --- |
| P/D worker异常 | 分别退出P、D或所在节点 | 谁发现故障、谁停止路由、谁释放KV，哪些请求失败，剩余容量是否达标 |
| 传输异常 | 在隔离测试环境限制带宽或中断链路 | 超时归属、重算/重试条件、是否重复输出及是否泄漏块 |
| 状态不新鲜 | 延迟缓存事件或服务发现更新 | 错误命中判断和过期endpoint怎样识别，回退是否引起重试风暴 |
| 控制面异常 | 暂停所选平台的容量控制或指标采集 | 已有worker是否继续服务，新请求/扩容怎样退化，缺失指标是否误触发缩容 |
| 发布与缩容 | 停止接流、排空、滚动替换与回滚 | 在途流式响应、模型/KV身份、恢复时间及缓存冷启动代价 |

演练前记录期望语义，再与所选平台的真实行为比对。N-1必须说明失去的是一个worker、一个并行实例还是一个节点；一台机器可能同时承载多个角色，容量不能简单扣一张卡。请求已经返回部分token时，自动重试不等于能够无缝续接，必须解释客户端看到的结果。

### 观测与运行手册

看板至少分为请求、引擎、KV传输、控制面和基础设施五层：

- 请求：到达/接纳/拒绝/完成数量，同请求TTFT、TPOT、ITL、超时与联合SLO；分别观察不同负载类别和租户。
- 引擎与KV：P/D队列、实际batch、活跃上下文、物理块用量、命中/重算、传输字节/等待/失败。
- 控制面：期望/启动中/Ready副本、路由耗时、缓存事件新鲜度、扩容和排空时间。
- 基础设施：GPU内存、GPU/NIC利用率、链路与节点异常；GPU利用率不能单独充当扩容依据。

指标名以固定版本实际导出为准。请求ID放在可关联的trace/日志中，避免把每个请求ID作为监控时序标签；采样率、时钟对齐和观测开销写入报告。告警应能定位“需要处理的SLO退化或容量不足”，并附操作步骤、停止条件和回滚方法。

### 结课交付包

1. 版本锁定、拓扑、容量预算和主平台部署清单，说明复现环境。
2. 同预算混部/PD/PD加远端缓存的基线与消融，报告稳态、冷启动和突发。
3. 真实请求轨迹、阶段时间线、合格请求吞吐及P50/P95/P99，并说明样本量和置信范围。
4. 故障演练、缩容排空、滚动升级/回滚手册及操作证据。
5. 预测残差、N-1余量、成本与适用边界。单位合格请求成本使用一致窗口下的实际资源成本除以合格完成数；合格数为零时标为不可用，不输出零成本。

前三阶段的离线模型和单卡曲线是输入材料；它们不能替代本节的多机服务证据。

1. 先定义输入数据、计数单位、阶段/窗口边界和参考结果。
2. 用实测数据校准本节模型；合成输入只做边界检查，单独标记数据来源与适用范围。
3. 接入真实数据时记录模型、版本、硬件、并行/缓存配置及原始证据。
4. 保留失败与不生效的条件，提供复现命令、边界案例和结果解释。

成功返回、各阶段单独P95达标与同一请求同时满足全部SLO是不同条件。失败、拒绝、超时及测量窗口内未完成请求分别记录。

## 练习与验收

<!-- BEGIN lesson-exercise -->

目标产出：**交付 SLO 看板设计、故障演练/排空回滚手册及同资源容量验收报告；报告恢复时间、联合 goodput 与预测误差**。

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
