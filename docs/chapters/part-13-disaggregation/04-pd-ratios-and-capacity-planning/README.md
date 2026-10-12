# 13.4 P/D 配比、卡数与集群容量规划

[本篇规划](../README.md) · [全部课程](../../README.md) · [上一节 13.3](../03-performance-surfaces-and-slo-lookup/README.md) · [下一节 13.5](../05-kv-volume-and-transfer-critical-path/README.md)

> 教学规划已扩充；正文、可运行代码与测量结果待交付。本页不代表已完成实验。

## 学习目标

req/s 与 token/s 换算；SLO 合格 qP/qD；目标流量与容量余量；整数实例数；TP/PP/EP/CP 资源组；网络/显存/成本约束；离线容量建议与在线 Planner 的分工及 GPU 预算。

## 前置知识与设备

面向已有服务部署与压测经验的工程师；缓存、调度、并行与测量基础分别见 Part 5、6、11、17。

离线规划与目标集群资源清单；配置需真实集群复核。离线推导用于筛选方案，不能代替真实集群验收；A100/C500 分别记录支持证据。

## 正文安排

<!-- BEGIN lesson-plan -->

1. req/s 与 token/s 换算：补充概念、推导、例子及实现边界。
2. SLO 合格 qP/qD：补充概念、推导、例子及实现边界。
3. 目标流量与容量余量：补充概念、推导、例子及实现边界。
4. 整数实例数：补充概念、推导、例子及实现边界。
5. TP/PP/EP/CP 资源组：补充概念、推导、例子及实现边界。
6. 网络/显存/成本约束：补充概念、推导、例子及实现边界。
7. 离线容量建议与在线 Planner 的分工及 GPU 预算：补充概念、推导、例子及实现边界。

<!-- END lesson-plan -->

## 代码与实验设计

工程对接任务：给13.10提供固定GPU预算下的初始P/D配置、上下限和不可行原因。离线规划输出建议，在线Planner处理负载与容量变化；不能把向上取整的静态配比直接当作自动扩缩容算法，也不能把待启动副本计入当前可用容量。

1. 先定义输入数据、计数单位、阶段/窗口边界和参考结果。
2. 用实测数据校准本节模型；合成输入只做边界检查，单独标记数据来源与适用范围。
3. 接入真实数据时记录模型、版本、硬件、并行/缓存配置及原始证据。
4. 保留失败与不生效的条件，提供复现命令、边界案例和结果解释。

qP/qD必须对应同一业务场景和各自预算；P/D实例比不等于卡数比。网络、排队和内存未纳入时只能给初估。

## 练习与验收

<!-- BEGIN lesson-exercise -->

目标产出：**设计容量规划器：输出 P/D 实例、GPU/节点数、预算与不可行原因，标明忽略传输的初估边界**。

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
