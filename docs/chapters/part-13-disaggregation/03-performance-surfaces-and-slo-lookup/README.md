# 13.3 性能曲线、SLO 查表与不确定性

[本篇规划](../README.md) · [全部课程](../../README.md) · [上一节 13.2](../02-isolated-measurement-and-workload-protocol/README.md) · [下一节 13.4](../04-pd-ratios-and-capacity-planning/README.md)

> 教学规划已扩充；正文、可运行代码与测量结果待交付。本页不代表已完成实验。

## 学习目标

长度/并发/速率性能表；可选 scan 与绝对速率 sweep；固定 q_ref；分位数与联合 SLO；重复测量；插值边界与不可行点；Roofline解释与实测拟合/留出残差；Dynamo Profiler 的测量/估算来源与适用范围。

## 前置知识与设备

面向已有服务部署与压测经验的工程师；缓存、调度、并行与测量基础分别见 Part 5、6、11、17。

离线分析机与可追溯实测数据；平台 Profiler 另需其支持环境。离线推导用于筛选方案，不能代替真实集群验收；A100/C500 分别记录支持证据。

## 正文安排

<!-- BEGIN lesson-plan -->

1. 长度/并发/速率性能表：补充概念、推导、例子及实现边界。
2. 可选 scan 与绝对速率 sweep：补充概念、推导、例子及实现边界。
3. 固定 q_ref：补充概念、推导、例子及实现边界。
4. 分位数与联合 SLO：补充概念、推导、例子及实现边界。
5. 重复测量：补充概念、推导、例子及实现边界。
6. 插值边界与不可行点：补充概念、推导、例子及实现边界。
7. Roofline解释与实测拟合/留出残差：补充概念、推导、例子及实现边界。
8. Dynamo Profiler 的测量/估算来源与适用范围：补充概念、推导、例子及实现边界。

<!-- END lesson-plan -->

## 代码与实验设计

工程对接任务：将本节性能表与[Dynamo Profiler](https://docs.nvidia.com/dynamo/dev/knowledge-base/modular-components/profiler/overview)的数据来源、插值维度和输出配置逐项对齐。在线实测和离线估算必须分列；自建数据转入平台前先核对schema、单位、模型/硬件身份和缺失点处理。13.10负责消费这些信息的控制面。

1. 先定义输入数据、计数单位、阶段/窗口边界和参考结果。
2. 用实测数据校准本节模型；合成输入只做边界检查，单独标记数据来源与适用范围。
3. 接入真实数据时记录模型、版本、硬件、并行/缓存配置及原始证据。
4. 保留失败与不生效的条件，提供复现命令、边界案例和结果解释。

scan可用于确定范围，直接配置绝对请求速率也可 sweep。已有工具若依赖q_ref，需先改代码和检查；课程规划不能冒充该功能已经实现。

## 练习与验收

<!-- BEGIN lesson-exercise -->

目标产出：**规划可追溯性能表和 SLO 查询器，保留失败/超时/拒绝与所有复测，不把 q_ref 当合格容量**。

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
