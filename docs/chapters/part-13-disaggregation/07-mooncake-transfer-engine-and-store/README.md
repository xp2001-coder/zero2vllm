# 13.7 Mooncake：Transfer Engine、Store 与 KV 调度

[本篇规划](../README.md) · [全部课程](../../README.md) · [上一节 13.6](../06-the-kv-connector-protocol/README.md) · [下一节 13.8](../08-connectors-and-tiered-storage/README.md)

> 教学规划已扩充；正文、可运行代码与测量结果待交付。本页不代表已完成实验。

## 学习目标

论文架构与开源组件边界；Segment/Buffer/BatchTransfer；内存注册与异步完成；RDMA/TCP 与拓扑；Store 元数据/副本/回收；重算或传输。

## 前置知识与设备

面向已有服务部署与压测经验的工程师；缓存、调度、并行与测量基础分别见 Part 5、6、11、17。

受所选 Mooncake 版本支持的双端环境与实际网络。离线推导用于筛选方案，不能代替真实集群验收；A100/C500 分别记录支持证据。

## 正文安排

<!-- BEGIN lesson-plan -->

1. 论文架构与开源组件边界：补充概念、推导、例子及实现边界。
2. Segment/Buffer/BatchTransfer：补充概念、推导、例子及实现边界。
3. 内存注册与异步完成：补充概念、推导、例子及实现边界。
4. RDMA/TCP 与拓扑：补充概念、推导、例子及实现边界。
5. Store 元数据/副本/回收：补充概念、推导、例子及实现边界。
6. 重算或传输：补充概念、推导、例子及实现边界。

<!-- END lesson-plan -->

## 代码与实验设计

1. 先定义输入数据、计数单位、阶段/窗口边界和参考结果。
2. 用实测数据校准本节模型；合成输入只做边界检查，单独标记数据来源与适用范围。
3. 接入真实数据时记录模型、版本、硬件、并行/缓存配置及原始证据。
4. 保留失败与不生效的条件，提供复现命令、边界案例和结果解释。

论文中的完整调度系统、开源Transfer Engine、Store与vLLM Connector分开核对；不能用一个组件的存在证明整套架构已部署。

## 练习与验收

<!-- BEGIN lesson-exercise -->

目标产出：**给出固定版本组件图、传输微基准方案与重算/远端读取决策，独立记录硬件兼容性**。

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
