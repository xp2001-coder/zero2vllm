# 13.5 KV 字节预算、传输成本与关键路径

[本篇规划](../README.md) · [全部课程](../../README.md) · [上一节 13.4](../04-pd-ratios-and-capacity-planning/README.md) · [下一节 13.6](../06-the-kv-connector-protocol/README.md)

> 教学规划已扩充；正文、可运行代码与测量结果待交付。本页不代表已完成实验。

## 学习目标

MHA/GQA 与模型专用状态；逻辑/分片/线上字节；KV dtype 与 scale；注册/打包/网络/重排；分层流水与重叠；网络争用。

## 前置知识与设备

面向已有服务部署与压测经验的工程师；缓存、调度、并行与测量基础分别见 Part 5、6、11、17。

双端设备和实际链路；单卡仅做字节预算与本地开销测量。离线推导用于筛选方案，不能代替真实集群验收；A100/C500 分别记录支持证据。

## 正文安排

<!-- BEGIN lesson-plan -->

1. MHA/GQA 与模型专用状态：补充概念、推导、例子及实现边界。
2. 逻辑/分片/线上字节：补充概念、推导、例子及实现边界。
3. KV dtype 与 scale：补充概念、推导、例子及实现边界。
4. 注册/打包/网络/重排：补充概念、推导、例子及实现边界。
5. 分层流水与重叠：补充概念、推导、例子及实现边界。
6. 网络争用：补充概念、推导、例子及实现边界。

<!-- END lesson-plan -->

## 代码与实验设计

1. 先定义输入数据、计数单位、阶段/窗口边界和参考结果。
2. 用实测数据校准本节模型；合成输入只做边界检查，单独标记数据来源与适用范围。
3. 接入真实数据时记录模型、版本、硬件、并行/缓存配置及原始证据。
4. 保留失败与不生效的条件，提供复现命令、边界案例和结果解释。

传输字节不能总按TP简单相除；分片、复制、padding和多个链路分别核算。重叠只扣掉实测隐藏部分。

## 练习与验收

<!-- BEGIN lesson-exercise -->

目标产出：**手算 KV 流量和链路上限，设计测量表区分传输总时长与暴露在关键路径上的等待**。

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
