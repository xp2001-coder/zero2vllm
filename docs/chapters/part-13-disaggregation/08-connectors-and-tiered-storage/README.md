# 13.8 连接器生态与分层 KV 存储

[本篇规划](../README.md) · [全部课程](../../README.md) · [上一节 13.7](../07-mooncake-transfer-engine-and-store/README.md) · [下一节 13.9](../09-heterogeneous-parallelism-and-transfer-compatibility/README.md)

> 教学规划已扩充；正文、可运行代码与测量结果待交付。本页不代表已完成实验。

## 学习目标

NIXL/Mooncake/LMCache/MoRIIO/FlexKV/HF3FS 分层职责；HBM/DRAM/SSD；缓存准入/驱逐/预取；对象/字节命中；网络热点与缓存污染；Dynamo KVBM 与 NIXL 的层次及版本能力矩阵。

## 前置知识与设备

面向已有服务部署与压测经验的工程师；缓存、调度、并行与测量基础分别见 Part 5、6、11、17。

目标缓存层与实际链路；按组件和硬件分别验证。离线推导用于筛选方案，不能代替真实集群验收；A100/C500 分别记录支持证据。

## 正文安排

<!-- BEGIN lesson-plan -->

1. NIXL/Mooncake/LMCache/MoRIIO/FlexKV/HF3FS 分层职责：补充概念、推导、例子及实现边界。
2. HBM/DRAM/SSD：补充概念、推导、例子及实现边界。
3. 缓存准入/驱逐/预取：补充概念、推导、例子及实现边界。
4. 对象/字节命中：补充概念、推导、例子及实现边界。
5. 网络热点与缓存污染：补充概念、推导、例子及实现边界。
6. Dynamo KVBM 与 NIXL 的层次及版本能力矩阵：补充概念、推导、例子及实现边界。

<!-- END lesson-plan -->

## 代码与实验设计

平台对照任务：纳入[Dynamo KVBM](https://docs.nvidia.com/dynamo/dev/reference/components/kvbm-configuration)的缓存层级，分别与Mooncake Store、LMCache比较对象身份、块生命周期、准入/驱逐和故障处理。NIXL归入传输层。每项都记录固定版本的连接器/后端支持；表中名称仅用于职责比较，不表示这些组件可任意组合或已在A100/C500验证。

1. 先定义输入数据、计数单位、阶段/窗口边界和参考结果。
2. 用实测数据校准本节模型；合成输入只做边界检查，单独标记数据来源与适用范围。
3. 接入真实数据时记录模型、版本、硬件、并行/缓存配置及原始证据。
4. 保留失败与不生效的条件，提供复现命令、边界案例和结果解释。

版本与平台支持逐项核验；先分析代价，再比较收益，不把待实现设计标为实机验证。

## 练习与验收

<!-- BEGIN lesson-exercise -->

目标产出：**按传输库/存储服务/运行时连接器比较能力，记录命中率与真实延迟收益的差异**。

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
