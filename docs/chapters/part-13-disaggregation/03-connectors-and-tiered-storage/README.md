# 13.3 连接器与分层存储地图

[本篇目录](../README.md) · [全部课程](../../README.md) · [上一节 13.2](../02-the-kv-connector-protocol/README.md) · [下一节 13.4](../04-heterogeneous-parallelism-and-transfer-compatibility/README.md)

> 📋 教学骨架；正文与实验待编写。目录存在不表示内容已经完成。

## 学习目标

NIXL、Mooncake/Store、LMCache、MoRIIO、FlexKV、HF3FS、原生 offload。

## 前置知识与设备

- 前置：Part 5、6、11。
- 设备：多机/专项。具体版本和执行命令随本节实验一起固定。
- A100/C500 的支持情况以实机记录为准，未执行的路径保持待验证。

## 正文安排

<!-- BEGIN lesson-plan -->

1. NIXL：补充概念、推导、例子及实现边界。
2. Mooncake/Store：补充概念、推导、例子及实现边界。
3. LMCache：补充概念、推导、例子及实现边界。
4. MoRIIO：补充概念、推导、例子及实现边界。
5. FlexKV：补充概念、推导、例子及实现边界。
6. HF3FS：补充概念、推导、例子及实现边界。
7. 原生 offload：补充概念、推导、例子及实现边界。

<!-- END lesson-plan -->

## 代码与实验

- [ ] 确定最小输入、预期输出与张量/请求契约。
- [ ] 补充可独立运行的实现、命令、参数解释和结果示例。
- [ ] 覆盖正常输入、边界输入和失败排查。
- [ ] 保存可复核结果；涉及性能时注明设备、版本、负载和计时范围。

## 练习与验收

<!-- BEGIN lesson-exercise -->

目标产出：**逐项核对传输协议、存储层与外部依赖**。

- [ ] 提供输入、命令、结果与解释。
- [ ] 补充参考答案和至少一个边界案例。

<!-- END lesson-exercise -->

完成正文、代码、命令和必要验证后，才能把课程数据中的状态改为已完成。填充方法见[章节模板](../../../maintenance/templates/lesson.md)。

## 参考资料

- [Disaggregated Prefilling](https://docs.vllm.ai/en/stable/features/disagg_prefill/)
- [Disaggregated Encoder](https://docs.vllm.ai/en/stable/features/disagg_encoder/)
- [KV Connector 注册表](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/kv_transfer/kv_connector/factory.py)
- [NIXL 兼容性矩阵](https://docs.vllm.ai/en/stable/features/nixl_connector_compatibility/)
- [KV Offloading](https://docs.vllm.ai/en/stable/features/kv_offloading_usage/)
