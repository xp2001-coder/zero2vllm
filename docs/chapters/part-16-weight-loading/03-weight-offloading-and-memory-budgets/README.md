# 16.3 权重 Offload 与内存预算

[本篇目录](../README.md) · [全部课程](../../README.md) · [上一节 16.2](../02-efficient-loading-and-startup/README.md) · [下一节 16.4](../04-sleep-wake-and-pause-resume/README.md)

> 📋 教学骨架；正文与实验待编写。目录存在不表示内容已经完成。

## 学习目标

CPU weight offload、prefetch/UVA 路径、KV offload、pinned memory、图内存。

## 前置知识与设备

- 前置：Part 3—5、14。
- 设备：CPU/GPU。具体版本和执行命令随本节实验一起固定。
- A100/C500 的支持情况以实机记录为准，未执行的路径保持待验证。

## 正文安排

<!-- BEGIN lesson-plan -->

1. CPU weight offload：补充概念、推导、例子及实现边界。
2. prefetch/UVA 路径：补充概念、推导、例子及实现边界。
3. KV offload：补充概念、推导、例子及实现边界。
4. pinned memory：补充概念、推导、例子及实现边界。
5. 图内存：补充概念、推导、例子及实现边界。

<!-- END lesson-plan -->

## 代码与实验

- [ ] 确定最小输入、预期输出与张量/请求契约。
- [ ] 补充可独立运行的实现、命令、参数解释和结果示例。
- [ ] 覆盖正常输入、边界输入和失败排查。
- [ ] 保存可复核结果；涉及性能时注明设备、版本、负载和计时范围。

## 练习与验收

<!-- BEGIN lesson-exercise -->

目标产出：**建立权重/激活/KV/workspace/graph 的显存账本**。

- [ ] 提供输入、命令、结果与解释。
- [ ] 补充参考答案和至少一个边界案例。

<!-- END lesson-exercise -->

完成正文、代码、命令和必要验证后，才能把课程数据中的状态改为已完成。填充方法见[章节模板](../../../maintenance/templates/lesson.md)。

## 参考资料

- [Model Resolution](https://docs.vllm.ai/en/stable/configuration/model_resolution/)
- [Conserving Memory](https://docs.vllm.ai/en/stable/configuration/conserving_memory/)
- [Sleep Mode](https://docs.vllm.ai/en/stable/features/sleep_mode/)
- [Weight Transfer](https://docs.vllm.ai/en/stable/training/weight_transfer/)
- [Async Reinforcement Learning](https://docs.vllm.ai/en/stable/training/async_rl/)
- [Layerwise Reloading](https://docs.vllm.ai/en/stable/training/layerwise/)
