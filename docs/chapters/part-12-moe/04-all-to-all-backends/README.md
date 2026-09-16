# 12.4 All-to-All 后端

[本篇目录](../README.md) · [全部课程](../../README.md) · [上一节 12.3](../03-fused-moe-kernels/README.md) · [下一节 12.5](../05-eplb-and-elastic-expert-parallelism/README.md)

> 📋 教学骨架；正文与实验待编写。目录存在不表示内容已经完成。

## 学习目标

allgather/reducescatter、DeepEP、MoRI、NIXL EP、FlashInfer NVLink。

## 前置知识与设备

- 前置：Part 7、8、11。
- 设备：多卡/专项。具体版本和执行命令随本节实验一起固定。
- A100/C500 的支持情况以实机记录为准，未执行的路径保持待验证。

## 正文安排

<!-- BEGIN lesson-plan -->

1. allgather/reducescatter：补充概念、推导、例子及实现边界。
2. DeepEP：补充概念、推导、例子及实现边界。
3. MoRI：补充概念、推导、例子及实现边界。
4. NIXL EP：补充概念、推导、例子及实现边界。
5. FlashInfer NVLink：补充概念、推导、例子及实现边界。

<!-- END lesson-plan -->

## 代码与实验

- [ ] 确定最小输入、预期输出与张量/请求契约。
- [ ] 补充可独立运行的实现、命令、参数解释和结果示例。
- [ ] 覆盖正常输入、边界输入和失败排查。
- [ ] 保存可复核结果；涉及性能时注明设备、版本、负载和计时范围。

## 练习与验收

<!-- BEGIN lesson-exercise -->

目标产出：**比较延迟/吞吐模式与拓扑条件，识别已移除后端**。

- [ ] 提供输入、命令、结果与解释。
- [ ] 补充参考答案和至少一个边界案例。

<!-- END lesson-exercise -->

完成正文、代码、命令和必要验证后，才能把课程数据中的状态改为已完成。填充方法见[章节模板](../../../maintenance/templates/lesson.md)。

## 参考资料

- [Expert Parallel Deployment](https://docs.vllm.ai/en/stable/serving/expert_parallel_deployment/)
- [并行配置与度数校验](https://github.com/vllm-project/vllm/blob/main/vllm/config/parallel.py)
- [Fused MoE Modular Kernel](https://docs.vllm.ai/en/stable/design/fused_moe_modular_kernel/)
- [MoE Kernel Features](https://docs.vllm.ai/en/stable/design/moe_kernel_features/)
- [Linear/MoE Kernel 配置](https://github.com/vllm-project/vllm/blob/main/vllm/config/kernel.py)
- [Dual Batch Overlap](https://docs.vllm.ai/en/stable/design/dbo/)
