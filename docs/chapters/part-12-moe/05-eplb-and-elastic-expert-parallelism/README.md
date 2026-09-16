# 12.5 EPLB 与弹性专家并行

[本篇目录](../README.md) · [全部课程](../../README.md) · [上一节 12.4](../04-all-to-all-backends/README.md) · [下一节 12.6](../06-moe-overlap-and-combined-tuning/README.md)

> 🧩 已有相关材料；本节正文与配套实验待完善。目录存在不表示内容已经完成。

## 学习目标

EPLB、重排/冗余、异步均衡、Elastic EP、扩缩容与故障恢复。

## 前置知识与设备

- 前置：Part 7、8、11。
- 设备：CPU/多卡。具体版本和执行命令随本节实验一起固定。
- A100/C500 的支持情况以实机记录为准，未执行的路径保持待验证。

## 正文安排

<!-- BEGIN lesson-plan -->

1. EPLB：补充概念、推导、例子及实现边界。
2. 重排/冗余：补充概念、推导、例子及实现边界。
3. 异步均衡：补充概念、推导、例子及实现边界。
4. Elastic EP：补充概念、推导、例子及实现边界。
5. 扩缩容与故障恢复：补充概念、推导、例子及实现边界。

<!-- END lesson-plan -->

## 代码与实验

- [ ] 确定最小输入、预期输出与张量/请求契约。
- [ ] 补充可独立运行的实现、命令、参数解释和结果示例。
- [ ] 覆盖正常输入、边界输入和失败排查。
- [ ] 保存可复核结果；涉及性能时注明设备、版本、负载和计时范围。

## 练习与验收

<!-- BEGIN lesson-exercise -->

目标产出：**模拟热点专家并评估迁移成本和再均衡收益**。

- [ ] 提供输入、命令、结果与解释。
- [ ] 补充参考答案和至少一个边界案例。

<!-- END lesson-exercise -->

完成正文、代码、命令和必要验证后，才能把课程数据中的状态改为已完成。填充方法见[章节模板](../../../maintenance/templates/lesson.md)。

## 已有材料

先读[相关专题导读](../../part-11-parallelism/guide.md)。它覆盖多个小节，不能替代本节的完整实现和验收。

## 参考资料

- [Expert Parallel Deployment](https://docs.vllm.ai/en/stable/serving/expert_parallel_deployment/)
- [并行配置与度数校验](https://github.com/vllm-project/vllm/blob/main/vllm/config/parallel.py)
- [Fused MoE Modular Kernel](https://docs.vllm.ai/en/stable/design/fused_moe_modular_kernel/)
- [MoE Kernel Features](https://docs.vllm.ai/en/stable/design/moe_kernel_features/)
- [Linear/MoE Kernel 配置](https://github.com/vllm-project/vllm/blob/main/vllm/config/kernel.py)
- [Dual Batch Overlap](https://docs.vllm.ai/en/stable/design/dbo/)
