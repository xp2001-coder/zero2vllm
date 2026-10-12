# 6.6 调度功能组合与演进

[本篇目录](../README.md) · [全部课程](../../README.md) · [上一节 6.5](../05-comparing-dbo-and-tbo/README.md) · [下一节 7.1](../../part-07-attention/01-flashattention-principles/README.md)

> 📋 教学骨架；正文与实验待编写。目录存在不表示内容已经完成。

## 学习目标

async scheduling × spec decode × graphs × PP/LoRA；旧 multi-step/best-of。

## 前置知识与设备

- 前置：Part 2—3、5。
- 设备：CPU/GPU/多卡。具体版本和执行命令随本节实验一起固定。
- A100/C500 的支持情况以实机记录为准，未执行的路径保持待验证。

## 正文安排

<!-- BEGIN lesson-plan -->

1. async scheduling × spec decode × graphs × PP/LoRA：补充概念、推导、例子及实现边界。
2. 旧 multi-step/best-of：补充概念、推导、例子及实现边界。
3. 混部与PD同预算对照，批处理/排队对阶段容量的影响：补充概念、推导、例子及实现边界。

<!-- END lesson-plan -->

## 代码与实验

- [ ] 确定最小输入、预期输出与张量/请求契约。
- [ ] 补充可独立运行的实现、命令、参数解释和结果示例。
- [ ] 覆盖正常输入、边界输入和失败排查。
- [ ] 保存可复核结果；涉及性能时注明设备、版本、负载和计时范围。

## 练习与验收

<!-- BEGIN lesson-exercise -->

目标产出：**制作所选版本组合矩阵，记录拒绝/回退原因**。

- [ ] 提供输入、命令、结果与解释。
- [ ] 补充参考答案和至少一个边界案例。

<!-- END lesson-exercise -->

完成正文、代码、命令和必要验证后，才能把课程数据中的状态改为已完成。填充方法见[章节模板](../../../maintenance/templates/lesson.md)。

## 参考资料

- [Optimization and Tuning](https://docs.vllm.ai/en/stable/configuration/optimization/)
- [调度配置](https://github.com/vllm-project/vllm/blob/main/vllm/config/scheduler.py)
- [Dual Batch Overlap](https://docs.vllm.ai/en/stable/design/dbo/)
- [SGLang EP：TBO/SBO](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/expert_parallelism.mdx)


## 与分离式服务的衔接

本节的相关结论将接入 [Part 13：分布式推理工程](../../part-13-disaggregation/README.md)。混部与PD同预算对照，批处理/排队对阶段容量的影响。先明确计量单位和验证范围，再用于集群推算。
