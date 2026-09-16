# 14.5 多模态执行优化

[本篇目录](../README.md) · [全部课程](../../README.md) · [上一节 14.4](../04-the-multimodal-input-pipeline/README.md) · [下一节 14.6](../06-lora-and-multi-tenant-adaptation/README.md)

> 📋 教学骨架；正文与实验待编写。目录存在不表示内容已经完成。

## 学习目标

encoder batching/DP、ViT attention/FP8、encoder compile/graphs、EC cache。

## 前置知识与设备

- 前置：Part 1、3、5。
- 设备：GPU/多卡/专项。具体版本和执行命令随本节实验一起固定。
- A100/C500 的支持情况以实机记录为准，未执行的路径保持待验证。

## 正文安排

<!-- BEGIN lesson-plan -->

1. encoder batching/DP：补充概念、推导、例子及实现边界。
2. ViT attention/FP8：补充概念、推导、例子及实现边界。
3. encoder compile/graphs：补充概念、推导、例子及实现边界。
4. EC cache：补充概念、推导、例子及实现边界。

<!-- END lesson-plan -->

## 代码与实验

- [ ] 确定最小输入、预期输出与张量/请求契约。
- [ ] 补充可独立运行的实现、命令、参数解释和结果示例。
- [ ] 覆盖正常输入、边界输入和失败排查。
- [ ] 保存可复核结果；涉及性能时注明设备、版本、负载和计时范围。

## 练习与验收

<!-- BEGIN lesson-exercise -->

目标产出：**拆开编码与文本生成耗时并验证组合条件**。

- [ ] 提供输入、命令、结果与解释。
- [ ] 补充参考答案和至少一个边界案例。

<!-- END lesson-exercise -->

完成正文、代码、命令和必要验证后，才能把课程数据中的状态改为已完成。填充方法见[章节模板](../../../maintenance/templates/lesson.md)。

## 参考资料

- [模型支持与任务](https://docs.vllm.ai/en/stable/models/supported_models/)
- [Pooling Models](https://docs.vllm.ai/en/stable/models/pooling_models/)
- [Multimodal Inputs](https://docs.vllm.ai/en/stable/features/multimodal_inputs/)
- [LoRA Adapters](https://docs.vllm.ai/en/stable/features/lora/)
- [Prompt Embedding Inputs](https://docs.vllm.ai/en/stable/features/prompt_embeds/)
- [ViT CUDA Graphs](https://docs.vllm.ai/en/stable/design/cuda_graphs_multimodal/)
- [离散扩散语言模型配置](https://github.com/vllm-project/vllm/blob/main/vllm/config/diffusion.py)
