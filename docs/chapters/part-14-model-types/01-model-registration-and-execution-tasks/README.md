# 14.1 模型注册与执行任务

[本篇目录](../README.md) · [全部课程](../../README.md) · [上一节 13.12](../../part-13-disaggregation/12-distributed-serving-lifecycles/README.md) · [下一节 14.2](../02-text-generation-and-scoring/README.md)

> 🧩 已有相关材料；本节正文与配套实验待完善。目录存在不表示内容已经完成。

## 学习目标

generative/pooling、runner/task、原生实现/Transformers backend、encoder-decoder。

## 前置知识与设备

- 前置：Part 1、3、5。
- 设备：CPU/GPU。具体版本和执行命令随本节实验一起固定。
- A100/C500 的支持情况以实机记录为准，未执行的路径保持待验证。

## 正文安排

<!-- BEGIN lesson-plan -->

1. generative/pooling：补充概念、推导、例子及实现边界。
2. runner/task：补充概念、推导、例子及实现边界。
3. 原生实现/Transformers backend：补充概念、推导、例子及实现边界。
4. encoder-decoder：补充概念、推导、例子及实现边界。

<!-- END lesson-plan -->

## 代码与实验

- [ ] 确定最小输入、预期输出与张量/请求契约。
- [ ] 补充可独立运行的实现、命令、参数解释和结果示例。
- [ ] 覆盖正常输入、边界输入和失败排查。
- [ ] 保存可复核结果；涉及性能时注明设备、版本、负载和计时范围。

## 练习与验收

<!-- BEGIN lesson-exercise -->

目标产出：**识别架构与任务匹配，制作能力清单**。

- [ ] 提供输入、命令、结果与解释。
- [ ] 补充参考答案和至少一个边界案例。

<!-- END lesson-exercise -->

完成正文、代码、命令和必要验证后，才能把课程数据中的状态改为已完成。填充方法见[章节模板](../../../maintenance/templates/lesson.md)。

## 已有材料

先读[相关专题导读](../../part-03-vllm/guide.md)。它覆盖多个小节，不能替代本节的完整实现和验收。

## 参考资料

- [模型支持与任务](https://docs.vllm.ai/en/stable/models/supported_models/)
- [Pooling Models](https://docs.vllm.ai/en/stable/models/pooling_models/)
- [Multimodal Inputs](https://docs.vllm.ai/en/stable/features/multimodal_inputs/)
- [LoRA Adapters](https://docs.vllm.ai/en/stable/features/lora/)
- [Prompt Embedding Inputs](https://docs.vllm.ai/en/stable/features/prompt_embeds/)
- [ViT CUDA Graphs](https://docs.vllm.ai/en/stable/design/cuda_graphs_multimodal/)
- [离散扩散语言模型配置](https://github.com/vllm-project/vllm/blob/main/vllm/config/diffusion.py)
