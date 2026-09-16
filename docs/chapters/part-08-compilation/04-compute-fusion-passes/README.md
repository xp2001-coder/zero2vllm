# 8.4 计算融合 Pass

[本篇目录](../README.md) · [全部课程](../../README.md) · [上一节 8.3](../03-shape-specialization-and-compilation-caches/README.md) · [下一节 8.5](../05-communication-fusion-and-async-tp/README.md)

> 🧩 已有相关材料；本节正文与配套实验待完善。目录存在不表示内容已经完成。

## 学习目标

Norm+Quant、Act+Quant、Attention+Quant、QK Norm+RoPE、RoPE+KV。

## 前置知识与设备

- 前置：Part 3、6、7。
- 设备：GPU。具体版本和执行命令随本节实验一起固定。
- A100/C500 的支持情况以实机记录为准，未执行的路径保持待验证。

## 正文安排

<!-- BEGIN lesson-plan -->

1. Norm+Quant：补充概念、推导、例子及实现边界。
2. Act+Quant：补充概念、推导、例子及实现边界。
3. Attention+Quant：补充概念、推导、例子及实现边界。
4. QK Norm+RoPE：补充概念、推导、例子及实现边界。
5. RoPE+KV：补充概念、推导、例子及实现边界。

<!-- END lesson-plan -->

## 代码与实验

- [ ] 确定最小输入、预期输出与张量/请求契约。
- [ ] 补充可独立运行的实现、命令、参数解释和结果示例。
- [ ] 覆盖正常输入、边界输入和失败排查。
- [ ] 保存可复核结果；涉及性能时注明设备、版本、负载和计时范围。

## 练习与验收

<!-- BEGIN lesson-exercise -->

目标产出：**对照融合前后 kernel 数和误差，不只看开关值**。

- [ ] 提供输入、命令、结果与解释。
- [ ] 补充参考答案和至少一个边界案例。

<!-- END lesson-exercise -->

完成正文、代码、命令和必要验证后，才能把课程数据中的状态改为已完成。填充方法见[章节模板](../../../maintenance/templates/lesson.md)。

## 已有材料

先读[相关专题导读](../guide.md)。它覆盖多个小节，不能替代本节的完整实现和验收。

## 参考资料

- [torch.compile 集成](https://docs.vllm.ai/en/stable/design/torch_compile/)
- [编译模式与融合 Pass](https://github.com/vllm-project/vllm/blob/main/vllm/config/compilation.py)
- [CUDA Graphs](https://docs.vllm.ai/en/stable/design/cuda_graphs/)
- [Optimization Levels](https://docs.vllm.ai/en/stable/design/optimization_levels/)
- [vLLM IR](https://docs.vllm.ai/en/stable/design/vllm_ir/)
- [ViT CUDA Graphs](https://docs.vllm.ai/en/stable/design/cuda_graphs_multimodal/)
