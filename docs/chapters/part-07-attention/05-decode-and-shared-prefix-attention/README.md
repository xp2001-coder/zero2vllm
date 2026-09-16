# 7.5 Decode 与共享前缀注意力

[本篇目录](../README.md) · [全部课程](../../README.md) · [上一节 7.4](../04-mla-and-sparse-attention/README.md) · [下一节 7.6](../06-linear-layers-and-fused-kernels/README.md)

> 🧩 已有相关材料；本节正文与配套实验待完善。目录存在不表示内容已经完成。

## 学习目标

分区归约、split-K、变长/分页、cascade attention、共享前缀。

## 前置知识与设备

- 前置：Part 1、3、5；补充 GPU 编程。
- 设备：GPU。具体版本和执行命令随本节实验一起固定。
- A100/C500 的支持情况以实机记录为准，未执行的路径保持待验证。

## 正文安排

<!-- BEGIN lesson-plan -->

1. 分区归约：补充概念、推导、例子及实现边界。
2. split-K：补充概念、推导、例子及实现边界。
3. 变长/分页：补充概念、推导、例子及实现边界。
4. cascade attention：补充概念、推导、例子及实现边界。
5. 共享前缀：补充概念、推导、例子及实现边界。

<!-- END lesson-plan -->

## 代码与实验

- [ ] 确定最小输入、预期输出与张量/请求契约。
- [ ] 补充可独立运行的实现、命令、参数解释和结果示例。
- [ ] 覆盖正常输入、边界输入和失败排查。
- [ ] 保存可复核结果；涉及性能时注明设备、版本、负载和计时范围。

## 练习与验收

<!-- BEGIN lesson-exercise -->

目标产出：**比较短/长上下文和高/低 batch 下的算子时间**。

- [ ] 提供输入、命令、结果与解释。
- [ ] 补充参考答案和至少一个边界案例。

<!-- END lesson-exercise -->

完成正文、代码、命令和必要验证后，才能把课程数据中的状态改为已完成。填充方法见[章节模板](../../../maintenance/templates/lesson.md)。

## 已有材料

先读[相关专题导读](../guide.md)。它覆盖多个小节，不能替代本节的完整实现和验收。

## 参考资料

- [Attention 后端注册表](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/registry.py)
- [Attention Backend 支持矩阵](https://docs.vllm.ai/en/stable/design/attention_backends/)
- [FlashAttention 官方实现与版本](https://github.com/Dao-AILab/flash-attention)
- [Linear/MoE Kernel 配置](https://github.com/vllm-project/vllm/blob/main/vllm/config/kernel.py)
- [CustomOp](https://docs.vllm.ai/en/stable/design/custom_op/)
- [Triton Fused Attention 教程](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)
