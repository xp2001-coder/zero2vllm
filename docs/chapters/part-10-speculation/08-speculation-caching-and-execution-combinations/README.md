# 10.8 推测的缓存与执行组合

[本篇目录](../README.md) · [全部课程](../../README.md) · [上一节 10.7](../07-dynamic-speculation-and-adaptive-verification/README.md) · [下一节 10.9](../09-acceptance-rates-and-actual-gains/README.md)

> 🧩 已有相关材料；本节正文与配套实验待完善。目录存在不表示内容已经完成。

## 学习目标

KV 分配/回滚、CUDA Graph、异步调度、LoRA/量化/多模态/CP。

## 前置知识与设备

- 前置：Part 1、3、5、6。
- 设备：GPU/多卡。具体版本和执行命令随本节实验一起固定。
- A100/C500 的支持情况以实机记录为准，未执行的路径保持待验证。

## 正文安排

<!-- BEGIN lesson-plan -->

1. KV 分配/回滚：补充概念、推导、例子及实现边界。
2. CUDA Graph：补充概念、推导、例子及实现边界。
3. 异步调度：补充概念、推导、例子及实现边界。
4. LoRA/量化/多模态/CP：补充概念、推导、例子及实现边界。

<!-- END lesson-plan -->

## 代码与实验

- [ ] 确定最小输入、预期输出与张量/请求契约。
- [ ] 补充可独立运行的实现、命令、参数解释和结果示例。
- [ ] 覆盖正常输入、边界输入和失败排查。
- [ ] 保存可复核结果；涉及性能时注明设备、版本、负载和计时范围。

## 练习与验收

<!-- BEGIN lesson-exercise -->

目标产出：**为支持的组合设计错误恢复和正确性检查**。

- [ ] 提供输入、命令、结果与解释。
- [ ] 补充参考答案和至少一个边界案例。

<!-- END lesson-exercise -->

完成正文、代码、命令和必要验证后，才能把课程数据中的状态改为已完成。填充方法见[章节模板](../../../maintenance/templates/lesson.md)。

## 已有材料

先读[相关专题导读](../guide.md)。它覆盖多个小节，不能替代本节的完整实现和验收。

## 参考资料

- [Speculative Decoding](https://docs.vllm.ai/en/stable/features/speculative_decoding/)
- [推测方法与模型类型](https://github.com/vllm-project/vllm/blob/main/vllm/config/speculative.py)
- [EAGLE](https://docs.vllm.ai/en/stable/features/speculative_decoding/eagle/)
- [MTP](https://docs.vllm.ai/en/stable/features/speculative_decoding/mtp/)
- [Dynamic Speculative Decoding](https://docs.vllm.ai/en/stable/features/speculative_decoding/dynamic_speculative_decoding/)
- [Adaptive Verification](https://docs.vllm.ai/en/stable/features/speculative_decoding/adaptive_verification/)
