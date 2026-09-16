# 2.7 HTTP 与流式输出

[本篇目录](../README.md) · [全部课程](../../README.md) · [上一节 2.6](../06-prefix-caching/README.md) · [下一节 3.1](../../part-03-vllm/01-a-request-s-journey-through-vllm/README.md)

> 📋 教学骨架；正文与实验待编写。目录存在不表示内容已经完成。

## 学习目标

异步迭代、背压、断连、取消传播、请求错误。

## 前置知识与设备

- 前置：Part 1。
- 设备：GPU。具体版本和执行命令随本节实验一起固定。
- A100/C500 的支持情况以实机记录为准，未执行的路径保持待验证。

## 正文安排

<!-- BEGIN lesson-plan -->

1. 异步迭代：补充概念、推导、例子及实现边界。
2. 背压：补充概念、推导、例子及实现边界。
3. 断连：补充概念、推导、例子及实现边界。
4. 取消传播：补充概念、推导、例子及实现边界。
5. 请求错误：补充概念、推导、例子及实现边界。

<!-- END lesson-plan -->

## 代码与实验

- [ ] 确定最小输入、预期输出与张量/请求契约。
- [ ] 补充可独立运行的实现、命令、参数解释和结果示例。
- [ ] 覆盖正常输入、边界输入和失败排查。
- [ ] 保存可复核结果；涉及性能时注明设备、版本、负载和计时范围。

## 练习与验收

<!-- BEGIN lesson-exercise -->

目标产出：**让断连请求释放资源并保存流式事件记录**。

- [ ] 提供输入、命令、结果与解释。
- [ ] 补充参考答案和至少一个边界案例。

<!-- END lesson-exercise -->

完成正文、代码、命令和必要验证后，才能把课程数据中的状态改为已完成。填充方法见[章节模板](../../../maintenance/templates/lesson.md)。

## 参考资料

- [Transformers](https://github.com/huggingface/transformers)
- [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)
- [mini-sglang](https://github.com/sgl-project/mini-sglang)
- [架构设计](https://docs.vllm.ai/en/stable/design/arch_overview/)
