# 15.4 Tool Calling

[本篇目录](../README.md) · [全部课程](../../README.md) · [上一节 15.3](../03-structured-outputs/README.md) · [下一节 15.5](../05-reasoning-and-interleaved-thinking/README.md)

> 📋 教学骨架；正文与实验待编写。目录存在不表示内容已经完成。

## 学习目标

tool_choice、自动/指定/required、parallel tools、parser、chat template。

## 前置知识与设备

- 前置：Part 3、14。
- 设备：CPU/GPU。具体版本和执行命令随本节实验一起固定。
- A100/C500 的支持情况以实机记录为准，未执行的路径保持待验证。

## 正文安排

<!-- BEGIN lesson-plan -->

1. tool_choice：补充概念、推导、例子及实现边界。
2. 自动/指定/required：补充概念、推导、例子及实现边界。
3. parallel tools：补充概念、推导、例子及实现边界。
4. parser：补充概念、推导、例子及实现边界。
5. chat template：补充概念、推导、例子及实现边界。

<!-- END lesson-plan -->

## 代码与实验

- [ ] 确定最小输入、预期输出与张量/请求契约。
- [ ] 补充可独立运行的实现、命令、参数解释和结果示例。
- [ ] 覆盖正常输入、边界输入和失败排查。
- [ ] 保存可复核结果；涉及性能时注明设备、版本、负载和计时范围。

## 练习与验收

<!-- BEGIN lesson-exercise -->

目标产出：**构建工具调用解析用例与流式增量组装检查**。

- [ ] 提供输入、命令、结果与解释。
- [ ] 补充参考答案和至少一个边界案例。

<!-- END lesson-exercise -->

完成正文、代码、命令和必要验证后，才能把课程数据中的状态改为已完成。填充方法见[章节模板](../../../maintenance/templates/lesson.md)。

## 参考资料

- [OpenAI-compatible Server](https://docs.vllm.ai/en/stable/serving/online_serving/openai_compatible_server/)
- [Structured Outputs](https://docs.vllm.ai/en/stable/features/structured_outputs/)
- [Tool Calling](https://docs.vllm.ai/en/stable/features/tool_calling/)
- [Reasoning Outputs](https://docs.vllm.ai/en/stable/features/reasoning_outputs/)
- [Interleaved Thinking](https://docs.vllm.ai/en/stable/features/interleaved_thinking/)
- [Custom Logits Processors](https://docs.vllm.ai/en/stable/features/custom_logitsprocs/)
