# 0.2 第一次模型生成

[本篇目录](../README.md) · [全部课程](../../README.md) · [上一节 0.1](../01-learning-paths-and-environment-setup/README.md) · [下一节 0.3](../03-working-with-experiments-and-source-code/README.md)

> 📋 教学骨架；正文与实验待编写。目录存在不表示内容已经完成。

## 学习目标

离线 LLM、在线服务、流式请求、Tokenizer 与 chat template。

## 前置知识与设备

- 前置：基础 Python。
- 设备：GPU。具体版本和执行命令随本节实验一起固定。
- A100/C500 的支持情况以实机记录为准，未执行的路径保持待验证。

## 正文安排

<!-- BEGIN lesson-plan -->

1. 离线 LLM：补充概念、推导、例子及实现边界。
2. 在线服务：补充概念、推导、例子及实现边界。
3. 流式请求：补充概念、推导、例子及实现边界。
4. Tokenizer 与 chat template：补充概念、推导、例子及实现边界。

<!-- END lesson-plan -->

## 代码与实验

- [ ] 确定最小输入、预期输出与张量/请求契约。
- [ ] 补充可独立运行的实现、命令、参数解释和结果示例。
- [ ] 覆盖正常输入、边界输入和失败排查。
- [ ] 保存可复核结果；涉及性能时注明设备、版本、负载和计时范围。

## 练习与验收

<!-- BEGIN lesson-exercise -->

目标产出：**画出输入消息到生成文本的数据流，保存一次服务记录**。

- [ ] 提供输入、命令、结果与解释。
- [ ] 补充参考答案和至少一个边界案例。

<!-- END lesson-exercise -->

完成正文、代码、命令和必要验证后，才能把课程数据中的状态改为已完成。填充方法见[章节模板](../../../maintenance/templates/lesson.md)。

## 参考资料

- [Datawhale zero-to-sglang](https://github.com/datawhalechina/zero-to-sglang)
- [vLLM Quickstart](https://docs.vllm.ai/en/stable/getting_started/quickstart/)
- [GPU 安装与设备条件](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)
