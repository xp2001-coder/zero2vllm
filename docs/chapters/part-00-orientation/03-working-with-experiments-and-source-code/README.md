# 0.3 实验与源码工作方法

[本篇目录](../README.md) · [全部课程](../../README.md) · [上一节 0.2](../02-your-first-model-generation/README.md) · [下一节 1.1](../../part-01-inference/01-next-token/README.md)

> 📋 教学骨架；正文与实验待编写。目录存在不表示内容已经完成。

## 学习目标

固定 revision、阅读配置/日志/调用链、最小复现与原始数据。

## 前置知识与设备

- 前置：基础 Python。
- 设备：CPU。具体版本和执行命令随本节实验一起固定。
- A100/C500 的支持情况以实机记录为准，未执行的路径保持待验证。

## 正文安排

<!-- BEGIN lesson-plan -->

1. 固定 revision：补充概念、推导、例子及实现边界。
2. 阅读配置/日志/调用链：补充概念、推导、例子及实现边界。
3. 最小复现与原始数据：补充概念、推导、例子及实现边界。

<!-- END lesson-plan -->

## 代码与实验

- [ ] 确定最小输入、预期输出与张量/请求契约。
- [ ] 补充可独立运行的实现、命令、参数解释和结果示例。
- [ ] 覆盖正常输入、边界输入和失败排查。
- [ ] 保存可复核结果；涉及性能时注明设备、版本、负载和计时范围。

## 练习与验收

<!-- BEGIN lesson-exercise -->

目标产出：**建立个人实验目录和源码阅读笔记模板**。

- [ ] 提供输入、命令、结果与解释。
- [ ] 补充参考答案和至少一个边界案例。

<!-- END lesson-exercise -->

完成正文、代码、命令和必要验证后，才能把课程数据中的状态改为已完成。填充方法见[章节模板](../../../maintenance/templates/lesson.md)。

## 参考资料

- [Datawhale zero-to-sglang](https://github.com/datawhalechina/zero-to-sglang)
- [vLLM Quickstart](https://docs.vllm.ai/en/stable/getting_started/quickstart/)
- [GPU 安装与设备条件](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)
