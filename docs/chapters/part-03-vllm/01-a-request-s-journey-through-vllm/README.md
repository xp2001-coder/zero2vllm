# 3.1 vLLM 中一次请求的旅程

[本篇目录](../README.md) · [全部课程](../../README.md) · [上一节 2.7](../../part-02-engine/07-http-and-streaming-output/README.md) · [下一节 3.2](../02-scheduling-and-kv-management-in-vllm/README.md)

> 🧩 已有相关材料；本节正文与配套实验待完善。目录存在不表示内容已经完成。

## 学习目标

输入处理→EngineCoreClient→EngineCore→Executor/Worker/ModelRunner；请求 ID、进程通信、流式队列与取消。

## 前置知识与设备

- 前置：Part 2；工程师可从本篇进入。
- 设备：CPU/GPU。具体版本和执行命令随本节实验一起固定。
- A100/C500 的支持情况以实机记录为准，未执行的路径保持待验证。

## 正文安排

<!-- BEGIN lesson-plan -->

1. 输入处理→EngineCoreClient→EngineCore→Executor/Worker/ModelRunner：补充概念、推导、例子及实现边界。
2. 请求 ID：补充概念、推导、例子及实现边界。
3. 进程通信：补充概念、推导、例子及实现边界。
4. 流式队列与取消：补充概念、推导、例子及实现边界。

<!-- END lesson-plan -->

## 代码与实验

- [ ] 确定最小输入、预期输出与张量/请求契约。
- [ ] 补充可独立运行的实现、命令、参数解释和结果示例。
- [ ] 覆盖正常输入、边界输入和失败排查。
- [ ] 保存可复核结果；涉及性能时注明设备、版本、负载和计时范围。

## 练习与验收

<!-- BEGIN lesson-exercise -->

目标产出：**为固定版本标出请求 ID、进程/线程边界、消息内容及取消后的资源回收**。

- [ ] 提供输入、命令、结果与解释。
- [ ] 补充参考答案和至少一个边界案例。

<!-- END lesson-exercise -->

完成正文、代码、命令和必要验证后，才能把课程数据中的状态改为已完成。填充方法见[章节模板](../../../maintenance/templates/lesson.md)。

## 已有材料

先读[相关专题导读](../guide.md)。它覆盖多个小节，不能替代本节的完整实现和验收。

## 参考资料

- [架构设计](https://docs.vllm.ai/en/stable/design/arch_overview/)
- [Model Runner V2](https://docs.vllm.ai/en/stable/design/model_runner_v2/)
- [Hugging Face 集成](https://docs.vllm.ai/en/stable/design/huggingface_integration/)
- [vLLM Features 与兼容性入口](https://docs.vllm.ai/en/stable/features/)
- [OpenAI-compatible Server](https://docs.vllm.ai/en/stable/serving/online_serving/openai_compatible_server/)
