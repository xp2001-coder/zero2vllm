# 18.1 平台与插件机制

[本篇目录](../README.md) · [全部课程](../../README.md) · [上一节 17.6](../../part-17-profiling/06-system-ablations-and-capstone-projects/README.md) · [下一节 18.2](../02-adapting-to-c500/README.md)

> 📋 教学骨架；正文与实验待编写。目录存在不表示内容已经完成。

## 学习目标

out-of-tree platform、模型/算子/量化/LoRA/I/O 插件、能力检测。

## 前置知识与设备

- 前置：Part 3、4、17。
- 设备：CPU/GPU。具体版本和执行命令随本节实验一起固定。
- A100/C500 的支持情况以实机记录为准，未执行的路径保持待验证。

## 正文安排

<!-- BEGIN lesson-plan -->

1. out-of-tree platform：补充概念、推导、例子及实现边界。
2. 模型/算子/量化/LoRA/I/O 插件：补充概念、推导、例子及实现边界。
3. 能力检测：补充概念、推导、例子及实现边界。

<!-- END lesson-plan -->

## 代码与实验

- [ ] 确定最小输入、预期输出与张量/请求契约。
- [ ] 补充可独立运行的实现、命令、参数解释和结果示例。
- [ ] 覆盖正常输入、边界输入和失败排查。
- [ ] 保存可复核结果；涉及性能时注明设备、版本、负载和计时范围。

## 练习与验收

<!-- BEGIN lesson-exercise -->

目标产出：**追踪 MetaX 插件如何覆盖平台行为与算子选择**。

- [ ] 提供输入、命令、结果与解释。
- [ ] 补充参考答案和至少一个边界案例。

<!-- END lesson-exercise -->

完成正文、代码、命令和必要验证后，才能把课程数据中的状态改为已完成。填充方法见[章节模板](../../../maintenance/templates/lesson.md)。

## 参考资料

- [Plugin System](https://docs.vllm.ai/en/stable/design/plugin_system/)
- [CustomOp](https://docs.vllm.ai/en/stable/design/custom_op/)
- [vLLM-MetaX](https://github.com/MetaX-MACA/vLLM-metax)
- [MetaX 模型列表](https://vllm-metax.readthedocs.io/en/stable/models/supported_models.html)
- [Kubernetes 部署](https://docs.vllm.ai/en/stable/deployment/k8s/)
- [vLLM Features 与兼容性入口](https://docs.vllm.ai/en/stable/features/)
- [SGLang Hyperparameter Tuning](https://docs.sglang.io/docs/advanced_features/hyperparameter_tuning)
