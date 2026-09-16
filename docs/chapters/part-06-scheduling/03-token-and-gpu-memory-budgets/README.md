# 6.3 Token 预算与显存预算

[本篇目录](../README.md) · [全部课程](../../README.md) · [上一节 6.2](../02-chunked-prefill/README.md) · [下一节 6.4](../04-async-scheduling-and-output-processing/README.md)

> 🧩 已有相关材料；本节正文与配套实验待完善。目录存在不表示内容已经完成。

## 学习目标

max_num_seqs、max_num_batched_tokens、版本中的 scheduled-token 上限、max_model_len、KV/encoder/推测预算。

## 前置知识与设备

- 前置：Part 2—3、5。
- 设备：CPU/GPU。具体版本和执行命令随本节实验一起固定。
- A100/C500 的支持情况以实机记录为准，未执行的路径保持待验证。

## 正文安排

<!-- BEGIN lesson-plan -->

1. max_num_seqs：补充概念、推导、例子及实现边界。
2. max_num_batched_tokens：补充概念、推导、例子及实现边界。
3. 版本中的 scheduled-token 上限：补充概念、推导、例子及实现边界。
4. max_model_len：补充概念、推导、例子及实现边界。
5. KV/encoder/推测预算：补充概念、推导、例子及实现边界。

<!-- END lesson-plan -->

## 代码与实验

- [ ] 确定最小输入、预期输出与张量/请求契约。
- [ ] 补充可独立运行的实现、命令、参数解释和结果示例。
- [ ] 覆盖正常输入、边界输入和失败排查。
- [ ] 保存可复核结果；涉及性能时注明设备、版本、负载和计时范围。

## 练习与验收

<!-- BEGIN lesson-exercise -->

目标产出：**手算 4096 token 预算中两个 decode 加一个长 prefill，标记理想上限与真实约束**。

- [ ] 提供输入、命令、结果与解释。
- [ ] 补充参考答案和至少一个边界案例。

<!-- END lesson-exercise -->

完成正文、代码、命令和必要验证后，才能把课程数据中的状态改为已完成。填充方法见[章节模板](../../../maintenance/templates/lesson.md)。

## 已有材料

先读[相关专题导读](../../part-05-kv-cache/guide.md)。它覆盖多个小节，不能替代本节的完整实现和验收。

## 参考资料

- [Optimization and Tuning](https://docs.vllm.ai/en/stable/configuration/optimization/)
- [调度配置](https://github.com/vllm-project/vllm/blob/main/vllm/config/scheduler.py)
- [Dual Batch Overlap](https://docs.vllm.ai/en/stable/design/dbo/)
- [SGLang EP：TBO/SBO](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/expert_parallelism.mdx)
