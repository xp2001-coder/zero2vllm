# 11.4 Data Parallelism

[本篇目录](../README.md) · [全部课程](../../README.md) · [上一节 11.3](../03-pipeline-parallelism/README.md) · [下一节 11.5](../05-context-parallelism-concepts/README.md)

> 🧩 已有相关材料；本节正文与配套实验待完善。目录存在不表示内容已经完成。

## 学习目标

DP、副本、内部/外部/混合负载均衡、dense 与 MoE 差异。

## 前置知识与设备

- 前置：Part 3、5—8；掌握 collective 基础。
- 设备：CPU/多卡。具体版本和执行命令随本节实验一起固定。
- A100/C500 的支持情况以实机记录为准，未执行的路径保持待验证。

## 正文安排

<!-- BEGIN lesson-plan -->

1. DP：补充概念、推导、例子及实现边界。
2. 副本：补充概念、推导、例子及实现边界。
3. 内部/外部/混合负载均衡：补充概念、推导、例子及实现边界。
4. dense 与 MoE 差异：补充概念、推导、例子及实现边界。

<!-- END lesson-plan -->

## 代码与实验

- [ ] 确定最小输入、预期输出与张量/请求契约。
- [ ] 补充可独立运行的实现、命令、参数解释和结果示例。
- [ ] 覆盖正常输入、边界输入和失败排查。
- [ ] 保存可复核结果；涉及性能时注明设备、版本、负载和计时范围。

## 练习与验收

<!-- BEGIN lesson-exercise -->

目标产出：**比较请求路由、排队与副本扩展吞吐**。

- [ ] 提供输入、命令、结果与解释。
- [ ] 补充参考答案和至少一个边界案例。

<!-- END lesson-exercise -->

完成正文、代码、命令和必要验证后，才能把课程数据中的状态改为已完成。填充方法见[章节模板](../../../maintenance/templates/lesson.md)。

## 已有材料

先读[相关专题导读](../guide.md)。它覆盖多个小节，不能替代本节的完整实现和验收。

## 参考资料

- [Parallelism and Scaling](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/)
- [并行配置与度数校验](https://github.com/vllm-project/vllm/blob/main/vllm/config/parallel.py)
- [Context Parallel Deployment](https://docs.vllm.ai/en/stable/serving/context_parallel_deployment/)
- [Data Parallel Deployment](https://docs.vllm.ai/en/stable/serving/data_parallel_deployment/)
- [Expert Parallel Deployment](https://docs.vllm.ai/en/stable/serving/expert_parallel_deployment/)
- [编译模式与融合 Pass](https://github.com/vllm-project/vllm/blob/main/vllm/config/compilation.py)
