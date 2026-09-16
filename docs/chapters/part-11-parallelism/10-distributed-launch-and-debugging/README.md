# 11.10 分布式启动与排错

[本篇目录](../README.md) · [全部课程](../../README.md) · [上一节 11.9](../09-combining-parallelism-and-planning-topology/README.md) · [下一节 12.1](../../part-12-moe/01-the-moe-computation-path/README.md)

> 📋 教学骨架；正文与实验待编写。目录存在不表示内容已经完成。

## 学习目标

mp/Ray/external launcher、节点发现、通信超时、NCCL/P2P、故障。

## 前置知识与设备

- 前置：Part 3、5—8；掌握 collective 基础。
- 设备：多卡/多机。具体版本和执行命令随本节实验一起固定。
- A100/C500 的支持情况以实机记录为准，未执行的路径保持待验证。

## 正文安排

<!-- BEGIN lesson-plan -->

1. mp/Ray/external launcher：补充概念、推导、例子及实现边界。
2. 节点发现：补充概念、推导、例子及实现边界。
3. 通信超时：补充概念、推导、例子及实现边界。
4. NCCL/P2P：补充概念、推导、例子及实现边界。
5. 故障：补充概念、推导、例子及实现边界。

<!-- END lesson-plan -->

## 代码与实验

- [ ] 确定最小输入、预期输出与张量/请求契约。
- [ ] 补充可独立运行的实现、命令、参数解释和结果示例。
- [ ] 覆盖正常输入、边界输入和失败排查。
- [ ] 保存可复核结果；涉及性能时注明设备、版本、负载和计时范围。

## 练习与验收

<!-- BEGIN lesson-exercise -->

目标产出：**设计启动、卡数不足、断连与通信失败的排查流程**。

- [ ] 提供输入、命令、结果与解释。
- [ ] 补充参考答案和至少一个边界案例。

<!-- END lesson-exercise -->

完成正文、代码、命令和必要验证后，才能把课程数据中的状态改为已完成。填充方法见[章节模板](../../../maintenance/templates/lesson.md)。

## 参考资料

- [Parallelism and Scaling](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/)
- [并行配置与度数校验](https://github.com/vllm-project/vllm/blob/main/vllm/config/parallel.py)
- [Context Parallel Deployment](https://docs.vllm.ai/en/stable/serving/context_parallel_deployment/)
- [Data Parallel Deployment](https://docs.vllm.ai/en/stable/serving/data_parallel_deployment/)
- [Expert Parallel Deployment](https://docs.vllm.ai/en/stable/serving/expert_parallel_deployment/)
- [编译模式与融合 Pass](https://github.com/vllm-project/vllm/blob/main/vllm/config/compilation.py)
