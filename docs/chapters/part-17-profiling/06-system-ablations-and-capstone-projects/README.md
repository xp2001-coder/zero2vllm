# 17.6 系统消融与综合项目

[本篇目录](../README.md) · [全部课程](../../README.md) · [上一节 17.5](../05-quality-reproducibility-and-batch-invariance/README.md) · [下一节 18.1](../../part-18-deployment/01-platforms-and-plugins/README.md)

> 🧩 已有相关材料；本节正文与配套实验待完善。目录存在不表示内容已经完成。

## 学习目标

APC×Chunked Prefill×Graphs×Quant×Spec、双平台成本、容量规划。

## 前置知识与设备

- 前置：Part 4；按实验选择 Part 5—16。
- 设备：双平台。具体版本和执行命令随本节实验一起固定。
- A100/C500 的支持情况以实机记录为准，未执行的路径保持待验证。

## 正文安排

<!-- BEGIN lesson-plan -->

1. APC×Chunked Prefill×Graphs×Quant×Spec：补充概念、推导、例子及实现边界。
2. 双平台成本：补充概念、推导、例子及实现边界。
3. 容量规划：补充概念、推导、例子及实现边界。
4. P/D容量预测残差：补充概念、推导、例子及实现边界。
5. 等卡数/等成本消融，衔接13.12：补充概念、推导、例子及实现边界。

<!-- END lesson-plan -->

## 代码与实验

- [ ] 确定最小输入、预期输出与张量/请求契约。
- [ ] 补充可独立运行的实现、命令、参数解释和结果示例。
- [ ] 覆盖正常输入、边界输入和失败排查。
- [ ] 保存可复核结果；涉及性能时注明设备、版本、负载和计时范围。

## 练习与验收

<!-- BEGIN lesson-exercise -->

目标产出：**交付可复现优化报告、原始数据及不生效的配置**。

- [ ] 提供输入、命令、结果与解释。
- [ ] 补充参考答案和至少一个边界案例。

<!-- END lesson-exercise -->

完成正文、代码、命令和必要验证后，才能把课程数据中的状态改为已完成。填充方法见[章节模板](../../../maintenance/templates/lesson.md)。

## 已有材料

先读[相关专题导读](../../part-08-compilation/guide.md)。它覆盖多个小节，不能替代本节的完整实现和验收。

## 参考资料

- [Benchmark CLI](https://docs.vllm.ai/en/stable/benchmarking/cli/)
- [Parameter Sweeps](https://docs.vllm.ai/en/stable/benchmarking/sweeps/)
- [Profiling vLLM](https://docs.vllm.ai/en/stable/contributing/profiling/)
- [Production Metrics](https://docs.vllm.ai/en/stable/usage/metrics/)
- [Per-Request Metrics](https://docs.vllm.ai/en/stable/features/per_request_metrics/)
- [Batch Invariance](https://docs.vllm.ai/en/stable/features/batch_invariance/)


## 与分离式服务的衔接

本节的相关结论将接入 [Part 13：分布式推理工程](../../part-13-disaggregation/README.md)。P/D容量预测残差、等卡数/等成本消融，衔接13.12。先明确计量单位和验证范围，再用于集群推算。
