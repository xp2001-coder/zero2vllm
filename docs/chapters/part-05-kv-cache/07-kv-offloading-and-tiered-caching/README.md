# 5.7 KV Offloading 与分层缓存

[本篇目录](../README.md) · [全部课程](../../README.md) · [上一节 5.6](../06-state-caching-for-mamba-and-linear-models/README.md) · [下一节 5.8](../08-context-extension-and-kv-sharing/README.md)

> 📋 教学骨架；正文与实验待编写。目录存在不表示内容已经完成。

## 学习目标

原生/LMCache、GPU↔CPU、异步换入换出、存储与连接器。

## 前置知识与设备

- 前置：Part 1—3。
- 设备：GPU/多机。具体版本和执行命令随本节实验一起固定。
- A100/C500 的支持情况以实机记录为准，未执行的路径保持待验证。

## 正文安排

<!-- BEGIN lesson-plan -->

1. 原生/LMCache：补充概念、推导、例子及实现边界。
2. GPU↔CPU：补充概念、推导、例子及实现边界。
3. 异步换入换出：补充概念、推导、例子及实现边界。
4. 存储与连接器：补充概念、推导、例子及实现边界。
5. 对象/字节命中与远端读取成本，衔接13.5/13.8：补充概念、推导、例子及实现边界。

<!-- END lesson-plan -->

## 代码与实验

- [ ] 确定最小输入、预期输出与张量/请求契约。
- [ ] 补充可独立运行的实现、命令、参数解释和结果示例。
- [ ] 覆盖正常输入、边界输入和失败排查。
- [ ] 保存可复核结果；涉及性能时注明设备、版本、负载和计时范围。

## 练习与验收

<!-- BEGIN lesson-exercise -->

目标产出：**比较重算和传输成本；核对命中、传输字节与显存**。

- [ ] 提供输入、命令、结果与解释。
- [ ] 补充参考答案和至少一个边界案例。

<!-- END lesson-exercise -->

完成正文、代码、命令和必要验证后，才能把课程数据中的状态改为已完成。填充方法见[章节模板](../../../maintenance/templates/lesson.md)。

## 参考资料

- [Automatic Prefix Caching](https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/)
- [Prefix Caching 设计](https://docs.vllm.ai/en/stable/design/prefix_caching/)
- [缓存配置与 dtype 清单](https://github.com/vllm-project/vllm/blob/main/vllm/config/cache.py)
- [Hybrid KV Cache Manager](https://docs.vllm.ai/en/stable/design/hybrid_kv_cache_manager/)
- [KV Offloading](https://docs.vllm.ai/en/stable/features/kv_offloading_usage/)
- [Context Extension](https://docs.vllm.ai/en/stable/features/context_extension/)
- [IndexCache](https://docs.vllm.ai/en/stable/features/index_cache/)


## 与分离式服务的衔接

本节的相关结论将接入 [Part 13：分布式推理工程](../../part-13-disaggregation/README.md)。对象/字节命中与远端读取成本，衔接13.5/13.8。先明确计量单位和验证范围，再用于集群推算。
