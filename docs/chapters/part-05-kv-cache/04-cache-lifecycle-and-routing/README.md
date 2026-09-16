# 5.4 缓存生命周期与路由

[本篇目录](../README.md) · [全部课程](../../README.md) · [上一节 5.3](../03-cache-keys-and-sharing-isolation/README.md) · [下一节 5.5](../05-cache-management-for-hybrid-models/README.md)

> 🧩 已有相关材料；本节正文与配套实验待完善。目录存在不表示内容已经完成。

## 学习目标

引用计数、LRU、逐出、抢占重算、reset、KV events、亲和路由。

## 前置知识与设备

- 前置：Part 1—3。
- 设备：CPU/GPU。具体版本和执行命令随本节实验一起固定。
- A100/C500 的支持情况以实机记录为准，未执行的路径保持待验证。

## 正文安排

<!-- BEGIN lesson-plan -->

1. 引用计数：补充概念、推导、例子及实现边界。
2. LRU：补充概念、推导、例子及实现边界。
3. 逐出：补充概念、推导、例子及实现边界。
4. 抢占重算：补充概念、推导、例子及实现边界。
5. reset：补充概念、推导、例子及实现边界。
6. KV events：补充概念、推导、例子及实现边界。
7. 亲和路由：补充概念、推导、例子及实现边界。

<!-- END lesson-plan -->

## 代码与实验

- [ ] 确定最小输入、预期输出与张量/请求契约。
- [ ] 补充可独立运行的实现、命令、参数解释和结果示例。
- [ ] 覆盖正常输入、边界输入和失败排查。
- [ ] 保存可复核结果；涉及性能时注明设备、版本、负载和计时范围。

## 练习与验收

<!-- BEGIN lesson-exercise -->

目标产出：**回放容量不足轨迹并解释命中率和排队变化**。

- [ ] 提供输入、命令、结果与解释。
- [ ] 补充参考答案和至少一个边界案例。

<!-- END lesson-exercise -->

完成正文、代码、命令和必要验证后，才能把课程数据中的状态改为已完成。填充方法见[章节模板](../../../maintenance/templates/lesson.md)。

## 已有材料

先读[相关专题导读](../guide.md)。它覆盖多个小节，不能替代本节的完整实现和验收。

## 参考资料

- [Automatic Prefix Caching](https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/)
- [Prefix Caching 设计](https://docs.vllm.ai/en/stable/design/prefix_caching/)
- [缓存配置与 dtype 清单](https://github.com/vllm-project/vllm/blob/main/vllm/config/cache.py)
- [Hybrid KV Cache Manager](https://docs.vllm.ai/en/stable/design/hybrid_kv_cache_manager/)
- [KV Offloading](https://docs.vllm.ai/en/stable/features/kv_offloading_usage/)
- [Context Extension](https://docs.vllm.ai/en/stable/features/context_extension/)
- [IndexCache](https://docs.vllm.ai/en/stable/features/index_cache/)
