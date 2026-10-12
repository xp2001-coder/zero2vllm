# 3.2 固定版本源码定位表

[正文](README.md) · [实验](lab.md) · [源码清单](data/source-manifest.json)

版本：vLLM v0.8.5 / V1；提交 `ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d`。

以下位置来自本次对固定源码的语法树检查。调用表达式存在，不等于分支执行过；请结合正文逐行阅读条件、退出点与更新顺序。

| 阅读顺序 | 关注问题 | 函数与固定位置 |
| --- | --- | --- |
| 1 | 先运行队列，后等待队列；预算、分配与抢占 | [`Scheduler.schedule`](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/v1/core/sched/scheduler.py#L137) |
| 2 | 完整前缀查找与末块重算 | [`KVCacheManager.get_computed_blocks`](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/v1/core/kv_cache_manager.py#L92) |
| 3 | 分配公式、命中保护与新增块 | [`KVCacheManager.allocate_slots`](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/v1/core/kv_cache_manager.py#L164) |
| 4 | 拆除请求块表并释放引用 | [`KVCacheManager.free`](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/v1/core/kv_cache_manager.py#L296) |
| 5 | 命中块增加引用并离开空闲队列 | [`BlockPool.touch`](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/v1/core/block_pool.py#L212) |
| 6 | 取得块并清除旧缓存身份 | [`BlockPool.get_new_blocks`](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/v1/core/block_pool.py#L158) |
| 7 | 引用归零才归还队列 | [`BlockPool.free_blocks`](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/v1/core/block_pool.py#L227) |
| 8 | 移除缓存身份 | [`BlockPool._maybe_evict_cached_block`](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/v1/core/block_pool.py#L190) |
| 9 | 按请求更新采样结果与停止状态 | [`Scheduler.update_from_output`](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/v1/core/sched/scheduler.py#L625) |
| 10 | 最终结束时清理完整请求状态 | [`Scheduler._free_request`](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/v1/core/sched/scheduler.py#L791) |
| 11 | 输出 token 同步进入完整序列 | [`Request.append_output_token_ids`](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/v1/request.py#L94) |

## 不要只读入口

- `Scheduler.schedule()`：先找两个队列循环，再标出 q=0、容量不足、发生抢占后的分支。
- `KVCacheManager.allocate_slots()`：先在纸上定义 R/D/F/E，再对应源码变量；注意容量判断前的专门管理器清理。
- `BlockPool`：分别跟踪命中、普通分配、解除引用和逐出，避免把它们合成一个“释放”操作。
- `Scheduler.update_from_output()`：核对请求 ID 和结果索引的映射，以及停止后的状态回收。

额外核验 [`SchedulerOutput`](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/v1/core/sched/output.py#L82) 的新请求、缓存请求、逐请求 token 数、总 token 数与结束请求 ID 字段。

源码文本未复制到公开课程目录；清单包含官方来源、完整提交与 LF 规范化后的 SHA256。下载器复用 3.1 已有实现，默认不会联网。
