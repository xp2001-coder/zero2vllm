# Part 5 · KV Cache、Prefix Cache 与长上下文

[课程总目录](../README.md) · [学习路线](../learning-paths.md)

前置知识：Part 1—3。

## 本篇章节

<!-- BEGIN part-index -->

| 节 | 课程 | 验收产出 | 进度 | 正文 |
| --- | --- | --- | --- | --- |
| 5.1 | [PagedAttention 与缓存布局](01-pagedattention-and-cache-layouts/README.md) | 计算不同块大小的浪费并解释寻址过程 | 🧩 | 教学骨架 |
| 5.2 | [Automatic Prefix Caching](02-automatic-prefix-caching/README.md) | 设计整块命中、尾块未命中、无共享、salt 隔离和容量逐出用例 | 🧩 | 教学骨架 |
| 5.3 | [缓存键与共享隔离](03-cache-keys-and-sharing-isolation/README.md) | 设计共享与不应共享的请求用例，检查命中正确性 | 🧩 | 教学骨架 |
| 5.4 | [缓存生命周期与路由](04-cache-lifecycle-and-routing/README.md) | 回放容量不足轨迹并解释命中率和排队变化 | 🧩 | 教学骨架 |
| 5.5 | [混合模型缓存管理](05-cache-management-for-hybrid-models/README.md) | 绘制混合层的内存分配与有效缓存范围 | 📋 | 教学骨架 |
| 5.6 | [Mamba 与线性模型状态缓存](06-state-caching-for-mamba-and-linear-models/README.md) | 用状态递推模拟说明缓存与回滚为何不同于普通 KV | 📋 | 教学骨架 |
| 5.7 | [KV Offloading 与分层缓存](07-kv-offloading-and-tiered-caching/README.md) | 比较重算和传输成本；核对命中、传输字节与显存 | 📋 | 教学骨架 |
| 5.8 | [上下文扩展与 KV Sharing](08-context-extension-and-kv-sharing/README.md) | 设计长上下文质量评估并区分容量与有效理解长度 | 📋 | 教学骨架 |
| 5.9 | [IndexCache 与稀疏索引缓存](09-indexcache-and-sparse-index-caching/README.md) | 分析索引复用成本与质量变化；不把它当 APC | 📋 | 教学骨架 |

<!-- END part-index -->
