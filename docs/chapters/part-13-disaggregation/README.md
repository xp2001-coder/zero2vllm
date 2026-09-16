# Part 13 · 分离式部署与 KV 传输

[课程总目录](../README.md) · [学习路线](../learning-paths.md)

前置知识：Part 5、6、11。

## 本篇章节

<!-- BEGIN part-index -->

| 节 | 课程 | 验收产出 | 进度 | 正文 |
| --- | --- | --- | --- | --- |
| 13.1 | [Prefill-Decode 分离](01-prefill-decode-disaggregation/README.md) | 推导传输成本与分离收益的临界条件 | 🧩 | 教学骨架 |
| 13.2 | [KV Connector 协议](02-the-kv-connector-protocol/README.md) | 绘制一次远端 KV 的生产/传输/消费流程 | 🧩 | 教学骨架 |
| 13.3 | [连接器与分层存储地图](03-connectors-and-tiered-storage/README.md) | 逐项核对传输协议、存储层与外部依赖 | 📋 | 教学骨架 |
| 13.4 | [异构并行与传输兼容](04-heterogeneous-parallelism-and-transfer-compatibility/README.md) | 做兼容性表与流量预算，不能假定任意配置可互传 | 🧩 | 教学骨架 |
| 13.5 | [Encoder 分离与缓存](05-encoder-disaggregation-and-caching/README.md) | 画出图像/音频编码结果的跨进程流转 | 📋 | 教学骨架 |
| 13.6 | [分布式服务生命周期](06-distributed-serving-lifecycles/README.md) | 分析故障注入轨迹与请求完成/回收语义 | 🧩 | 教学骨架 |

<!-- END part-index -->
