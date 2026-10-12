# 专题导读

[课程总目录](../chapters/README.md) · [参考资料](README.md)

以下七篇导读串联课程中的相关章节，提供概念解释、手算问题和实验设计。导读表示已有部分材料；完整章节与 GPU 实测的进度以课程目录为准。

## 选择一个学习入口

| 材料 | 解决的问题 | 当前可交付的练习 |
| --- | --- | --- |
| [请求、状态与张量](../chapters/part-03-vllm/guide.md) | 一次请求经过哪些组件，谁准备张量、谁持有状态？ | 请求链路、加载流程和版本对照表 |
| [调度与缓存](../chapters/part-05-kv-cache/guide.md) | token 预算、KV 容量和前缀命中如何共同影响调度？ | 手算调度轮次、块寻址与缓存实验设计 |
| [Attention 与算子](../chapters/part-07-attention/guide.md) | FlashAttention、分页存储和后端选择分别解决什么？ | Online Softmax 推导、Triton 实现顺序 |
| [编译、图执行与 Profiling](../chapters/part-08-compilation/guide.md) | 如何确认某项优化真的生效？ | 编译/图模式消融、同 shape 时间线分析 |
| [量化策略与算子](../chapters/part-09-quantization/guide.md) | 从算法选择走到布局、内核和 vLLM 集成 | [CPU INT4 Linear 实验](../chapters/part-09-quantization/08-bit-packing-serialization-and-repack/lab.md)与 GPU 后续任务 |
| [并行、MoE 与分离式部署](../chapters/part-11-parallelism/guide.md) | 切什么、怎么通信、需要几张卡？ | TP/CP 推导、专家路由和 KV 传输账本 |
| [推测解码与服务闭环](../chapters/part-10-speculation/guide.md) | 接受率、缓存提交和流式结果怎样对应？ | 接受/拒绝手算、停止与取消场景设计 |

初学者先完成 [Part 1](../chapters/part-01-inference/README.md) 和 [2.1 最小模型](../chapters/part-02-engine/01-pytorch-model/README.md)，再进入请求与缓存。工程师可以先阅读编译与 Profiling 导读，确定测量方法；量化方向可从量化导读和 INT4 实验进入。

## 配合正文使用

1. 从导读选择问题，再进入对应章节完成推导、代码和练习。
2. 涉及源码时，使用正文列出的官方链接并核对 tag/commit；动态文档页面不能代替运行版本记录。
3. 区分数学参考、模拟结果和 GPU 测量，实验结果以相应章节的原始记录为准。

公开来源集中在[参考资料总表](README.md)，各章另列与当前内容直接相关的论文、官方文档和源码。
