# Part 2 · 从零构建教学推理引擎

[课程总目录](../README.md) · [学习路线](../learning-paths.md)

前置知识：Part 1。

本篇从可检查的 PyTorch 模型出发，逐步加入真实权重、请求管理、连续批处理、分页/前缀缓存和流式服务。当前 **2.1—2.7 均已提供完整正文和 CPU 验证**，2.7 还包含真实本机 HTTP 实验。真实权重的增量 KV、张量批执行与 GPU 服务仍需后续实现和验证。

## 本次可运行内容

1. 阅读 [2.1 正文](01-pytorch-model/README.md)，理解完整前向和生成循环。
2. 按[实验页](01-pytorch-model/lab.md)准备 CPU PyTorch 环境。
3. 在仓库根目录运行 `python -m scripts.experiments.lesson0201`，检查七个案例与本节测试。
4. 完成[六组练习](01-pytorch-model/exercises.md)，再读 [2.2 加载真实模型权重](02-loading-real-model-weights/README.md)。
5. 按 [2.2 实验页](02-loading-real-model-weights/lab.md)下载固定版本的 SmolLM2-135M，在 CPU 上对齐 33 个位置的输出和独立生成序列。
6. 完成 [2.2 八组练习](02-loading-real-model-weights/exercises.md)，解释权重布局、GQA、RoPE 和共享参数，再进入 2.3 的请求生命周期。
7. 按 [2.3 实验页](03-request-and-cache-lifecycles/lab.md)运行状态与缓存回收案例，观察取消和失败后其他请求是否仍可继续。
8. 按 [2.4 实验页](04-from-static-to-continuous-batching/lab.md)运行静态与连续调度，解释预算、队首阻塞和长 prefill 的输出间隔。
9. 按 [2.5 实验页](05-paged-caching/lab.md)追踪物理 slot 与 K/V 张量，核对整块额度、旧数据不可见和终态回收。
10. 按 [2.6 实验页](06-prefix-caching/lab.md)比较冷暖请求，观察分叉请求共享 K/V、末块重算、取消回收与压力逐出。
11. 按 [2.7 实验页](07-http-and-streaming-output/lab.md)启动 HTTP 服务与流式客户端，检查事件顺序、慢读背压、断连取消和错误终态，再完成 14 组练习。

本篇共用模型放在 `zero2vllm/modeling/`，请求管理放在 `zero2vllm/engine/`，张量缓存放在 `zero2vllm/cache/`，HTTP 适配与客户端放在 `zero2vllm/serving/`，数学对照放在 `zero2vllm/reference/`。各节复用已有实现，并明确当前使用的模型和缓存路径。

## 本篇章节

<!-- BEGIN part-index -->

| 节 | 课程 | 验收产出 | 进度 | 正文 |
| --- | --- | --- | --- | --- |
| 2.1 | [构建最小 PyTorch 模型](01-pytorch-model/README.md) | 用同一组权重逐层比较 logits | ✅ | 完整课程 |
| 2.2 | [加载真实模型权重](02-loading-real-model-weights/README.md) | 固定 SmolLM2-135M revision，映射权重并逐层对齐参考输出 | ✅ | 完整课程 |
| 2.3 | [请求与缓存生命周期](03-request-and-cache-lifecycles/README.md) | 回放 11 个生命周期案例，验证取消隔离与资源回收 | ✅ | 完整课程 |
| 2.4 | [从静态批处理到连续批处理](04-from-static-to-continuous-batching/README.md) | 回放六组双策略负载，核对输出、预算与调度边界 | ✅ | 完整课程 |
| 2.5 | [分页缓存](05-paged-caching/README.md) | 连接物理块与 KV 张量，回放 11 个案例并对齐完整重算 logits | ✅ | 完整课程 |
| 2.6 | [前缀缓存](06-prefix-caching/README.md) | 运行 12 个 CPU 案例，核对冷暖 logits、共享块和取消后的引用回收 | ✅ | 完整课程 |
| 2.7 | [HTTP 与流式输出](07-http-and-streaming-output/README.md) | 运行 14 个服务案例及可选前缀后端，核对事件顺序、客户端断连与资源回收 | ✅ | 完整课程 |

<!-- END part-index -->
