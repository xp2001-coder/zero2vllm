# Part 2 · 从零构建教学推理引擎

[课程总目录](../README.md) · [学习路线](../learning-paths.md)

前置知识：Part 1。

本篇从可检查的 PyTorch 模型出发，逐步加入真实权重、请求管理、连续批处理、分页/前缀缓存和流式服务。当前仅 **2.1 已完成 CPU 验证**，2.2—2.7 按下表逐节推进。

## 本次可运行内容

1. 阅读 [2.1 正文](01-pytorch-model/README.md)，理解完整前向和生成循环。
2. 按[实验页](01-pytorch-model/lab.md)准备 CPU PyTorch 环境。
3. 在仓库根目录运行 `python -m scripts.experiments.lesson0201`，检查七个案例与本节测试。
4. 完成[六组练习](01-pytorch-model/exercises.md)，再进入 2.2。

本篇共用实现放在 `zero2vllm/modeling/`，数学对照放在 `zero2vllm/reference/`。后续小节复用同一套模型代码。

## 本篇章节

<!-- BEGIN part-index -->

| 节 | 课程 | 验收产出 | 进度 | 正文 |
| --- | --- | --- | --- | --- |
| 2.1 | [构建最小 PyTorch 模型](01-pytorch-model/README.md) | 用同一组权重逐层比较 logits | ✅ | 完整课程 |
| 2.2 | [加载真实模型权重](02-loading-real-model-weights/README.md) | 加载候选小模型并对齐参考输出 | 📋 | 教学骨架 |
| 2.3 | [请求与缓存生命周期](03-request-and-cache-lifecycles/README.md) | 画出状态机并验证取消后资源回收 | 📋 | 教学骨架 |
| 2.4 | [从静态批处理到连续批处理](04-from-static-to-continuous-batching/README.md) | 回放长短请求混合的调度轨迹 | 📋 | 教学骨架 |
| 2.5 | [分页缓存](05-paged-caching/README.md) | 把已有逻辑块模拟与 Attention 张量存储连接 | 🧩 | 教学骨架 |
| 2.6 | [前缀缓存](06-prefix-caching/README.md) | 构建共享前缀树形请求并检查复用与回收 | 📋 | 教学骨架 |
| 2.7 | [HTTP 与流式输出](07-http-and-streaming-output/README.md) | 让断连请求释放资源并保存流式事件记录 | 📋 | 教学骨架 |

<!-- END part-index -->
