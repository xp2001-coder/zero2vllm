# 中文课程数据

[数据导航](../README.md)

[catalog.json](catalog.json) 只保存全局来源、审阅范围和 part_files 索引。每篇独立维护标题、前置、小节和进度；不要把 parts 重新合并进总索引。

| 篇 | 数据文件 | 节数 |
| --- | --- | --- |
| 0 | [开始之前](parts/00-orientation.json) | 3 |
| 1 | [理解大模型推理](parts/01-inference.json) | 6 |
| 2 | [从零构建教学推理引擎](parts/02-engine.json) | 7 |
| 3 | [走进真实 vLLM](parts/03-vllm.json) | 5 |
| 4 | [CUDA 与 MACA 单卡基线](parts/04-backends.json) | 3 |
| 5 | [KV Cache、Prefix Cache 与长上下文](parts/05-kv-cache.json) | 9 |
| 6 | [调度、批处理与重叠执行](parts/06-scheduling.json) | 6 |
| 7 | [Attention 与高性能算子](parts/07-attention.json) | 7 |
| 8 | [编译、CUDA Graph 与执行优化](parts/08-compilation.json) | 7 |
| 9 | [量化算法、算子实现与推理加速](parts/09-quantization.json) | 11 |
| 10 | [Speculative Decoding 推测解码](parts/10-speculation.json) | 9 |
| 11 | [并行切分与分布式执行](parts/11-parallelism.json) | 10 |
| 12 | [MoE 与通信优化](parts/12-moe.json) | 6 |
| 13 | [分离式部署与 KV 传输](parts/13-disaggregation.json) | 6 |
| 14 | [模型类型、多模态与适配器](parts/14-model-types.json) | 7 |
| 15 | [服务接口与生成能力](parts/15-serving.json) | 7 |
| 16 | [权重加载、内存与 RL 推理生命周期](parts/16-weight-loading.json) | 6 |
| 17 | [评测、性能分析与优化方法](parts/17-profiling.json) | 6 |
| 18 | [平台扩展、部署生态与开源贡献](parts/18-deployment.json) | 5 |

directory 是篇目录，reading 是本节 README，guide 是相关导读。content=scaffold/lesson 区分骨架和正文；status=planned/partial/complete 区分完成程度。完成小节须同时有正文、代码、命令和验证证据。
