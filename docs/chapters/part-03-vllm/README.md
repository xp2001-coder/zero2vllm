# Part 3 · 走进真实 vLLM

[课程总目录](../README.md) · [学习路线](../learning-paths.md)

前置知识：Part 2；工程师可从本篇进入。

## 本篇章节

<!-- BEGIN part-index -->

| 节 | 课程 | 验收产出 | 进度 | 正文 |
| --- | --- | --- | --- | --- |
| 3.1 | [vLLM 中一次请求的旅程](01-a-request-s-journey-through-vllm/README.md) | 为固定版本标出请求 ID、进程/线程边界、消息内容及取消后的资源回收 | 🧩 | 教学骨架 |
| 3.2 | [vLLM 调度与 KV 管理](02-scheduling-and-kv-management-in-vllm/README.md) | 把教学状态机映射到正式实现 | 🧩 | 教学骨架 |
| 3.3 | [模型执行与注意力后端](03-model-execution-and-attention-backends/README.md) | 追踪 [K,N] 权重加载及一次调度到张量化、实际 Attention/Linear 内核的完整链路 | 🧩 | 教学骨架 |
| 3.4 | [V1 与 ModelRunner 演进](04-v1-and-modelrunner-evolution/README.md) | 制作 runner×图模式对照；按固定源码版本核对能力，避免把设计目标当已支持 | 🧩 | 教学骨架 |
| 3.5 | [配置与能力的解析流程](05-resolving-configuration-and-capabilities/README.md) | 追踪用户参数如何变成实际运行配置 | 🧩 | 教学骨架 |

<!-- END part-index -->
