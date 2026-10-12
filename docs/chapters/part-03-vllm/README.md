# Part 3 · 走进真实 vLLM

[课程总目录](../README.md) · [学习路线](../learning-paths.md)

前置知识：Part 2；工程师可从本篇进入。

## 从这里开始

先完成 [3.1 vLLM 中一次请求的旅程](01-a-request-s-journey-through-vllm/README.md)：把第二篇的教学引擎对应到固定 v0.8.5 的真实源码。第一遍只追提交、执行、返回与取消，随后进入 [3.2 调度与 KV 管理](02-scheduling-and-kv-management-in-vllm/README.md)，用 token 与块账本解释分配、前缀引用和抢占。接着读 [3.3 模型执行与注意力后端](03-model-execution-and-attention-backends/README.md)，把计划逐项转换成张量并验证输出。

- [正文与 5 张中文图](01-a-request-s-journey-through-vllm/README.md)
- [源码准备、离线核验与双请求示例](01-a-request-s-journey-through-vllm/lab.md)
- [12 组练习与答案](01-a-request-s-journey-through-vllm/exercises.md)
- [源码定位表](01-a-request-s-journey-through-vllm/source-map.md)与[核验记录](01-a-request-s-journey-through-vllm/results/README.md)

```bash
python -m scripts.experiments.lesson0301 --fetch
python -m scripts.experiments.lesson0301 --focus cancel
```

基础部分只需 Python 3.11+，不安装或执行 vLLM。3.1 提供请求路径源码核验；3.2 提供 [10 个计数实验](02-scheduling-and-kv-management-in-vllm/lab.md)与 [14 组练习](02-scheduling-and-kv-management-in-vllm/exercises.md)。3.3 提供 [12 个 CPU 案例](03-model-execution-and-attention-backends/lab.md)与 [16 组练习](03-model-execution-and-attention-backends/exercises.md)，数值部分需要 CPU PyTorch。[3.4 Runner 演进](04-v1-and-modelrunner-evolution/README.md)提供 10 个状态与数学案例，[3.5 配置解析](05-resolving-configuration-and-capabilities/README.md)提供 12 个字段追踪案例；两节各有 6 张中文图与 14 组练习，均只需标准库。A100/C500 的执行与性能另行验证。

```bash
python -m scripts.experiments.lesson0302
python -m scripts.experiments.lesson0303 --mode metadata
python -m scripts.experiments.lesson0304
python -m scripts.experiments.lesson0305
# 安装实验页指定的 CPU PyTorch 后
python -m scripts.experiments.lesson0303
```

## 本篇章节

<!-- BEGIN part-index -->

| 节 | 课程 | 验收产出 | 进度 | 正文 |
| --- | --- | --- | --- | --- |
| 3.1 | [vLLM 中一次请求的旅程](01-a-request-s-journey-through-vllm/README.md) | 核对固定提交的 25 个源文件、38 个函数与 4 类消息，绘制请求路径并解释取消回收 | ✅ | 完整课程 |
| 3.2 | [vLLM 调度与 KV 管理](02-scheduling-and-kv-management-in-vllm/README.md) | 手算逐轮 token/块账本；运行 10 个计数案例并核验固定源码分支 | ✅ | 完整课程 |
| 3.3 | [模型执行与注意力后端](03-model-execution-and-attention-backends/README.md) | 手算双请求张量与缓存地址；运行 12 个 CPU 案例并核验 27 个固定源码函数 | ✅ | 完整课程 |
| 3.4 | [V1 与 ModelRunner 演进](04-v1-and-modelrunner-evolution/README.md) | 手算状态映射与增量载荷；运行 10 个 CPU 案例，核验两版本 26 个函数并设计受控对照 | ✅ | 完整课程 |
| 3.5 | [配置与能力的解析流程](05-resolving-configuration-and-capabilities/README.md) | 运行 12 个配置案例，追踪字段覆盖与改写；核验 17 个固定源码函数并列出待验证能力 | ✅ | 完整课程 |

<!-- END part-index -->
