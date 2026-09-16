# Part 16 · 权重加载、内存与 RL 推理生命周期

[课程总目录](../README.md) · [学习路线](../learning-paths.md)

前置知识：Part 3—5、14。

## 本篇章节

<!-- BEGIN part-index -->

| 节 | 课程 | 验收产出 | 进度 | 正文 |
| --- | --- | --- | --- | --- |
| 16.1 | [模型解析与权重加载](01-model-resolution-and-weight-loading/README.md) | 分解读取、解码、拷贝和首次生成的时间 | 🧩 | 教学骨架 |
| 16.2 | [高效加载与启动](02-efficient-loading-and-startup/README.md) | 比较冷/热文件缓存下启动时间与主机峰值内存 | 📋 | 教学骨架 |
| 16.3 | [权重 Offload 与内存预算](03-weight-offloading-and-memory-budgets/README.md) | 建立权重/激活/KV/workspace/graph 的显存账本 | 📋 | 教学骨架 |
| 16.4 | [Sleep、Wake 与 Pause/Resume](04-sleep-wake-and-pause-resume/README.md) | 检查释放后的占用与恢复后输出一致性 | 📋 | 教学骨架 |
| 16.5 | [权重更新与传输](05-weight-updates-and-transfer/README.md) | 画训练器到推理器的更新流程和同步点 | 📋 | 教学骨架 |
| 16.6 | [强化学习推理集成](06-reinforcement-learning-inference-integration/README.md) | 分析权重滞后、采样分布和吞吐；明确 vLLM 的推理角色 | 📋 | 教学骨架 |

<!-- END part-index -->
