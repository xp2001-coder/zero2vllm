# 2.6 CPU 实测记录

[正文](../README.md) · [复现实验](../lab.md)

2026-09-17，Windows / Python 3.11.9 / PyTorch 2.7.1+cpu，单线程 CPU。模型是词表 16、宽度 8、最大上下文 32、seed=7 的单层 TinyAttentionLM。K/V 与 Attention 使用实际张量；投影沿用 Python 参考计算。

[baseline/summary.json](baseline/summary.json)包含 12 个案例、全部断言、冷暖逐步 logits、实际共享块表、引用计数、逐出与取消记录、环境和源码摘要。

```bash
python -m scripts.experiments.lesson0206 --output-dir docs/chapters/part-02-engine/06-prefix-caching/results/baseline
```

日常复跑请使用默认 `.local/runs/`，保留这份公开基线。

| 观察项 | 本次结果 |
| --- | --- |
| 默认冷暖输出 | 均为 [12,3,8]，与独立完整重算一致 |
| 热轮命中 | P=6、B=2，命中 4 个位置 |
| 实际计算位置 | 冷 8，热 4 |
| 单头 Attention 分数项 | 冷 36，热 26 |
| float64 最大 logits 误差 | 冷暖均约 1.67e-16，阈值 1e-12 |
| float32 最大 logits 误差 | 冷暖均约 1.97e-8，阈值 1e-6 |
| 活跃共享 | A 表 [0,1,2,3]，B 表 [0,1,4,5]，前两块 refs=2 |
| A 释放后 | B 的共享前缀 refs 降为 1，继续输出 [14,13,1] |
| 压力逐出 | 先逐出块 1，再逐出块 2；活跃块内容不变 |
| 碰撞、OOM、reset、重复冷启动 | 全部断言通过 |
| 调度案例 | 取消、完成、失败终态正确，保守额度约束成立，活跃引用最终归零 |

`source_sha256` 标识本轮代码、测试与输入；Git commit 仍指向已有提交，工作区改动尚未提交。2.2—2.5 的历史记录保留原始摘要，不改写成当前状态。

这些结果证明了当前参考路径的命中、计算减少、数值对齐和所有权回收。没有测量 GPU 性能、服务延迟或吞吐量，也没有运行 A100/C500。CPU 位置计数不能转换成硬件加速倍数。
