# 2.4 交付时的仓库验证

[质量导航](../README.md) · [课程正文](../../../docs/chapters/part-02-engine/04-from-static-to-continuous-batching/README.md)

2026-09-17，Windows / Python 3.11.9。全仓库 **180 项测试全部通过，0 项跳过**，包含本节新增的 23 项调度与回放检查。

- [tests.txt](tests.txt)：完整测试输出。
- [regression.json](regression.json)：命令、环境、实际测试数量与代码指纹。
- [本节 CPU 实测](../../../docs/chapters/part-02-engine/04-from-static-to-continuous-batching/results/README.md)：六组负载、两种策略，共十二次回放均通过。

本节测试、两个教学示例、默认回放与改参命令另使用 Python `-S` 执行，确认不依赖 site-packages。改参核对结果：

| 输入与修改 | 静态 / 连续逻辑轮数 | 结果 |
| --- | --- | --- |
| mixed_lengths，max_running=1 | 12 / 12 | 两者无批内空位差异 |
| mixed_lengths，max_tokens_per_round=3 | 9 / 8 | 预算限制改变选中集合 |
| fifo_capacity，capacity_tokens=16 | 4 / 4 | 原先的队首可在第一轮进入 |
| mixed_lengths，max_tokens_per_round=2 | 不产生成功报告 | 按预期非零退出，拒绝无法容纳的完整 prompt |

本回归和本节公开基线的 `source_sha256` 相同。Git commit 指向父版本，工作区尚未提交。2.2、2.3 的历史记录保持各自的运行结果。

四张 PNG 已逐张查看并核对数据，SVG 为可编辑原稿。课程同步、公开引用、本地链接与目录检查在交付前另行执行。已在工作流中加入本节运行器和示例，但本轮没有运行远程 GitHub Actions。

范围为 CPU 参考执行；没有执行 A100/C500，没有把串行参考计划当作张量 batch，也没有根据逻辑轮数推算吞吐、毫秒延迟或 GPU 利用率。
