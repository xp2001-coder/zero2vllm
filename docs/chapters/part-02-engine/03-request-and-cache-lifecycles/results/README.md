# 2.3 CPU 实测记录

[正文](../README.md) · [复现命令](../lab.md)

日期：2026-09-17。Windows、Python 3.11.9；真实模型扩展使用上一节的固定快照与 CPU PyTorch 2.7.1+cpu。

[baseline/summary.json](baseline/summary.json)保存完整输入、事件、状态快照、资源断言、源码指纹与真实权重摘要。生成命令：

```bash
python -m scripts.experiments.lesson0203 --real-checkpoint --output-dir docs/chapters/part-02-engine/03-request-and-cache-lifecycles/results/baseline
```

日常复跑请使用默认的 `.local/runs/` 目录，保留本基线。

## 基础 11 案例

固定 TinyAttentionLM：词表 16、宽度 8、最大上下文 32、种子 7。它实际计算并保存 Python K/V 列表，不生成自然语言。

- 正常四步生成 `[14,1,5,12]`，与独立完整重算一致；缓存峰值为 6，终态为 0。
- 生成两项后取消保留 `[14,1]`；计算中取消不提交本次新 token。
- 模型错误在 K/V 已经追加后注入，仍完成回收。
- 容量等待/重试、请求交错隔离、初始化错误、EOS、零输出、等待取消与引擎关闭均通过。
- 每个案例均检查所有请求终态、无活动缓存、预算回到 0，以及每请求只收尾一次。

## 真实权重两案例

模型 revision 为 `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`。CPU float32、两线程、完整前缀重算；没有 Llama 增量 KV。

1. 取消 A 后，A 保留 `[28]`；B 完成 `[260,3575,282,260]`，与 2.2 生成循环相同。
2. 同一模型重新接收故事输入，生成 `[28,665,436,253]`，与 2.2 生成循环相同。

这两项验证生命周期封装与共享模型的继续可用性。参考是 2.2 的生成循环；不声称重新执行了 Transformers 的逐层对照，也不把缓存峰值 0 描述为真实模型 KV 回收。

## 复现范围

源码记录中的 Git commit 为已提交的父版本，工作区仍有未提交内容；`source_sha256` 描述本轮实际代码。2.2 的历史报告保留自己的源码摘要，不因本节新增代码而改写。

A100/C500 和远程 GitHub Actions 本轮未运行；事件序号没有时间单位。本节没有吞吐、延迟、显存或并发安全的实测结论。
