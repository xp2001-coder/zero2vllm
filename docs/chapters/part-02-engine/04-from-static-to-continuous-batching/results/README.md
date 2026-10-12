# 2.4 CPU 实测记录

[正文](../README.md) · [复现命令](../lab.md)

2026-09-17，Windows / Python 3.11.9，标准库 TinyAttentionLM，seed=7、词表 16、宽度 8、最大上下文 32。实际独立 KV 列表，逐请求串行执行。

[baseline/summary.json](baseline/summary.json)保留六组输入、每组两种策略的逐轮计划、实际执行成员、逐请求输出、边界计时、资源断言、环境和源码摘要。生成命令：

```bash
python -S -m scripts.experiments.lesson0204 --output-dir docs/chapters/part-02-engine/04-from-static-to-continuous-batching/results/baseline
```

平时复跑请使用默认 `.local/runs/`，保留这份已核验基线。

| 负载 | 静态逻辑轮数 | 连续逻辑轮数 | 检查结果 |
| --- | --- | --- | --- |
| mixed_lengths | 8 | 6 | 输出一致，均为 12 次逐请求前向、18 个输入 token |
| token_pressure | 6 | 6 | 两个已活跃请求首轮只选一个 |
| fifo_capacity | 7 | 7 | 等待队首挡住本可放下的小请求 |
| staggered_arrivals | 6 | 6 | 包含 3 个空轮；晚到请求没有被漏掉 |
| stop_and_cancel | 2 | 2 | 运行/等待取消、EOS、零输出与资源回收通过 |
| long_prefill_turn | 8 | 7 | long 更早得到首项，A/B 最大输出间隔从 1 轮变 2 轮 |

十二次回放均与独立完整重算路径对照，终态预留为 0，上下文关闭，收尾次数正确。取消请求比较已提交输出的参考前缀。异常注入、初始化失败和冻结计划边界另由本节单元测试检查。

报告中的 `source_sha256` 标识本轮代码；Git commit 仍指向已提交的父版本，工作区含未提交修改。2.2、2.3 的历史报告保留原始源码指纹，不因新增本节而改写。

本记录没有测量 GPU batch、tokens/s、毫秒延迟或显存。A100、C500 和远程 GitHub Actions 本轮未执行。参考模型的调度计划由 Python 串行执行，轮次和选中位置比例不能用来宣称设备加速或 GPU 利用率。
