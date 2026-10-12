# 2.5 CPU 实测记录

[正文](../README.md) · [复现命令](../lab.md)

2026-09-17，Windows / Python 3.11.9 / PyTorch 2.7.1+cpu，单线程 CPU。TinyAttentionLM 固定词表 16、宽度 8、最大上下文 32、seed=7；投影使用原 Python 参考路径，实际 K/V 与 Attention 使用 PyTorch 张量。

[baseline/summary.json](baseline/summary.json)保留 11 个案例、实际输出、全部 logits 对照、块表与逐轮资源快照、断言、环境和源码指纹。生成命令：

```bash
python -m scripts.experiments.lesson0205 --output-dir docs/chapters/part-02-engine/05-paged-caching/results/baseline
```

日常复跑使用默认 `.local/runs/`，不要覆盖公开基线。

| 案例 | 实际观察 |
| --- | --- |
| mapping | A 块表 [0,2]，slot [0,1,4]；Attention=[20,10]，手算对照最大误差 0 |
| reuse | 旧尾部仍为 [99,99]；新请求只读 [3,7]，旧句柄被拒绝 |
| oom | 增长失败后元数据与空闲块不变 |
| logits-float64，B=2/4 | 两例最大误差均约 1.11e-16，小于 1e-12 |
| logits-float32，B=2/4 | 两例最大误差均约 2.97e-8，小于 1e-6 |
| admission | 6 个逻辑轮；第二请求等待两个完整物理块 |
| fragmented | 3 个逻辑轮；D 实际获得非相邻块 [0,2] |
| cleanup | 4 个逻辑轮；取消请求保留两个已提交输出，失败请求不影响同伴 |
| block-sizes | B=1/2/4/8 时，分配 slot 为 17/20/24/32，尾部空位为 0/3/7/15 |

四个生成案例的输出均为 `[14,12,3,8,13]`，最后有效长度为 9。各案例全部断言通过。引擎案例的物理块账目逐轮匹配额度，最终所有块归还、上下文关闭、请求仅收尾一次。

`source_sha256` 对应本轮代码与输入；Git commit 指向已有提交，`working_tree_dirty=true` 表示实验包含工作区尚未提交的改动。更早课程的历史报告保持原始指纹。

本记录没有测量 GPU 算子、tokens/s、毫秒延迟或显存；A100/C500 未运行。这里的 gather 包含历史 K/V 复制，实验用于检验地址与结果，不是分页性能基准。
