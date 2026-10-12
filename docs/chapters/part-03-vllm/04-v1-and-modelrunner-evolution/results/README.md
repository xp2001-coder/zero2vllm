# 3.4 实际运行记录

[正文](../README.md) · [复现步骤](../lab.md) · [跨章节质量记录](../../../../../reports/quality/part-03-completion/README.md)

记录时间（UTC）：`2026-09-20T07:17:41.115269+00:00`。Windows / Python 3.11.9，使用 `python -S` 禁用 site 初始化，标准库完成 10 个案例。没有加载 vLLM 或执行 GPU 路径。

| 文件 | 内容 |
| --- | --- |
| [cpu.json](cpu.json) | 10 个案例的输入、检查项与实际输出 |
| [source-inspection.json](source-inspection.json) | 同样的 CPU 案例及固定源码文件、函数和调用检查 |
| [commands.json](commands.json) | 10 条实际命令、预期/实际退出码与记录时间 |
| [cpu.txt](cpu.txt)、[source-inspection.txt](source-inspection.txt) | 两个实验入口的原始终端输出 |

## 短示例与结论

- [状态映射](mapping.txt)：A/B/C 长度分别到达 5/4/0；A 恢复后移到行 2。
- [增量写入](staged.txt)：两个不重叠更新，逻辑载荷 36 字节，整表 72 字节。
- [存储所有权](copy.txt)：共享引用读到 `[9,7]`，快照保留 `[3,7]`。
- [采样数学](sampling.txt)：展开索引 `[2,0,0,0]`；固定噪声下 argmax 一致，top-k logprob 使用完整分母。
- [普通 decode](graph-decode.txt) 与 [混合形状](graph-mixed.txt)：分别选择 FULL / PIECEWISE，桶为 4。
- [未捕获整图](graph-uncaptured.txt) 与 [仅 decode 图遇到混合形状](graph-decode-only.txt)：均回到 NONE。

图相关结果只是限定规则的决策，PIECEWISE 包装器准备状态未模拟；没有实际图捕获与重放。generation 校验和重叠写拒绝是教学保护，不能描述成上游实现字段。

## 固定源码与快照身份

- `v0.8.5` / `ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d`：3 个文件、4 个函数通过静态检查。
- `v0.17.1` / `95c0f928cdeeaa21c4906e73cee6a156e1b3b995`：12 个文件、22 个函数通过静态检查。

本轮先准备官方固定提交的文本，再在已有缓存上离线检查。因此检查报告的 `downloaded=[]` 表示这次没有新下载；完整身份由清单提交和文件摘要确定。阅读器不执行上游 Python。

本节记录的教学源码摘要为 `ea9e60f8bd9b2989ed8a63a25b3d52ef19ab601d048b669f3dbfa4c2359c0497`。CPU、源码检查和命令记录使用相同快照；`git_commit` 是未提交工作区的基点，不代替 `source_sha256`。历史课程结果保留各自当时的快照。

这些记录证明 CPU 教学程序与静态定位完成，不能据此声明 A100/C500 运行、真实并发、GPU 内核吞吐或平台兼容性通过。
