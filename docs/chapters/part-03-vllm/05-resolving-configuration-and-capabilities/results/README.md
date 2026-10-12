# 3.5 实际运行记录

[正文](../README.md) · [复现步骤](../lab.md) · [跨章节质量记录](../../../../../reports/quality/part-03-completion/README.md)

记录时间（UTC）：`2026-09-20T07:17:44.521972+00:00`。Windows / Python 3.11.9，使用 `python -S` 禁用 site 初始化，标准库完成 12 个案例。没有加载 vLLM 或执行 GPU 路径。

| 文件 | 内容 |
| --- | --- |
| [cpu.json](cpu.json) | 12 个案例的输入、检查项与实际输出 |
| [source-inspection.json](source-inspection.json) | 同样的 CPU 案例及固定源码文件、函数和调用检查 |
| [commands.json](commands.json) | 15 条实际命令、预期/实际退出码与记录时间 |
| [cpu.txt](cpu.txt)、[source-inspection.txt](source-inspection.txt) | 两个实验入口的原始终端输出 |

## 短示例与预期错误

- [覆盖记录](config-override.txt)：预算最终 3072，prefix cache 为 false，分块 prefill 被 V1 策略设为 true。
- 默认值：[40/server](defaults-40-server.txt)、[40/offline](defaults-40-offline.txt)、[80/server](defaults-80-server.txt)、[80/offline](defaults-80-offline.txt)。容量是输入情景，不是设备检测结果。
- 引擎意图：[自动且支持](engine-auto-supported.txt)、[自动且实验性](engine-auto-experimental.txt)、[显式 V1 且实验性](engine-1-experimental.txt)。
- [强制 V1 但不支持](engine-1-unsupported.txt)：按预期退出 2。
- [分块预算 512](chunk-budget.txt)：通过；[预算小于序列上限](invalid-budget.txt)：按预期退出 2。
- [eager 意图](eager-intent.txt)：记录 disabled；[直接校验不分块预算](unchunked-rejection.txt)：按预期抛出 ValueError、退出 1。

三个非零退出都来自特意安排的错误输入，已在命令记录中逐项核对；不能把它们当成回归测试意外失败。未完成的后端、缓存、图捕获和数值验证保留在报告中。

## 固定源码与快照身份

- `v0.8.5` / `ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d`：8 个文件、17 个函数通过静态检查。

本轮先准备官方固定提交的文本，再在已有缓存上离线检查。因此检查报告的 `downloaded=[]` 表示这次没有新下载；完整身份由清单提交和文件摘要确定。阅读器不执行上游 Python。

本节记录的教学源码摘要为 `ea9e60f8bd9b2989ed8a63a25b3d52ef19ab601d048b669f3dbfa4c2359c0497`。CPU、源码检查和命令记录使用相同快照；`git_commit` 是未提交工作区的基点，不代替 `source_sha256`。历史课程结果保留各自当时的快照。

这些记录证明 CPU 教学程序与静态定位完成，不能据此声明 A100/C500 运行、真实并发、GPU 内核吞吐或平台兼容性通过。
