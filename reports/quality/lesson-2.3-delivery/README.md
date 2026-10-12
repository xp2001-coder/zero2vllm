# 2.3 交付时的仓库验证

[质量导航](../README.md) · [课程正文](../../../docs/chapters/part-02-engine/03-request-and-cache-lifecycles/README.md)

2026-09-17，Windows / Python 3.11.9。全仓库 **157 项测试全部通过，0 项跳过**，包括本节新增的 24 项生命周期检查。

- [tests.txt](tests.txt)：完整测试输出。
- [regression.json](regression.json)：执行命令、实际测试数量、环境、源码摘要和实验记录位置。
- [本节实测](../../../docs/chapters/part-02-engine/03-request-and-cache-lifecycles/results/README.md)：11 个基础案例与两个真实模型生命周期案例均通过。

基础运行器、两个教学示例和 24 项测试另在 Python `-S` 模式下运行，确认它们不依赖 site-packages。真实模型扩展仍使用 2.2 的依赖与固定快照。

本回归与本节公开实验报告的源码摘要相同。工作区尚未提交，Git commit 指向父版本；2.2 的历史验证文件没有被改写。课程/链接检查与目录检查在交付前另外执行。

范围为本地 CPU。没有执行远程 CI 或 A100/C500，也没有将同步回调的取消实验描述成线程或设备并发验证。
