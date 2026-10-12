# 2.5 交付时的仓库验证

[质量导航](../README.md) · [课程正文](../../../docs/chapters/part-02-engine/05-paged-caching/README.md)

2026-09-17，Windows / Python 3.11.9。全仓库 **206 项测试全部通过，0 项跳过**，包含本节新增的 15 项分页存储检查和 11 项引擎连接检查。

- [tests.txt](tests.txt)：完整测试输出。
- [regression.json](regression.json)：实际命令、环境、测试数量与源码指纹。
- [本节 CPU 实测](../../../docs/chapters/part-02-engine/05-paged-caching/results/README.md)：11 个案例及其逐项断言，全部通过。

两个教学示例均执行通过。实验页中的 float32、B=3 改参命令另行执行，logits 最大误差约 2.97e-8，输出与完整重算一致。原生命周期与调度 47 项测试另用 Python `-S` 通过；新增 26 项在 `-S` 下全部明确跳过，确认没有把 PyTorch 强加给基础标准库路径。跳过检查不计入张量验证通过的证据。

本回归与本节公开基线的 `source_sha256` 相同。Git commit 指向已有提交，工作区尚未提交；2.2—2.4 的历史记录保留原始源码指纹。

四张白底 PNG 已逐张查看，图内地址和数值与实验核对，SVG 保留可编辑原稿。课程同步、公开引用、本地链接和目录检查另行执行。本节测试、运行器与两个示例已加入 CPU PyTorch 工作流，远程 GitHub Actions 本轮未执行。

验证范围为单线程 CPU 张量与参考 Attention。没有运行 A100/C500，没有测量 GPU 分页算子、真实模型增量缓存、吞吐量或毫秒延迟。
