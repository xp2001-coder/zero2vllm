# 仓库目录重构验证

[质量检查索引](../README.md) · [目录规范](../../../docs/maintenance/repository-layout.md)

本轮在 Windows / Python 3.11.9 下实际执行。全套测试使用 PyTorch 2.7.1+cpu、NumPy 2.2.6；第一篇案例使用不含 PyTorch 的标准库环境。

| 检查 | 结果 | 证据 |
| --- | --- | --- |
| 全套单元测试 | 108 项通过，0 跳过 | [原始输出](tests.txt) |
| 第一篇实验 | 28 个案例通过；运行器跳过重复单元测试，测试另行完整执行 | [回归摘要](regression.json) |
| 2.1 实验 | 7 个案例与本节 13 项测试通过 | [回归摘要](regression.json) |
| 迁移后的其他入口 | 本节示例、INT4、环境采集、PyTorch CLI 通过 | [命令记录](regression.json) |
| 课程同步与文档链接 | 19 篇、126 节；检查结果以 JSON 为准 | [课程检查](course.json) |
| 目录与公开结果引用 | 无旧入口残留、无空目录；检查结果以 JSON 为准 | [目录检查](layout.json) |

复现顺序（已安装 CPU PyTorch 的虚拟环境）：

```bash
python -m unittest discover -s tests -v
python -m scripts.experiments.part01 --skip-tests
python -m scripts.experiments.lesson0201
python -m scripts.docs.course --check
python -m scripts.docs.layout
```

第一篇案例也可用未安装第三方依赖的 Python 独立运行。测试摘要保存本轮源码指纹；验证时 Git 工作区存在未提交改动，commit 单独不能定位当时的实现。

本轮原始运行产物在维护者本地 .local/，公开摘要列出执行项和状态。章节 results/ 的历史 baseline 保留原始测量、时间与源码摘要；不将它们标为本轮新测量。A100/C500 与远程 CI 均未在本轮验证。
