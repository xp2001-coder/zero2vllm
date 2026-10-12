# 第一篇教学修缮记录

[第一篇学习入口](../../../docs/chapters/part-01-inference/README.md) · [质量导航](../README.md)

本轮修缮已完成的第一篇，重点补齐 1.4—1.6 的推导、阅读步骤与可执行手算；1.1—1.3 补阅读路线、时序/GQA 图解及第二篇衔接。共享数学实现与历史实验基线保留原样。

## 实际验证

2026-09-17，Windows / Python 3.11.9：

- 6 个新增短示例全部通过，使用 Python `-S` 禁用 site packages，确认只需要标准库。完整输出见 [examples.txt](examples.txt)。
- 27 项第一篇数学参考与命令行测试全部通过，见 [tests.txt](tests.txt)。
- 原有 28 个 CPU 命令案例全部重跑通过。命令、退出码和各自验证结果见 [cases.json](cases.json)。本次运行器跳过了全仓库测试，第一篇专项测试单独执行并记录。
- 课程、双语目录、公开引用与链接检查见 [course.json](course.json)；目录及已有结果链接检查见 [layout.json](layout.json)。
- 环境、源码指纹和具体检查范围见 [revision.json](revision.json)。10 张新图均已查看，SVG 保留可编辑源文件。

本次修订没有改变共享数学算法，使用第一篇专项回归和命令案例；2.7 的 264 项全仓库测试保留在其原始交付记录。历史基线仍对应各自原始源码与时间。

验证范围为标准库 CPU 数学、统计与文档，没有设备性能测量。CI 已加入六个示例，远程 GitHub Actions 本轮未运行。工作区保留供 review，尚未提交。
