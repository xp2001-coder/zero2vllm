# 第三篇收尾：3.4—3.5 质量记录

[返回质量导航](../README.md)

记录时间（UTC）：`2026-09-20T07:25:24.718119+00:00`。Windows / Python 3.11.9。本轮完成 3.4 Runner 演进与 3.5 配置解析，补齐第三篇最后两节。

## 实际验证

| 检查 | 结果 | 记录 |
| --- | --- | --- |
| 全仓库回归 | 380 项通过，0 跳过，退出 0 | [regression.json](regression.json)、[完整日志](tests.txt) |
| 3.4 标准库案例 | 10 项通过，含状态映射、增量写、所有权、采样数学与图决策 | [本节记录](../../../docs/chapters/part-03-vllm/04-v1-and-modelrunner-evolution/results/README.md) |
| 3.5 标准库案例 | 12 项通过，含覆盖、默认值、强制规则和预期拒绝 | [本节记录](../../../docs/chapters/part-03-vllm/05-resolving-configuration-and-capabilities/results/README.md) |
| 文档中的短示例与入口 | 合计 25 条命令均符合预期退出状态；包含 3 条有意失败输入 | 两节各自 commands.json |
| 固定源码 | 3.4：v0.8.5 的 3 文件/4 函数 + v0.17.1 的 12 文件/22 函数；3.5：v0.8.5 的 8 文件/17 函数 | 两节各自 source-inspection.json |
| 课程与目录 | 中英文状态、公开引用、链接与分级目录检查通过 | [course.json](course.json)、[layout.json](layout.json) |

新增 45 项测试：22 项 runner/数学边界、18 项配置规则、5 项共享源码阅读与默认离线检查。旧版源码阅读器保持默认严格模式；新增按所需符号筛选的模式，用于处理无关 property/overload 重名，目标符号有歧义仍然报错。

## 教学交付与视觉检查

- [3.4 正文](../../../docs/chapters/part-03-vllm/04-v1-and-modelrunner-evolution/README.md)、实验手册、14 组练习答案、固定源码定位表与 6 张中文原创图。
- [3.5 正文](../../../docs/chapters/part-03-vllm/05-resolving-configuration-and-capabilities/README.md)、实验手册、14 组练习答案、固定源码定位表与 6 张中文原创图。
- 12 张图保留 SVG 和 PNG，已逐张检查标签、数据、箭头与排版。
- 中英文课程目录、章节导航、学习路线和 CI 中的离线入口同步更新。

## 快照与验证范围

源码摘要：`ea9e60f8bd9b2989ed8a63a25b3d52ef19ab601d048b669f3dbfa4c2359c0497`。两节的 CPU 记录、源码核验和全仓库测试使用同一教学代码/数据快照。测试期间摘要未改变；历史课程记录保留原始快照。

`git_commit` 是尚未提交工作区的基点，不能用它单独识别本轮新增代码。文档与插图不在这个源码摘要的覆盖范围内，另经课程链接/结构检查和人工视觉核验。

真实运行的是课程 CPU 程序；上游 vLLM 只做静态文本核验。没有运行 A100/C500、DMA、Triton 或 CUDA Graph，也没有执行 GitHub 远程 CI。CI 配置新增了标准库实验入口，远程结果需后续提交后查看。
