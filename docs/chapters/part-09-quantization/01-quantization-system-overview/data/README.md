# 9.1 固定输入

[正文](../README.md) · [实验](../lab.md)

| 文件 | 用途 |
| --- | --- |
| [linear.json](linear.json) | 非方阵六权重 Linear、手算 q/DQ 与预期输出 |
| [affine.json](affine.json) | 非零 zero-point、两套参数间的重量化、舍入反例 |
| [terms.json](terms.json) | 20 个带明确上下文的术语和分类答案 |
| [source-manifest.json](source-manifest.json) | 固定 vLLM 提交的文件摘要、九个符号与选定调用 |
| [autoround-source.json](autoround-source.json) | 固定 AutoRound README 的来源提交与摘要；仅文档阅读 |

数值输入为课程自行设计，使用能精确表示的二进制分数，便于逐项手算。没有模型权重和校准语料；terms 中的指标值只是分类示例。运行结果保存在 results，不覆盖这些输入。
