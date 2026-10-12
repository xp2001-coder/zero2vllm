# 3.3 图稿

[返回正文](../README.md)

本节 7 张图均由课程自行绘制，使用白底、中文标注和蓝/橙/绿分区。图中数字来自本节输入契约及 CPU 实验；不使用外部文章插图。SVG 保留文字与图形供编辑，PNG 用于 GitHub 正文。

| 图稿 | 可编辑版本 | 阅读版本 |
| --- | --- | --- |
| 模型解析、参数写入与后处理 | [SVG](model-loading.svg) | [PNG](model-loading.png) |
| Linear 存储方向及融合 QKV | [SVG](weight-layout.svg) | [PNG](weight-layout.png) |
| 双请求的三套索引 | [SVG](packed-inputs.svg) | [PNG](packed-inputs.png) |
| 带历史偏移的因果可见范围 | [SVG](causal-prefix.svg) | [PNG](causal-prefix.png) |
| 后端选择、缓存写入与注意力计算 | [SVG](backend-path.svg) | [PNG](backend-path.png) |
| 真实工作与填充缓冲区 | [SVG](padding-boundary.svg) | [PNG](padding-boundary.png) |
| 末行选择、候选过滤与请求归属 | [SVG](output-routing.svg) | [PNG](output-routing.png) |

图 7 的候选 24、16 对应默认种子 303、float64 的实际运行；不是模型能力示例。图 5 的算子名称对应固定 v0.8.5 源码入口，不表示已做 GPU 性能验证。
