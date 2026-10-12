# 9.1 原创插图

[正文](../README.md)

采用白底中文图、蓝色流程/存储、橙色数值操作和绿色还原/验证分区。SVG 为可编辑源稿，PNG 供课程直接显示。图中含量化参数与明确的 Q、DQ、RQ；对称示例显式标记 z=0。

| 图 | 可编辑源稿 | 显示版本 |
| --- | --- | --- |
| 量化生产与消费 | [SVG](production-consumption.svg) | [PNG](production-consumption.png) |
| 术语分层 | [SVG](terminology-layers.svg) | [PNG](terminology-layers.png) |
| 非零 zero-point 与 Q/DQ | [SVG](affine-qdq.svg) | [PNG](affine-qdq.png) |
| 重量化与 repack | [SVG](requantization-vs-repack.svg) | [PNG](requantization-vs-repack.png) |
| 对称 Linear 手算 | [SVG](qdq-hand-calculation.svg) | [PNG](qdq-hand-calculation.png) |
| Linear 生命周期 | [SVG](linear-lifecycle.svg) | [PNG](linear-lifecycle.png) |

六张图均由课程自行绘制，没有复制第三方论文或产品图片。非零 zero-point 和重量化数字对应 [affine.json](../data/affine.json)，Linear 数字对应 [linear.json](../data/linear.json)，均由 CPU 案例复核。源码调用关系见 [公开定位](../source-map.md)。

图中的 DQ→Q 是 RQ 的数学参考路径，实际 kernel 不一定物化浮点中间张量。纯 repack 保持逻辑数值，不能与改变网格的 RQ 混用。图不表示 A100/C500 已运行，也不把整数仿射表示推广为所有 FP8/FP4 的解码规则。
