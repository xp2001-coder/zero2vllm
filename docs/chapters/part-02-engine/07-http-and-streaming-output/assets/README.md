# 2.7 中文原创插图

[课程正文](../README.md)

沿用白底、细线、蓝橙绿强调与子图说明。全部根据本节实现重新绘制，保留可编辑 SVG 与正文使用的 PNG；图形长度不表示耗时。

| 图 | SVG | 内容 |
| --- | --- | --- |
| [请求路径](request-journey.png) | [源文件](request-journey.svg) | HTTP 到引擎，以及模型提交、网络交接、客户端读取的区别 |
| [事件边界](sse-framing.png) | [源文件](sse-framing.svg) | SSE 行与空行，任意字节分块 |
| [异步边界](async-boundaries.png) | [源文件](async-boundaries.svg) | 事件循环、任务等待与同步前向 |
| [有界背压](bounded-backpressure.png) | [源文件](bounded-backpressure.svg) | 两项排队、一项在途、慢请求暂缓、同伴继续 |
| [断连收尾](disconnect-cleanup.png) | [源文件](disconnect-cleanup.svg) | 取消传回引擎，归还 KV 与请求所有权 |
| [错误边界](error-boundary.png) | [源文件](error-boundary.svg) | HTTP 错误、流内终态与截断 |

背压数量与[运行记录](../results/README.md)对应。代码名和事件字段保留原文，解释文字使用中文。没有复制第三方论文或文章的图片。
