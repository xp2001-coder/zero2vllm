# 3.1 中文原创插图

[课程正文](../README.md)

沿用白底、细线、蓝橙绿强调的图解风格。所有图依据本节固定版本的源码阅读与原创教学数据绘制，保留可编辑 SVG 和正文使用的 PNG。

| 图 | 可编辑源文件 | 说明 |
| --- | --- | --- |
| [进程与任务](process-boundaries.png) | [SVG](process-boundaries.svg) | 前端、核心、线程队列与本地执行，限定单卡 uni 路径 |
| [消息对象](message-objects.png) | [SVG](message-objects.svg) | 请求、调度、模型结果与按请求返回的对象 |
| [两请求迭代](iteration-example.png) | [SVG](iteration-example.svg) | 位置、物理槽位与 ID 结果映射；数字为教学设定 |
| [输出边界](output-boundaries.png) | [SVG](output-boundaries.svg) | 增量合并、消费者读取与 SSE 事件的区别 |
| [取消回收](cancellation-path.png) | [SVG](cancellation-path.svg) | 消息、核心终态、块引用和 runner 清理的交接 |

解释文字使用中文，代码符号保留原名。图中距离和箭头长度不表示实测耗时。没有复制上游或其他课程插图，也没有将示意数据作为设备运行记录。
