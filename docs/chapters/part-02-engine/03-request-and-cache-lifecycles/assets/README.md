# 2.3 原创图稿

四幅图依据本节实现与实际事件绘制，沿用深色网格和蓝青紫配色。SVG 可编辑；正文使用两倍分辨率 PNG。

| 图 | SVG | PNG | 绘制依据 |
| --- | --- | --- | --- |
| 请求状态机 | [源稿](request-states.svg) | [图片](request-states.png) | WAITING/RUNNING、三种终态、零输出与初始化失败出口 |
| 资源归属 | [源稿](request-ownership.svg) | [图片](request-ownership.png) | 共享模型、独立 RequestContext、实际 K/V 清理 |
| 缓存时序 | [源稿](cache-clock.svg) | [图片](cache-clock.png) | P=3、N=4、预留 6；四次提交时 KV 为 3/4/5/6，收尾后为 0 |
| 取消边界 | [源稿](cancel-boundary.svg) | [图片](cancel-boundary.png) | 同线程回调中收到取消、返回后丢弃 token、统一收尾 |

依据：[实现](../../../../../zero2vllm/engine/lifecycle.py)、[教学示例](../examples/README.md)、[运行记录](../results/README.md)。上游公开源码集中列在[正文](../README.md#参考资料)，图不复制上游素材。

本轮 Figma MCP 仍返回 Starter 套餐调用额度限制，因此图稿在本地制作，未同步至 Figma。仓库中的 SVG/PNG 是当前版本。
