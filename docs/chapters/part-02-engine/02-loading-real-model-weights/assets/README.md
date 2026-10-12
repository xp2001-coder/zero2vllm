# 2.2 原创图稿

本节四幅图按模型配置、教学实现与实测结果重新绘制，沿用 2.1 的深色网格、蓝青紫配色与等宽张量标注。SVG 是可编辑源稿；正文使用两倍分辨率 PNG。

| 图 | 源稿 | 图片 | 内容与校核依据 |
| --- | --- | --- | --- |
| 检查点数据流 | [SVG](checkpoint-route.svg) | [PNG](checkpoint-route.png) | 本节下载、配置、参数映射、编码与对照运行器 |
| 权重映射与共享 | [SVG](weight-contract.svg) | [PNG](weight-contract.png) | 模型固定配置、272 个张量的实际形状及共享关系 |
| GQA 与 RoPE 布局 | [SVG](head-layout.svg) | [PNG](head-layout.png) | 实际 9:3 头分组；Dh=4 的缩小 RoPE 示例，不把它误标成真实 head_dim |
| 数值验证路径 | [SVG](validation-ladder.svg) | [PNG](validation-ladder.png) | 每个输入 33 个阶段、全部 logits 与独立贪心生成；容差 1e-4/1e-4 |

公式和形状由本节[布局示例](../examples/layout_traps.py)、[模型实现](../../../../../zero2vllm/modeling/llama.py)和[公开实测记录](../results/README.md)校核。原模型配置与上游源码链接集中在[正文参考资料](../README.md#参考资料)。

本轮 Figma MCP 返回套餐调用额度限制，因此这四幅图在本地完成，尚未同步到 Figma 文件。仓库内 SVG/PNG 是当前正文采用的版本。
