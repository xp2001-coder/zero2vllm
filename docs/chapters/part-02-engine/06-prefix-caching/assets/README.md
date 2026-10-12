# 2.6 中文原创插图

沿用白底、细线、少量蓝橙绿强调与子图编号。标题和说明使用中文，变量、代码字段与必要技术名称保留原写法；SVG 包含中文无障碍描述。图中内容根据本节代码和固定案例重新绘制。

| 图 | 可编辑源文件 | 对应内容 |
| --- | --- | --- |
| [共享前缀](shared-prefix.png) | [SVG](shared-prefix.svg) | branching-live 的 A/B 输入、实际物理块表与共享引用 |
| [末块边界](hit-boundary.png) | [SVG](hit-boundary.svg) | P=6、B=2 的命中上界，及长度 5—8 的边界对照 |
| [哈希链](hash-chain.png) | [SVG](hash-chain.svg) | 不同父前缀与相同当前块、计算身份、强制碰撞校验 |
| [共享块生命周期](block-lifecycle.png) | [SVG](block-lifecycle.svg) | 引用 1→2→1→0、留存、再次命中和逐出 |
| [计算量节省](work-savings.png) | [SVG](work-savings.svg) | 冷暖运行的 8→4 个位置、36→26 个 Attention 分数项 |

输入见 [cases.json](../data/cases.json)，数值见[公开运行记录](../results/README.md)。箭头表示逻辑关系或操作顺序，图形宽度不表示耗时。展示在块中的 token ID 用于说明内容身份，实际张量保存 K/V 向量。

全部图为课程原创，不复用第三方论文或文章图片。修改时同步更新 SVG 与 PNG，并核对图中数值与固定案例。
