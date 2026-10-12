# 2.5 原创插图

本节采用白底、细线、少量蓝橙绿配色与子图编号。所有图形按本节代码、公式和固定案例自行绘制；没有复制论文或第三方文章的图片。

PNG 用于正文阅读；SVG 保留可编辑文字、图形和无障碍描述。图内标题、说明和子图标签使用中文，公式、变量与必要的代码字段名保留原写法；SVG 的无障碍标题与描述同步使用中文。

| 图 | SVG 源文件 | 对应内容 |
| --- | --- | --- |
| [地址映射](address-mapping.png) | [SVG](address-mapping.svg) | B=2，A 的块表 [0,2]，有效 slot [0,1,4]；池中间的块归 B |
| [有效长度与额度](validity-budget.png) | [SVG](validity-budget.svg) | L/R/整块容量的区别；两个 R=5 请求在 B=4 时各需两块 |
| [Attention 路径](attention-path.png) | [SVG](attention-path.svg) | mapping 案例的 q/K/V、手算概率与输出、因果读取和 CPU 实现边界 |
| [释放与复用](reuse-boundary.png) | [SVG](reuse-boundary.svg) | reuse 案例中保留的旧尾部与新前缀；错误读取的反例 |

输入见 [cases.json](../data/cases.json)，可复核数值见[实测记录](../results/README.md)。物理块的矩形宽度表示固定位置容量，不表示内存字节比例或实际耗时。正文图号依次为地址、额度、Attention、复用。
