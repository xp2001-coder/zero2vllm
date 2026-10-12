# 9.2 中文插图与源稿

[正文](../README.md)

七张图均为本课程原创绘制的中文 SVG，并渲染为 PNG。数值来自本节公式和固定输入；未复制公开资料的图片。白底、蓝橙绿区域和分面标记延续课程既有图解风格。

| 图 | PNG | 可编辑源稿 | 对应内容 |
| --- | --- | --- | --- |
| 1 | [预览](integer-grids.png) | [SVG](integer-grids.svg) | Q/DQ 网格、范围与非零 z |
| 2 | [预览](granularity.png) | [SVG](granularity.svg) | 四种粒度的共享参数与 shape |
| 3 | [预览](groups-and-padding.png) | [SVG](groups-and-padding.svg) | 非方阵分组、尾组和 q_pad=z |
| 4 | [预览](float-bit-fields.png) | [SVG](float-bit-fields.svg) | 位域、bias、同字节的不同解释 |
| 5 | [预览](float-grids.png) | [SVG](float-grids.svg) | FP4 非均匀格点、subnormal 与 FTZ |
| 6 | [预览](scaled-fp4.png) | [SVG](scaled-fp4.svg) | MXFP4 / NVFP4 的缩放结构 |
| 7 | [预览](effective-bpw.png) | [SVG](effective-bpw.svg) | 15 个权重的完整位数账单 |

整数图显式保留 zero-point；浮点图按具体编码规则解读，不添加无关的仿射 z。图 3 的手算采用 FP64 scale，FP16 的实际参数舍入见运行结果。图 6 只画数学分解，不表示真实文件布局或硬件能力。
