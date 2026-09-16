# 2.1 教学插图

[返回正文](../README.md)

本目录四张图均为本项目原创绘制，图中的数据与本节代码、手算例子一致。

## 图稿与文件

当前版本采用深蓝背景、矩阵网格和有方向的数据流连线。每张图均保留本地 SVG 源文件，正文使用对应 PNG，便于在 GitHub 上稳定显示。

| 插图 | 课程文件 | 内容 |
| --- | --- | --- |
| 从 token ID 到下一个 token | [PNG](decoder-forward.png)、[SVG](decoder-forward.svg) | 完整前向、Block 内两次残差与最后位置的预测 |
| Attention：从位置打分到信息聚合 | [PNG](attention-matrices.png)、[SVG](attention-matrices.svg) | 三位置、四维单头的打分、缩放、因果遮罩、逐行 softmax 与 Value 加权 |
| 多头拆分：沿特征维分组 | [PNG](multi-head-attention.png)、[SVG](multi-head-attention.svg) | 八维特征拆成两个四维头，保留三个 token 位置 |
| RoPE：在每对特征上旋转 | [PNG](rope-rotation.png)、[SVG](rope-rotation.svg) | 相邻配对、不同频率、单位圆旋转和位置 1 的结果 |

Attention 与多头拆分图的早期白底草稿保存在 [Figma 画布](https://www.figma.com/design/9oP6m3yZweWle1jjyKQymy)。Figma 工具额度用尽后，新版和新增图均在本地绘制，尚未同步至该画布；当前图稿以本目录的 SVG 为准，可导入 Figma 继续编辑。

## 数据约定

- 图 1 对应正文 1、2、6 和 [walk_forward.py](../examples/walk_forward.py)。三 token 前向的各位置 argmax 为 `[[5,9,10]]`，本轮只追加 10；这些 ID 来自 `seed=7` 的未训练模型。第二次残差保留第一次更新后的 U。
- 图 2 对应正文 3.5 和 [attention_math.py](../examples/attention_math.py)。Q、K 直接作为进入点积阶段的输入，不再额外执行 RoPE。红色表示未来位置被填为负无穷，绿色表示注意力权重，紫色表示输出。输出由未舍入的权重计算，图中显示四位小数；因此显示出的权重相加可能与 1 相差最后一位。
- 图 3 对应正文 3.2 的 `reshape(B,T,H,Dh).transpose(1,2)`。数字 `0–23` 是用于跟踪元素的编号，不是模型激活值。青色表示特征 0–3，紫色表示特征 4–7；两个头均保留全部三个位置。
- 图 4 对应正文 3.3 和 [rope_math.py](../examples/rope_math.py)。`Dh=4`、`base=10000`；单头输入 `[1,0,1,0]`，位置 1 的两对旋转角为 1 和 0.01 弧度。单位圆中的角度按真实比例绘制，数值显示六位小数。

修改代码中的教学形状或手算数据时，请同步修改正文、图稿和导出文件。
