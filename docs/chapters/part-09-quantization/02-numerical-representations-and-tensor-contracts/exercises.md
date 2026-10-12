# 9.2 练习与参考答案

[正文](README.md) · [实验手册](lab.md)

先独立手算，再展开答案。凡是题中没有说明的范围、轴或存储精度，都应指出缺项。

## 1. 范围并不只有一张表

INT3 的完整补码范围、无符号范围、本课对称量化范围分别是什么？少用一个编码会不会自动改变存储位宽？

<details><summary>参考答案</summary>

分别是 `[-4,3]`、`[0,7]`、`[-3,3]`。窄范围仍按 3-bit 逻辑编码计量，不会因为未使用 -4 就自动变成更小的定长容器；物理 packing 另行定义。

</details>

## 2. 舍入顺序

固定 `s=0.5,z=3,range=[0,15]`。量化 `x=0.25`。如果把 `round(x/s)+z` 改为 `round(x/s+z)`，结果为什么不同？

<details><summary>参考答案</summary>

本课先 round 得 `round(0.5)+3=3`；后者 `round(3.5)=4`。ties-to-even 的奇偶性落在不同整数上。DQ 分别是 0 与 0.5，都距输入 0.25，但编码规则不相同。

</details>

## 3. 非对称参数

对 `[-1,0,1,2]` 做 unsigned INT4 MinMax，包括实数零，使用 FP64 scale。求 s、z、q、DQ，并说明 q=0 是否意味着实数零。

<details><summary>参考答案</summary>

`s=0.2,z=5,q=[0,5,10,15]`。DQ 为原输入（浮点实现按容差检查）；q=0 对应 -1，q=5 才对应实数零。

</details>

## 4. 常数和全零组

对 `[2,2]`、`[-2,-2]`、`[0,0]` 分别求本课非对称参数。为什么不能一律用原始 `max-min`？

<details><summary>参考答案</summary>

前两组纳入零，得到 `(s,z)=(2/15,0)` 与 `(2/15,15)`；全零采用 `(1,0)`。原始常数组的 max-min 为零，直接当作 scale 会导致除零。不同框架可有其他明示策略。

</details>

## 5. scale 也要量化

对于 amax 为 `1e-8` 的 INT4 对称组，将 scale 存成 FP16 会有什么问题？把 scale 悄悄改成 1 是否合理？

<details><summary>参考答案</summary>

理想 s 约 `1.43e-9`，FP16 下溢为零。本课拒绝。可以明确选择更宽 scale 或 epsilon/额外缩放，并重新评估误差；偷偷改成 1 会把大部分值编码为零，改变原有契约。

</details>

## 6. 非方阵与分组轴

W[5,3]、G=2，沿 K 与沿 N 切组分别需要什么 scale shape？若转成 PyTorch `[N,K]` 存储，还能不变地使用原 axis 吗？

<details><summary>参考答案</summary>

沿 K 是 `[3,3]`，沿 N 是 `[5,2]`。转置后原 K 变成索引 1，分组轴、参数索引和布局都要随之转换。不能用方阵 shape 恰好一致掩盖这个问题。

</details>

## 7. per-channel 与 ONNX axis

W[K,N] 每个输出列一套参数。本课怎样设置？ONNX per-axis 的一维 scale 与 axis 应怎样解释？

<details><summary>参考答案</summary>

本课 `axis=0,group_size=None`，完整 K 覆盖一组，参数 `[1,N]`。ONNX per-axis 可用长度 N 的 scale、axis=1，表示参数随输出列变化。两个 axis 的语义不同，需按广播关系映射。

</details>

## 8. per-token 与 per-head

X[8,4096] 做 per-token，需要多少 scale？KV[B,H,T,D] 的“per-head”为什么仍可能描述不完整？

<details><summary>参考答案</summary>

X 需要 8 个，例如 shape `[8,1]`。KV 若为 `[1,H,1,1]`，跨样本和时间共享；若 `[B,H,T,1]`，每样本/时间步也独立。仅写 per-head 无法确认统计与广播作用域。

</details>

## 9. 非零 z 的 padding

尾组只有 -0.5，非对称 INT4，FP64 scale。把它补到两个元素，求正确的两个 code；若填成 `[0,0]` 会发生什么？

<details><summary>参考答案</summary>

`s=0.5/15,z=15`，正确 code `[0,15]`，DQ `[-0.5,0]`。错误 `[0,0]` 得 `[-0.5,-0.5]`。即使外部激活也可能补零而暂时掩盖输出差异，权重自身仍违反“padding 表示零”的契约。

</details>

## 10. FP8 手算

把 E4M3FN 的 `0x3D` 分成 S/E/F 并求值；为什么 E4M3FN 的全一指数不都表示 Inf？

<details><summary>参考答案</summary>

`0 | 0111 | 101`，S=0、E=7、F=5、bias=7、m=3，值为 1.625。该格式没有 Inf；指数 15 仍有有限值，指数 15 且尾数字段 7 才为 NaN。需按 FN 格式定义处理特殊码。

</details>

## 11. FP4、subnormal 与 FTZ

列出 E2M1 的非负格点；编码 2.5 和 5 得什么？E4M3FN `0x01` 默认与本课 FTZ 模式各是什么？

<details><summary>参考答案</summary>

格点 `[0,.5,1,1.5,2,3,4,6]`。nearest-even 分别选 2（code4）和 4（code6）。`0x01` 默认是 `2^-9`，FTZ 为正零；这说明改变转换/解码策略会损失信息，不能只记录 dtype。

</details>

## 12. 块缩放

MXFP4 一块 32 个元素，scale code 为 128，一个元素 code 为 7；其 DQ 是多少？NVFP4 若同一元素、块 scale 为 `0x38`、全局 DQ scale 为 .5 呢？

<details><summary>参考答案</summary>

E2M1 code7 为 6。MX 的 E8M0 code128 为 2，所以得 12。NV 的 E4M3FN code0x38 为 1，故 `.5×1×6=3`。E8M0 code255 是 NaN，不可按 `2^128` 解读；它没有零编码。

</details>

## 13. 算清 25.6 bpw

复算正文的 15 个权重案例。若把 z 改为每个 INT32 保存，其他声明不变，effective bpw 是多少？

<details><summary>参考答案</summary>

原案例共 48 字节，48×8/15=25.6。改后 q9B、scale18B、z36B、头4B，总67B；整体16B对齐到80B，effective bpw=640/15≈42.6667。不能仍按 z 的逻辑4bit来计算物理32bit开销。

</details>

## 14. 量化后的误差属于哪一层

主例权重与输出 max_abs 都为 .25，能否认定 Linear 输出总与权重误差一样？把 X 整体乘 10 会怎样？

<details><summary>参考答案</summary>

不能。`ΔY=XΔW`。在相同权重量化误差下，将 X 乘10会使输出误差也乘10，max_abs 变为2.5，权重误差仍是.25。模型任务质量还需要完整前向与独立评测。

</details>

## 15. 切分与重量化

K=5,G=2，在全局行3之前切分。右片包含哪些全局组？为什么直接重新按本地行分组不是单纯 repack？

<details><summary>参考答案</summary>

右片全局行3、4属于组1、2。局部 `floor(local_k/2)` 会把两行放同组0。若保留原数值，需要带上正确组映射与参数；如果重新选择参数，需从浮点重新 Q 或从已有表示 RQ，并承担新的误差。repack 本身只调整布局。

</details>

## 16. Passport 审计

有人把本节 CPU 结果写成“Qwen3-8B INT4 已验证、显存降低4倍、A100原生FP4加速”。逐项指出证据缺口，并写出本节可以成立的结论。

<details><summary>参考答案</summary>

本节没有加载 Qwen、没有测显存、没有运行 A100，也没有原生 FP4 kernel。可成立的是：固定二维输入的整数 Q/DQ 和张量契约通过 CPU 参考检查；两种 FP8 的位模式与指定 PyTorch CPU 解码一致；所声明的缓冲区计划能逐项核算。模型质量与 GPU 性能应保持 unknown 或未验证。

</details>
