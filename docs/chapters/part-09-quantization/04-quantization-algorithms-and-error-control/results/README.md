# 9.4 CPU 算法运行记录

[正文](../README.md) · [复现命令](../lab.md) · [质量记录](../../../../../reports/quality/lesson-9.4-delivery/README.md)

记录时间（UTC）：`2026-09-29T06:48:32.232984+00:00`。本次在 Windows / Python 3.11.9 运行；可选数学核验实际使用 PyTorch 2.7.1+cpu CPU、float64。输入是固定的原创合成矩阵。

| 文件 | 实际结果与用途 |
| --- | --- |
| [summary.json](summary.json) | 23 个固定案例全部通过，保存候选、GPTQ 逐步补偿、缩放与校准/留出误差 |
| [cpu-passport.json](cpu-passport.json) | 主矩阵的 GPTQ 核心参考参数；模型质量和性能为 unknown |
| [commands.json](commands.json) | 10 条命令符合预期，包括两次非法参数的预期退出 2 |

## 受控对照

W[3,2]，校准 X[4,3]，留出 X[3,3]。INT4 [-7,7]、z=0；每个输出通道共享一个全 K 范围的网格，浮点运算为 float64。舍入/GPTQ 保持原始网格，范围搜索与缩放会重算网格。比较各方法时固定数据与位宽，但没有声称优化变量或预算相同。

| 教学实现 | 权重 MSE | 校准输出 MSE | 留出输出 MSE |
| --- | --- | --- | --- |
| RTN | 0.009103061 | 0.053635842 | 0.015052891 |
| clipping search | 0.009103061 | 0.053635842 | 0.015052891 |
| floor/ceil exhaustive | 0.009415986 | 0.007753189 | 0.051117517 |
| GPTQ core reference | 0.009415986 | 0.007753189 | 0.051117517 |
| AWQ-inspired scaling | 0.006474807 | 0.002040827 | 0.030709092 |

范围搜索选 ratio=1；固定网格舍入枚举 64 个组合；GPTQ 使用 relative damping=.01、block_size=2、act_order=False；AWQ 启发式缩放在五个 α 中选择 .75。所有选择仅依据校准目标，留出数据不参与调参。

GPTQ 与穷举在本例结果相同，不代表一般情况下全局最优。缩放的校准误差更小，也没有同时改善相对 RTN 的留出误差。这不是完整 GPTQ/AWQ 算法、真实模型或论文质量排名。

## 可手算的独立案例

- 二维舍入例：W=[.49,.49]ᵀ、步长1。RTN=[0,0]，输出 SSE=1.2005；最优候选[0,1]为 .2405。
- GPTQ 先把第一个权重 .49 舍入到0，再将第二个补偿为 .98，最终舍入到1。增量目标 .2401+.0004=.2405。
- 同一舍入例的留出 MSE：RTN 为 .60025，优化后 .65025；校准改善并不保证泛化。
- 缩放例五个 α 对应 MSE 约为 .010580/.101821/.029647/.006237/.001600，收益不是单调的。
- 二维 Hadamard 旋转保持平方能量64，配套变换权重后输出仍为2；非线性反例说明不能任意跨越 ReLU。

## 独立数学核验

两个 GPTQ 核验分别关闭/启用 act-order：用 PyTorch 直接求剩余 Hessian 子矩阵的逆，逐步求约束补偿，与本课 Cholesky 路径比较，最大绝对差均为0。

四个 STE 核验使用显式 detach 的 autograd 表达式，覆盖 x=−4/−.6/.6/4、s=.5 的内部与饱和区间。它核对本课选定的替代梯度，不是量化台阶的真实导数，也不是完整 LSQ 训练。

## 复现与范围

```bash
python -m scripts.experiments.lesson0904
python -m scripts.experiments.lesson0904 --torch-check
```

第一条只需标准库；第二条显式要求 PyTorch。默认输出进入 `.local/runs/part-09-quantization/04-algorithms/`。数值和检查项应一致，时间及环境摘要可不同。

源码摘要：`18960990a50965d3605514de5e516bfb93d5dc73841c7718f551113e777ae3f4`。覆盖课程 Python、测试、章节示例与输入 JSON；图文经额外检查。git_commit 是未提交工作区的基点，不代替源码摘要。

本次没有执行上游完整 GPTQ/AWQ 包、Qwen 模型质量、量化 checkpoint 生产、GPU kernel 或 A100/C500 服务性能；对应字段为 false，未知指标不填推测值。
