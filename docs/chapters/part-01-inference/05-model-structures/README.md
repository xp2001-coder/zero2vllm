# 1.5 模型结构与长上下文基础

[本篇目录](../README.md) · [全部课程](../../README.md) · [上一节 1.4](../04-performance-metrics/README.md) · [下一节 1.6](../06-sampling-correctness/README.md)

> 基础必读，含工程进阶。提供 RoPE、ALiBi、滑窗、线性 Attention、latent 内容投影和标量 SSM 的独立 CPU 数学例子。它们不是完整的生产模型实现。建议 120—150 分钟，可分两次学习。

[Part 1 总表](../README.md) · [上一课：性能指标](../04-performance-metrics/README.md) · [下一课：采样与正确性](../06-sampling-correctness/README.md)

## 学习目标与准备

这一课主要回答：**模型结构改变后，哪些表示需要保留，缓存多大，每个新 token 又需要读取什么？**

完成后应能推导二维旋转、手算带偏置/滑窗的 Attention、比较线性 Attention 的显式与递推形式，并区分 KV 序列、latent 序列和固定维度状态。

前置：1.1 的点积/softmax、1.3 的缓存形状。Python 3.11+；不需要 PyTorch、真实权重或 GPU。适用含 `structure-demo` 的 `0.1.0.dev0` checkout。

```bash
python -m zero2vllm structure-demo --output .local/runs/part-01-inference/05-model-structures/structure-default.json
python -m zero2vllm structure-demo --help
```

实现位于 [positions.py](../../../../zero2vllm/reference/positions.py)。默认序列长度 8、滑窗宽度 3；所有输入矩阵都保存在报告中。几个实验独立运行，不是把所有结构开关叠加到同一个语言模型里。

可对照[本地原始结果](results/baseline/05-structures.json)查看矩阵、位置对数量及各数学恒等式误差。

## 1. 先分清三类变化

| 变化 | 例子 | 主要改变 |
| --- | --- | --- |
| 表达位置或修改打分 | RoPE、ALiBi | query/key 或 attention score 如何体现位置 |
| 限制可见范围、改变缓存表示 | Sliding Window、稀疏 Attention、MLA | 访问哪些位置、保存哪些表示 |
| 改变聚合/递推结构 | 线性 Attention、SSM | 是否必须保留逐 token K/V，如何更新状态 |

这些是结构层面的选择。FlashAttention 主要优化某种 Attention 的计算与访存；PagedAttention 处理块式 KV 的存储/寻址；APC 复用匹配前缀。读源码时先确定模型结构，再检查可用算子后端。

## 2. RoPE：在每对坐标上旋转

把向量的一对坐标 `[a,b]` 旋转角度 `θ`：

```text
a' = a*cos(θ) - b*sin(θ)
b' = a*sin(θ) + b*cos(θ)
```

本例向量维度 `d` 必须是正偶数，使用**相邻配对**：`(0,1)、(2,3)…`。第 `j` 对的频率与位置角度为：

```text
frequency_j = base^(-2*j/d)
angle_j = position * frequency_j
```

默认 `base=10000`。对 `d=2`、向量 `[1,0]`、位置 1，输出是 `[cos(1),sin(1)]`，约 `[0.540302,0.841471]`。

### 2.1 两个可检查的性质

**旋转保持长度。** 同一对坐标的平方和不变，因此整个向量的平方范数也保持。

**旋转后的点积携带相对位置。** 使用同一组频率时：

```text
dot(R(m)q, R(n)k) = dot(q, R(n-m)k)
```

所以把两边位置同时加上同一个偏移，点积保持不变。默认实验比较 `(m,n)=(7,10)` 和 `(0,3)`，误差约 `1.11e-16`；这是 CPU float 的数值误差，不要求符号级零误差。[RoFormer 论文](https://arxiv.org/abs/2104.09864)

### 2.2 位置与缓存的连接

若缓存保存的是已经旋转后的 K，新 token 的 Q 和 K 必须使用正确的绝对位置；不能把 Decode 的位置每轮重置为 0。滑窗丢掉历史 token 后，缓存长度也不再等于总位置编号，需另外保存累计位置。

实际模型可能采用“前后半段配对”、部分维度旋转、不同 scaling、不同 cos/sin buffer 布局。看见公式相似还不足以直接共用 kernel；先对齐权重、配对方式和位置。

### 2.3 位置编码不等于长上下文质量保证

能为位置 100000 算出 sin/cos，不意味着模型在那么长的输入上表现可靠。训练长度、频率扩展方法、任务分布、注意力后端和内存预算都影响可用长度。本实验只验证旋转恒等式，没有验证外推质量。

## 3. ALiBi：在分数里加距离偏置

单个 head、因果位置 `j≤i` 下，可写成：

```text
score(i,j) = dot(q_i,k_j)/sqrt(d) - slope*(i-j)
```

正的 slope 使远位置获得更负的距离项。各 head 可使用不同斜率；本例只用手工 slope，不实现论文中的整套 head slope 分配。[ALiBi 论文](https://arxiv.org/abs/2108.12409)

假设两位置的 QK 点积都为 0，第二个位置的 slope 为 `ln(2)`：

```text
scores = [-ln(2),0]
softmax = [1/3,2/3]
V = [1,3]
output = (1/3)*1 + (2/3)*3 = 7/3
```

ALiBi 的距离惩罚不是因果掩码。未来位置仍必须排除；给它一个有限负数，仍可能获得非零权重。报告 `alibi_slope_0_5` 使用 slope=0.5，对全部默认输入位置计算结果。

## 4. Sliding Window：明确窗口包含谁

本课窗口 `W` **包含当前 token**。位置 `i` 只看：

```text
max(0,i-W+1) … i
```

例如 `T=8,W=3`：

| 查询位置 | 可见位置 | 位置对数量 |
| --- | --- | --- |
| 0 | `[0]` | 1 |
| 1 | `[0,1]` | 2 |
| 2 | `[0,1,2]` | 3 |
| 3 | `[1,2,3]` | 3 |
| 7 | `[5,6,7]` | 3 |

全因果 Attention 共 `T(T+1)/2=36` 对；滑窗共 `1+2+6*3=21` 对。报告分别放在 `dense_attention.score_pairs` 和 `sliding_window.score_pairs`。

```bash
python -m zero2vllm structure-demo --window 1 --output .local/runs/part-01-inference/05-model-structures/structure-self.json
python -m zero2vllm structure-demo --window 8 --output .local/runs/part-01-inference/05-model-structures/structure-full.json
```

`W=1` 时只看自己；`W≥T` 时可见范围与全因果一致。小窗口通常改变模型输出，不能要求它与完整 Attention 一样。

### 4.1 看得少，不自动等于分配得少

本 Python 代码限制参与计算的索引，仍保留完整输入列表。若要把显存降到窗口规模，需要缓存逐出、环形 buffer 或相应块管理，并保留绝对位置。本例没有实现设备缓存逐出。

多层局部 Attention 可以通过层间传播间接利用更远信息，但不等于每层直接读取所有历史。不同实现对“窗口大小是否含当前 token”也可能有不同约定；对齐公式和边界后再比较。[Mistral 7B 论文](https://arxiv.org/abs/2310.06825)

## 5. 稀疏 Attention：少计算哪些位置

滑窗是结构化稀疏的一种。其他图案可包括全局 token、固定块、周期连接、内容选择或检索得到的位置。

对每个 query 实际访问 `s` 个位置时，注意力主体工作可从与 `T²` 相关变成与 `T*s` 相关；但索引生成、路由、非连续读取和负载不均也有成本。一个 sparse score 图不保证 GPU 更快。

**稀疏计算不一定减少全部 KV 存储。** 如果未来 query 可能选中很早的任意位置，就仍可能需要保留完整历史 KV。只有结构/策略允许安全丢弃某些位置时，才可相应减少存储。

本课可运行的稀疏例子就是滑窗；内容检索或动态选择的完整 kernel 留在 Attention 专题。

## 6. MLA：压缩内容表示与吸收线性投影

MLA 的重要问题是如何避免逐 token 保存展开后的大量多头 K/V。理解时先看一个简化的内容分支恒等式：

```text
一个 token 的 latent 行向量：c
展开内容 key：k = c Wk
展开 value：v = c Wv

q · k = q · (c Wk) = (Wk q) · c
sum_j p_j (c_j Wv) = (sum_j p_j c_j) Wv
```

这里按本课程的行向量布局解释：`Wk` 为 `[r,d_key]`，`q` 为长度 `d_key` 的向量，吸收后的 query 为长度 `r`。可以先把 query 映射到 latent 空间，对 latent 打分；汇总 latent 后再做 value 展开。

报告 `latent_content_identity` 使用两个 token、3 维 latent、2 维 key/value：

```text
两项未缩放点积分数约为 [0.121, -0.003]
两个 value 输出分量约为 [-0.059962, 0.123543]
```

代码比较显式展开与线性变换后的结果，默认最大差约 `2.43e-17`。小维度是为方便检查，不用于代表任何真实模型的压缩率。

真实 MLA 还要处理多头布局、解耦位置分支、投影吸收、特定 attention backend 等。若实现确实缓存每 token 一个 `r` 维共享 latent 及 `d_rope` 维位置 key，可用 `L*T*(r+d_rope)*b` 作对应缓存分量的估算；展开缓存、其他分支或不同实现会有不同账本。不能直接套用 MHA 的 `2*Hkv*d`，也不能只看到名字 MLA 就认定采用了压缩布局。[DeepSeek-V2 论文](https://arxiv.org/abs/2405.04434)

## 7. 线性 Attention：把历史汇总进状态

本课选择一个始终为正的特征映射：

```text
phi(x) = max(x,0)+1，逐元素作用
weight(q,k) = phi(q) · phi(k)
```

对第 `t` 步因果前缀，显式计算为：

```text
y_t = sum_j [(phi(q_t)·phi(k_j))*v_j] / sum_j [phi(q_t)·phi(k_j)]
```

利用结合律，可以递推：

```text
S_t = S_(t-1) + outer(phi(k_t), v_t)
z_t = z_(t-1) + phi(k_t)
y_t = (phi(q_t)^T S_t) / (phi(q_t)·z_t)
```

状态 `S` 为 `[d_phi,d_v]`，`z` 为 `[d_phi]`，其大小不随序列长度增长。默认两维例子只需 `2*2+2=6` 个持久状态元素。代码为了展示还保留输入与所有输出，它的整个 Python 进程内存并不是常数。

**显式形式与递推形式是在同一套 feature-map 算法内对齐。** 它们通常不等于 `softmax(QKᵀ)V`，不能当作后者的无损 kernel 优化。模型训练和质量需要单独考虑。[Linear Attention 论文](https://arxiv.org/abs/2006.16236)

手算单维例子：`q=[1,2]、k=[0,1]、v=[1,3]`。第二步 `phi(q)=3`，历史 `phi(k)=[1,2]`，权重比例为 `[3,6]`，输出为 `7/3`。

## 8. SSM：状态递推不等于保存 K/V 列表

先看一个标量线性时不变系统：

```text
h_t = a*h_(t-1) + b*x_t，h_initial=0
y_t = c*h_t
```

当 `a=0.5,b=1,c=2,x=[1,2,3]`：

| 输入 | 更新后状态 | 输出 |
| --- | --- | --- |
| 1 | 1 | 2 |
| 2 | 2.5 | 5 |
| 3 | 4.25 | 8.5 |

也能用展开形式 `y_t=Σ_j c*a^(t-j)*b*x_j` 检查递推是否正确，报告保留两条路径。

这只演示状态更新。Mamba 等选择性状态空间模型会让部分参数依赖输入，还涉及向量/矩阵状态、卷积和门控；不能把本例的固定卷积展开直接用于所有选择性 SSM。真实缓存还可能包括卷积状态；混合模型也可能同时保存 Attention KV 与 SSM 状态。[Mamba 论文](https://arxiv.org/abs/2312.00752)

## 9. 把缓存对象与复杂度放到一张表

下面为单层、单请求的示意，省略 batch、层数、字节和部分 head 常数。`T` 是长度，`W` 是窗口，`r` 是 latent 宽度。

| 结构 | 主要持久状态 | 随长度的增长 | 需要注意 |
| --- | --- | --- | --- |
| MHA/GQA/MQA | 历史 K、V | `O(T*Hkv*d)` | 新 query 仍访问历史；GQA 主要改变 KV head 数 |
| 滑窗且实现逐出 | 最近窗口 K、V | `O(min(T,W)*Hkv*d)` | 只 mask 而不逐出并不会自动节省分配 |
| 一般稀疏 Attention | 可能仍是完整历史 KV | 依算法而定 | 访问数量少不等于可丢弃历史 |
| 压缩 MLA 路径 | latent 与位置 key 等 | 常与 `T*(r+d_rope)` 相关 | 必须核对所用实现真正保存什么 |
| 本课线性 Attention | `S` 与 `z` | 与 T 无关，`O(d_phi*d_v+d_phi)` | 改变了 Attention 公式；不等于 softmax |
| 本课标量 SSM | 一个状态 h | 与 T 无关 | 真实向量状态和卷积 buffer 有不同形状 |

RoPE/ALiBi 主要改变位置表示或 score，不会仅凭自身把完整 KV 变成固定大小。长上下文优化要同时考虑模型结构、缓存布局、kernel 和任务质量。

## 10. 练习与参考答案

### A. 旋转与位置

解释为何旋转保持范数；比较 `(7,10)` 和 `(0,3)` 的点积。若 Decode 每次使用位置 0，是否还能保持与完整前向一致？

<details>
<summary>参考答案 A</summary>

二维旋转中交叉项抵消，且 `sin²+cos²=1`。两组位置差均为 3，因此同频率旋转点积相同。Decode 错用位置会改变与历史 key 的相对位置，通常导致不一致。

</details>

### B. 窗口计数

`T=5,W=2` 时有多少位置对？第 4 号 query 看谁？

<details>
<summary>参考答案 B</summary>

`1+2+2+2+2=9` 对；第 4 号 query 看位置 `[3,4]`。完整因果为 15 对。本实现窗口含当前 token。

</details>

### C. ALiBi 与 mask

手算 `[-ln2,0]`、V=`[1,3]` 的输出，并解释为何有限负分数不能代替未来 mask。

<details>
<summary>参考答案 C</summary>

概率为 `[1/3,2/3]`，输出 `7/3`。有限负值的指数仍大于零，因此未来位置可能泄漏。

</details>

### D. 线性状态

特征维度 4、value 维度 3，本课线性 Attention 需要几个状态元素？能否把结果当成完整 softmax Attention 的正确性参考？

<details>
<summary>参考答案 D</summary>

`S` 有 12 个、`z` 有 4 个，总计 16。它只作为同一 feature-map 算法的参考，不能要求与 softmax 版本相同。

</details>

### E. SSM

若固定递推中的 `a=0`，状态和输出怎样变化？

<details>
<summary>参考答案 E</summary>

`h_t=b*x_t`，过去状态完全不影响当前结果；`y_t=c*b*x_t`。状态维度虽不变，记忆行为已经不同，不能只根据缓存小就断言模型能力相同。

</details>

## 11. 排错与验收

RoPE 奇数宽度会报错；窗口必须为正整数；Q/K/V 的长度、维度与数值必须有效。命令的 `--sequence-length` 限制在 1—256，避免把 Python 展示代码当作长序列压测器。

- [ ] 完成五道练习，写出每种结构保存的状态。
- [ ] 默认 `verification.passed=true`，各数学对照误差 ≤ `1e-12`。
- [ ] 跑通 window1 与 window8 两个边界。
- [ ] 能指出哪些对照应该一致，哪些输出不同是预期。
- [ ] 不将压缩率、位置扩展能力或计数直接当成 GPU 性能/模型质量结论。

测试入口为 [test_part1.py](../../../../tests/reference/test_part1.py) 的 `StructureTests`。上述论文来源与查阅日期：2026-09-15；本课矩阵、标量例子和代码为独立教学构造。
