# 1.5 实验：逐步追踪位置、可见范围和递推状态

[正文](README.md) · [10 组练习](exercises.md)

从仓库根目录运行，Python 3.11+，不需要第三方包、GPU 或模型下载。各例子验证独立数学关系，不将所有结构混合成一个模型。

## 1. 第一遍：只看位置与可见范围

```bash
python docs/chapters/part-01-inference/05-model-structures/examples/walk_positions.py
```

主要输出：

```text
rotate [1,0] at position 1: [0.540302, 0.841471]
query position 7: score=0.540302; wrongly reset to 0: 0.960170
query 0: keys [0]
query 1: keys [0, 1]
query 2: keys [0, 1, 2]
query 3: keys [1, 2, 3]
...
query 7: keys [5, 6, 7]
pairs: full=36, window3=21; ALiBi second output=7/3
```

这里的省略号仅压缩展示，实际命令会打印每个位置。先解释两项：

- query 在位置 7、key 在位置 6，位置差为 1；错误地把 query 重置为 0 后，关系变了。
- query 7 只读 `[5,6,7]`，但它仍是绝对位置 7，窗口长度不会把绝对位置改为 2。

ALiBi 例子用两个相同的内容分数，单独显示距离项的作用。可在正文重算 `[1/3,2/3]` 与输出 `7/3`。

## 2. 保存默认结构报告

```bash
python -m zero2vllm structure-demo --output .local/runs/part-01-inference/05-model-structures/review-default.json
```

第一次只检查：

| 字段 | 预期 |
| --- | --- |
| `input.sequence_length / window` | 8 / 3 |
| `dense_attention.score_pairs` | 36 |
| `sliding_window.score_pairs` | 21 |
| `sliding_window.visible_positions[7]` | `[5,6,7]` |
| `rope.pair_convention` | adjacent，相邻坐标配对 |
| `verification.errors` | 各个独立数学对照在 `1e-12` 容差内 |

`verification` 没有要求滑窗输出与完整 softmax 相同，因为它们看见的历史不同。报告没有记录真实 GPU 耗时。

## 3. 改动窗口，预测结果

```bash
python -m zero2vllm structure-demo --window 1 --output .local/runs/part-01-inference/05-model-structures/review-window1.json
python -m zero2vllm structure-demo --window 8 --output .local/runs/part-01-inference/05-model-structures/review-window8.json
python -m zero2vllm structure-demo --sequence-length 5 --window 2 --output .local/runs/part-01-inference/05-model-structures/review-window2-length5.json
```

| 参数 | 最后一个 query 看见谁 | 滑窗总位置对 |
| --- | --- | --- |
| T=8，W=1 | `[7]` | 8 |
| T=8，W=8 | `[0,1,2,3,4,5,6,7]` | 36 |
| T=5，W=2 | `[3,4]` | 9 |

W=1 时每行只有一个可见位置，softmax 权重为 1，因此输出等于当前位置 V。W=8 时可见集合与完整因果相同，两条路径输出应一致。W=2 通常改变输出，这并不是错误。

## 4. 第二遍：跟着整数例子走状态更新

```bash
python docs/chapters/part-01-inference/05-model-structures/examples/walk_states.py
```

实际关键结果：

```text
latent: expanded K=[1, 2, 3]; transformed Q=[3, 4]; both scores=11
linear step 0: S=1, z=1, output=1.000000
linear step 1: S=7, z=3, output=2.333333
SSM states: [1.0, 2.5, 4.25] outputs: [2.0, 5.0, 8.5]
```

不要把这三行视为三个模型的优劣比较：

1. latent 例子检查同一个点积的两种计算次序。
2. 线性 Attention 例子检查显式遍历与 `S/z` 递推的等价关系。
3. 标量 SSM 检查固定系数递推与展开求和。

建议在纸上划出“上一步状态”“新输入贡献”“新状态”三栏。第 2 步的线性状态增加 `2×3=6`，所以 S 从 1 变成 7；归一化状态增加 2，所以 z 从 1 变成 3。

## 5. 再读完整报告中的状态分支

回到默认 JSON，依次查看：

- `latent_content_identity`：矩阵、query、显式分数与变换后分数。这套小数输入与上面的整数例子不同，分别对齐各自参考。
- `linear_attention.recurrent / explicit`：各位置输出应相同；`persistent_state_elements=6` 指默认两维状态的元素数。
- `ssm.states / outputs / convolution_reference`：状态与输出不是一回事，本例输出是状态的 2 倍。

特别注意默认 latent 例子取 r=3、key/value 宽度为 2，只用于检查代数关系，没有用它证明压缩比例。实际压缩需要对真实缓存布局和各分支做完整计数。

## 6. 有意检查几个边界

```bash
python -m zero2vllm structure-demo --sequence-length 1 --window 1 --output .local/runs/part-01-inference/05-model-structures/review-single.json
python -m zero2vllm structure-demo --window 0
```

第一条有一个可见位置对，第二条应拒绝零窗口。报告中的 RoPE、latent 和标量 SSM 是独立例子，修改序列长度不会自动改变所有这些示例的手工输入。

| 问题 | 查哪里 |
| --- | --- |
| 窗口差一个位置 | 本节 W 包含当前 token；核对 `i-W+1` |
| Decode 与完整前向不一致 | 优先查绝对位置、RoPE 配对和 mask 偏移 |
| mask 变小但进程内存没降 | 本代码保留输入列表，没有实现环形缓存逐出 |
| 线性 Attention 与 softmax 不一致 | 算法不同；改为检查同一 feature map 的显式与递推 |
| 固定状态大小但数值不稳定 | 状态大小不保证系数、归一化或训练质量合理 |

独立检查：

```bash
python -m unittest tests.reference.test_part1.StructureTests -v
```

## 交付一张状态对照表

对每种结构写：读取的历史、保留的状态、每步新增或更新的内容、可以与哪条数学参考对齐。不要只抄复杂度符号，至少附上一个具体维度或手算结果，再完成[练习](exercises.md)。
