# 2.2 练习与答案

[正文](README.md) · [实验](lab.md)

先完成前四题，再运行布局示例与错误注入测试。工程进阶题不要求持有 GPU。

## 1. 从配置推出形状

模型 `D=576`、`Hq=9`、`Hkv=3`、`F=1536`。求 head_dim、K 投影权重形状、MLP down 权重形状，以及五个 token 的 Attention 分数形状。若 batch 改为 2，哪些权重形状要改？

<details>
<summary>答案</summary>

- head_dim 为 `576/9=64`。
- K 输出宽度为 `3×64=192`，权重 `[192,576]`。
- down 从 1536 回到 576，权重 `[576,1536]`。
- 分数是 `[1,9,5,5]`；教学 GQA 展开 K/V 后与 9 个 Q 头分别计算。
- batch 改为 2 后中间量的第一维变为 2，权重形状不变。

</details>

## 2. 为什么模型少了一个 LM Head 权重

文件没有 `lm_head.weight`，加载器是否应一律报错？共享关系只做 `head.weight.copy_(embedding.weight)` 是否足够？如果两份权重都存在但不相同呢？

<details>
<summary>答案</summary>

先看 `tie_word_embeddings`。为 true 时，本节允许省略 LM Head，并将 `head.weight` 指向 `embedding.weight`。为 false 时缺少 LM Head 必须报错。

`copy_` 只复制数值，不建立参数别名，还多占一份内存。若配置要求共享而文件保存的两份值不相同，加载器应报冲突，不能随文件遍历顺序任选最后一份。

运行 `python -m unittest tests.modeling.test_checkpoint.CheckpointTests.test_explicit_tied_head_must_equal_embedding -v` 检查正反两种情况。

</details>

## 3. 形状没错，为什么输出错了

用 `X=[1,2]` 与 `W=[[1,2],[3,4]]` 手算 `X@W.T` 和 `X@W`。再解释为什么误转置真实模型的 Q 权重可能不报 shape 错误。

<details>
<summary>答案</summary>

结果分别为 `[5,11]` 和 `[7,10]`。`W` 是方阵，转置后仍为 `[2,2]`。真实模型 Q 的 `[576,576]` 同理，所以必须比较实际数值。

运行 `python -m unittest tests.modeling.test_checkpoint.CheckpointTests.test_transposing_square_q_is_caught_by_numeric_oracle -v`。这个测试故意让加载检查成功，要求独立参考发现 logits 不一致。

</details>

## 4. 两种 RoPE 能否互换

对四维向量 `[a,b,c,d]`，写出 2.1 与本节的配对；构造让相邻配对得到 Llama 布局结果的置换。为什么只测位置 0 发现不了错误？

<details>
<summary>答案</summary>

2.1 配对 `(a,b)`、`(c,d)`；本节配对 `(a,c)`、`(b,d)`。先按索引 `[0,2,1,3]` 置换，执行相邻配对，再使用同一置换恢复顺序。四维例子中该置换恰好等于自己的逆；一般维度应显式求逆，不能照搬这个巧合。

位置 0 的旋转角为零，两条路径都返回原向量。必须选非零位置，并用非对称输入。运行 `examples/layout_traps.py` 验证。

</details>

## 5. GQA 的 repeat 能随便替换吗

三个 K/V 头各用编号 0、1、2 表示。写出 `repeat_interleave(3,dim=1)` 与 `repeat(1,3,1,1)` 后九个头的顺序，解释哪一个符合本节权重布局。

<details>
<summary>答案</summary>

前者是 `[0,0,0,1,1,1,2,2,2]`，后者是 `[0,1,2,0,1,2,0,1,2]`。本节每三个连续 Q 头共享一个 K/V 头，使用前者。两者头数都变为 9，仅检查 shape 不足以识别错误。

进一步思考：正式 GPU kernel 若直接按 `q_head // 3` 找到 KV 头，就无需把 K/V 存储扩大三倍。本节没有验证这个 GPU 实现或其收益。

</details>

## 6. 判断一份报告究竟证明了什么

分别判断以下说法：

1. `--inspect-only` 退出码为 0，说明生成结果已对齐。
2. 三个输入的 logits 与生成 ID 一样，说明模型已经能正确回答算术题。
3. 这次 CPU 误差是 0，后续 A100/C500 也必须是 0。
4. 两份结果固定同一模型 revision，就不必记录 Tokenizer 和推理库版本。

<details>
<summary>答案</summary>

四项都不成立。只检查权重时数值验证为 `not_run`；实现一致不等于模型质量高；换计算后端可能改变浮点结果；文本编码和框架实现也是复现条件。

本轮 `1 + 1 =` 的两个模型均生成 ` 1000000`，正好说明质量问题不能被“两个实现相同”排除。

</details>

## 7. 分片与配置错误注入

阅读 [test_checkpoint.py](../../../../tests/modeling/test_checkpoint.py)，找出测试如何在临时目录生成分片。分别构造：丢失分片、索引 key 与文件不一致、删除某层 Norm、把 K 权重转置。写下应该在哪个阶段报错。

<details>
<summary>提示与验收</summary>

丢失分片和索引错误在 `checkpoint_files` 被发现；缺少 Norm 在复制前的名称检查被发现；K 的非方阵 `[192,576]` 被转置后在 shape 检查被发现。错误必须明确失败，不返回可用于前向的半加载模型。

测试应修改临时生成的随机小模型，不修改下载的真实快照；Windows 上重写之前先复制脱离文件映射。

</details>

## 8. 工程进阶：融合 QKV 前先定义什么

如果要把本节三个独立投影合成一个，融合输出应有多宽？三个输出区间是什么？能否平均分为三块？

<details>
<summary>答案</summary>

输出宽度为 `576+192+192=960`。沿输出维拼接权重，得到 `[960,576]`，Q/K/V 的区间分别为 `[0:576]`、`[576:768]`、`[768:960]`。

GQA 下三个投影宽度不同，不能平均切三份。应先用相同输入核对融合前后的 Q/K/V，再验证整层与模型结果。是否加速要测量，不能只凭三个 Linear 变成一个就下结论。

对应阅读：正文列出的固定版本 vLLM `LlamaModel.load_weights`。本题只要求给出布局和验证方案，融合算子实现留给后续章节。

</details>
