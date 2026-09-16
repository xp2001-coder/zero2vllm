# 引导实验：KV Cache 前后是否计算一致

**层次：基础必读。设备：CPU。状态：教程与代码已实现，本地验证通过。**

## 学习目标

1. 知道 KV Cache 保存了什么。
2. 用每一步的 logits 检查缓存实现是否正确。
3. 区分重复计算减少与实际 GPU 性能提升。

前置知识：Python 列表、循环、点积。可以先运行，再阅读公式。

## 运行

在工程根目录使用 Python 3.11+：

```bash
python -m zero2vllm demo --output runs/cpu-kv-cache.json
```

代码入口：[attention.py](../../zero2vllm/reference/attention.py)。验证入口：[test_attention.py](../../tests/test_attention.py)。

## 模型做了什么

每个 token 的输入向量由 token embedding 和位置 embedding 相加得到。单个注意力头计算：

```text
q_i = x_i W_q
k_i = x_i W_k
v_i = x_i W_v
a_ij = softmax_j(q_i · k_j / sqrt(d)), 仅允许 j <= i
h_i = x_i + (sum_j a_ij v_j) W_o
logits_i = h_i W_vocab
```

这是固定随机权重的单层模型，用于观察数学行为。它没有训练、MLP、归一化或真实 Tokenizer，生成的数字没有语言意义。

## 比较两种生成方式

### 完整重算

每生成一个 token，就把目前整个序列重新送入模型，计算所有位置的 Q/K/V 与因果注意力。

### 使用缓存

首次处理整个 prompt 并保留 K/V。此后每次只送入刚生成的一个 token，计算它的 Q/K/V，并读取已有的 K/V。位置编号从缓存当前长度继续，不能每一步从零开始。

对每次生成，实验保存整个词表上的 logits，而不只检查最终 token。只比较 token 可能漏掉“分数已经错误，但最大值位置碰巧相同”的问题。

## 默认结果如何解释

4 个 prompt token，生成 8 个新 token，需要 8 次模型调用。

- 完整重算投影位置数：`4 + 5 + ... + 11 = 60`。
- 缓存路径投影位置数：`4 + 7 = 11`。
- 最后生成的 token 已经返回，无需再送入模型，因此不是 12。
- 默认实验的注意力 query/key 配对次数：完整重算 276，缓存路径 66。

查看 `verification`：`same_token_ids` 应为 `true`，`max_abs_logit_error` 应不超过 `1e-12`。已有 [本地结果](../../reports/cpu-kv-cache.json) 可对照。

KV Cache 避免了已有 token 的 K/V 重算，但新 query 仍然要访问已有上下文。这个实验的计数不是 FLOPs，也不包含内存传输、GPU 并行、内核启动等成本，不能用 `60/11` 作为真实 GPU 加速比。

## 动手练习

1. 把 prompt 改为 `1,2`，保留 8 个输出，先预测投影位置数再运行。
2. 把输出改为 1 个，观察两种方式的工作量是否相同。
3. 尝试超过词表范围的 token，例如 `99`，阅读错误信息。
4. 阅读测试：为什么在末尾添加 token 不应影响前面位置的 logits？

```bash
python -m zero2vllm demo --prompt 1,2 --max-new-tokens 8
python -m zero2vllm demo --max-new-tokens 1
```

答案要点：第 1 题完整重算为 `2+...+9=44`，缓存为 `2+7=9`；第 2 题只做一次 prefill，尚无跨生成步骤的重复工作可避免。

## 工程进阶

测试覆盖多种 prompt、多个随机种子、分块 prefill、位置偏移、缓存不能跨模型实例混用、超出容量时不部分写入缓存。

将来接入 PyTorch 和 GPU 半精度时，需要根据 dtype 设置合理误差范围，并增加真实权重的逐层比较。当前两条参考路径的严格容差不能直接作为所有 GPU 算子的要求。

逻辑块分配模拟在 [blocks.py](../../zero2vllm/reference/blocks.py)。它目前独立于此 Attention 实现，后续分页缓存章节再把两者连接起来。

## 下一步

能解释 60 与 11 的来源后，可先读 [1.3 完整缓存课](../chapters/01-kv-cache.md)的分块、GQA 和容量推导，再继续[调度与缓存](../chapters/02-scheduling-cache.md)。基础内容的完整顺序见 [Part 1 总表](../chapters/part1-guide.md)，跨专题学习见[现有材料短路线](../learning-paths.md#先用现有材料走通一条短路线)。
