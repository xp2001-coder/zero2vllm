# 1.6 实验：先检查分布，再检查生成与数值

[正文](README.md) · [12 组练习](exercises.md)

所有命令从仓库根目录运行，Python 3.11+，无需额外依赖。先处理一个固定四候选分布，再运行五词项模型，最后检查停止与误差。两个词表属于不同的小实验。

## 1. 把筛选的每一步算出来

```bash
python docs/chapters/part-01-inference/06-sampling-correctness/examples/walk_distribution.py
```

示例把 `[0.4,0.3,0.2,0.1]` 取自然对数作为 logits，逐组构建参数。主要结果：

| 方案 | 保留 ID | 最终概率，约 |
| --- | --- | --- |
| T=1，无筛选 | 0、1、2、3 | `[0.4,0.3,0.2,0.1]` |
| T=0.5，无筛选 | 0、1、2、3 | `[0.533333,0.3,0.133333,0.033333]` |
| top-k=2 | 0、1 | `[0.571429,0.428571,0,0]` |
| top-p=0.8 | 0、1、2 | `[0.444444,0.333333,0.222222,0]` |
| min-p=0.6 | 0、1 | `[0.571429,0.428571,0,0]` |
| top-k=2 后 top-p=0.5 | 0 | `[1,0,0,0]` |

最后两次用手工指定的 u=0.65 和 0.75 检查累计区间，输出分别为 ID 1、2。`FixedUniform` 只为这项手算返回指定值，不是生产随机数生成器。

## 2. 先建立生成的确定性基线

```bash
python -m zero2vllm sampling-demo --temperature 0 --max-new-tokens 3 --output .local/runs/part-01-inference/06-sampling-correctness/review-greedy.json
```

预期 `generation.new_token_ids=[2,2,2]`，`generated_text` 为“推理推理推理”。这里只包含新文本，不包含 prompt“我喜欢”。检查第一步输入为 `[0,1]`，后续每次只输入上一轮选出的 ID 2。

阅读 `generation.steps[0].distribution` 的顺序：

1. `raw_logits`：模型原始分数。
2. `penalized_logits`：历史惩罚与屏蔽后的结果。
3. `kept_token_ids`：当前仍可能被选择的候选。
4. `probabilities`：本例是只在最大项为 1 的分布。
5. `selected_token_id` 在本步外层：最终选中了谁。

本课 temperature=0 时跳过 top-k/top-p/min-p，因此不要用这个设置观察三种筛选参数的效果。

## 3. 回到随机采样，单独改一个条件

```bash
python -m zero2vllm sampling-demo --temperature 1 --top-p 1 --seed 7 --output .local/runs/part-01-inference/06-sampling-correctness/review-base.json
python -m zero2vllm sampling-demo --temperature 0.5 --top-p 1 --seed 7 --output .local/runs/part-01-inference/06-sampling-correctness/review-temp05.json
python -m zero2vllm sampling-demo --temperature 1 --top-p 1 --top-k 2 --seed 7 --output .local/runs/part-01-inference/06-sampling-correctness/review-topk.json
```

三份报告先比较第一步，因为它们的模型输入相同。第二步开始若已选出不同 ID，输入就发生分叉，后续 raw logits 不再具有“完全相同输入”的前提。

`frequency_experiment` 使用另一个固定四候选分布，适合单独观察筛选与抽样频率。它没有 prompt/生成历史，penalty 在其中没有可作用的历史项。不要将它的四项概率与五词项模型的 logits 按下标混着比。

## 4. 复现 seed，比较正确的字段

```bash
python -m zero2vllm sampling-demo --seed 7 --output .local/runs/part-01-inference/06-sampling-correctness/review-seed7-a.json
python -m zero2vllm sampling-demo --seed 7 --output .local/runs/part-01-inference/06-sampling-correctness/review-seed7-b.json
python -m zero2vllm sampling-demo --seed 8 --output .local/runs/part-01-inference/06-sampling-correctness/review-seed8.json
```

同 seed 的前两份应比较 `generation`、配置和频率结果；创建时间可能不同，所以整份文件字节不必相同。不同 seed 可能碰巧抽到相同序列，不能把“必须不同”写成正确性要求。

## 5. 一次对比四种结束方式

```bash
python -X utf8 docs/chapters/part-01-inference/06-sampling-correctness/examples/walk_stopping.py
```

| 情形 | 原始新 ID | 展示文本 | 缓存长度 | 原因 |
| --- | --- | --- | --- | --- |
| 零输出 | `[]` | 空 | 0 | length |
| 指定 EOS=2 | `[2]` | 空 | 2 | eos |
| stop=“推理推理” | `[2,2]` | 空 | 3 | stop_string |
| 输出长度 3 | `[2,2,2]` | 推理推理推理 | 4 | length |

这里四条请求都以两个 prompt token 开始。零输出不做前向，所以缓存为 0；其他情况最终缓存为 `P+已生成数量−1`。它们返回的空文本也有不同原因。

独立保存跨 token 停止案例：

```bash
python -m zero2vllm sampling-demo --temperature 0 --stop "推理推理" --output .local/runs/part-01-inference/06-sampling-correctness/review-stop.json
```

这个采样参考先完整缓冲结果，再裁去停止字符串。2.7 的 HTTP 示例目前传数字 ID，也没有直接实现这里的文本 stop 缓冲；扩展文本流时需单独处理尚未确定的尾部。

## 6. 用反例检查数值结论

```bash
python docs/chapters/part-01-inference/06-sampling-correctness/examples/walk_correctness.py
```

主要结果：

```text
same_token_wrong_scores: within_tolerance=False, same_argmax=True, max_error=18
close_scores_different_token: within_tolerance=True, same_argmax=False, max_error=2e-07
gap_exceeds_twice_error: within_tolerance=False, same_argmax=True, max_error=0.1
addition order: left=1, right=0
```

第三行是专门保留的反例：误差虽足够小，不会翻转最大项，但它仍大于当前数值容差。“输出稳定”和“数值容差通过”是两个独立判定。

如要定位量化或 kernel 问题，先用相同输入检查中间量与 logits，再分析采样和完整生成；不要先通过放宽容差把失败标记消掉。

## 7. 参数边界与排查

```bash
python -m zero2vllm sampling-demo --top-p 0
python -m zero2vllm sampling-demo --temperature -1
```

两条都应因非法参数失败。更多参数范围见正文速查表。

| 现象 | 优先检查 |
| --- | --- |
| top-k 好像没有效果 | 是否 temperature=0，或 k 已大于可用候选数量 |
| 相同 seed 在另一个后端结果不同 | 随机算法、消耗顺序、数值误差和筛选边界是否一致 |
| 相同 token 但 logits 差很大 | 不能据 token 相同判正确，检查全向量误差 |
| stop 后文本为空但有 ID | 停止内容被从展示文本移除，原始 ID 仍保留 |
| 同时改许多参数后无法解释 | 回到固定四候选例子，每次只改变一项 |
| 中文显示乱码 | 使用 `python -X utf8` 或打开 UTF-8 的输出文件 |

```bash
python -m unittest tests.reference.test_part1.SamplingTests -v
```

## 验收产出

交付一份分布手算、一组同 seed 对照、一张停止记录表和两个数值反例解释。完成[12 组练习](exercises.md)后，再把这些约定带到第二篇的真实权重与生成检查中。
