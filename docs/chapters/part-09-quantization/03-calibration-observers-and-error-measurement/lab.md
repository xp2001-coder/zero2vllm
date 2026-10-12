# 9.3 实验：让校准过程可以复核

[正文](README.md) · [练习](exercises.md) · [发布结果](results/README.md)

从仓库根目录运行。Python 3.11+，默认无需 GPU、模型、网络或第三方包；终端中文编码有问题时加 `-X utf8`。

## 1. 完整运行

```bash
python -m scripts.experiments.lesson0903
```

预期 20 个 PASS、退出 0，生成：

```text
.local/runs/part-09-quantization/03-calibration/
├── summary.json
├── protocol.json
└── cpu-passport.json
```

需要保留一次独立运行时：

```bash
python -m scripts.experiments.lesson0903 --output-dir .local/runs/part-09-quantization/03-calibration/replay
```

默认只覆盖本地实验目录，不覆盖本节发布的 results。时间、Python 版本和源码摘要可以变化，固定输入的数值应一致。

## 2. 检查协议的实际数量

```bash
python docs/chapters/part-09-quantization/03-calibration-observers-and-error-measurement/examples/inspect_protocol.py
python docs/chapters/part-09-quantization/03-calibration-observers-and-error-measurement/examples/inspect_protocol.py --token-budget 16
python docs/chapters/part-09-quantization/03-calibration-observers-and-error-measurement/examples/inspect_protocol.py --domain code --token-budget 16
```

默认 general 的 4 条样本是 24 个有效 token、8 个 padding。设 16 token 预算后两种分布都实际用到 3 条样本，并对最后一条保留前缀。检查 `sample_ids`、`attention_mask`、`bos_tokens/eos_tokens`，确认 EOS 是否被预算截去。

申请多于所选样本能提供的有效 token：

```bash
python docs/chapters/part-09-quantization/03-calibration-observers-and-error-measurement/examples/inspect_protocol.py --token-budget 1000
```

预期退出 2，说明预算不可满足。程序不把 PAD、重复文档或不存在的 token 算入预算。

## 3. 跟踪 Observer 状态

```bash
python docs/chapters/part-09-quantization/03-calibration-observers-and-error-measurement/examples/walk_observer.py
python docs/chapters/part-09-quantization/03-calibration-observers-and-error-measurement/examples/walk_observer.py --reverse
python docs/chapters/part-09-quantization/03-calibration-observers-and-error-measurement/examples/walk_observer.py --mode minmax
```

| 设置 | 最终极值 |
| --- | --- |
| EMA α=.25，先 [-1,1] 再 [-4,8] | [-1.75,2.75] |
| EMA 相同系数，顺序相反 | [-3.25,6.25] |
| 全局 MinMax | [-4,8] |

示例冻结为 unsigned affine [0,15]，所以还会输出非零 zero-point。这与完整样本数实验的默认 symmetric [-7,7] 是两个已声明的设置。结尾故意在 freeze 后尝试 update，预期捕获明确异常；整体示例仍退出 0。

## 4. 检查裁剪与误差排名

```bash
python docs/chapters/part-09-quantization/03-calibration-observers-and-error-measurement/examples/compare_objectives.py
```

应观察到：

- 裁剪候选 `[.5,1,2,4,7]` 中 T=4 的校准 MSE 约 .039946，T=7 为 .248062；另一验证分布中两者分别约 4.502551 与 .125。
- 权重例 A/B 的权重 MSE 为 .0078125 / .03125，层输出 MSE 为 1 / 0。
- FP 输入的重构误差 A/B 为 0 / 1，上游量化输入的误差为 1 / .0625。
- logits 反例 A/B 的 MSE 为 .0078125 / .28515625，合成准确率为 .5 / 1。

这些候选是人为数值例，不是任何论文算法的性能比较；不能写成“某算法提升了 50% 准确率”。

## 5. 对照两组正式受控实验

查看 `summary.json` 的 `studies`：

| 对照 | 改变的变量 | 冻结参数与相同 test 上的 MSE |
| --- | --- | --- |
| `sample_count` | general 样本数 2/4/8，固定嵌套顺序 | scale .142857/.214286/1；MSE 3.850553/3.013287/.09375 |
| `fixed_token_budget` | 16 个有效 token 下的 domain | general/code：scale .214286/1；MSE 3.013287/.09375 |

两组均在同一预定义 test 上报告，共 24 个有效 token、48 个元素。默认 MinMax，per-tensor symmetric INT4，z=0，FP64 scale。没有用 test 反复搜索阈值。

`protocol.json` 对应第一组 8 条 general 校准样本，实际 50 个有效 token；其余协议完整嵌入各自 `studies` 条目，不能只看这一份协议就解释所有行。

## 6. 可选 PyTorch 对照

若已有 PyTorch，可运行：

```bash
python -m scripts.experiments.lesson0903 --torch-check
```

本轮实际为 **2.7.1+cpu**。MinMax 与 MovingAverageMinMax 各用两批输入，统一 affine [0,15]，极值与 z 一致，scale 按 1e-6 相对容差比较。该容差覆盖 FP32/FP64 存储差异，不是对量化质量的宽松门槛。

没有安装 torch 却显式传入此开关，会报告失败；默认不导入 torch。它不是 MQBench、真实模型或 GPU 检查。安装版本可参考 [PyTorch 历史版本说明](https://pytorch.org/get-started/previous-versions/#v271)。

## 7. 跑针对性测试

```bash
python -m unittest tests.quantization.test_calibration -v
```

共 39 项：协议 12、Observer 14、指标 10、实验入口/对照 3。没有 torch 时独立框架对照 1 项会明确跳过；本轮环境全部运行。

## 8. 自己扩展时一次只改一个因素

可以复制输入到自己的实验目录，增加更少见的 outlier，再观察 MinMax 与裁剪的变化；或保持所有 token 不变，只改变 Observer 的批次切法，观察 EMA。每次保存原始数据摘要、参数、结果与解释。

切勿拿扩展后的结果覆盖发布记录。对于实际模型，另外提供浮点基线、模型/tokenizer revision 和独立质量数据；本节合成实验的 Passport 不能替代这些证据。
