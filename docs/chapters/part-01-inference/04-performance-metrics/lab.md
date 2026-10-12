# 1.4 实验：从四条事件记录读懂一份性能报告

[正文](README.md) · [10 组练习](exercises.md)

本节不启动模型服务。所有时间都是明确标记的教学模拟，Python 只负责重算。先在包含 `pyproject.toml` 的仓库根目录打开终端；使用 Python 3.11+，无需第三方包。环境准备见[第一篇总表](../README.md)。

## 1. 先找输入，不急着看总分

打开 [request-events.json](data/request-events.json)，找请求 A：

```json
{"id":"A","arrival_s":0.0,"work_started_s":0.2,"token_times_s":[0.5,0.7,1.0],"finished_s":1.1,"status":"ok"}
```

在纸上写出五类对象：提交、开始服务、输出时间数组、完成、状态。`token_times_s` 有三个数，所以 A 有三个输出、两个相邻间隔。

先手算：排队 0.2 秒、TTFT 0.5 秒、ITL 为 0.2/0.3 秒、TPOT 0.25 秒、E2E 1.1 秒。最后输出到完成的 0.1 秒单独保留。

## 2. 生成完整报告

```bash
python -m zero2vllm metrics-demo --output .local/runs/part-01-inference/04-performance-metrics/review-default.json
```

输出目录会自动创建。终端会显示完整 JSON，阅读时更适合打开保存的文件：

| 顺序 | 字段 | 此时核对 |
| --- | --- | --- |
| 1 | `record_type`、`input` | 仍是 simulation，输入保留四条请求 |
| 2 | `request_metrics` | 找 id=A，与刚才手算一致 |
| 3 | `counts` | 总数 4、成功 3、失败 1、成功输出 6 |
| 4 | `window.duration_s` | 2.3，包含 D 的失败结束时间 |
| 5 | `rates` | 成功率 0.75、输出吞吐约 2.608696、Goodput 约 0.869565 |
| 6 | `distributions` | TTFT 有 3 个样本，TPOT 有 2 个样本，ITL 有 3 个样本 |

这张表中的小数是显示用近似值。JSON 保留计算精度，`0.7−0.5` 可能显示为 `0.19999999999999996`，不表示它算出了另一个事件顺序。

## 3. 一次只改一个条件

```bash
python docs/chapters/part-01-inference/04-performance-metrics/examples/compare_timelines.py
```

这个示例在内存中复制输入，保持原文件不变。主要输出：

```text
base: window=2.3s success=3/4 tokens/s=2.608696 good=2
late_failure: window=3.0s success=3/4 tokens/s=2.000000 good=2
partial_failure: window=2.3s success=3/4 tokens/s=2.608696 good=2
strict_slo: window=2.3s success=3/4 tokens/s=2.608696 good=0
request-mean TPOT=0.425000s; interval-mean ITL=0.366667s
pooled P95=90.1; average of round P95=50.5
```

逐行回答：

1. `late_failure` 只把 D 的完成延迟到 3 秒，成功请求并没有变慢，为什么总吞吐下降？因为公共窗口变长。
2. `partial_failure` 给 D 两个部分输出，但它仍失败。本工具只统计成功输出吞吐，所以分子不增加。
3. `strict_slo` 只改变服务目标，没有改原始事件，因此只有达标判定改变。
4. 两种平均和两种分位数汇总不同，原因是样本与权重不同。

示例里的断言还会检查这些解释是否与实际报告一致。

## 4. 亲手修改一份输入副本

先创建个人目录并复制数据，下面一行可用于 Git Bash、PowerShell 或常见 Linux shell：

```bash
python -c "from pathlib import Path; import shutil; p=Path('.local/runs/part-01-inference/04-performance-metrics'); p.mkdir(parents=True,exist_ok=True); shutil.copyfile('docs/chapters/part-01-inference/04-performance-metrics/data/request-events.json',p/'my-events.json')"
```

打开副本，将 D 的 `finished_s` 从 2.3 改为 3.0，其他字段保留。先预测哪些字段改变，再运行：

```bash
python -m zero2vllm metrics-demo --input .local/runs/part-01-inference/04-performance-metrics/my-events.json --output .local/runs/part-01-inference/04-performance-metrics/my-report.json
```

此时成功率仍为 75%，输出吞吐为 2 token/s，Goodput 为 `2/3 request/s`；A/B/C 的 TTFT 不变。D 的 E2E 则从 2.0 变为 2.7 秒，因为它在 0.3 秒提交。

修改后仍保持 `record_type=simulation`。自己填进数据的数字不会因为程序接受它就变成硬件测量。

## 5. 运行一个明确应失败的输入

在个人副本中把 A 的时间数组改为 `[0.5,0.4,1.0]`，再运行相同命令。它应拒绝倒序事件，不能通过排序悄悄修正日志。

可以继续单独试：重复请求 ID、把一个时间填成字符串、将 `timestamp_kind` 改成 `stream_chunk`。每次只破坏一个条件，阅读具体错误，再恢复。输入失败时不要把此前已存在的输出文件当成这次新产生的有效报告。

## 6. 参数与故障定位

| 参数或现象 | 解释 |
| --- | --- |
| `--input` | 替换事件文件；不改变采集器，也不启动压测 |
| `--ttft-slo` / `--tpot-slo` | 秒；只改变达标判定 |
| 文件不存在 | 当前目录是否为仓库根目录，副本是否已经保存 |
| B 的 TPOT 为 null | 正确；只有一项输出，间隔没有定义 |
| P99 看起来很好但成功率很低 | 本工具的成功延迟分布不包含失败请求，需两项一起报告 |
| 与另一工具 TPOT 不同 | 核对终点是最后 token 还是完成事件，及样本选择与单位 |

独立检查：

```bash
python -m unittest tests.reference.test_part1.MetricsTests -v
```

## 交付一页解释

保存默认报告和修改后的报告。写出改变的一个条件、保持不变的量、发生变化的量，以及你为什么这样预测。再完成[练习](exercises.md)，用原始数据说明结论，而不是只贴一个吞吐数字。
