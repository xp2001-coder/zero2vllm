# 1.4 如何读性能结果

[本篇目录](../README.md) · [全部课程](../../README.md) · [上一节 1.3](../03-kv-cache/README.md) · [下一节 1.5](../05-model-structures/README.md)

> 基础必读，含工程进阶。提供可重算的事件数据与指标代码。默认数据是手工构造的教学模拟，不是 A100/C500 测量。建议 90—120 分钟。

[Part 1 总表](../README.md) · [上一课：KV Cache](../03-kv-cache/README.md) · [下一课：模型结构](../05-model-structures/README.md)

## 学习目标与准备

本课结束时，你应能从原始事件重算 TTFT、TPOT、ITL、吞吐、Goodput 与 P50/P95/P99，并明确说明每个指标的分母、样本和时间窗口。

前置：知道 Prefill/Decode。只需 Python 3.11+；不需要部署模型。代码为含 `metrics-demo` 的 `0.1.0.dev0` checkout。

先运行：

```bash
python -m zero2vllm metrics-demo --output .local/runs/part-01-inference/04-performance-metrics/metrics-default.json
python -m zero2vllm metrics-demo --help
```

输入：[request-events.json](data/request-events.json)。实现：[metrics.py](../../../../zero2vllm/reference/metrics.py)。

已运行的[默认指标报告](results/baseline/04-metrics.json)保留完整模拟输入和全部重算结果。

## 1. 从请求的一条时间线开始

我们记录每个请求的：提交时间、开始服务时间、每个输出 token 的时间、完成时间和成功/失败状态。**以下时间都在同一个时钟下，以秒为单位。**

```text
提交             开始服务        第一个 token      后续 token       完成
 arrival ---------- start -------- t0 ------ t1 ------ t2 ---------- finish
         排队                首次处理与其他开销       生成与传输
```

不要拿客户端 wall clock 减服务端未校准的 wall clock 来计算排队时间。生产采集可能需要单调时钟、同进程事件或分布式追踪；没有这些证据时，排队时间应标为不可得。

本例为四个并发且长度不同的请求：

| 请求 | 提交 | 开始服务 | 输出 token 时刻 | 完成 | 状态 |
| --- | --- | --- | --- | --- | --- |
| A | 0.0 | 0.2 | `[0.5,0.7,1.0]` | 1.1 | 成功 |
| B | 0.1 | 0.4 | `[0.9]` | 1.0 | 成功，只有 1 个输出 |
| C | 0.2 | 0.7 | `[1.4,2.0]` | 2.1 | 成功 |
| D | 0.3 | 0.8 | `[]` | 2.3 | 失败 |

这些值独立构造，目的是暴露常见统计问题。报告保留 `record_type=simulation` 和完整输入，运行指标代码不会把它变成真实性能数据。

## 2. TTFT：等到第一个 token 多久

```text
TTFT = 首个输出 token 时间 - 请求提交时间
```

默认例子：A 为 `0.5-0.0=0.5s`，B 为 `0.9-0.1=0.8s`，C 为 `1.4-0.2=1.2s`。D 没有输出 token，TTFT 为 `null`，不是 0。

TTFT 包含排队、首次处理以及所选观测位置之前的传输/调度等开销，不能直接当作 Prefill kernel 时间。若采集到的是第一个有内容的网络事件，应称“首内容延迟”，并记录测量口径。

### 排队和端到端延迟

```text
queue = 开始服务 - 提交
E2E = 完成 - 提交
```

A 的 queue 为 0.2s，E2E 为 1.1s。最后 token 在 1.0s，但完成事件是 1.1s；这 0.1s 仍属于 E2E，不属于相邻 token 间隔。

## 3. ITL 与 TPOT：逐间隔和每请求平均

对输出时刻 `t0,t1,…,t(N-1)`：

```text
ITL_i = t_i - t_(i-1)，i=1…N-1
TPOT = (t_(N-1)-t0)/(N-1)，要求 N>=2
```

| 请求 | ITL | TPOT |
| --- | --- | --- |
| A | `[0.2,0.3]` 秒 | `(1.0-0.5)/2=0.25s` |
| B | `[]` | `null`，没有相邻 token |
| C | `[0.6]` 秒 | `(2.0-1.4)/1=0.6s` |

**为什么分母是 N−1？** N 个 token 之间只有 N−1 个间隔。第一个 token 的等待已由 TTFT 描述。如果使用 `(finish-arrival)/N`，会把排队、首 token 和完成尾部混入另一个量。

### TPOT 均值不等于所有 ITL 的均值

默认成功请求中，TPOT 有两个样本：`[0.25,0.6]`，均值 0.425s。ITL 有三个样本：`[0.2,0.3,0.6]`，均值约 0.366667s。

前者每个请求各有一票；后者每个 token 间隔各有一票。长输出请求在第二种口径中权重更高，两者不同是正常结果。字段分别为 `distributions.tpot_s` 与 `distributions.itl_s`。

### 网络 chunk 不能直接当作一个 token

一个流式事件可能包含多个 token，甚至只有角色元数据。若只能记录 chunk 到达时间，不能把 chunk 间隔直接称为模型 ITL，也不应为一个 chunk 内的 token 编造不同时间。

本工具要求 `timestamp_kind=per_token_same_clock`，拒绝 `stream_chunk`。它能检查字段和值，无法替你证明采集过程真的产生了逐 token 时间；原始日志和采集器仍需要人工核对。[vLLM 指标设计](https://docs.vllm.ai/en/stable/design/metrics/)

## 4. 吞吐先确定同一个测量窗口

本课窗口定义：

```text
begin = 所有请求中最早提交时间
end = 所有请求中最晚完成时间，包含失败请求
duration = end-begin
```

默认 `duration=2.3-0.0=2.3s`。在此窗口内：

```text
请求总数 = 4，成功数 = 3，失败数 = 1
成功输出 token 数 = 3+1+2=6
成功率 = 3/4 = 75%
成功请求吞吐 = 3/2.3 ≈ 1.304348 request/s
输出 token 吞吐 = 6/2.3 ≈ 2.608696 token/s
```

失败请求即使输出了部分 token，本工具也不将其计入“成功输出 token 吞吐”，但保留该请求的事件和失败数量。这是明确的统计契约，不是所有基准工具的唯一口径。

不能把各请求的 E2E 相加作为此处的窗口。并发请求在时间上重叠；相加得到的是请求时间总量。也不能平均各请求的 token/s 来替代窗口内总吞吐。

## 5. Goodput：成功且达到延迟目标的请求

先确定目标再看结果。本例默认：

- TTFT ≤ 0.9s。
- 对至少两个输出 token 的请求，TPOT ≤ 0.35s。
- 只有一个输出 token 时，TPOT 不定义，本例只应用 TTFT 条件。
- 请求必须成功。

A、B 满足目标；C 超过目标；D 失败。因此：

```text
good_requests = 2
Goodput = 2/2.3 ≈ 0.869565 request/s
占全部请求的达标比例 = 2/4 = 50%
```

若只对成功请求计算达标比例，则是 `2/3`，但不能把它替换成“全部请求中有 66.7% 满足服务承诺”。报告使用 `good_fraction_all_requests` 明示分母。

重新设定目标：

```bash
python -m zero2vllm metrics-demo --ttft-slo 0.4 --tpot-slo 0.2 --output .local/runs/part-01-inference/04-performance-metrics/metrics-strict.json
```

成功率、吞吐和原始延迟应完全不变，达标请求数变为 0。这条命令只重新统计，没有优化服务器。SLO 是服务目标，不能为了让结果好看而事后悄悄放宽。

## 6. P50/P95/P99：样本与插值方法

本代码使用排序后的线性插值，零基下标为：

```text
index = (n-1)*p
lo=floor(index)，hi=ceil(index)
percentile = sorted[lo] + (sorted[hi]-sorted[lo])*(index-lo)
```

TTFT 样本 `[0.5,0.8,1.2]` 中，`n=3`：

```text
P50: index=1      → 0.8
P95: index=1.9    → 0.8+(1.2-0.8)*0.9=1.16
P99: index=1.98   → 1.192
```

这是三个教学样本的插值，**不是已经获得稳定的线上尾延迟结论**。记录样本数，真实测量应覆盖足够请求和重复轮次。其他库可能采用 nearest-rank、不同插值或直方图估算，必须先统一算法再对比。

### 不要平均多轮 P99

第一轮样本 `[1,1]`，第二轮 `[100]`，各自 P95 为 1 与 100，平均得到 50.5。合并样本 `[1,1,100]` 的 P95 却是 90.1。

如果要给出总体分位数，应合并原始样本再计算；如果报告“每轮 P95 的中位数”，就明确这是轮间汇总，不能称作合并请求 P95。

## 7. 自己提供事件数据

复制示例文件到 `.local/runs/part-01-inference/04-performance-metrics/my-events.json`，修改一个变量后运行：

```bash
python -m zero2vllm metrics-demo --input .local/runs/part-01-inference/04-performance-metrics/my-events.json --output .local/runs/part-01-inference/04-performance-metrics/my-metrics.json
```

需要保留的字段：

| 字段 | 含义与约束 |
| --- | --- |
| `schema_version` | 1 |
| `record_type` | 教学数据为 `simulation`；真正设备记录才可使用 `measured_gpu` 并附外部证据 |
| `timestamp_kind` | `per_token_same_clock` |
| `requests[].id` | 非空且唯一 |
| `arrival_s/work_started_s/finished_s` | 非负有限数，提交 ≤ 开始 ≤ 完成 |
| `token_times_s` | 非递减数组，元素位于开始与完成之间 |
| `status` | `ok` 或 `error`；本课成功请求至少有一个输出 token |

时间不能填字符串、毫秒混入秒、NaN 或无穷大。相等的 token 时间允许表示观测精度有限，但需要说明采集含义。读取不存在文件、非法 JSON、事件倒序均以错误码 2 退出。

## 8. 练习与答案

### A. 手算 A 的所有量

<details>
<summary>参考答案 A</summary>

queue=0.2s，TTFT=0.5s，E2E=1.1s，ITL=[0.2,0.3]s，TPOT=0.25s。完成尾部为 `1.1-1.0=0.1s`。

</details>

### B. 把 D 的完成时间改成 3.0s

哪些值改变？先预测，再修改副本运行。

<details>
<summary>参考答案 B</summary>

窗口变为 3s，成功输出吞吐变为 `6/3=2 token/s`，Goodput 变为 `2/3 request/s`。A/B/C 的 TTFT、TPOT 和分位数不变，成功率仍为 75%；D 的 E2E 增加。

</details>

### C. 给失败请求 D 添加 token 时间 `[1.5,1.6]`

<details>
<summary>参考答案 C</summary>

D 的局部 token 数与时间指标可以计算，但状态仍失败，不进入成功延迟分布、不贡献成功输出 token 吞吐，也不满足 Goodput。默认窗口不变。

</details>

### D. 四个样本 `[0,10,20,30]` 的 P95

<details>
<summary>参考答案 D</summary>

下标 `3*0.95=2.85`，结果 `20+(30-20)*0.85=28.5`。必须说明使用的是线性插值，不能只说“P95 就是第 95 个请求”。

</details>

## 9. 工程进阶与完成标准

真实 A100/C500 实验另记录模型/Tokenizer 版本、实际 token 长度、预热与测量区间、请求到达率/并发限制、错误和原始日志。客户端 E2E、服务端排队、GPU kernel 三个时间口径分别保存。字段名相同不能证明工具版本的定义相同，阅读所固定版本的统计实现。

- [ ] 手算全部四个请求，说明 `null` 不等于 0。
- [ ] 解释 TPOT 均值和 ITL 均值为何不同。
- [ ] 用统一窗口重算吞吐和 Goodput，不丢掉失败请求。
- [ ] 复算 P95，并报告样本数及算法。
- [ ] 修改一份事件副本，预测和验证变化。
- [ ] 保持模拟数据的来源标记，说明真实时间戳如何采集。

测试见 [test_part1.py](../../../../tests/reference/test_part1.py) 的 `MetricsTests`。进一步阅读[实验规范](../../../engineering/experiments.md)与[vLLM 指标设计](https://docs.vllm.ai/en/stable/design/metrics/)。资料查阅：2026-09-15。
