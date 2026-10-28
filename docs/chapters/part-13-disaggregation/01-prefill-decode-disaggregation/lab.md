# 13.1 实验：审计一次PD方案对照

[正文](README.md) · [练习与答案](exercises.md) · [输入轨迹](data/comparison.json) · [默认结果](results/baseline.json)

本实验是**离线请求轨迹分析**，不是GPU模拟、流量发生器或Connector实现。默认6个场景均为原创合成数据。程序检查声明的对照条件，逐请求计算延迟与联合SLO，再报告完成吞吐和合格吞吐。

## 1. 环境与运行位置

在包含`pyproject.toml`的仓库根目录运行，Python≥3.11，仅使用标准库，无需安装PyTorch/vLLM，不下载模型或连接GPU。

```bash
python --version
python -m scripts.experiments.lesson1301
```

已有项目虚拟环境时可先激活；Windows PowerShell也可直接使用`.venv/Scripts/python.exe`替代`python`。Git Bash/Linux命令相同。默认输入在本节`data/`，默认输出在项目本地区，执行不会改动公开样例。

预期控制台核心输出：

```text
Source: synthetic; observation: 1s
name                          complete qualified goodput(req/s)
mixed_uncontrolled                  4         2          2.000
mixed_chunked                       4         3          3.000
pd_fast_transfer                    4         4          4.000
pd_slow_transfer                    4         0          0.000
pd_first_token_from_p                4         0          0.000
pd_pool_imbalance                   3         2          2.000
```

列间空格可能随场景名长度变化。精确字段以输出JSON为准：

```text
.local/runs/part-13-disaggregation/01-pd-boundaries/comparison.json
```

退出码0只说明输入通过协议检查、报告成功生成；goodput可以为0，它不代表实验程序失败。非法输入或不可写输出使用退出码2。

## 2. 先核对一个请求

找到`mixed_uncontrolled`中的`r1`：发送时刻0，token时刻`[100,130,260,290]`。

1. TTFT=100ms。
2. 三个ITL分别为30、130、30ms，最大值130ms。
3. TPOT=(290−100)/3≈63.33ms。
4. 它成功完成，但同时违反TPOT≤50和最大ITL≤80两个条件，因此不计入goodput。

JSON中的`violations`应为`["tpot_ms", "max_itl_ms"]`。`qualified`应为`false`，`status`仍为`completed`；成功结束和满足SLO要分开记录。

对`pd_pool_imbalance`，最后一条请求只观察到1个token。它的TTFT可计算，TPOT和最大ITL为`null`，状态是`incomplete`。报告保存4个offered、3个completed、2个qualified，联合达标比例2/4=0.5。部分token也计入`observed_output_tokens_per_second`，因此该字段是13token/s，不能解释成成功请求的产出。

## 3. 不改轨迹，只改SLO

收紧首token要求：

```bash
python -m scripts.experiments.lesson1301 --ttft-ms 150 --output .local/runs/part-13-disaggregation/01-pd-boundaries/strict-ttft.json
```

6个场景的合格请求数应依次为`[2,1,0,0,0,0]`。`pd_fast_transfer`仍完成4个请求，但TTFT都是160ms，因此goodput变为0。这个变化来自评价条件，而不是服务变慢。

放宽TTFT至300ms：

```bash
python -m scripts.experiments.lesson1301 --ttft-ms 300 --output .local/runs/part-13-disaggregation/01-pd-boundaries/relaxed-ttft.json
```

合格请求数应为`[3,4,4,4,0,3]`。`pd_first_token_from_p`仍因输出间隔超标而为0；未完成的请求仍不能计入goodput。

单独检查平均间隔怎样掩盖停顿：

```bash
python -m scripts.experiments.lesson1301 --tpot-ms 70 --output .local/runs/part-13-disaggregation/01-pd-boundaries/relaxed-tpot.json
```

此时`pd_first_token_from_p`的TPOT满足70ms，但最大ITL=140ms仍违反80ms限制。查看每请求`violations`，应只剩`max_itl_ms`。不要仅看场景总行就结束分析。

## 4. 输入协议是什么

唯一解析实现见[disaggregation.py](../../../../zero2vllm/reference/disaggregation.py)，命令入口见[lesson1301.py](../../../../scripts/experiments/lesson1301.py)。

| 字段 | 含义和限制 |
| --- | --- |
| schema_version | 当前为1 |
| source_kind / provenance | `synthetic`或`measured`，以及数据来源说明；标签本身不是实测证明 |
| clock | 固定为`client_monotonic_ms`，所有时间单位都是ms，来自同一客户端时钟 |
| contract | 所有场景共享的模型/tokenizer、精度、负载、生成参数、缓存策略、GPU型号/总数和runtime身份 |
| window | 正观察时长；请求到达在[start,end)，token与结束事件允许等于end |
| slo | `ttft_ms`、`tpot_ms`、`max_itl_ms`，非负有限数；阈值相等算达标 |
| workload | 每条请求的唯一ID、到达时刻、输入和要求输出token数；本协议要求输出至少2个token |
| runs[].allocation | 混部使用mixed角色，PD使用prefill/decode角色；GPU总数必须等于统一预算 |
| runs[].configuration | 此场景配置与允许变化的因素；实测需记录并行布局、调度预算、Connector/网络等 |
| runs[].requests | 每个offered ID都必须出现一次，不能丢掉失败请求 |
| token_arrivals_ms | 非递减token到达时间；与发送时刻、结束时刻及观察窗口一致 |
| status / end_ms | 完成/失败等状态及观察到该状态的时刻；未完成记录的end_ms必须是观察截止时刻 |

支持的状态为`completed`、`rejected`、`error`、`timeout`、`cancelled`和`incomplete`。完成请求必须恰好达到要求的固定输出长度；拒绝请求不能已经输出token。失败可以发生在部分token返回以后，但仍不计入goodput。

**为什么暂不支持EOS提前结束和单token请求？** 本节先用固定输出长度隔离比较条件，让TPOT有明确定义。不是说这些生产请求非法，而是输入协议尚未覆盖它们。真实业务转换器需要先定义合法终止和SLO适用范围；不能为了通过校验伪造token。

程序可以拒绝缺记录、错顺序和不等卡数，但无法验证输入声明是否真实。它也不检查模型输出质量。实际采集还必须保存版本、配置、内容身份和正确性基线。

## 5. 观察窗口和统计总体

默认窗口是[0,1000]ms，1秒作为所有场景共同分母，不使用“最快场景最后一个请求结束时刻”。输入中没有窗口之前到达的请求；到达恰好等于end的请求不属于本次offered。截止时仍在执行的请求标为`incomplete`，保存截止前的部分token。

不要把end之后的完成时刻硬塞入本次窗口。如果要观察排空，将所有场景的窗口统一延长，并补齐真实状态；或另建到达窗口与排空窗口协议。不能一边扩大完成计数，一边保留原来较短的时间分母。

这是有限批请求的观察协议，含启动与截尾效应，**不估算稳态最大容量**。正式13.2/13.3还需稳定供压、预热、充分时长、复测与失败率。4条样本不支持可信的生产P99结论。

报告的`completed_request_percentiles`只覆盖成功完成请求，采用nearest-rank：排序后取第ceil(p×n)项，序号从1开始。例如`[100,110,120,230]`的P50为110，P95/P99为230。没有完成请求时返回`null`，不用0表示“没有延迟”。其中`max_itl_ms.p95`是**每请求最大ITL的P95**，不是全体token的ITL P95。

## 6. 常见错误与测试

```bash
python -m unittest tests.reference.test_disaggregation -v
```

22项测试覆盖手算指标、联合达标、缺失失败请求、不同GPU预算、窗口边界、部分token、阈值相等、NaN/Infinity、时间顺序、固定输出契约及命令入口。

| 现象 | 应检查什么 |
| --- | --- |
| `missing offered requests` | 是否只导出了成功请求，丢掉超时或拒绝 |
| `unequal GPU budget` | 是否把1P+2D与两个混部GPU比较 |
| `completed output length mismatch` | 是否发生EOS提前结束，或把网络chunk误当token |
| `token times must be ordered before end` | 单位/时钟是否混用，是否把服务端时间混进客户端事件 |
| `incomplete records must end at observation cutoff` | 未完成记录是否真正按统一窗口截尾 |

流式HTTP/SSE的一次chunk可能包含多个token；不能直接把chunk间隔当token间隔。本协议需要能核验的逐token到达数据，或在另一个明确标注chunk级指标的工具里分析。不要凭空均分chunk内token时间。

## 7. 换成自己的记录

先复制输入到`.local/`，保留公开fixture。补全同一负载/同一资源预算下各场景的请求记录和配置，保留所有失败；确认时间、token计数与状态契约后执行：

```bash
python -m scripts.experiments.lesson1301 --input .local/my-pd-comparison.json --output .local/my-pd-report.json
```

只有真实采集且满足协议时才标`measured`；填入的估算值仍标`synthetic`并说明构造方法。手工补出的时刻不能伪装成测量数据。`input_sha256`对输入文件的CRLF换行规范化为LF后计算，避免Git在Windows/Linux之间转换换行时导致摘要变化；其他字节保持原样。

交付时保留输入、命令、阈值、报告以及不适用范围。公开基线与收紧TTFT的结果分别见[baseline.json](results/baseline.json)、[strict-ttft.json](results/strict-ttft.json)，全部为离线合成轨迹分析。
