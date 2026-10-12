# 2.3 实验：跟踪请求、取消与资源回收

[正文](README.md) · [练习](exercises.md) · [本轮结果](results/README.md)

所有命令在仓库根目录执行。基础实验只用 Python 3.11+ 标准库，不需要下载模型。真实模型扩展复用上一节的虚拟环境和快照。

## 1. 选择解释器

已经做完 2.2 的读者，继续使用原环境：

```bash
# Windows Git Bash
source .venv/Scripts/activate

# Linux Bash
source .venv/bin/activate
```

只执行适合自己系统的那一行。PowerShell 可以直接用 `.\.venv\Scripts\python.exe` 替换下文的 `python`，无需激活。

基础实验也可以使用系统 Python。下面这条检查不依赖 PyTorch：

```bash
python -m unittest tests.integration.test_lifecycle -v
```

本轮 24 项测试通过。需要确认标准库边界时，可加 Python 的 `-S` 选项跳过 site-packages：

```bash
python -S -m unittest tests.integration.test_lifecycle -v
```

这只适用于本节基础实验；真实模型扩展需要第三方依赖，不能加 `-S`。

## 2. 先手动推进两条请求

```bash
python docs/chapters/part-02-engine/03-request-and-cache-lifecycles/examples/walk_requests.py
```

观察四个时刻：

1. 提交 A、B：都在等待，已用额度为 0。
2. A 准入并生成一项：已用 6，A 的实际 K/V 长度为 3，B 仍在等待。
3. 取消 A：A 保留一个输出，K/V 和预留额度清零。
4. 再次尝试准入 B：B 得以运行，正常结束后总已用额度回到 0。

这里每条请求最多生成四项，但只预留六个输入位置：`3+4-1=6`。改成七个或五个之前，先解释多出来或少掉的位置分别意味着什么。

示例结束时应出现：

```text
PASS: cancellation returned capacity; b completed.
```

## 3. 在计算返回前发出取消

```bash
python docs/chapters/part-02-engine/03-request-and-cache-lifecycles/examples/cancellation_boundary.py
```

预期事件：

```text
During forward: cancellation latched; KV=3, reserved=6.
1 submitted waiting
2 admitted running
3 forward_started running
4 cancel_requested running
5 forward_returned running
6 finalized cancelled
PASS: no token committed after cancellation; release happened after return.
```

重点看三处：收到取消时状态仍为 `running`；计算返回之前额度仍为 6；整个事件序列没有 `token_committed`。

这是同线程回调实验，用来固定先后顺序。它不发起 HTTP 请求，不创建并发线程，也不证明 GPU kernel 可以被立即中断。

## 4. 一次运行全部基础案例

```bash
python -m scripts.experiments.lesson0203
```

默认结果位置：`.local/runs/part-02-engine/03-request-and-cache-lifecycles/summary.json`。

| 案例 | 应观察的结果 |
| --- | --- |
| `length` | 四项生成与完整前缀参考相同；缓存峰值 6，终态缓存 0 |
| `eos` | 将首个参考 token 指定为 EOS；生成一项后结束 |
| `zero_output` | 无前向、无上下文、无运行额度 |
| `cancel_waiting` | 不分配缓存就进入取消终态 |
| `cancel_running` | 保留已经提交的两项，清理缓存 |
| `cancel_in_forward` | 取消先记录，返回后收尾，本次 token 不提交 |
| `model_error` | 在真实 KV 追加后注入异常，进入 FAILED 并回收 |
| `initialization_error` | 创建上下文失败，撤销已预留额度 |
| `capacity_wait_resume` | B 暂缓准入，A 取消后再次准入 B |
| `interleaved_isolation` | 手动交错 A、B，两者分别与独立重算一致 |
| `engine_close` | 同时关闭运行和等待中的请求；重复关闭不重复释放 |

Tiny 模型固定 `vocab_size=16`、`width=8`、`max_context=32`、`seed=7`。默认输入 `[1,2,3]` 的四项输出是 `[14,1,5,12]`；它们是教学模型 ID，不代表自然语言质量。交错案例 B 使用 `[4,2]`，输出 `[7,14,1]`。

只有所有案例的状态、输出和资源断言同时满足，基础部分才通过。默认输出的 `real_checkpoint.verification=not_run`，不会把尚未运行的真实模型扩展标成通过。

## 5. 复用 2.2 的 SmolLM2

先确认已经完成 [2.2 的安装与下载](../02-loading-real-model-weights/lab.md)。如果尚未准备，运行：

```bash
python -m scripts.experiments.lesson0202 --download
```

本节真实扩展只读取已有快照，并核对上一节的文件摘要；它不隐式下载：

```bash
python -m scripts.experiments.lesson0203 --real-checkpoint
```

基础 11 案例仍会运行，随后增加：

```text
PASS real_cancel_isolation
PASS real_reuse_after_cancel
Real checkpoint: executed; summary: ...
```

两个输入分别为 `Once upon a time` 和 `The capital of France is`，最多四项输出。A、B 各推进一步后取消 A，再让 B 完成；随后复用同一模型运行一条新请求。

| 观察对象 | 本轮结果 |
| --- | --- |
| 被取消的 A | 保留 `[28]`，状态 CANCELLED |
| 正常完成的 B | `[260,3575,282,260]`，与 2.2 循环相同 |
| 重新运行故事输入 | `[28,665,436,253]`，与 2.2 循环相同 |

这里没有 Llama 增量 KV，因此 `cache_kind=none_full_prefix_recompute`，缓存峰值为 0。不得用这些数字声称测到了真实模型的 KV 回收或节省了显存。真实扩展检查的是请求封装、取消隔离与共享模型的继续可用性。

本轮环境为 Windows、Python 3.11.9、PyTorch 2.7.1+cpu；使用 CPU float32 和 2 个线程。Linux CI 只配置离线基础检查，运行是否成功以实际记录为准。A100/C500 未在本节实测。

## 6. 报告怎样读

```bash
python -m scripts.experiments.lesson0203 --output-dir .local/runs/part-02-engine/03-request-and-cache-lifecycles/my-run
```

参数只有两个：`--real-checkpoint` 开启已有真实权重扩展，`--output-dir` 更改保存目录。重复使用同一目录会覆盖同名 `summary.json`；仓库中的公开基线应保留原日期与源码指纹。

| 字段 | 含义 |
| --- | --- |
| `source` | 执行时 Git commit、工作区修改状态和实际源码摘要 |
| `tiny_model` | 随机教学模型的维度、上下文与种子 |
| `cases[].requests[]` | 原始输入、输出上限、EOS、终态、错误、最终额度与缓存峰值 |
| `cases[].timeline[]` | 指定观察时刻的请求快照与全局额度 |
| `cases[].events` | 每条请求的先后事件；序号不是性能计时 |
| `cases[].assertions` | 独立记录每条验收条件，便于发现只改状态没回收的情况 |
| `real_checkpoint` | 是否执行、权重摘要、输入和与 2.2 循环的比较 |
| `passed` | 本次实际执行的案例是否全部通过；同时查看真实扩展的执行状态 |

请求信息与事件会保存输入 token ID。这里使用公开的合成输入；把运行器用于自己的文本时，应先确认报告适合公开，不直接提交整份个人运行目录。

## 7. 自己做两次错误实验

先在临时副本中试验，完成后恢复代码：

1. 将 `_finish` 中的额度归还去掉。观察 `capacity_wait_resume`：A 已经取消，B 却仍无法准入，`budget_returned` 也应失败。
2. 将取消判断移到 `r.outputs.append(token)` 之后。观察 `cancel_in_forward`：计算中收到取消后，结果仍多了一项。

这些错误不会一定让 Python 崩溃，因此需要同时检查事件、输出与资源。提交前恢复实现，再运行：

```bash
python -m unittest tests.integration.test_lifecycle -v
python -m scripts.experiments.lesson0203
python -m scripts.docs.course --check
python -m scripts.docs.layout
```

完整错误注入由测试自动在内存中完成，不会修改已经下载的模型权重。

## 常见问题

| 现象 | 排查方向 |
| --- | --- |
| `Admit the request before stepping it` | 还在 WAITING；先尝试准入并检查返回值 |
| `admit` 返回 False | 看状态：可能暂缺额度、已经准入/终态，或初始化失败；读取 view 与事件 |
| `Request can never fit...` | 单条请求所需额度大于总容量；缩短输出上限或调整教学预算 |
| 取消后进程内存没有明显下降 | 检查请求引用与实际缓存；不要把分配器/RSS 当作唯一的回收证据 |
| `No snapshot receipt` | 先完成 2.2；真实扩展不会自动联网下载 |
| 重复提交同一 ID 失败 | 旧结果仍在；先读取再 forget，或使用新的唯一 ID |
| `Nested execution is not supported` | 在一次 step 中又调用了 step/admit；本节只允许受控的取消回调，不支持嵌套执行 |
