# 3.2 实验：给每轮调度记一本账

[正文](README.md) · [练习](exercises.md) · [源码定位](source-map.md) · [运行记录](results/README.md)

本实验有两种证据：**运行我们自己的计数程序**，检查数值和状态；**读取 vLLM 固定源码**，检查文件和函数位置。两者都不执行上游 vLLM、不加载模型，也不证明 GPU 支持或吞吐提升。

## 1. 环境与工作目录

在包含 `pyproject.toml` 的 zero2vllm 仓库根目录运行。需要 Python 3.11+；PowerShell、Git Bash、Linux 均可使用下面的单行命令。如果你的系统只有 `python3`，把命令开头的 `python` 替换掉。

```bash
python --version
python -m scripts.experiments.lesson0302 --help
```

已有虚拟环境可以继续使用。本实验仅用标准库，无需安装 vLLM、PyTorch 或其他包。Windows 中文终端显示不正确时，可用 `python -X utf8 ...`。

## 2. 第一次运行：不要改参数

```bash
python -m scripts.experiments.lesson0302
```

预期打印以下 10 行 PASS，最后再次说明这是计数模拟：

```text
mixed: PASS
chunked: PASS
sequence_limit: PASS
head_blocking: PASS
preemption: PASS
self_preemption: PASS
prefix_boundary: PASS
prefix_capacity: PASS
shared_release: PASS
eviction: PASS
```

结果保存到 `.local/runs/part-03-vllm/02-scheduling-kv/summary.json`。默认不会联网；也不要求已经下载官方源码。可以用 `--output-dir` 为每次运行单独保存记录，同一目录下的 `summary.json` 会被新运行覆盖。

### 先读结果的外层

| 字段 | 应怎样理解 |
| --- | --- |
| `passed` | 所选案例及可选源码检查是否通过 |
| `verification_level` | `teaching_count_simulation`，表示教学计数模拟 |
| `executed_upstream_code` / `model_executed` / `gpu_validated` | 均为 false，不能把 PASS 当成模型或设备执行结果 |
| `course_source.source_sha256` | 课程代码和输入数据的摘要，帮助识别你是否运行了不同版本 |
| `cases` | 每个案例的具体状态；不能只看最后一个布尔值 |

如果只想查看一项：

```bash
python -m scripts.experiments.lesson0302 --case preemption --output-dir .local/runs/part-03-vllm/02-scheduling-kv/preemption
```

## 3. 三轮混合调度：先预测，再看输出

```bash
python docs/chapters/part-03-vllm/02-scheduling-and-kv-management-in-vllm/examples/walk_schedule.py
```

初始 A=N5/C4，B=N7/C0；预算 4、阈值 0，块长 4。预期：

| 轮次 | A 安排量 | B 安排量 | 反馈后的 A：N/C | 反馈后的 B：N/C | 剩余预算 |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | 3 | 6/5 | 7/3 | 0 |
| 2 | 1 | 3 | 7/6 | 7/6 | 0 |
| 3 | 1 | 1 | 8/7 | 8/7 | 2 |

输出中的 `新块` 是新增物理块编号，不是全部历史块表。A 第一轮跨过第 4 个位置，需要新块；第二轮写入已有尾块，不需要继续新增。块编号来自教学账本，并不保证和真实 vLLM 某次运行的编号一样。

在 JSON 中，依次看 `before`、`scheduled`、`after_schedule`、`after_feedback`。`synthetic_output_counts` 是教师设定的反馈：只有追平的请求给 1，部分 Prefill 给 0。这不是由模型预测的 token，更不是一个真实性能测量。

### 实验 A：只给一个 token 预算

```bash
python docs/chapters/part-03-vllm/02-scheduling-and-kv-management-in-vllm/examples/walk_schedule.py --token-budget 1 --rounds 3
```

三轮都由 A 用完预算，B 的 C 保持 0。这里 A 被设定为持续生成，所以只能据此解释这个有限回放中 B 被推迟；不能声称真实系统的所有 B 都会永久饥饿。

### 实验 B：增大总预算，同时限制单请求块长

```bash
python docs/chapters/part-03-vllm/02-scheduling-and-kv-management-in-vllm/examples/walk_schedule.py --token-budget 8 --threshold 2 --rounds 1
```

预期 A=1、B=2，剩余预算=5。阈值是单请求安排量的限制，不能看到剩余预算就继续在同一轮给 B 追加一块。

### 实验 C：独立长提示

```bash
python -m scripts.experiments.lesson0302 --case chunked --output-dir .local/runs/part-03-vllm/02-scheduling-kv/chunked
```

9 个提示 token、预算 4，对应三轮 `[4,4,1]`。前两轮的输出计数为空，第三轮才给出假设的新 token。请用“最后位置 logits 还没有完成”解释前两轮为什么没有用户输出。

## 4. 两种“预算明明没用完”

```bash
python -m scripts.experiments.lesson0302 --case sequence_limit --output-dir .local/runs/part-03-vllm/02-scheduling-kv/seq-limit
python -m scripts.experiments.lesson0302 --case head_blocking --output-dir .local/runs/part-03-vllm/02-scheduling-kv/head-blocking
```

| 案例 | 参数要点 | 应观察的结果 |
| --- | --- | --- |
| `sequence_limit` | 预算 4；最多 1 条运行请求；A 已在运行 | A 用 1，剩余 3，B 仍等待；`stop_reason=max_seqs` |
| `head_blocking` | 预算 8；只有 1 个可用块；队首需 8 token，后面需 1 token | 两条都没有被调度；剩余预算 8；`waiting_head_no_blocks` |

第二项没有偷偷给队首分一半，也没有跳过它去调度后面的短请求。队首的 q 先按预算得到 8，容量检查需要 2 块，失败后退出等待循环。这是固定版本普通主路径的一个具体行为。

## 5. 抢占：释放的是状态，保留的是请求

```bash
python -m scripts.experiments.lesson0302 --case preemption --output-dir .local/runs/part-03-vllm/02-scheduling-kv/preemption
```

本例关闭前缀缓存，配置 `num_blocks=3`，其中编号 0 保留；A、B 各占一块。

逐项核对：

1. 第一轮的 `preempted` 为 `["B"]`。
2. B 的 known 仍是 5，computed 变成 0，status 变成 PREEMPTED。
3. A 本轮安排 1，剩余预算 3；这一轮不接回 B。
4. 实验显式设定 A 在反馈中结束，释放其两块。
5. 第二轮 B 的 kind 为 `resumed`，从 C=0 安排 4 个输入位置；它还差第 5 个位置，不能把它当作只需计算 1 个 token 的普通 Decode。

再运行极小容量：

```bash
python -m scripts.experiments.lesson0302 --case self_preemption --output-dir .local/runs/part-03-vllm/02-scheduling-kv/self-preemption
```

只有一个可用块，而 A 要从 4 个位置扩到 5 个。A 抢占自己后结束本轮，本轮没有计算工作。此案例只回放这一轮，**不宣称请求随后能完成**。容量不足不能被抢占机制消除。

## 6. 前缀边界与容量：这是独立块账本

下面四项使用显式确认的前缀块编号，不做哈希查找、不执行 Attention；和前面的关闭前缀缓存调度器分开运行。

```bash
python -m scripts.experiments.lesson0302 --case prefix_boundary --output-dir .local/runs/part-03-vllm/02-scheduling-kv/prefix-boundary
python -m scripts.experiments.lesson0302 --case prefix_capacity --output-dir .local/runs/part-03-vllm/02-scheduling-kv/prefix-capacity
python -m scripts.experiments.lesson0302 --case shared_release --output-dir .local/runs/part-03-vllm/02-scheduling-kv/shared-release
python -m scripts.experiments.lesson0302 --case eviction --output-dir .local/runs/part-03-vllm/02-scheduling-kv/eviction
```

| 案例 | 先写下你的预测 | 再对照结果 |
| --- | --- | --- |
| `prefix_boundary` | 长度 7、8、9，B=4，缓存前两块 | 复用长度为 4、4、8 |
| `prefix_capacity` | 9 token 命中 2 块；空闲队列只有这 2 块 | 分配失败，前后状态完全相同；增加第 3 块后成功 |
| `shared_release` | 两条请求共享块 1，分别持有块 2、3 | 块 1 的引用从 2→1→0；A 结束时不能把块 1 放回空闲队列 |
| `eviction` | 唯一可用块有旧缓存身份，现在分配给不同内容 | 同一物理块被使用，旧缓存身份移除 |

`references` 数组下标是物理块编号；编号 0 是特殊占位，其零值不能按普通空闲块处理。`free_ids` 是当前可覆盖的候选队列，可能包含仍有前缀身份的零引用块；`cached_ids` 不表示所有块都被请求持有。

练习时先回答“空间够不够”，再回答“省了多少计算”。它们不是同一个问题。

## 7. 回到真实源码

### 首次准备：显式联网

```bash
python -m scripts.experiments.lesson0302 --fetch-source
```

下载来自 vLLM 官方仓库固定完整提交的 6 个文件（含 LICENSE），默认复用 3.1 的源码保存位置。已经存在的正确文件不重复下载；已有文件若与摘要不匹配，会报错并保留，不会覆盖你的修改。

该目录是**阅读快照的文件子集**，不是完整可安装的 vLLM 工程。

### 以后重复检查：离线

```bash
python -m scripts.experiments.lesson0302 --inspect-source
```

预期额外出现：

```text
Static source inspection: PASS (11 functions)
```

打开结果的 `source_inspection`，应看到：6 份文件摘要一致、11 个函数和调用表达式检查通过、1 个 SchedulerOutput 结构字段检查通过。你也可以指定已有完整 checkout：

```bash
python -m scripts.experiments.lesson0302 --inspect-source --source-dir .local/sources/vllm-v0.8.5
```

如果目录指向其他版本，即使碰巧也有 Scheduler 类，也不能当作本节同一份源码。工具会比较内容摘要。

### 你仍需要亲手读的分支

语法树核验能证明函数和调用表达式存在，但不能证明某个输入会走哪个分支。按[源码定位表](source-map.md)，逐项找出：

1. 运行队列中的 q 如何受到阈值、总预算和长度上限限制。
2. 分配失败时的 `running.pop()` 和 `waiting.appendleft()`。
3. 等待队首分配失败后为什么是退出循环。
4. 命中块中的零引用块数为什么要从空闲数量中扣掉。
5. `touch()` 为什么在取得新块之前。
6. `num_computed_tokens` 在构造输出前还是后更新。
7. 结束清理与抢占清理有哪些不同。

把答案写成你的阅读笔记；不要仅粘贴一长段函数。

## 8. 验证自己的改动

```bash
python -m unittest tests.reference.test_scheduling -v
python -m scripts.docs.course --check
python -m scripts.docs.layout
```

新增测试共 25 项，包括失败时账本保持一致、共享块活跃引用不能归还、随机分配/释放容量守恒、抢占后历史长度保留、错误反馈拒绝、默认离线及错误源码目录等。它们验证本课工具；已有真实 CPU 模型测试继续保留在全仓库回归中。

若修改了脚本，应重新保存实验结果，而不是手工把 JSON 中的 `passed` 改为 true。代码和结果不一致时，以重新运行的记录为准。

## 9. 常见问题

| 现象 | 检查方向 |
| --- | --- |
| `No module named scripts` | 是否在仓库根目录运行，Python 是否是 3.11+ |
| 源码缺失 | 默认模拟仍可运行；需要源码核验时先加 `--fetch-source`，网络失败可稍后重试 |
| 源码摘要不一致 | 检查是否误用了另一个提交；保留你改过的源码，另选新目录准备快照 |
| 3 个块只能用 2 个 | 本节保留编号 0 的 null 块 |
| 尚有 token 预算却没有新请求加入 | 检查序列上限、KV 容量、队首阻塞和本轮抢占 |
| 部分 Prefill 的输出列表为空 | 提示末尾尚未算到，不是请求已经结束 |
| 计数案例 PASS，却不能推断延迟 | 程序没有模型执行、墙钟服务负载或 GPU 计时 |

最终交付一张三轮账本、一份前缀容量手算、一份抢占状态表，并用自己的话解释它们分别对应哪些源码分支。
