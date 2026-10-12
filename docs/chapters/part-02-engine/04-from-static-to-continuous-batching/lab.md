# 2.4 实验：让同一批请求走两种调度路径

[正文](README.md) · [练习](exercises.md) · [已记录结果](results/README.md)

本实验从已经可用的 2.3 请求引擎出发。你会实际运行小模型和独立 KV，观察调度集合怎样变化，并解释三种预算分别在哪里起作用。无需下载模型、安装 PyTorch 或准备 GPU。

## 1. 环境与入口

所有命令在仓库根目录执行，使用 Python 3.11 或更新版本。本轮在 Windows / Python 3.11.9 验证。

如果已经在做 2.1—2.3，沿用现有 `.venv`。新读者可以先运行 `python -m venv .venv`。激活方式如下，按自己的终端选一行：

```bash
# Windows Git Bash
source .venv/Scripts/activate
# Linux / macOS
source .venv/bin/activate
```

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

若 PowerShell 不允许激活脚本，可以直接把后续命令中的 `python` 换成 `.\.venv\Scripts\python.exe`，无需修改系统执行策略。先检查解释器：

```bash
python --version
python -S -m unittest tests.integration.test_scheduler -v
```

本节包含 23 项测试。`-S` 禁用自动加载 site-packages，验证这些代码只依赖标准库。它适用于本节，不要把它直接加到需要 PyTorch 的 2.1/2.2 命令上。

| 位置 | 用途 |
| --- | --- |
| [scheduler.py](../../../../zero2vllm/engine/scheduler.py) | FIFO 准入、轮转选取、冻结计划与边界计时 |
| [workloads.json](data/workloads.json) | 六组固定输入和参数 |
| [lesson0204.py](../../../../scripts/experiments/lesson0204.py) | 回放、独立参考对照、结构化报告 |
| [walk_batches.py](examples/walk_batches.py) | 最短的静态/连续比较 |
| [inspect_budgets.py](examples/inspect_budgets.py) | 计算额度与容量额度的区别 |

## 2. 先看四个请求如何补位

```bash
python docs/chapters/part-02-engine/04-from-static-to-continuous-batching/examples/walk_batches.py
```

输出先列 `static`，再列 `continuous`。`prefill/3` 表示送入三个 prompt token；`decode/1` 表示送入上一项输出。结尾应该出现：

```text
rounds=8; reserved=0
...
rounds=6; reserved=0
PASS: same outputs, different admission boundaries.
```

对照正文的第一张图，找到这三个位置：

1. B 在第 1 轮结束，结果记在边界 2。
2. 连续策略在第 2 轮补入 C；静态策略此时只运行 A。
3. 连续策略在第 4 轮补入 D；静态策略要等到第 6 轮。

程序比较了两种策略的全部输出 ID；实验运行器还会分别比较独立完整重算参考，避免两种策略共享同一个错误而彼此一致。

## 3. 保存完整回放

```bash
python -m scripts.experiments.lesson0204 --trace
```

默认输出保存在 `.local/runs/part-02-engine/04-from-static-to-continuous-batching/summary.json`。这个目录用于自己的实验，不覆盖课程公开基线。默认运行六组负载，每组两种策略，应全部 PASS：

| 负载 | 静态轮数 | 连续轮数 | 重点观察 |
| --- | --- | --- | --- |
| `mixed_lengths` | 8 | 6 | 短请求结束后补位 |
| `token_pressure` | 6 | 6 | RUNNING 不等于本轮被选中 |
| `fifo_capacity` | 7 | 7 | 有空闲额度，仍可能被等待队首挡住 |
| `staggered_arrivals` | 6 | 6 | 1、2、3 轮确实没有工作到达 |
| `stop_and_cancel` | 2 | 2 | 边界取消、EOS、零输出 |
| `long_prefill_turn` | 8 | 7 | 新请求首项等待与已有请求输出间隔的取舍 |

这里只列默认参数的已验证结果。修改输入和上限之后，应重新分析轨迹，不能继续要求轮数等于这张表。

### 参数含义

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--case` | `all` | 选择上表一组负载 |
| `--policy` | `both` | `static`、`continuous` 或两者 |
| `--max-running` | 使用负载配置 | 覆盖活跃请求上限 S |
| `--max-tokens-per-round` | 使用负载配置 | 覆盖每轮输入额度 T |
| `--capacity-tokens` | 使用负载配置 | 覆盖跨轮持有的逻辑容量 C |
| `--max-rounds` | `1000` | 防止异常负载无限回放；达到上限视为失败 |
| `--trace` | 关闭 | 打印逐轮计划与等待原因 |
| `--output-dir` | 本节 `.local/runs/` | 保存 `summary.json` 的目录 |

每次运行会覆盖指定目录里的同名报告。比较不同配置时使用不同输出目录。

## 4. 把三个预算分开改

先运行最小预算示例：

```bash
python docs/chapters/part-02-engine/04-from-static-to-continuous-batching/examples/inspect_budgets.py
```

第一部分第 0 轮有两个 active，却只执行 A。第 1 轮先执行 B 的两个 prompt token，再执行 A 的一个 decode token。两轮打印的计算额度都是 `3/3`，持有额度都是 `10/32`。

第二部分打印 `wait=('b', 'c')` 和 `available=2`。B 需要 5，C 只需要 1；FIFO 不越过 B。取消 B 后，C 下一轮准入。示例结束时上下文管理器会取消未完成的请求并回收资源。

接着只改一个上限，保存三份结果：

```bash
python -m scripts.experiments.lesson0204 --case mixed_lengths --max-running 1 --trace --output-dir .local/runs/lesson0204/one-slot
python -m scripts.experiments.lesson0204 --case mixed_lengths --max-tokens-per-round 3 --trace --output-dir .local/runs/lesson0204/three-tokens
python -m scripts.experiments.lesson0204 --case fifo_capacity --capacity-tokens 16 --trace --output-dir .local/runs/lesson0204/more-capacity
```

先预测再查看报告：一个活跃位置时，是否还有“某成员先结束留下空位”的现象？T 变小时，哪些活跃请求开始出现在 `skipped_budget`？C 变大后，原先的 FIFO 队首能否在第一轮就进入？

## 5. 长 prompt 与输出间隔

```bash
python -m scripts.experiments.lesson0204 --case long_prefill_turn --trace --output-dir .local/runs/lesson0204/prefill
```

连续策略的关键几轮：

```text
r=01 [b:D1 a:D1] ... skip=long
r=02 [a:D1 b:D1] ... skip=long
r=03 [long:P4] ... skip=b,a
r=04 [b:D1 a:D1 long:D1] ...
```

查看 A 的 `token_boundaries`：连续策略为 `[1,2,3,5,6,7]`，其中 3→5 间隔为两轮。静态策略为 `[1,2,3,4,5,6]`。再查看 long：连续首项在边界 4，静态首项在边界 7。用一句话同时解释两边的变化。

## 6. 停止、取消与失败诊断

```bash
python -m scripts.experiments.lesson0204 --case stop_and_cancel --trace --output-dir .local/runs/lesson0204/stops
```

应得到：运行取消保留 `[14,1]`；等待取消保留空输出；EOS 请求生成 `[14]`；零输出请求不创建上下文。后两种无首项输出的情况，其首项等待字段是 `null`。这些请求终止后预留都回到零。

下面是一条**有意失败**的命令：

```bash
python -m scripts.experiments.lesson0204 --case mixed_lengths --max-tokens-per-round 2
```

A 的 prompt 有三个 token，单轮额度只有两个；本节没有分块 prefill，所以命令应以非零退出码退出，提示 `Prompt exceeds one round`。不要通过忽略错误或把上限检查删除来得到 PASS。

| 现象 | 先检查 |
| --- | --- |
| `No module named zero2vllm` 或 `scripts` | 当前是否在仓库根目录；不要把示例单独复制到别的目录运行 |
| `Prompt exceeds one round` | T 是否小于某个需要输出的请求的完整 P |
| `Request can never fit` | 单请求 P+N−1 是否已经超过总容量 C |
| `Request exceeds ... context limit` | 输入与最大生成量是否超过 Tiny 模型的上下文 32 |
| `requires P-then-1 cached steps` | 是否错误接入了完整前缀重算运行器 |
| `max_rounds reached` | 上限是否小于到达时间/必要轮数；先看轨迹，错误退出不会保存部分成功报告 |
| 多轮空执行 | 是没有请求到达，还是活动请求缺少执行机会；观察 arrival、waiting、running |

初始化失败、KV 追加后的前向失败、已选请求被取消、嵌套调度拒绝和资源回收，由本节测试覆盖。修改核心逻辑后重新执行第 1 步的测试命令。

## 7. 交一份可核对的实验说明

建议保留主实验报告和一个改参报告，并填写：

| 项目 | 你的记录 |
| --- | --- |
| 输入与策略 | case、到达边界、P/N/EOS、取消边界 |
| 三个上限 | S、T、C |
| 一轮手算 | 当时谁在等待，谁已活跃，谁被选中，成本各是多少 |
| 请求输出 | 与完整重算是否一致；取消保留几项 |
| 调度差异 | 首项边界、终态边界、最大相邻输出间隔 |
| 资源 | 是否只收尾一次，末尾预留是否为零 |
| 解释边界 | 明确是逻辑轮次，未据此计算毫秒、tokens/s 或 GPU 利用率 |

保存结果后完成[练习与答案](exercises.md)。后续改执行器时，这份轨迹就是可以继续比较的参照。
