# 2.6 实验：亲眼看见前缀复用

[正文](README.md) · [练习](exercises.md) · [运行记录](results/README.md)

本节所有命令都在仓库根目录运行。先看两个短示例，再跑案例集，最后尝试改变块大小。输入是数字 token ID，输出不表示自然语言回答。

## 1. 准备环境

沿用 2.5 的虚拟环境即可：Python 3.11/3.12、CPU PyTorch 2.7.1、NumPy 2.2.6。本节不需要下载权重，也不需要 Transformers。首次配置环境的完整步骤见 [2.5 环境说明](../05-paged-caching/lab.md#1-环境)。

Windows Git Bash 中激活已有环境：

```bash
source .venv/Scripts/activate
python -c "import sys, torch; print(sys.executable); print(torch.__version__)"
```

PowerShell 使用 `.\.venv\Scripts\Activate.ps1`，Linux 使用 `source .venv/bin/activate`。不激活时，Windows 下也可以把下文 `python` 换成 `.venv/Scripts/python.exe`。

若尚未安装依赖，在激活后执行：

```bash
python -m pip install torch==2.7.1 numpy==2.2.6 --index-url https://download.pytorch.org/whl/cpu
```

检查解释器路径确实指向当前 `.venv`。已有可运行环境不必重新安装。

## 2. 实验一：同一条请求跑两遍

```bash
python docs/chapters/part-02-engine/06-prefix-caching/examples/walk_prefix.py
```

本次实际输出：

```text
cold hits= 0 projected= 8 attention_pairs= 36 output= [12, 3, 8]
after release: active_refs= 0 idle_cached= 4
warm hits= 4 projected= 4 attention_pairs= 26 output= [12, 3, 8]
after release: active_refs= 0 idle_cached= 4
PASS: identical outputs; four projected positions saved on the warm request.
```

### 第一步：先解释 cold

输入 `[1,2,3,4,5,6]`，输出上限 3，块大小 2。冷轮没有已有缓存，所以 `hits=0`。

首轮计算 6 个 prompt 位置，后面两轮各计算 1 个位置，所以 `projected=8`。最后输出 8 没有再送进模型。代码中的 `projected_positions` 在实际完成新位置 K/V 写入后逐项增加，不是单纯用命中量倒推出来的计数。

### 第二步：解释请求结束后的状态

`active_refs=0` 表示所有请求对块的使用权已经归还。`idle_cached=4` 表示四个完整块仍保留可复用内容：

```text
[1,2] [3,4] [5,6] [12,3]
```

不要把这四块当作泄漏。它们没有活跃引用，空间不够时可以逐出。预分配张量继续存在，因此也不要求进程内存马上下降。

### 第三步：解释 warm

第二条请求输入完全相同。因为需要重新得到最后位置的 logits，又只共享完整块，实际命中前四个位置：`hits=4`。

热轮计算 `[5,6]` 两个 prompt 位置，再计算两个 decode 输入，共 4 个位置。Attention 仍要读取完整前缀，因此分数项是 `5+6+7+8=26`。

### 第四步：确认答案没有换掉

两轮都与独立完整重算生成结果对照，不拿“上一轮输出”当唯一参考。两次输出相同只是第一步；完整案例集还比较每一项 logits。

## 3. 实验二：A 结束之后，B 能否继续

```bash
python docs/chapters/part-02-engine/06-prefix-caching/examples/share_and_release.py
```

应看到：

```text
A table: [0, 1, 2, 3]
B table: [0, 1, 4, 5]
B refs before A closes: [2, 2, 1, 1]
B refs after A closes: [1, 1, 1, 1]
B output: [14, 13, 1]
final active refs: 0
retained idle blocks: 5
PASS: releasing A did not invalidate B; completed prefixes remain reusable.
```

这里 A、B 的前四个位置相同。对照块表，前两项是相同物理编号 0、1；后两项不同，所以并不是把两条完整请求混在一起。

A 结束时，只把自己的引用减掉。B 仍然持有共享前缀，随后两个生成步骤还会读取它们。若 release 把共享块直接当空白块交给别人，B 的后续结果就可能变化。

本例最后留存五个完整块：两个公共前缀块、A 的一个完整后缀块，以及 B 的后缀块与生成形成的完整块。A 为未来生成预留的空块没有写满，不进入缓存索引。

## 4. 运行全部案例并保存证据

```bash
python -m scripts.experiments.lesson0206
```

默认输出 `.local/runs/part-02-engine/06-prefix-caching/summary.json`。目录被 Git 忽略，不改公开基线。预期打印 12 个 PASS；任一断言失败，退出状态非零。

| 报告案例名 | 检查内容 |
| --- | --- |
| `cold-warm-float64` | 冷暖两轮的全部生成 logits、命中长度与工作量 |
| `cold-warm-float32` | 同一对照，缓存和 Attention 使用 float32 |
| `branching-live` | 活跃请求共享相同物理块，A 退出后 B 继续 |
| `hit-boundaries` | 18 组长度/块大小组合，检查末块重算边界 |
| `parent-and-namespace` | 相同前缀、不同历史、不同共享域和中途分叉 |
| `forced-collision` | 故意让摘要全部相同，仍要逐项核对精确前缀 |
| `lru-and-protection` | 命中块被保护，只回收闲置候选；数据未被同伴覆盖 |
| `oom-atomic` | 空间不足时，引用计数、索引、回收次序与事件均不改变 |
| `reset` | 活跃句柄存在时拒绝重置；全部释放后清空，下一次冷启动 |
| `duplicate-cold` | 两条请求先各自冷启动，后发布时不改换活跃块表 |
| `scheduler-cleanup` | 请求取消、正常结束与注入异常共存；资源和同伴输出正确 |
| `no-reuse` | 关闭查找复用后，两次都计算完整输入，不应出现节省量 |

`no-reuse` 仅关闭前缀查找，仍执行相同的写入和完整块发布流程。它便于验证“少算的位置确实来自复用”；不是彻底关闭缓存管理所有开销的性能基准。

## 5. 单独运行一个问题

| CLI 选择值 | 对应报告 |
| --- | --- |
| `cold-warm` | 按 dtype 生成一个或两个冷暖案例 |
| `branching` / `boundaries` | 活跃共享 / 命中边界表 |
| `scope` / `collision` | 计算身份 / 强制碰撞 |
| `pressure` / `oom` | 逐出压力 / 原子失败 |
| `reset` / `duplicates` | 清空 / 重复冷启动 |
| `scheduler` / `no-reuse` | 请求引擎连接 / 无查找复用对照 |

例如：

```bash
python -m scripts.experiments.lesson0206 --case pressure --output-dir .local/runs/part-02-engine/06-prefix-caching/pressure
python -m scripts.experiments.lesson0206 --case collision --output-dir .local/runs/part-02-engine/06-prefix-caching/collision
python -m scripts.experiments.lesson0206 --case scheduler --output-dir .local/runs/part-02-engine/06-prefix-caching/scheduler
```

同一个输出目录下的 `summary.json` 会被下一次运行覆盖，所以这里为单项实验指定独立子目录。要保留完整结果，可以最后重新跑默认命令。

## 6. 改参：块大小是否越大越好

先预测，再运行：

```bash
python -m scripts.experiments.lesson0206 --case cold-warm --dtype float32 --block-size 3 --output-dir .local/runs/part-02-engine/06-prefix-caching/b3
python -m scripts.experiments.lesson0206 --case cold-warm --dtype float64 --block-size 8 --output-dir .local/runs/part-02-engine/06-prefix-caching/b8
```

`--block-size` 允许 1—16，`--dtype` 为 `both/float64/float32`。这两个参数只影响冷暖对照组；其他固定案例仍使用自己的教学输入。

对于固定 P=6、N=3：

| B | 热轮预期命中 | 热轮预期计算位置 | 预期 Attention 分数项 |
| --- | --- | --- | --- |
| 2 | 4 | 4 | 26 |
| 3 | 3 | 5 | 30 |
| 8 | 0 | 8 | 36 |

B=8 时，prompt 长度还不够一个可复用完整块，即使池里有别的完整历史，也不能把它截成任意长度使用。这个例子说明复用粒度会影响收益，但没有测量不同 B 的设备内核效率。

## 7. 报告字段怎么读

建议先打开 `cold-warm-float64`：

- `cold/warm.hit_positions`：真正复用的位置数。
- `projected_positions`：当前请求实际计算 Q/K/V 的新位置数，包含 prompt 后缀和 decode 输入。
- `attention_pairs`：单头参考计算的 query-key 配对数；不是 GPU FLOPs。
- `logits/reference_logits/max_abs_error`：每一步全词表对照，而不只是 argmax。
- `snapshots`：每步结束后的块表、有效 token 历史和各块当前 refs。
- `pool_after_release`：释放请求后，活跃、闲置缓存与空白块的状态。

再打开 `scheduler-cleanup`：

- `requests`：A 取消、B 正常结束、bad 失败。失败请求没有提交注入异常之后的输出。
- `rounds[].reserved_after_execution`：引擎保守预留额度，已结束请求的额度已经归还。
- `rounds[].pool_after.active_unique_blocks`：实际被活跃请求引用的不同物理块数。
- 两者应满足 `active_unique_blocks × B ≤ reserved_after_execution`，不是必须相等。
- 终态 `active_block_references=0`，但 `idle_cached_blocks` 可以大于 0。

`source.source_sha256` 标识生成报告时的代码与输入。它与历史章节的摘要不同是正常现象；不要修改旧报告来伪装成同一轮执行。

## 8. 跟着代码走一遍

| 顺序 | 文件与入口 | 建议停下来观察 |
| --- | --- | --- |
| 1 | [`prefix_runner.py`](../../../../zero2vllm/engine/prefix_runner.py) 的 `new_context/reserve` | 此时只记额度，还没有查找或绑定块 |
| 2 | `advance` 首次调用 | 为什么输入改成 `prompt[hit_positions:]` |
| 3 | [`prefix.py`](../../../../zero2vllm/cache/prefix.py) 的 `acquire` | 命中块如何从新块候选中排除 |
| 4 | `append` | K、V 写完之后才推进 token 历史；完整块何时进入索引 |
| 5 | `gather/attention` | 后缀 query 仍读完整逻辑前缀 |
| 6 | `release/_evict` | 一个撤销使用权，一个移除内容身份 |

只有一个 store 对应一个固定 runner 的池。不要同时让独立引擎或旁路请求使用同一个池，再假设原引擎的保守额度可以约束那些额外占用。

## 9. 自动检查

```bash
python -m unittest tests.cache.test_prefix tests.integration.test_prefix_engine -v
python -m scripts.docs.course --check
python -m scripts.docs.layout
```

新增 29 项测试中，18 项检查缓存管理，11 项检查模型和引擎连接。包含同 prompt 不同块大小/dtype、改变父前缀、强制哈希碰撞、物理共享、逐出旧索引、OOM 不改状态、取消/异常、EOS、零输出以及权重身份隔离。

没有 PyTorch 时这些测试会明确跳过。`OK (skipped=29)` 不是张量路径验证成功；先确认环境再运行。完整仓库回归还需要 2.2 的固定可选依赖，见[对应实验页](../02-loading-real-model-weights/lab.md)。

## 10. 常见现象与定位方法

| 现象 | 原因与检查顺序 |
| --- | --- |
| 热轮仍然 hits=0 | 检查 P 与 B、实际 token 前缀、namespace、是否在每次调用前新建 runner 或 reset |
| 相同六个 token 只命中四个 | B=2 且没有缓存末尾 logits，所以最后一块重新计算 |
| 第二条请求仍等在队列里 | 当前引擎按完整 R 保守收费，共享收益没有扩展准入名额 |
| A 结束后 B 输出改变 | 检查 release 是否错误地把 refs>0 的共享块回收，再核对块表和逐出事件 |
| 活跃引用为 0，仍有缓存块 | 这是闲置缓存留存；继续用压力实验检查它能否被逐出 |
| 相同块 ID 对应内容变了 | 块 ID 是地址；内容身份看摘要与精确前缀，回收后地址可以复用 |
| 只有 token 输出一致，误差却很大 | argmax 可能掩盖 logits 错误；检查读取前缀、位置偏移、dtype 与计算身份 |
| 修改外部模型后旧 runner 输出没变 | runner 复制了固定权重；创建新 runner 才使用新模型 |
| reset 被拒绝 | 仍有请求持有句柄；先结束或关闭上下文，不能强行抹掉活跃索引 |

## 验收产出

保存一份完整 12 案例报告，回答：默认热轮少算哪四个位置、首个输出从哪次前向得到、A 退出后 refs 怎么变、为什么结束后仍可保留 K/V，以及 B=8 时为什么不能复用。这些都讲得通，再做[十二组练习](exercises.md)。
