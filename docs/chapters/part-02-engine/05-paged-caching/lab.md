# 2.5 实验：从 slot 到 Attention，再回到调度器

[正文](README.md) · [练习](exercises.md) · [公开记录](results/README.md)

全部命令在仓库根目录执行。先完成两个短示例，再运行案例集；不要跳过块表与张量输出直接看 PASS。

## 1. 环境

沿用 2.1 的 CPU 虚拟环境即可。需要 Python 3.11 或 3.12、PyTorch 2.7.1、NumPy 2.2.6；本节不依赖 Transformers，不读取真实权重，也不需要 GPU。

首次准备环境时：

```bash
python -m venv .venv
```

只选择你正在使用的终端对应的一行激活：

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# Windows Git Bash
source .venv/Scripts/activate
# Linux / macOS
source .venv/bin/activate
```

然后安装 CPU 依赖并确认解释器：

```bash
python -m pip install torch==2.7.1 numpy==2.2.6 --index-url https://download.pytorch.org/whl/cpu
python -c "import sys, torch; print(sys.executable); print(torch.__version__)"
```

Windows 若不激活，也可以把后续命令中的 `python` 换成 `.venv/Scripts/python.exe`。已有环境不用重建；不要在这个 CPU 教学环境里混装 MACA 或另一套 CUDA PyTorch。

## 2. 示例一：手算每个地址

```bash
python docs/chapters/part-02-engine/05-paged-caching/examples/walk_pages.py
```

应看到：

```text
A block table: (0, 2)
A slots: (0, 1, 4)
A values in logical order: [[10.0, 0.0], [20.0, 10.0], [30.0, 20.0]]
Attention: [[20.0, 10.0]]
...
Expected read rejection: Read length must stay within the written prefix.
PASS: mapping, attention and cleanup.
```

打开 [walk_pages.py](examples/walk_pages.py)，依次对照四个动作：

1. A、B 各预留一块。B 虽然没有写入数据，仍然占着块 1。
2. A 写前两个位置，再增长到三个位置，得到块 2。
3. 按逻辑位置 gather，读回的 V 应保持输入顺序；计算输出 `[20,10]`。
4. 故意请求读取四个位置。A 只写了三个，读取被拒绝。最后 A、B 的块全部归还。

这里的异常是示例预期行为，已在示例中捕获。若先看到 traceback、没有最后的 PASS，才是运行失败。

## 3. 示例二：两个请求为什么不能一起进入

```bash
python docs/chapters/part-02-engine/05-paged-caching/examples/paged_schedule.py
```

池有 3 块，`B=4`，共 12 个 slot。A、B 都是 `P=3、N=3`：

```text
a logical= 5 rounded= 8
b logical= 5 rounded= 8
r=0 run=('a',) wait=('b',) reserved=8 free_blocks=1
r=1 run=('a',) wait=('b',) reserved=8 free_blocks=1
r=2 run=('a',) wait=('b',) reserved=0 free_blocks=3
r=3 run=('b',) wait=() reserved=8 free_blocks=1
r=4 run=('b',) wait=() reserved=8 free_blocks=1
r=5 run=('b',) wait=() reserved=0 free_blocks=3
```

`run` 是本轮实际执行者，`wait` 是本轮准入后的等待队列，`reserved/free_blocks` 是本轮执行结束后的资源状态。第 2 轮 A 完成后归还块，但该轮计划早已冻结，B 在第 3 轮才进入。不能看到 `reserved=0` 就推断该轮执行前池也是空的。

示例在每轮结束检查：

```text
engine.used_tokens == (num_blocks - free_blocks) * block_size
```

这是逻辑额度和实际分配之间的对应关系，不能用 Attention 读到的 `L` 代替右侧。

## 4. 运行全部 11 个案例

```bash
python -m scripts.experiments.lesson0205
```

默认写入 `.local/runs/part-02-engine/05-paged-caching/summary.json`。这个路径被 Git 忽略，不覆盖公开基线。报告保存输入、输出、逐轮记录、断言、环境和源码摘要；任一案例未通过，命令返回非零状态。

| 案例 | 数量 | 应核对的行为 |
| --- | --- | --- |
| `mapping` | 1 | A 的块表 `[0,2]`、slot `[0,1,4]`、K/V 顺序与手算 Attention 一致 |
| `reuse` | 1 | 相同物理块中的旧尾部确实仍在，但新请求只读新前缀；旧句柄失效 |
| `oom` | 1 | 增长失败不改变原块表、预留长度与空闲块数 |
| `logits` | 4 | float64/float32 × B=2/4，全部生成步的全部 logits 对照完整重算 |
| `admission` | 1 | 两个逻辑长度 5 的请求按每个 8 个 slot 计费，共 6 轮完成 |
| `fragmented` | 1 | 短请求归还块 0、2，后来的 D 在非相邻两块中运行，共 3 轮 |
| `cleanup` | 1 | 写入后异常、取消与普通完成共存，其他请求结果不受影响，共 4 轮 |
| `block-sizes` | 1 | B=1/2/4/8 的空间账目；有效、未来预留、取整尾部三者之和等于已分配 slot |

三个引擎案例均比较独立完整重算的输出。异常和取消请求只比较已经提交的前缀；还要检查预期终态、上下文关闭、只收尾一次和所有块归还。

### 只研究一个问题

```bash
python -m scripts.experiments.lesson0205 --case mapping
python -m scripts.experiments.lesson0205 --case reuse
python -m scripts.experiments.lesson0205 --case fragmented
python -m scripts.experiments.lesson0205 --case logits --dtype float32 --block-size 3 --output-dir .local/runs/part-02-engine/05-paged-caching/b3
```

| 参数 | 含义 |
| --- | --- |
| `--case` | `all` 或表中的一个案例组；默认 `all` |
| `--dtype` | `both/float64/float32`，只影响 logits 对照组 |
| `--block-size` | 1—32，覆盖 logits 对照组的块大小；不改变其他案例的固定输入 |
| `--output-dir` | 报告目录；同目录下的 `summary.json` 会被本次结果覆盖 |

例如只跑 `reuse` 后，默认报告只包含这个案例。要保留完整报告，请分别指定目录，或最后重新运行默认命令。参数不涉及吞吐量、GPU 利用率或真实时间的测量。

## 5. 怎样读报告

输入见 [cases.json](data/cases.json)，结构说明见 [data/README.md](data/README.md)。

- `snapshot` / `snapshots`：块表、R、L、已分配 slot、尾部空位、有效 slot 映射。
- `logits` / `reference_logits`：每一步全部词表项；`max_abs_error` 是这些项的最大绝对误差。
- `rounds`：引擎案例每轮的准入、冻结计划、执行和回收；`physical_allocated_slots_after` 应等于 `reserved_after_execution`。
- `forward_observations`：前向写入之后、生命周期收尾之前的缓存快照。终态时句柄已失效，所以另存这些观察点。
- `requests`：终态、已提交输出与逻辑时间；轮数不是毫秒延迟。
- `assertions` / `passed`：逐项断言及合并结果。不要只抽看其中一项。

默认生成实验的 prompt 是 `[1,2,3,4,5]`，输出为 `[14,12,3,8,13]`，最后有效 KV 长度是 9。float64 阈值 `1e-12`，float32 阈值 `1e-6`；本次实测分别约 `1.11e-16` 和 `2.97e-8`。这里只把 K/V 与 Attention 改为对应 torch dtype，Python 参考投影仍用原来的数值路径，因此不要将它描述成整个模型的 float32/float64 实现对比。

## 6. 测试和回归

```bash
python -m unittest tests.cache.test_paged tests.integration.test_paged_engine -v
python -m unittest tests.integration.test_lifecycle tests.integration.test_scheduler -v
python -m scripts.docs.course --check
python -m scripts.docs.layout
```

新增 26 项测试：15 项检查张量存储，11 项检查模型与引擎连接。测试包含非连续块、错误句柄、GQA 逐头参考、因果读取、不同输入分段、取消时机、写入后失败与初始化失败回滚。另运行原生命周期和调度测试，确认块大小为 1 的旧路径保持行为。

未安装 PyTorch 时，这 26 项会跳过；`OK (skipped=...)` 不能当作分页张量已验证。完整仓库回归还需要 2.2 的固定依赖，环境准备见[2.2 实验页](../02-loading-real-model-weights/lab.md)。

## 7. 常见问题

| 现象 | 先检查什么 |
| --- | --- |
| 找不到 torch | 查看 `sys.executable`，确认安装和运行使用同一个环境 |
| `Append exceeds...` | 写入总长度超过 R；直接操作 store 时先增长预留，不要只看块尾有空位 |
| `Read length must stay...` | 读取超过 L；检查有效长度与 query 因果前缀，不能改用 R 掩盖错误 |
| `Cache handle is foreign...` | 是否使用了另一池、复制的或已释放的句柄 |
| `Engine capacity must match...` | 引擎额度与实际池不一致；通常让引擎自动读取 runner 的容量即可 |
| slot 正确、logits 不对 | 是否把 slot 当 position ID，或 prefill query 看到了未来位置；再检查 K/V 顺序、dtype 和 GQA 头顺序 |
| 输出正确但空闲块越来越少 | 检查失败/取消是否关闭上下文，而非只修改状态；观察收尾次数和物理池账目 |
| 回收了块但内存没有下降 | 块回到预分配池，K/V 张量仍然存在；`free_blocks` 和进程内存是不同指标 |

## 验收

能手算 A 的 slot、解释 `L/R/QB`、复现 11 个案例，并说明为什么 `B=4` 时 `5+5` 个逻辑位置不能放进 12 个物理 slot，就可以进入[练习](exercises.md)。继续前应能指出本节尚未实现的三件事：跨请求共享、融合 GPU Attention、真实模型增量 KV。
