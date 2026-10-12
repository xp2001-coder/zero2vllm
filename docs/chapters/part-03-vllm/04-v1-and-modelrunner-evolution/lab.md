# 3.4 实验：稳定状态行、增量更新与图模式

[返回正文](README.md) · [练习与答案](exercises.md) · [源码路线图](source-map.md) · [已保存记录](results/README.md)

## 1. 环境与实验边界

在仓库根目录运行下面的命令，`python` 指向 Python 3.11+。本节只用标准库，不需要 PyTorch、vLLM、CUDA 或模型权重。Windows 可将 `python` 替换为 `.venv/Scripts/python.exe`。

所有缓冲区、状态行和图分支都是 CPU 教学对象。这里没有 DMA、GPU kernel、线程并发或 CUDA Graph 捕获；耗时不应用来判断 MRV2 的加速比。

默认运行完全离线。只有明确使用 `--fetch-source` 时，才向官方仓库下载固定提交的源码文本；下载不等于执行源码。

## 2. 一次运行全部十个案例

```bash
python -m scripts.experiments.lesson0304
```

输出十行 `PASS`，随后提示实验边界。详细 JSON 默认写入 `.local/runs/part-03-vllm/04-runner-evolution/summary.json`，日常运行不会覆盖本节保存的公开记录。

| 案例 | 要检查的事实 |
|---|---|
| `stable_rows` | 请求保持自己的状态行，执行顺序 `[B,A]` 通过索引 `[1,0]` 访问 |
| `scatter_progress` | 以执行顺序提交增量，按状态行正确更新 A 与 B |
| `row_reuse` | B 释放后 D 复用空行；旧视图被教学代际检查拒绝 |
| `preempt_and_resume` | runner 状态释放后重新建立，请求历史仍可以恢复 |
| `staged_writes` | 打包两个不重叠的小更新，应用后其余位置不变 |
| `overlap_guard` | 教学实现拒绝同一批的重叠写入 |
| `copy_ownership` | 延迟读取共享列表会读到新值，快照保留旧值 |
| `expanded_sampling_map` | 一条请求需要多行 logits 时，展开到正确的状态行 |
| `sampling_algebra` | 固定随机数下的 Gumbel 等价关系与完整词表归一化 |
| `graph_dispatch` | 图模式、运行形状与捕获状态共同决定路径 |

这里的“代际检查”和“拒绝重叠写”是课程主动增加的边界保护，不是声称上游存在同样的实现。

## 3. 手算状态，再查看代码结果

```bash
python docs/chapters/part-03-vllm/04-v1-and-modelrunner-evolution/examples/walk_state.py --case mapping
```

先在纸上画三行：A 在 0、B 在 1、C 在 2。已计算长度分别为 2、3、0。执行顺序是 `[B,A]`，所以索引是 `[1,0]`。

本轮 B 计算 1 个、A 计算 3 个，增量 `[1,3]` 必须散写回状态行：

```text
A：2 + 3 = 5
B：3 + 1 = 4
C：0，没有参加本轮
```

继续观察 B 移除、D 入驻后的行复用，以及 A 被移出、其他请求占位、A 带历史返回的过程。请求 ID 才是业务身份；状态行只是当前占用的存储位置。

程序会用旧视图访问复用后的行并捕获预期错误。如果自己修改代码，将校验删除，旧视图就可能把 D 当成 B。不要仅凭索引仍在范围内判断引用有效。

## 4. 增量写入与数据所有权

```bash
python docs/chapters/part-03-vllm/04-v1-and-modelrunner-evolution/examples/walk_state.py --case staged
python docs/chapters/part-03-vllm/04-v1-and-modelrunner-evolution/examples/walk_state.py --case copy
```

### 两次小更新如何打包

初始表为 `3 × 6` 的零表。更新 `(行0, 起点2, [8,3])` 和 `(行2, 起点0, [5])`：

```text
indices         = [0,2]
starts          = [2,0]
contents        = [8,3,5]
cumulative_ends = [2,3]
```

第一个更新取 `contents[0:2]`，第二个取 `contents[2:3]`。应用后第 0 行为 `[0,0,8,3,0,0]`，第 2 行为 `[5,0,0,0,0,0]`。

教学打包大小：`(3 个内容元素 + 3 × 2 个元数据元素) × 4 字节 = 36 字节`。整表逻辑大小为 `3 × 6 × 4 = 72 字节`。这只是逻辑载荷对照，没有计入对齐、分配、启动或传输协议开销；上游的 UVA 元数据与内容传输路径也不同。

### 为什么 non_blocking 不等于源数据被冻结

列表先为 `[3,7]`，任务安排后、第一个元素被改为 9，随后才真正读取。

- 共享同一列表：读取 `[9,7]`。
- 安排时保存副本：读取 `[3,7]`。

这是确定性的顺序程序，用来解释“提交工作”和“消费者读完”是两个时刻。真正异步拷贝还要处理设备流、事件、缓冲区生命周期与复用时机；本实验没有模拟这些运行机制。

## 5. 采样映射和概率分母

```bash
python docs/chapters/part-03-vllm/04-v1-and-modelrunner-evolution/examples/walk_state.py --case sampling
```

请求状态行 `[2,0]`，各需要 `[1,3]` 行 logits，展开后为 `[2,0,0,0]`，各请求内部位置为 `[0,0,1,2]`。

对 logits `[2,1,0,-1]`，完整词表的 `logsumexp` 约为 `2.44019`。前两个 token 的 logprob 约为 `-0.44019`、`-1.44019`，而不是只拿 `[2,1]` 重算分母得到的值。

Gumbel 例子使用固定 uniforms `[0.13,0.81,0.4,0.73]`，不会证明两个 GPU sampler 的随机流逐位一致。它仅核对同一组噪声下的代数关系。

## 6. 改变图模式和本轮形状

```bash
python docs/chapters/part-03-vllm/04-v1-and-modelrunner-evolution/examples/inspect_graphs.py --mode FULL_AND_PIECEWISE --query-lengths 1 1 1 --captured 4
python docs/chapters/part-03-vllm/04-v1-and-modelrunner-evolution/examples/inspect_graphs.py --mode FULL_AND_PIECEWISE --query-lengths 1 3
python docs/chapters/part-03-vllm/04-v1-and-modelrunner-evolution/examples/inspect_graphs.py --mode FULL_AND_PIECEWISE --query-lengths 1 1 1 --captured
python docs/chapters/part-03-vllm/04-v1-and-modelrunner-evolution/examples/inspect_graphs.py --mode FULL_DECODE_ONLY --query-lengths 1 3
```

依次预期：

1. 三条普通 decode，3 个实际 token 向上匹配到 bucket 4，已捕获，`FULL`。
2. `[1,3]` 为混合形状，4 个 token，`PIECEWISE`。
3. `--captured` 后无数字表示空列表；bucket 4 没有完整图，回退 `NONE`。
4. 配置只允许 decode 完整图，混合形状回到 `NONE`。

本模拟只覆盖单卡、普通解码 `q=1`。它不构建上游编译包装器，也没有验证 PIECEWISE 的捕获准备工作。输出是决策解释，不是 GPU 执行报告。

## 7. 阅读两个固定提交

已有源码缓存时：

```bash
python -m scripts.experiments.lesson0304 --inspect-source
```

第一次明确联网取文本时：

```bash
python -m scripts.experiments.lesson0304 --fetch-source
```

默认目录是 `.local/sources/vllm-v0.8.5` 与 `.local/sources/vllm-v0.17.1`。若使用 `--source-root <目录>`，该目录下也要保留这两个版本子目录。

检查范围为旧版 3 个文件、4 个函数，新版 12 个文件、22 个函数。清单固定 SHA 和文件摘要，检查时不执行下载的 `.py` 文件。

`--inspect-source` 不会自动补下载。缺文件、错误版本、内容损坏或所需符号变化都会失败；请核对清单与目录，不要直接把预期 hash 改成当前文件的 hash 来消除错误。

## 8. 测试与验收

```bash
python -m unittest tests.reference.test_runner_state tests.maintenance.test_later_source_reading -v
```

22 项 runner/数学边界测试，加 5 项供两节共用的源码阅读测试。后者检查只筛选目标符号时如何处理无关重名方法，默认严格模式仍保留。

验收标准：

- 能手算状态行、执行顺序、展开采样行，并解释它们为何不是同一组索引。
- 能解释两次更新的载荷，指出小载荷不等于已测得加速。
- 能用一次失败例子解释状态复用或数据所有权问题。
- 能区分引擎、runner 和图模式；把结论绑定到具体提交。
- 不把 CPU 演示写成 A100/C500 验证报告。

常见问题：从子目录执行找不到 `scripts` 时回到仓库根目录；Python 版本过低时换到 3.10+；源码缓存缺失时先检查目录或显式取源码；想判断 MRV2 性能时，需要另做同一提交、同一负载的 GPU 对照实验。
