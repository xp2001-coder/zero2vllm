# Part 1 · 理解大模型推理：学习与命令总表

> 六节正文、CPU 参考代码、逐项命令、参数解释、练习答案与检查入口。本篇先完成数学和数据分析基础；A100/C500 的真实服务与性能实验在平台章节独立验收。

[项目首页](../../../README.md) · [完整目录](../README.md) · [学习路线](../learning-paths.md)

## 1. 学完这一篇要交付什么

把“输入文字 → 模型前向 → 选 token → 更新状态 → 停止”连成一条能解释的计算链，并能判断一份性能或正确性报告是否采用了合理口径。

| 节 | 学习材料 | 可运行入口 | 主要产出 | 建议时间 |
| --- | --- | --- | --- | --- |
| 1.1 | [从 token 到下一步预测](01-next-token/README.md) | `next-token-demo` | 张量形状表、Attention 手算、逐步生成 | 60—90 分钟 |
| 1.2 | [Prefill 与 Decode](02-prefill-decode/README.md) | `phase-demo` | 每步输入/缓存轨迹、FLOPs/字节与 Roofline 估算 | 90—120 分钟 |
| 1.3 | [KV Cache 的作用](03-kv-cache/README.md) | `demo`、`kv-demo` | 缓存对齐、绝对位置、GQA 与容量账本 | 90—120 分钟 |
| 1.4 | [如何读性能结果](04-performance-metrics/README.md) | `metrics-demo` | 从原始事件重算延迟、吞吐、分位数与 Goodput | 90—120 分钟 |
| 1.5 | [模型结构与长上下文基础](05-model-structures/README.md) | `structure-demo` | RoPE/ALiBi/滑窗/latent/线性/SSM 的状态对照 | 120—150 分钟 |
| 1.6 | [采样与数值正确性](06-sampling-correctness/README.md) | `sampling-demo` | 筛选分布、penalty、seed、停止与数值反例 | 120—150 分钟 |

时间是学习安排建议，未通过学生试学计时。初学者按 1.1→1.6 顺序；工程师可先运行全部命令，再按薄弱部分回读。量化方向重点关注 1.2 的算术强度、1.3 的字节预算和 1.6 的误差定义。

## 后半篇怎样分段学习

1.4—1.6 已补充逐步推导、中文原创图与短示例。建议每次完成一组“预测 → 手算 → 运行 → 解释”，再进入下一组。

| 小节 | 先抓住的问题 | 实验 | 练习 |
| --- | --- | --- | --- |
| 1.4 | 时间点怎样变成延迟？样本、分母和窗口各是什么？ | [四请求报告与单变量对照](04-performance-metrics/lab.md) | [10 组](04-performance-metrics/exercises.md) |
| 1.5 | 下一步需要读取什么，上一轮需要留下什么？ | [位置、窗口、latent 与递推](05-model-structures/lab.md) | [10 组](05-model-structures/exercises.md) |
| 1.6 | 概率怎样选成一个 ID，误差和停止怎样检查？ | [分布、生成、停止与数值](06-sampling-correctness/lab.md) | [12 组](06-sampling-correctness/exercises.md) |

1.2 增加了输入/缓存/新输出时序图，1.3 增加了 GQA 读取图。后半篇另有 8 张图，图与手算、代码使用相同约定。六个新增短示例只用标准库；原有 28 案例入口继续保留，修订记录另存，不改写历史基线。

## 2. 准备一个可以直接运行的环境

### 2.1 检查 Python 与目录

下载/克隆仓库后，在有 `pyproject.toml` 的根目录运行：

```bash
python --version
python -m zero2vllm --help
```

要求 Python 3.11+。这一篇只依赖标准库，直接从源码运行，不必安装 PyTorch、vLLM、CUDA/MACA 或下载模型。Windows 有多个 Python 时可用 `py -3.11` 替换命令中的 `python`；Linux 常用 `python3`，先确认版本。

如果 `--help` 没有 `phase-demo` 等命令，当前 checkout 不包含本篇代码，先更新到相应课程提交。

### 2.2 虚拟环境可选

```bash
python -m venv .venv
```

激活方式取决于终端：

| 终端 | 激活 |
| --- | --- |
| Windows PowerShell | `.\.venv\Scripts\Activate.ps1` |
| Windows Git Bash | `source .venv/Scripts/activate` |
| Linux/macOS shell | `source .venv/bin/activate` |

PowerShell 若限制激活脚本，可以直接用 `.\.venv\Scripts\python.exe -m zero2vllm ...`，无需为了本课修改系统执行策略。虚拟环境内不必安装额外依赖。

### 2.3 保留复现信息

```bash
git rev-parse HEAD
git status --short
```

`0.1.0.dev0` 是开发版本，不足以唯一定位代码。记录 commit 和未提交改动；统一运行器还会保存 CPU 源码摘要。`.local/runs/` 存个人输出，默认不提交。

## 3. 逐节运行：先看结果，再看原理

```bash
python -m zero2vllm next-token-demo --output .local/runs/part-01-inference/01-next-token/01-next-token.json
python -m zero2vllm phase-demo --output .local/runs/part-01-inference/02-prefill-decode/02-phases.json
python -m zero2vllm kv-demo --output .local/runs/part-01-inference/03-kv-cache/03-cache.json
python -m zero2vllm metrics-demo --output .local/runs/part-01-inference/04-performance-metrics/04-metrics.json
python -m zero2vllm structure-demo --output .local/runs/part-01-inference/05-model-structures/05-structures.json
python -m zero2vllm sampling-demo --output .local/runs/part-01-inference/06-sampling-correctness/06-sampling.json
```

所有命令自动创建父目录。终端会打印 JSON，可以用编辑器打开同名文件；数字字段保留实际精度，正文中的手算值常只显示六位小数。

| 节 | 先看这些字段 | 默认检查要点 |
| --- | --- | --- |
| 1.1 | `input`、`trace`、`generation` | 下一 token 是 ID 2；最后一行有 5 个分数 |
| 1.2 | `steps`、`cached_work`、`gemm_estimates` | 投影 60→11；CPU 计数和 GEMM 假设分开 |
| 1.3 | `chunks`、`grouped_query_example`、`capacity_estimates` | 分块 offset 0/2/4；默认 GQA 理论 payload 512 MiB |
| 1.4 | `request_metrics`、`rates`、`distributions` | 成功率 75%；模拟 Goodput 为 2/2.3 request/s |
| 1.5 | `rope`、`sliding_window`、`linear_attention`、`ssm` | 配对 36/21；同算法数学对照误差在容差内 |
| 1.6 | `generation.steps`、`frequency_experiment`、`numerical_examples` | 看概率、停止原因和误差，不能只看文本 |

每节还有参数变体、非法输入解释和折叠的参考答案。不要只收集 `passed=true`，还需要解释为何结果应该这样。

## 4. 一次运行整个 Part 1

```bash
python -m scripts.experiments.part01
```

统一入口会依次运行 **28 个 CPU 案例**、整个仓库单元测试和文档一致性检查，保存到 `.local/runs/part-01-inference/`。控制台只打印每个案例的 PASS/FAIL；完整结果在文件中。

```text
.local/runs/part-01-inference/
├── 01-next-token/             第 1 节默认结果
├── 02-prefill-decode/         第 2 节默认结果与参数变体
├── 03-kv-cache/               第 3 节默认结果与参数变体
├── 04-performance-metrics/    第 4 节指标结果
├── 05-model-structures/       第 5 节结构对照
├── 06-sampling-correctness/   第 6 节采样与停止边界
├── checks/
│   ├── tests.txt              完整单元测试输出
│   └── course-check.txt       文档同步与链接检查
└── summary.json              每个命令、退出码、结果与源码摘要
```

脚本说明：[part01.py](../../../scripts/experiments/part01.py)。它调用当前 Python、固定在此仓库根目录运行，不访问网络，不下载依赖，不执行 GPU 操作，也不提交 Git。

维护者本地已运行的[28 案例摘要](results/baseline/summary.json)和[87 项测试原始输出](results/baseline/tests.txt)可直接对照。详细范围见[验证记录](../../../reports/history/validation-before-restructure.md)。这些记录来自 Windows / CPython 3.11.9，GPU 仍为 `not_run`。

如果刚运行过测试，单独重跑案例可以使用：

```bash
python -m scripts.experiments.part01 --skip-tests --output-dir .local/runs/part-01-inference-rerun
```

此时摘要会明确标记 `tests_skipped=true`，不能把它当成这次又运行了全套测试。每个输出目录用于一轮结果；再次使用相同目录会更新同名文件，需要保留对比时换一个目录。

### 4.1 六个新增短示例

```bash
python docs/chapters/part-01-inference/04-performance-metrics/examples/compare_timelines.py
python docs/chapters/part-01-inference/05-model-structures/examples/walk_positions.py
python docs/chapters/part-01-inference/05-model-structures/examples/walk_states.py
python docs/chapters/part-01-inference/06-sampling-correctness/examples/walk_distribution.py
python -X utf8 docs/chapters/part-01-inference/06-sampling-correctness/examples/walk_stopping.py
python docs/chapters/part-01-inference/06-sampling-correctness/examples/walk_correctness.py
```

每个示例把关键中间结果直接打印出来，并用独立手算值检查。它们不计入原有的 28 个命令案例；复现记录见[第一篇修订检查](../../../reports/quality/part-01-revision/README.md)。

## 5. 独立检查与代码阅读顺序

```bash
python -m unittest discover -s tests -v
python -m scripts.docs.course --check
```

| 阅读顺序 | 文件 | 核心问题 |
| --- | --- | --- |
| 1 | [next_token.py](../../../zero2vllm/reference/next_token.py) | 如何从手工权重追踪每个中间矩阵 |
| 2 | [attention.py](../../../zero2vllm/reference/attention.py) | 完整前向与缓存前向如何保持语义相同 |
| 3 | [phases.py](../../../zero2vllm/reference/phases.py) | 每步工作量与 GEMM 估算分别如何计算 |
| 4 | [kv_geometry.py](../../../zero2vllm/reference/kv_geometry.py) | 如何分块、按组读 K/V 与估计容量 |
| 5 | [metrics.py](../../../zero2vllm/reference/metrics.py) | 时间戳如何变成可审查指标 |
| 6 | [positions.py](../../../zero2vllm/reference/positions.py) | 位置、窗口和递推状态的数学关系 |
| 7 | [sampling.py](../../../zero2vllm/reference/sampling.py) | 如何定义候选筛选、停止与数值比较 |
| 8 | [test_part1.py](../../../tests/reference/test_part1.py) | 独立手算、边界、负例和命令输出如何验收 |

## 6. 综合练习：提交一份自己的实验说明

不修改权重，先完成以下六项，每项附命令、结果位置和自己的解释：

1. 将第一课输入改成三个词项，画出张量形状表。
2. 设 `P=2,O=3`，预测重算与缓存的投影/配对数，再运行验证。
3. 对 24 层、4 个 KV heads、head_dim=128、2 字节元素，计算 2048 token 的 payload。
4. 将模拟失败请求的完成时间从 2.3 改为 3.0，指出哪些统计量改变。
5. 比较 window1/window8，并解释为何线性 Attention 不应强制对齐 softmax。
6. 复现跨 token stop string，并解释原始 token 数、展示文本和缓存长度的差别。

各题参考答案在对应章节。工程进阶再附一页“迁移到 A100/C500 前需要确认的事项”：实际模型结构、dtype、软件版本、真实负载、计时位置与误差标准。此处列方法，不填入未测量的性能。

## 7. 常见问题

| 现象 | 处理 |
| --- | --- |
| 找不到 Python / 版本不足 | 先安装或选择 Python 3.11+，核对当前解释器 |
| `No module named zero2vllm` | 回到项目根目录，不要在 `docs/` 下运行 |
| 指令里出现 `...` | 文中带省略号的说明是占位示意，使用每节给出的完整命令 |
| 中文终端显示异常 | 用 UTF-8 编辑器打开 `--output` 文件；必要时使用 `python -X utf8` |
| 找不到指标输入文件 | 从仓库根目录运行，或通过 `--input` 指定正确文件 |
| 生成一直重复词项 | 教学权重未训练，不将输出文本用于评价语言质量 |
| `--check` 提示目录不同步 | 维护者修改清单后运行 `python -m scripts.docs.course --write`，再检查 |
| 需要向 GPU 安装什么 | Part 1 无需 GPU；正式环境按 Part 4 的实机记录准备 |

## 8. 什么算这一篇完成

课程交付需要六节正文、对应代码/命令、可审阅结果、练习答案和有效测试；学生完成需要能解释综合练习，不能以“命令退出码为 0”替代掌握原理。

当前 CPU 数学参考不包含训练、完整 Transformer 堆栈、生产 Tokenizer、真实 vLLM 服务或 GPU 性能。Part 2 的 2.1—2.7 已在这些小例子上提供模型、权重、请求、缓存与 HTTP 的 CPU 教学链路；GPU 环境、源码与量化/算子专题分别沿后续课程推进。

## 本篇章节

<!-- BEGIN part-index -->

| 节 | 课程 | 验收产出 | 进度 | 正文 |
| --- | --- | --- | --- | --- |
| 1.1 | [从 token 到下一步预测](01-next-token/README.md) | 手算小矩阵的 Attention 并标出张量维度 | ✅ | 完整课程 |
| 1.2 | [Prefill 与 Decode](02-prefill-decode/README.md) | 估算两阶段的 FLOPs、访存量与主要瓶颈 | ✅ | 完整课程 |
| 1.3 | [KV Cache 的作用](03-kv-cache/README.md) | 复现已有 logits 对齐实验并推导 KV 容量 | ✅ | 完整课程 |
| 1.4 | [如何读性能结果](04-performance-metrics/README.md) | 用原始时间序列重算指标并识别错误口径 | ✅ | 完整课程 |
| 1.5 | [模型结构与长上下文基础](05-model-structures/README.md) | 比较不同结构的缓存对象和复杂度 | ✅ | 完整课程 |
| 1.6 | [采样与数值正确性](06-sampling-correctness/README.md) | 实现小词表采样并检查边界、分布和停止规则 | ✅ | 完整课程 |

<!-- END part-index -->
