# Part 1 · 理解大模型推理：学习与命令总表

> 六节正文、CPU 参考代码、逐项命令、参数解释、练习答案与检查入口。本篇先完成数学和数据分析基础；A100/C500 的真实服务与性能实验在平台章节独立验收。

[项目首页](../../README.md) · [完整目录](../curriculum.md) · [学习路线](../learning-paths.md)

## 1. 学完这一篇要交付什么

把“输入文字 → 模型前向 → 选 token → 更新状态 → 停止”连成一条能解释的计算链，并能判断一份性能或正确性报告是否采用了合理口径。

| 节 | 学习材料 | 可运行入口 | 主要产出 | 建议时间 |
| --- | --- | --- | --- | --- |
| 1.1 | [从 token 到下一步预测](01-next-token.md) | `next-token-demo` | 张量形状表、Attention 手算、逐步生成 | 60—90 分钟 |
| 1.2 | [Prefill 与 Decode](01-prefill-decode.md) | `phase-demo` | 每步输入/缓存轨迹、FLOPs/字节与 Roofline 估算 | 90—120 分钟 |
| 1.3 | [KV Cache 的作用](01-kv-cache.md) | `demo`、`kv-demo` | 缓存对齐、绝对位置、GQA 与容量账本 | 90—120 分钟 |
| 1.4 | [如何读性能结果](01-performance-metrics.md) | `metrics-demo` | 从原始事件重算延迟、吞吐、分位数与 Goodput | 90—120 分钟 |
| 1.5 | [模型结构与长上下文基础](01-model-structures.md) | `structure-demo` | RoPE/ALiBi/滑窗/latent/线性/SSM 的状态对照 | 120—150 分钟 |
| 1.6 | [采样与数值正确性](01-sampling-correctness.md) | `sampling-demo` | 筛选分布、penalty、seed、停止与数值反例 | 120—150 分钟 |

时间是学习安排建议，未通过学生试学计时。初学者按 1.1→1.6 顺序；工程师可先运行全部命令，再按薄弱部分回读。量化方向重点关注 1.2 的算术强度、1.3 的字节预算和 1.6 的误差定义。

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

`0.1.0.dev0` 是开发版本，不足以唯一定位代码。记录 commit 和未提交改动；统一运行器还会保存 CPU 源码摘要。`runs/` 存个人输出，默认不提交。

## 3. 逐节运行：先看结果，再看原理

```bash
python -m zero2vllm next-token-demo --output runs/part1/01-next-token.json
python -m zero2vllm phase-demo --output runs/part1/02-phases.json
python -m zero2vllm kv-demo --output runs/part1/03-cache.json
python -m zero2vllm metrics-demo --output runs/part1/04-metrics.json
python -m zero2vllm structure-demo --output runs/part1/05-structures.json
python -m zero2vllm sampling-demo --output runs/part1/06-sampling.json
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
python scripts/run_part1.py
```

统一入口会依次运行 **28 个 CPU 案例**、整个仓库单元测试和文档一致性检查，保存到 `runs/part1/`。控制台只打印每个案例的 PASS/FAIL；完整结果在文件中。

```text
runs/part1/
├── 01-next-token.json … 06-sampling.json   六节默认结果
├── phase-short.json / cache-chunk3.json 等 参数变体
├── sampling-eos.json / sampling-stop-string.json 等 边界行为
├── tests.txt                             完整单元测试输出
├── course-check.txt                      文档同步与链接检查
└── summary.json                          每个命令、退出码、结果与源码摘要
```

脚本说明：[run_part1.py](../../scripts/run_part1.py)。它调用当前 Python、固定在此仓库根目录运行，不访问网络，不下载依赖，不执行 GPU 操作，也不提交 Git。

维护者本地已运行的[28 案例摘要](../../reports/part1/summary.json)和[87 项测试原始输出](../../reports/part1/tests.txt)可直接对照。详细范围见[验证记录](../../reports/validation.md)。这些记录来自 Windows / CPython 3.11.9，GPU 仍为 `not_run`。

如果刚运行过测试，单独重跑案例可以使用：

```bash
python scripts/run_part1.py --skip-tests --output-dir runs/part1-rerun
```

此时摘要会明确标记 `tests_skipped=true`，不能把它当成这次又运行了全套测试。每个输出目录用于一轮结果；再次使用相同目录会更新同名文件，需要保留对比时换一个目录。

## 5. 独立检查与代码阅读顺序

```bash
python -m unittest discover -s tests -v
python scripts/course_docs.py --check
```

| 阅读顺序 | 文件 | 核心问题 |
| --- | --- | --- |
| 1 | [next_token.py](../../zero2vllm/reference/next_token.py) | 如何从手工权重追踪每个中间矩阵 |
| 2 | [attention.py](../../zero2vllm/reference/attention.py) | 完整前向与缓存前向如何保持语义相同 |
| 3 | [phases.py](../../zero2vllm/reference/phases.py) | 每步工作量与 GEMM 估算分别如何计算 |
| 4 | [kv_geometry.py](../../zero2vllm/reference/kv_geometry.py) | 如何分块、按组读 K/V 与估计容量 |
| 5 | [metrics.py](../../zero2vllm/reference/metrics.py) | 时间戳如何变成可审查指标 |
| 6 | [positions.py](../../zero2vllm/reference/positions.py) | 位置、窗口和递推状态的数学关系 |
| 7 | [sampling.py](../../zero2vllm/reference/sampling.py) | 如何定义候选筛选、停止与数值比较 |
| 8 | [test_part1.py](../../tests/test_part1.py) | 独立手算、边界、负例和命令输出如何验收 |

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
| `--check` 提示目录不同步 | 维护者修改清单后运行 `python scripts/course_docs.py --write`，再检查 |
| 需要向 GPU 安装什么 | Part 1 无需 GPU；正式环境按 Part 4 的实机记录准备 |

## 8. 什么算这一篇完成

课程交付需要六节正文、对应代码/命令、可审阅结果、练习答案和有效测试；学生完成需要能解释综合练习，不能以“命令退出码为 0”替代掌握原理。

当前 CPU 数学参考不包含训练、完整 Transformer 堆栈、生产 Tokenizer、真实 vLLM 服务或 GPU 性能。Part 2 在这些可检查的小例子上实现 PyTorch 教学引擎；GPU 环境、源码与量化/算子专题分别沿后续课程推进。
