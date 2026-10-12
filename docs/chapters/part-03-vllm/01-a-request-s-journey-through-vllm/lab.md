# 3.1 实验：从请求路径图走到固定版本源码

[正文](README.md) · [练习](exercises.md) · [源码定位表](source-map.md) · [实际核验记录](results/README.md)

目标：亲手定位一条请求的提交、迭代、返回与取消，标清消息和进程边界。基础实验不需要 vLLM 环境、PyTorch、模型权重或 GPU。

## 1. 在哪里运行

在 zero2vllm 仓库根目录执行，即包含 `pyproject.toml` 的目录。Windows PowerShell、Git Bash、Linux 均可使用下面的单行命令；若系统使用 `python3`，替换开头的 `python`。

```bash
python --version
python -m scripts.experiments.lesson0301 --help
```

需要 Python 3.11+。已有虚拟环境可以继续使用，无需给它安装 vLLM。工具通过标准库解析源码文本，不执行被读取文件中的 import、装饰器或函数。

## 2. 先运行不用联网的手算例子

```bash
python docs/chapters/part-03-vllm/01-a-request-s-journey-through-vllm/examples/walk_iteration.py
```

先在纸上写出 B、A 顺序下的四个输入位置，再对照：

| 输出 | 预期 |
| --- | --- |
| `input_ids` | `[23,12,13,14]` |
| `positions` | `[3,2,3,4]` |
| `slot_mapping` | `[23,34,35,12]` |
| `illustrative_boundaries` | `[0,1,4]` |
| 按 ID 归还 | A 得到空列表，B 得到 `[24]` |
| 错误分配示例 | A 错拿 `[24]`，B 错拿空列表 |

为什么槽位从 35 跳到 12？A 从逻辑块 0 进入逻辑块 1，它们映射到物理块 8 和 3；物理块不必相邻。为什么 24 没有出现在输入中？它是本轮假设产生的新输出，尚待下一轮读入。

代码还检查 A、B 顺序。排列变化会改变拼接后的数组，但不应改变结果属于哪条请求。这里的输出为教学设定，程序不调用任何模型。

## 3. 首次准备源码

```bash
python -m scripts.experiments.lesson0301 --fetch
```

`--fetch` 才会联网，从 vLLM 官方仓库的固定完整提交下载缺少的 25 个文件（包含 LICENSE）。文件保存到：

```text
.local/sources/vllm-v0.8.5/
├── LICENSE
└── vllm/
    ├── entrypoints/
    ├── executor/
    └── v1/
```

这是供阅读的部分源码快照，不能当作完整 vLLM 安装包。它与项目自己的 `zero2vllm/` 代码分开，也不会上传到 GitHub。

每个文件下载后先核对 SHA256，再保存。已有且匹配的文件直接复用；已有文件若被修改，工具报错并保留原文件，不会覆盖你的笔记或源码修改。网络中断后可重新运行同一命令，继续补齐缺少文件。

如果你已有这个版本的完整 checkout，可以直接指定仓库根目录：

```bash
python -m scripts.experiments.lesson0301 --source-dir /path/to/vllm
```

请把示例路径替换成实际路径。工具核对所需文件的内容，不要求该目录一定有 `.git`；它也不声称检查了未列入清单的其他文件。

## 4. 后续默认离线运行

```bash
python -m scripts.experiments.lesson0301
```

成功结尾应为：

```text
Static source inspection: PASS; 25 files, 38 functions, 4 structures
No vLLM import, model execution or GPU validation was performed.
```

这证明固定源码的文件摘要、函数位置、指定调用表达式和指定结构字段满足本课约定。它不证明部署成功、调用分支实际执行、GPU 正确性或性能。

## 5. 分四次读，不用一次打开所有文件

下面的 `--focus` 只筛选终端导航，完整源码仍会检查，JSON 中仍保存全部检查点。

### 第一轮：提交

```bash
python -m scripts.experiments.lesson0301 --focus input
```

先看 `http → chat → register → submit → send-add → receive-input → core-add`。这是一条人工选择的阅读顺序，不是工具捕获的运行时堆栈。

在每处写一句话：输入对象是什么，输出对象是什么。然后核对 `submit` 中的调用先后，解释为什么先登记输出状态。

补读 `front-init`，确认实际构造的客户端。`client-choice` 是对照工厂分支，不在本节 AsyncLLM 的直接初始化链上。

### 第二轮：执行

```bash
python -m scripts.experiments.lesson0301 --focus core
```

从 `step` 的三个动作开始，再到 `executor → uni-rpc → worker → runner → feedback`。`collective_rpc` 动态调用不能靠静态名称自动证明目标类型，要同时阅读执行器选择和继承关系。

查找 `SchedulerOutput` 的字段，对照双请求示例解释：本轮计算 4 个输入位置，与返回 1 个有效新 token，为什么并不矛盾。

### 第三轮：返回

```bash
python -m scripts.experiments.lesson0301 --focus output
```

跟踪 `send-output → receive-output → output-loop → output-process → collector → sse`。分别标出线程、异步任务、普通函数调用。

阅读 `collector`，找到未取走结果的合并分支；阅读 `output-process`，找到查不到请求状态时的行为。不要把 2.7 的有界队列策略直接填到这两个位置。

### 第四轮：取消

```bash
python -m scripts.experiments.lesson0301 --focus cancel
```

按 `abort-front → abort-send → abort-core → finish → release → kv-free → block-free` 阅读，再补 `runner-cleanup`。

必须把中间没有以普通函数调用连接的部分也标出：ABORT 消息、输入线程、进程内队列和主循环分派。定位 `core-loop`，说明主循环在什么时候处理取消。

## 6. 怎样读生成的报告

默认文件放在：

```text
.local/runs/part-03-vllm/01-request-journey/
├── summary.json
└── source-map.md
```

| 字段 | 怎样读 |
| --- | --- |
| `commit` | 固定上游版本的完整标识 |
| `files[].sha256_lf`、`matched` | 本次读取的源码摘要是否符合清单 |
| `checkpoints[].symbol`、`line` | 函数限定名与实际源码行 |
| `checkpoints[].call_sites` | 在该函数体中找到的指定调用表达式和行号 |
| `structures[].declared_fields` | 从对应类定义中读到的带类型注解的字段 |
| `verification_level` | 必须是 `static_source_inspection` |
| `executed_upstream_code`、`gpu_validated` | 本节都为 `false` |
| `course_source` | 当前课程代码版本与源码指纹，区别于上游提交 |

需要保留不同阅读轮次时，指定独立目录：

```bash
python -m scripts.experiments.lesson0301 --focus cancel --output-dir .local/runs/part-03-vllm/01-request-journey/cancel-reading
```

## 7. 错误也要能解释

| 现象 | 原因与处理 |
| --- | --- |
| `Missing source file` | 尚未准备快照；联网时显式加 `--fetch`，或指定完整本地路径 |
| `Source hash mismatch` | 版本不一致或文件修改；保留原目录，在新的专用目录重新取固定版本 |
| 下载超时、证书或代理错误 | 网络未完成；恢复连接后重试，保持证书校验，不必安装 vLLM |
| `Missing function` / 缺少调用表达式 | 清单或被读取的源码不符合阅读约定；按报告先核对摘要，再核对限定名 |
| `No module named scripts` | 没有在 zero2vllm 仓库根目录运行 |
| 报告 PASS，但启动 GPU 仍失败 | 源码静态核对与硬件部署属于不同验证范围 |

想练习版本不匹配：复制一份源码快照到个人目录，只在副本的目标 `.py` 末尾加一行注释，再用 `--source-dir` 检查副本。应得到非零退出状态与摘要不匹配；不需要改原始快照。

## 8. 自动检查和人工验收

```bash
python -m unittest tests.maintenance.test_source_journey -v
python -m scripts.docs.course --check
python -m scripts.docs.layout
```

16 项源码工具测试不联网，使用独立临时文件覆盖：换行差异、源码改动、同名函数、嵌套作用域、假调用文本、消息字段、下载校验及默认离线行为。它们验证的是课程工具；对真实上游快照的核验需另运行本节命令。

交付三份内容：

1. `summary.json` 与一段解释：这个 PASS 能证明什么、不能证明什么。
2. 一张自己标注的请求路径图，含进程/线程/任务边界与消息结构。
3. 一份取消回收表：前端状态、核心请求、块引用、runner 状态各由谁释放。

源码实验全部通过即可完成本节的基础验收。实机扩展应在已经验证可用的 A100 或 C500 环境中，记录实际版本、进程 PID、模型/Tokenizer revision、执行器与注意力后端，再用正常结束和中途断连核对这张图。该扩展尚待平台章节实测，不要求初学者现在安装历史版本。
