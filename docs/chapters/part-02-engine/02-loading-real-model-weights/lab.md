# 2.2 实验：下载、检查与逐层对齐

[正文](README.md) · [练习与答案](exercises.md) · [本轮实测](results/README.md)

所有命令从仓库根目录运行。实验使用 CPU，不需要 GPU、Hugging Face 登录或付费服务。首次下载需要访问 Hugging Face；下载完成后可离线重复执行。

## 1. 准备环境

已做完 2.1 的读者继续使用原 `.venv`。新环境先按 [2.1 安装步骤](../01-pytorch-model/lab.md#1-安装-cpu-学习环境)安装 Python 3.11 和 CPU PyTorch 2.7.1。本节固定以下组合，不使用未固定的最新版本：

| 依赖 | 版本 | 作用 |
| --- | --- | --- |
| Python | 本轮 3.11.9 | 运行环境 |
| torch | 2.7.1+cpu | 我们的模型和官方参考计算 |
| numpy | 2.2.6 | PyTorch/Transformers 依赖 |
| transformers | 4.51.3 | 参考模型、Tokenizer 与参考生成循环 |
| safetensors | 0.5.3 | 读取张量文件 |
| huggingface-hub | 0.30.2 | 固定 revision 的快照下载 |
| tokenizers | 0.21.1 | 快速 Tokenizer |

### Windows Git Bash

```bash
source .venv/Scripts/activate
python -m pip install transformers==4.51.3 safetensors==0.5.3 huggingface-hub==0.30.2 tokenizers==0.21.1
```

### Windows PowerShell

直接调用虚拟环境解释器，无需修改执行策略：

```powershell
.\.venv\Scripts\python.exe -m pip install transformers==4.51.3 safetensors==0.5.3 huggingface-hub==0.30.2 tokenizers==0.21.1
```

### Linux Bash

```bash
source .venv/bin/activate
python -m pip install transformers==4.51.3 safetensors==0.5.3 huggingface-hub==0.30.2 tokenizers==0.21.1
```

下面统一写 `python`。PowerShell 未激活环境时，替换为 `.\.venv\Scripts\python.exe`。也可以在已装 CPU torch 的环境中使用 `python -m pip install -e ".[checkpoint]"` 安装本项目的可选依赖组。

本轮实测是 Windows CPU；Linux 已加入 CI 离线测试步骤，是否通过以实际运行记录为准。不要把 CPU wheel 安装进用于 A100/C500 的厂商环境。

## 2. 不下载模型，先看三个容易算错的布局

```bash
python docs/chapters/part-02-engine/02-loading-real-model-weights/examples/layout_traps.py
```

检查三件事：

1. 正确 Linear 输出 `[5,11]`，错误转置后是 `[7,10]`。
2. 同一个四维向量，相邻配对与前后半区配对的 RoPE 在位置 1 不同；经过正确坐标置换可以对上。
3. 正确 GQA 头顺序为 `[0,0,0,1,1,1,2,2,2]`，普通 `repeat` 的顺序不同。

末尾应显示 `PASS: all three mistakes preserve shape but change values or meaning.`。这三个例子都不能仅凭 shape 判断正确。

## 3. 下载固定版本并完成第一次对照

```bash
python -m scripts.experiments.lesson0202 --download
```

默认下载到 `.local/models/smollm2-135m/`，结果写到 `.local/runs/part-02-engine/02-loading-real-model-weights/`。

权重文件实际为 **269,060,552 字节**，加上配置与 Tokenizer 文件约 273 MB；Python 依赖另计。对照时同时持有两份约 538 MB 的 FP32 参数，还会用到激活和运行库内存，建议为进程留出数 GB 余量。下载时间受网络影响；本节不把耗时当作性能指标。

终端应出现：

```text
Loaded 272 tensors; 134,515,008 parameters.
PASS 'The capital of France is': [260, 3575, 282, 260, 1798, 30, 198, 198]
PASS 'Once upon a time': [28, 665, 436, 253, 1838, 8180, 3365, 14176]
PASS '1 + 1 =': [216, 33, 32, 32, 32, 32, 32, 32]
```

这些是本轮 Windows CPU 记录，不是对所有版本与设备的逐位承诺。验收以误差检查和两条路径生成 ID 一致为准，不能只比较终端中的某一句文本。

### 本次命令执行了什么

1. 读取 [data/model.json](data/model.json) 中的完整模型 revision。
2. 下载指定的九个文件，保存摘要清单。
3. 检查配置与所有权重，将 BF16 权重复制为 CPU FP32 参数。
4. 使用同一 Tokenizer 产生输入 ID。
5. 用我们自己的模型与 Transformers 分别计算，比较 33 个阶段。
6. 两条路径分别进行最多八步贪心生成，比较生成 ID，写入报告。

模型源文件不进入 Git；`.local/` 已被忽略。本节报告不保存令牌或登录凭据。

## 4. 离线复跑与检查快照

下载完成后，去掉 `--download`：

```bash
python -m scripts.experiments.lesson0202
python docs/chapters/part-02-engine/02-loading-real-model-weights/examples/inspect_snapshot.py
```

第二条命令只读取配置、张量头信息与 Tokenizer，不构建完整 Decoder。观察：

```text
Layers=30, hidden=576, head_dim=64
Q heads=9, KV heads=3, tensors=272
model.layers.0.self_attn.k_proj.weight: shape=[192, 576], dtype=BF16
lm_head.weight stored separately: False
```

也可以只验证配置与权重加载，不运行 Transformers 对照：

```bash
python -m scripts.experiments.lesson0202 --inspect-only --output-dir .local/runs/part-02-engine/02-loading-real-model-weights/inspect
```

这个模式的 `verification` 为 `not_run`，`passed` 为 `null`，退出码 0 只表示读取和加载检查成功。**它不能用作数值对齐通过的证据。**

## 5. 改输入与生成长度

```bash
python -m scripts.experiments.lesson0202 --prompt "Once upon a time" --max-new-tokens 4 --output-dir .local/runs/part-02-engine/02-loading-real-model-weights/story
python -m scripts.experiments.lesson0202 --prompt "Hello" --max-new-tokens 0 --output-dir .local/runs/part-02-engine/02-loading-real-model-weights/forward-only
```

零输出案例仍比较首次前向，但两条路径的生成 ID 都为空。`--prompt` 可以重复传入，每条依次执行，不是变长 batch。

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--download` | 关闭 | 显式允许本次访问公开模型仓库 |
| `--snapshot` | `.local/models/smollm2-135m` | 含下载回执的完整快照目录 |
| `--inspect-only` | 关闭 | 只检查权重，不运行数值参考 |
| `--prompt` | 配方中的三个输入 | 可重复；每条编码后 1–128 token，不截断长输入 |
| `--max-new-tokens` | 8 | 允许 0–32；遇到 EOS 可提前停止 |
| `--threads` | 2 | CPU 线程数，至少 1；不代表最优性能配置 |
| `--output-dir` | 本节 `.local/runs/` 目录 | 相对仓库根目录；同名报告会覆盖 |

128 token 是本节密集 Attention 实验的输入控制，不是模型的上下文上限。模型配置的 8192、实验输入限制与生成步数是三个不同约束。

## 6. 读懂报告

打开 `summary.json`，先核对这些字段：

| 字段 | 应如何解释 |
| --- | --- |
| `checkpoint.repo_id / revision / files` | 模型身份与本轮每个文件的摘要 |
| `source` | 课程 Git commit、工作区是否有修改、源文件摘要 |
| `environment` | Python、包版本、CPU、FP32 与线程数 |
| `cases[].input_ids` | 两条路径共用的输入 |
| `cases[].comparisons` | 每个阶段的 shape、最大/平均绝对误差和通过状态 |
| `cases[].last_top5` | 最后输入位置的五个高分候选；完整 logits 已参与比较 |
| `cases[].our_generated_ids / reference_generated_ids` | 两套生成循环的独立结果 |
| `passed` | 所有输入的数值和生成结果都通过时才为 true |
| `gpu_status` | A100/C500 均为 `not_run` |

`weights.json` 逐项记录源权重名、目标参数名、shape、存储 dtype 与计算 dtype。比较两次实验时，可先查看 `checkpoint.files` 与 `source.source_sha256`，再检查数值；不要先用不同版本的结果比较速度。

## 7. 离线回归测试

```bash
python -m unittest tests.modeling.test_checkpoint -v
```

测试在临时目录中用固定种子创建随机小模型，再保存为真实 safetensors 文件；它不下载 SmolLM2。覆盖：

- GQA/MHA 的所有 logits，以及输入 batch 和因果前缀不变性。
- 共享与不共享 LM Head；显式保存的两份共享权重发生冲突。
- 单文件和分片索引、缺失文件、索引与实际内容不符。
- 丢失/额外参数、形状、整数 dtype、非有限值、不支持的配置。
- BF16 到 FP32 的数值保留，未加载模型禁止前向。
- EOS、零生成、上下文边界和 Tokenizer 到生成的完整对照。
- 错误转置 Q 的反例，以及文件被改动后回执校验失败。

只安装标准库时该测试组会明确跳过；跳过不是通过。CI 执行这些离线小模型测试，完整真实权重对照由本节运行器与公开报告提供。

## 8. 常见问题

| 现象 | 处理方法 |
| --- | --- |
| `No snapshot receipt` | 第一次先加 `--download`；迁移快照时把九个文件和回执一起复制 |
| 下载超时、无法连接 | 检查到 Hugging Face 的网络访问；保留已下载文件，恢复网络后重跑下载命令 |
| 提示没有 `hf_xet`，随后继续下载 | 固定版本库可回退到普通 HTTP；以最后命令结果为准 |
| `Snapshot file missing or changed` | 核对是否改动了配置/Tokenizer；在新的快照目录重新下载固定版本再比较 |
| `Checkpoint keys: missing=...` | 对照映射和共享配置，不要直接忽略缺失项 |
| `Shape mismatch` | 先看 GQA 头数、MLP 宽度，以及是否错误转置 |
| `Unsupported ...` | 超出本节支持范围；不要删掉检查后宣称兼容 |
| 内存不够 | 先运行不下载模型的布局示例和离线小模型测试；完整对照需为两份模型及运行库预留内存 |
| 输出不像正确答案，但报告全通过 | 小型基础模型的生成质量有限；检查实现与质量评测是两项工作 |
| Windows 测试改写权重文件失败 | 从 safetensors 读取的张量可能仍引用文件映射；先 `clone()` 脱离映射，再改写测试文件 |

最后做一次文档与目录检查：

```bash
python -m scripts.docs.course --check
python -m scripts.docs.layout
```

完成后，提交你自己的 `summary.json` 观察笔记：模型版本是什么、输入有哪些 token、从哪层开始比较、共享关系怎样确认。不要把下载的权重或整个 `.local/` 目录加入提交。
