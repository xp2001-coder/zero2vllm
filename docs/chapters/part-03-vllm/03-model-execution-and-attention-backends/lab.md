# 3.3 实验：从一份调度计划走到张量结果

[正文](README.md) · [练习](exercises.md) · [源码定位](source-map.md) · [运行记录](results/README.md)

本节分三个层次：①标准库手算与打包；②真实 CPU 张量对照；③固定官方源码阅读。前两项运行的是课程代码，第三项只读取源码。三者都不代表上游 vLLM 或 GPU 内核已运行。

## 1. 从仓库根目录开始

下面命令在包含 `pyproject.toml` 的 zero2vllm 根目录执行。需要 Python 3.11+，适用于 PowerShell、Git Bash 和 Linux shell；若系统使用 `python3`，替换命令开头的 `python`。

```bash
python --version
python -m scripts.experiments.lesson0303 --help
python -m scripts.experiments.lesson0303 --mode metadata
```

最后一条只使用标准库，预期得到 `packing: PASS` 和 `packing_reordered: PASS`。默认不联网，也不需要下载模型。

结果默认保存在 `.local/runs/part-03-vllm/03-model-execution/summary.json`。重复运行会覆盖同目录下的记录；下面的对照实验分别指定目录，便于保留结果。

## 2. 先打印输入坐标，暂时不算模型

```bash
python docs/chapters/part-03-vllm/03-model-execution-and-attention-backends/examples/walk_inputs.py
```

输入来自[requests.json](data/requests.json)。B 先执行，A 后执行，预期为：

| 字段 | 预期值 | 自查问题 |
| --- | --- | --- |
| `input_ids` | `[23,12,13,14]` | 是否只取本轮 C:C+q，而不是所有历史？ |
| `positions` | `[3,2,3,4]` | 是否保留请求内的绝对位置？ |
| `slot_mapping` | `[23,34,35,12]` | 是否按各请求自己的块表寻址？ |
| `query_start_loc` | `[0,1,4]` | 最后元素是否等于本轮实际 token 数？ |
| `seq_lens` | `[4,5]` | 是否包含历史 C？ |
| `logits_indices` | `[0,3]` | 是否取每段本轮输入的最后一行？ |

把 A 的位置 4 单独算一遍：`4//4=1`，块表第 1 项是 3，偏移 `4%4=0`，所以 slot=12。这个位置存放的 token ID 是 14；14、位置 4 和 slot 12 分别代表词表下标、序列位置与缓存槽位。

### 对照：交换执行顺序

```bash
python docs/chapters/part-03-vllm/03-model-execution-and-attention-backends/examples/walk_inputs.py --order AB
```

预期 token 为 `[12,13,14,23]`，位置为 `[2,3,4,3]`，槽位为 `[34,35,12,23]`，Query 边界为 `[0,3,4]`，logits 行为 `[2,3]`。请求没变，只是数组排列变了。

## 3. 安装 CPU 数值实验依赖

如果你已经完成第二篇的 PyTorch 实验，可以复用原环境。本节使用此前固定的 **PyTorch 2.7.1 CPU**；已有 GPU 开发环境建议单独使用课程虚拟环境，以便保留自己的设备依赖。

```bash
python -m venv .venv
```

新环境选择对应激活方式，仅执行适合当前 shell 的一条：

```powershell
.venv\Scripts\Activate.ps1
```

```bash
# Git Bash on Windows
source .venv/Scripts/activate
# Linux
source .venv/bin/activate
```

再安装并核对：

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -c "import torch; print(torch.__version__)"
```

如果 PowerShell 的执行策略阻止激活，可以直接使用 `.venv/Scripts/python.exe` 代替后面的 `python`；本实验不需要修改系统执行策略。

不必安装 Transformers、vLLM 或下载 checkpoint。夹具在 CPU 上生成固定种子的浮点参数，词表 ID 只是整数，不能把输出解读成有语言能力的回答。

## 4. 执行完整 12 个案例

```bash
python -m scripts.experiments.lesson0303 --output-dir .local/runs/part-03-vllm/03-model-execution/float64
```

预期 12 个案例全部 PASS：

| 案例 | 验证什么 |
| --- | --- |
| `packing` | B/A 顺序的 token、位置、slot、边界和 logits 行 |
| `packing_reordered` | A/B 顺序下所有数组一起更新 |
| `weight_orientation` | `[K,N]` 数学乘法与 `[N,K]` 普通 Linear 权重对应 |
| `fused_qkv` | 一次融合投影与三次独立投影一致 |
| `paged_matches_dense` | 分页批次中每条请求的 logits 与独立 dense 参考一致 |
| `reordered_results` | 换执行排列不改变结果归属和每条请求的 logits |
| `causal_future` | 修改同请求后面的输入，不影响前面 Query；被修改位置确实改变 |
| `isolated_requests` | 修改 B 的全部历史，A 的结果不受影响 |
| `gqa_head_groups` | 按 KV 组索引等价于参考实现的头重复 |
| `graph_padding` | 8 行投影缓冲只对应 4 个真实 token 的写入和结果 |
| `partial_prefill_output` | A 的候选丢弃，B 的候选保留 |
| `unwritten_tail` | 只写 9 个历史及新增位置，未使用槽仍是 NaN，输出保持有限 |

`graph_padding` 的名称指这个案例解释图模式相关的补齐概念；**没有进行 CUDA Graph 捕获或回放**。

### 读取 JSON，不只看 PASS

打开上述输出目录的 `summary.json`，先看：

- `verification_level=cpu_tensor_fixture`：CPU 单层数值夹具。
- `upstream_executed=false`、`full_llama_executed=false`、`gpu_validated=false`。
- `torch_version`、`dtype`、`device` 与 `course_source`：记录实际环境和课程源码摘要。
- 每个案例的 `checks`：是哪条数值或状态约束通过了。

在 `paged_matches_dense` 中，默认应有以下形状：

```text
hidden          [4,8]
q               [4,4,2]
k / v           [4,2,2]
cache           [2,10,4,2,2]
logits          [4,32]
selected_logits [2,32]
```

这里的 `logits=[4,32]` 是课程夹具为了逐行对照而保留的全部结果。正文解释了正式 Runner 通常先选择隐藏状态，再只对必要行计算采样 logits。

维护者本次 float64 运行的最大绝对误差约 `2.22e-16`，小于 `1e-11` 的验收容差；以你当前运行的误差记录为准。默认候选 `[24,16]` 对应 `[B,A]`，实际保留 `{B:[24], A:[]}`。

## 5. 改排列与缓冲长度，观察不变量

```bash
python docs/chapters/part-03-vllm/03-model-execution-and-attention-backends/examples/walk_inputs.py --tensors
python docs/chapters/part-03-vllm/03-model-execution-and-attention-backends/examples/walk_inputs.py --order AB --tensors
python docs/chapters/part-03-vllm/03-model-execution-and-attention-backends/examples/walk_inputs.py --pad-to 8 --tensors
```

第一条按默认顺序执行；第二条换成 A/B，此时候选顺序随之改变，但按 ID 还原后仍应是 A 无输出、B 得到 24。

第三条的 `num_input_tokens=8`，而 `num_actual_tokens=4`。Q/K/V 投影有 8 行，但 `slot_mapping` 仍只有 4 项，Query 边界仍以 4 结束，最终只检查 4 个真实输入位置。

完整案例中的 padding token 使用另一个合法 ID 31，以证明尾部内容变化不影响结果；短示例默认用 ID 0。两者都必须排除对无效行的缓存写入。

## 6. float32 是另一份数值记录

```bash
python -m scripts.experiments.lesson0303 --dtype float32 --output-dir .local/runs/part-03-vllm/03-model-execution/float32
```

预期同样全部通过，容差为 `2e-5`。本节分别在各 dtype 下构造参数，并与该次参数的 dense 参考比较；**不要把 float32 和 float64 的随机参数输出直接相减**，同一 seed 在不同 dtype 下不承诺生成逐元素相同的随机权重。

也不要只看 greedy token 一样就通过。每条请求的完整有效 logits 误差、因果隔离、输出归属和写入边界都应该同时成立。

## 7. 源码准备与离线核验

第一次显式从官方固定提交准备本节 15 个文件（包含 LICENSE）：

```bash
python -m scripts.experiments.lesson0303 --mode metadata --fetch-source --output-dir .local/runs/part-03-vllm/03-model-execution/source
```

下载目录默认复用 3.1—3.2 的 `.local/sources/vllm-v0.8.5`。已有且摘要正确的文件不会重复下载；与摘要不一致的已有文件会保留并报错。它是源码阅读子集，不是完整可安装的 checkout。

之后只做离线检查：

```bash
python -m scripts.experiments.lesson0303 --mode metadata --inspect-source --output-dir .local/runs/part-03-vllm/03-model-execution/source
```

预期额外显示 `Static inspection: PASS; 27 functions`，JSON 中还有 15 个文件摘要及 2 类结构字段的检查。默认的 metadata / tensor 模式都不会自动联网，只有 `--fetch-source` 会请求下载。

语法树检查验证的是文件身份、符号、函数体调用表达式和结构字段，**不证明任何上游函数执行过**。请对照[源码定位表](source-map.md)完成下面的阅读任务：

| 阅读链 | 亲手回答的问题 |
| --- | --- |
| `load_model → get_model → DefaultModelLoader` | 类实例化、参数写入和后处理分别在哪里？ |
| `LlamaModel.load_weights → QKVParallelLinear.weight_loader` | 分开的 q/k/v 如何找到融合区域？ |
| `_update_states → _prepare_inputs` | 为什么请求身份和 batch 下标不能等同？ |
| `Attention.forward → FlashAttentionImpl.forward` | 元数据从哪里来？KV 写入和 Attention 计算是哪两个入口？ |
| `execute_model → compute_logits → 采样结果过滤` | A 的候选为什么不能进入正式输出历史？ |

## 8. 修改实验时的边界

输入数据可在[requests.json](data/requests.json)阅读，但默认案例的精确预期也来自这份固定输入。若想探索其他尺寸，建议先在短示例中构造新 `RequestSlice` 并保留默认数据，用 `AttentionFixture.execute()` 与它内部的 dense 参考检查数值；不要改完输入后把原来的 PASS 预期硬改掉。

夹具的模型尺寸固定为 H=8、Hq=4、Hkv=2、D=2。不要只改配置说明而不改投影和 reshape。它支持本节的 float32/float64、普通 RoPE 和不重叠块表，不包含前缀共享、分布式、MLA、滑动窗口、量化或真正的图执行。

每次运行重新生成此前 C 个位置的缓存，是为了构造一个明确的单轮起点；不会测量连续服务吞吐或缓存命中收益。

## 9. 测试与故障排查

```bash
python -m unittest tests.reference.test_execution tests.modeling.test_packed_attention -v
python -m scripts.docs.course --check
python -m scripts.docs.layout
```

新增检查共 30 项：13 项标准库输入/工具检查，17 项 PyTorch 数值检查。没有安装 PyTorch 时，后者会显式跳过；只有“0 项跳过”的记录才表示本次把所有数值检查实际运行了。

| 现象 | 排查方向 |
| --- | --- |
| 找不到 `scripts` | 是否在仓库根目录、是否用对 Python 环境 |
| 缺少 PyTorch | 先完成 metadata 模式，或在课程环境中安装固定 CPU 版本 |
| block table 不足 | 是否覆盖到 `C+q`，而不只是本轮 q |
| token 或 block ID 越界 | 区分词表 ID、请求内位置、物理块 ID 与 slot |
| shape 正确但数值不同 | 检查方阵转置、Q/K/V 拼接顺序、GQA 分组与 RoPE 位置 |
| 修改未来 token 改变了过去输出 | 检查因果掩码是否带了历史前缀偏移 |
| 换请求排列后答案混了 | 按本轮 ID/行映射还原，不要按登记顺序 zip |
| padding 后出现 NaN | 无效槽被读入了，或实际长度和缓冲长度混用 |
| 源码缺失或摘要不同 | 用 `--fetch-source` 准备固定提交；保留自己修改过的源码，另选目录复核 |

## 10. 以后在 GPU 上应该补什么证据

本节不提供未经实机确认的 C500 启动参数。A100 和 C500 分别按对应平台环境课准备：固定软件栈、模型、精度和源码版本，记录实际后端选择，再对齐输入、位置、块布局和输出。

验收至少应补齐：实际后端类与启动日志、设备上的前向结果、支持的 dtype/块长/头维度、与参考的数值差异，以及真实执行模式。需要声称某个内核或某项加速生效时，还要保存相应追踪与明确计时范围。

先把本节的输入契约和数值边界解释清楚，再迁移到设备；CPU 夹具的 PASS 不应被填成 GPU 验证结果。
