# 2.1 实验：运行、对齐与定位误差

[返回正文](README.md) · [练习与答案](exercises.md)

## 1. 安装 CPU 学习环境

所有命令均在仓库根目录执行。本次本地实测环境为 Windows、Python 3.11.9、PyTorch 2.7.1+cpu。Linux 的 PyTorch 课程任务曾在原仓库通过一次 [GitHub Actions 验证](https://github.com/xpxpxp2001xpxpxp/zero2vllm/actions/runs/35063302785)。该链接是原账号的历史记录，账号停用后可能无法访问；新仓库的 CI 状态以 Actions 页面为准。安装会下载约 216 MB 的 Windows torch wheel，另有依赖；无需下载模型权重。

### Windows Git Bash

```bash
cd ~/Documents/ChatGPT/zero2vllm
python -m venv .venv
source .venv/Scripts/activate
python -m pip install torch==2.7.1 numpy==2.2.6 --index-url https://download.pytorch.org/whl/cpu
python -c "import sys, torch; print(sys.executable); print(torch.__version__)"
```

### Windows PowerShell

无需修改执行策略，直接使用虚拟环境解释器：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 numpy==2.2.6 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m zero2vllm torch-demo --output .local/runs/part-02-engine/01-pytorch-model/float64.json
```

### Linux Bash

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.7.1 numpy==2.2.6 --index-url https://download.pytorch.org/whl/cpu
```

已有 `.venv` 且装好依赖时直接使用，无需重复创建。下面统一写 `python`，代表虚拟环境解释器。PowerShell 未激活时将其替换为 `.\.venv\Scripts\python.exe`。

NumPy 用于避免 PyTorch 启动时提示缺少 NumPy 桥接依赖，本节计算没有调用 NumPy。第一篇依然只需要标准库。MACA/CUDA 实机使用各自核验过的环境，不要把本节 CPU wheel 覆盖到 GPU 环境；见[平台说明](../../../platforms/README.md)。

## 2. 第一次前向

### 三位置 Attention 手算

```bash
python docs/chapters/part-02-engine/01-pytorch-model/examples/attention_math.py
```

这段程序使用正文指定的 Q、K、V，依次打印分数、因果 mask、softmax 概率和输出。Attention 输出矩阵的最后一行约为 `[0.2447,0.1801,1.9957,0]`，未来位置的概率均为零。程序还演示了错误地加上 0/1 可见性矩阵的结果：第一个位置分给未来的总权重约为 `0.5317`。

### 两头 RoPE 与位置检查

```bash
python docs/chapters/part-02-engine/01-pytorch-model/examples/rope_math.py
```

这段程序复用模型的 `apply_rope`，再用 `math.sin/cos` 逐对计算作对照。输入形状为 `[1,2,3,4]`，两个头的原始向量不同，以检查旋转是否跨头混合。head 0 在位置 1 的结果应为 `[0.540302,0.841471,0.999950,0.010000]`，与正文配图一致。

程序还会检查每对向量的平方和、位置 0 不变以及输入未被原地修改。末尾应出现：

```text
Head 0 dot, Q@2 and K@1: 1.851421
Head 0 dot, Q@5 and K@4: 1.851421
Joint position shift / relative rotation: PASS
Wrong: one-token Q resets to 0; dot with K@1: 0.148479
Different score detected: PASS (KV cache needs explicit positions)
```

前两行保持原始 Q/K 内容不变，只同时移动位置；最后两行故意只把新 Q 的位置重置为 0。当前函数根据输入长度生成位置，不能直接用于只传一个新 token 的增量路径。这里演示的是位置接口问题，尚未实现 KV Cache。

### 三 token 模型前向

```bash
python docs/chapters/part-02-engine/01-pytorch-model/examples/walk_forward.py
```

输入是 `[[1,2,3]]`，与[正文生成示例](README.md#6-从-logits-到生成循环)相同。观察 `[1,3] → [1,3,8] → [1,3,16]`；Block 的 MLP 内部也会短暂出现 `[1,3,16]`，但那里的 16 表示中间特征数，最终 logits 的 16 表示词表大小。

末尾应显示 `Walkthrough matches model(tokens): PASS`，三个位置的贪心 ID 为 `[[5,9,10]]`，完整 prompt 后追加的一个 ID 为 `[10]`。这项检查验证模块串接路径；数值正确性还要运行第 3 部分的独立参考对齐。

### 再观察默认输入和完整记录

```bash
python docs/chapters/part-02-engine/01-pytorch-model/examples/run.py
```

开头应看到：

```text
logits shape: (1, 4, 16)
parameters: 1576
embedding (1, 4, 8)
layers.0.norm1 (1, 4, 8)
```

观察 gate/up/swiglu 的末尾维度为 16，down/output 恢复到 8，最终 logits 为 16。终端会打印生成 ID，它们只是未训练模型的输出。

## 3. 对齐报告

```bash
python -m zero2vllm torch-demo --output .local/runs/part-02-engine/01-pytorch-model/float64.json
python -m zero2vllm torch-demo --dtype float32 --output .local/runs/part-02-engine/01-pytorch-model/float32.json
```

两条命令的 `verification.passed` 都应为 `true`。默认两层的 `comparisons` 数组有 31 项，每项给出 stage、shape、max_abs_error 和 passed。

```bash
python -c "import json; r=json.load(open('.local/runs/part-02-engine/01-pytorch-model/float64.json',encoding='utf-8')); print(r['verification']); print(max(x['max_abs_error'] for x in r['comparisons'])); print(r['torch_generated_ids'])"
```

公开证据见 [float64](results/baseline/float64.json)、[float32](results/baseline/float32.json)。数值末位可以随环境略有变化；以逐元素容差和全部检查结果为准。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--prompt` | `1,2,3,4` | 逗号分隔的 ID，0–15 |
| `--max-new-tokens` | `4` | 新预测 token 数；0 不执行生成 |
| `--seed` | `7` | 权重初始化种子，不是采样种子 |
| `--dtype` | `float64` | 可选 float64/float32 |
| `--layers` | `2` | 正整数 Decoder 层数 |
| `--output` | 不写文件 | 额外保存 JSON；同时输出到终端 |

更多说明：`python -m zero2vllm torch-demo --help`。退出码 0 表示验证通过，1 表示数值验证失败，2 表示输入/依赖错误。

## 4. 七个完整案例

```bash
python -m scripts.experiments.lesson0201
```

依次运行 float64、float32、seed8、单 token/单层、三层、零输出、满上下文预测一个 token；再运行本节 13 项测试。输出在 `.local/runs/part-02-engine/01-pytorch-model/`，summary.json 保存状态、源码指纹和测试结果。

```text
.local/runs/part-02-engine/01-pytorch-model/
├── cases/          七个案例各自的 JSON
├── checks/tests.txt
└── summary.json
```

```bash
python -m unittest tests.reference.test_decoder tests.modeling.test_decoder -v
python -m scripts.docs.course --check
```

`--skip-tests` 仅适合已经独立运行测试的情况。若解释器没装 torch，PyTorch 测试会显示 skipped，这不能算本节通过。

## 5. 三个排错实验

### A. 权重方向

阅读 `TorchDecoder.__init__` 的 `source.T`。取消 LM Head 的转置会遇到 `[8,16]` 与 `[16,8]` 的形状冲突；取消方阵 Q 的转置则不会发生形状错误，却会导致 Q 开始偏离参考。做完后恢复该行，再运行对齐。

### B. 因果遮罩

把 `future` 临时改为同形状的全 False 张量，再运行 `test_future_mask_and_prefix_extension`：后面 ID 改变会影响前两个位置。恢复上三角遮罩后测试应重新通过。

### C. 输入与容量

```bash
python -m zero2vllm torch-demo --prompt 16
python -m zero2vllm torch-demo --prompt 1,2 --max-new-tokens -1
python -m zero2vllm torch-demo --prompt 1,2 --max-new-tokens 32
```

三条都应返回输入错误。第三条最后一次前向需要 `2+32-1=33` 个位置，超过容量 32；把生成数改为 31 即为合法边界。

## 6. 常见问题

| 现象 | 检查方法 |
| --- | --- |
| 找不到 torch | 打印 sys.executable，确认安装和运行使用同一个解释器 |
| 下载失败 | 核对官方源、网络和 Python 版本，保留安装错误 |
| tokens 类型错误 | 必须是二维 torch.long；单请求也要写 `[[1,2,3]]` |
| 形状对但数值错 | 找首个失败中间层，再查权重方向、RoPE、softmax 轴 |
| float32 不逐位相等 | 检查容差；不要用 Python `==` 比较浮点结果 |
| 生成内容不可读 | 随机 16-token 教学模型没有语言能力，真实权重在 2.2 |
| 第一次启动较慢 | 导入依赖有开销；不要用启动耗时评价推理吞吐 |

## 7. 作业产出

提交模型配置、权重哈希、两种 dtype 报告、一个故障的首个失败点与修复说明，并完成[六组练习](exercises.md)。个人结果放进已忽略的 `.local/runs/part-02-engine/01-pytorch-model/`；核验后的公开结果放进本节 [results/](results/README.md)，保留运行时间、环境与来源摘要。
