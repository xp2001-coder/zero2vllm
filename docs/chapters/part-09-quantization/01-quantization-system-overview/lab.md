# 9.1 实验：解释整数，再解释模型

[正文](README.md) · [练习与答案](exercises.md) · [已运行结果](results/README.md)

## 1. 环境和工作目录

本节必做实验使用 Python 3.11+ 标准库，无需 PyTorch、模型下载或 GPU。在仓库根目录运行下面的命令。已使用课程环境的读者直接沿用；新环境可先运行：

```bash
python -m venv .venv
```

然后选择当前系统的解释器。以下命令均写作 `python`：Windows 可替换为 `.venv/Scripts/python.exe`，Linux/macOS 可替换为 `.venv/bin/python`。这避免激活脚本因终端类型不同而产生歧义。

## 2. 先看非零 zero-point

```bash
python docs/chapters/part-09-quantization/01-quantization-system-overview/examples/walk_affine.py
```

入口读取固定输入 [affine.json](data/affine.json)，使用源参数 `s1=0.5,z1=3` 和目标参数 `s2=1,z2=2`。两者的逻辑整数范围均为 `[0,15]`。

先只看输出最前面的五行：

```text
x -> Q1 -> DQ1 -> Q2 -> DQ2
-1.5 -> 0  -> -1.5 -> 0 -> -2.0
 0.0 -> 3  ->  0.0 -> 2 ->  0.0
 0.5 -> 4  ->  0.5 -> 2 ->  0.0
 2.5 -> 8  ->  2.5 -> 4 ->  2.0
 6.0 -> 15 ->  6.0 -> 8 ->  6.0
```

空格只是便于阅读。这里两次出现的编码 `0` 分别代表 `−1.5` 与 `−2`，因为它们绑定不同的参数；实数零则由 `3` 变成了 `2`。请把图里的 `(q,s,z)` 当作整体阅读。

随后查看 JSON 的三个反例：

| 字段 | 应看到什么 | 解释 |
| --- | --- | --- |
| `wrong_dq_without_z` | `[0,1.5,2,4,7.5]` | 故意漏减 z，每个值偏大 1.5 |
| `rounding_order` | 正确编码 3，改动顺序后为 4 | 中点取偶数时，先 round 再加 z 的约定不能省略 |
| `double_rounding` | 经第一套表示得到 2，直接量化原值得到 3 | RQ 不能恢复第一次 Q 丢失的信息 |

`saturated_q2` 使用较小的目标范围 `[0,3]`，结果为 `[0,2,2,3,3]`；这是目标范围的饱和，不是 repack。

## 3. 再看对称 Linear

```bash
python docs/chapters/part-09-quantization/01-quantization-system-overview/examples/walk_qdq.py
```

固定输入在 [linear.json](data/linear.json)。每行会打印权重坐标、FP32 输入值、除以 scale 的结果、逻辑 q、反量化值和误差。

预期：

```text
scale             = 0.5
zero_point        = 0
q                 = [[-7,-2],[0,2],[5,7]]
original_output   = [[-5.5,-1.75]]
quantized_output  = [[-6.0,-2.5]]
weight max error  = 0.25
weight MSE        = 0.03125
output max error  = 0.75
```

对照正文，把第二列的原始和量化点积各算一次。确认权重误差与输出误差是不同量，再进入边界实验。

### 改小 scale，观察饱和

```bash
python docs/chapters/part-09-quantization/01-quantization-system-overview/examples/walk_qdq.py --scale 0.25
```

现在实数端点变成 `[-1.75,1.75]`。看 `out_of_range_count`、`half_scale_bound_applicable` 和 `dequantized`：原来较大的权重被压到端点，因此不能继续使用无裁剪条件下的半格误差界。此命令应正常退出，因为饱和是我们有意设置的合法实验。

### 非法输入应明确报错

```bash
python docs/chapters/part-09-quantization/01-quantization-system-overview/examples/walk_qdq.py --scale 0
```

预期退出码 **2**，提示 scale 必须为正且有限。它是故意失败的演示；不需要改代码去“让这条也通过”。

## 4. 一次运行完整数值案例

```bash
python -m scripts.experiments.lesson0901
```

默认写入 `.local/runs/part-09-quantization/01-system-overview/`：

| 文件 | 内容 |
| --- | --- |
| `summary.json` | 输入、中间值、误差、17 个案例、验证范围和源码摘要 |
| `cpu-passport.json` | 六权重对称 Linear 的记录 |
| `qwen-plan-passport.json` | Qwen3-8B W4A16 G128 的计划记录 |

程序应返回 0、17 行 `PASS`，且 `summary.json` 的 `passed=true`。不要仅检查总状态：`packed`、`gpu_validated`、`upstream_executed` 都应为 `false`。

17 项中包含有意设置的错误情况，例如漏减 zero-point、错误 scale、目标饱和。案例的 `PASS` 表示程序**观察到了预期差异或拒绝**，不表示那些错误参数也得到正确模型。

可以另指定输出目录，以保留多轮运行：

```bash
python -m scripts.experiments.lesson0901 --output-dir .local/runs/part-09-quantization/01-system-overview/replay
```

## 5. 读取 Passport

```bash
python docs/chapters/part-09-quantization/01-quantization-system-overview/examples/inspect_passport.py .local/runs/part-09-quantization/01-system-overview/cpu-passport.json
python docs/chapters/part-09-quantization/01-quantization-system-overview/examples/inspect_passport.py .local/runs/part-09-quantization/01-system-overview/qwen-plan-passport.json
```

本版结构有 56 个字段。对称 Linear 的预期统计为：26 个 verified、21 个 not_applicable、9 个 unknown。Qwen 计划只有 5 个 declared、51 个 unknown，没有 verified。

未知不等于失败。我们确实没有加载 Qwen 权重，因此 format、kernel 和 accumulator 就应该未知。CPU Passport 的实际 kernel 标为“不适用”，原因是该实验没有 GPU kernel；不要将这个值照搬到真实 Qwen 记录。

为一个真实观察填 `verified` 时，指向能够核查的结果、源码或日志。这个类只校验记录结构，不能认证引用内容的真伪。字段规范见 [Passport 模板](../templates/model-passport.md)。

## 6. 可选：固定源码静态核验

数值实验默认离线，不下载上游源码。需要核验 vLLM 接口时，可显式执行：

```bash
python -m scripts.experiments.lesson0901 --fetch-source
```

它按 [source-manifest.json](data/source-manifest.json) 下载缺失的固定提交文件到 `.local/sources/`，验证 SHA-256、九个函数及选定调用；不会安装或运行 vLLM。已有缓存时使用：

```bash
python -m scripts.experiments.lesson0901 --inspect-source
```

也可通过 `--source-root` 指定缓存父目录。网络不可用且没有缓存时，保留数值结果，先读公开链接；源码检查失败不应被写成已验证。自动检查不覆盖 AutoRound 的所有导出行为，后者本节只做固定文档阅读。

完整定位和阅读问题见 [source-map.md](source-map.md)。尤其观察后端是在何时被选择的，而不是预设它总在 `apply()` 内选择。

## 7. 检查命令

```bash
python -m unittest tests.quantization.test_passport tests.reference.test_quantization_overview -v
python -m scripts.docs.course --check
python -m scripts.docs.layout
```

第一条目前包含 33 项针对性检查：Passport、FP32 转换、Q/DQ、zero-point、RQ、舍入、饱和和命令入口。文档和结构检查用于发现断链或导航不一致，不会测量模型质量。

## 8. 交作业时至少保留什么

保留三份 JSON、自己的手算过程以及一段解释：这次验证到哪一层，还缺哪些证据。可用 [14 组练习](exercises.md) 检查理解程度。

这节的终点是“能够正确解释一个量化张量和一份配置记录”。实际文件压缩、GPU kernel 和服务指标在后续各节单独验收。
