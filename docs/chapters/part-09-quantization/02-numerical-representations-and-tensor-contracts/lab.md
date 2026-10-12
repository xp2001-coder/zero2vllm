# 9.2 实验：先把数值和形状对齐

[正文](README.md) · [练习](exercises.md) · [运行记录](results/README.md)

所有命令在仓库根目录执行。默认 Python 3.11+，无需 GPU、PyTorch、模型下载或网络；Windows 若终端显示中文异常，可使用 `python -X utf8`。可选 PyTorch 对照单独运行。

## 一、运行固定案例

```bash
python -m scripts.experiments.lesson0902
```

预期 20 行 PASS，退出码 0。生成：

```text
.local/runs/part-09-quantization/02-tensor-contracts/
├── summary.json
├── symmetric-passport.json
└── asymmetric-passport.json
```

默认可重复运行，覆盖的是本地实验目录；不会覆盖课程发布的历史结果。要保留一次尝试：

```bash
python -m scripts.experiments.lesson0902 --output-dir .local/runs/part-09-quantization/02-tensor-contracts/replay
```

主输入是 [grouped-linear.json](data/grouped-linear.json)。先对照这些值：

| 字段 | 预期 |
| --- | --- |
| 对称 q | `[[-7,0,1],[-2,7,7],[1,3,-7],[7,7,2],[7,7,-7]]` |
| `linear.y_original` | `[[-3.5,6.5,-7.75],[-1.5,-3.5,3]]` |
| `linear.y_qdq` | `[[-3.5,6.5,-8],[-1.75,-3.5,3.25]]` |
| 权重 max_abs / MSE | `0.25 / 0.008333333333333333` |
| 输出 max_abs | `0.25` |
| 非对称物理 shape | `[6,3]`；逻辑仍是 `[5,3]` |
| 非对称最后一行 padding q | `[0,0,15]`，DQ 是 `[0,0,0]` |
| 声明存储总量 | `384 bit = 48 byte`，`25.6 bpw` |

注意 `linear.y_qdq` 属于 **symmetric FP64 scale** 例；非对称 FP16 例用于参数、补齐与存储实验，不能把两个实验的误差混在一份表里。

## 二、只改一个变量：分组轴

```bash
python docs/chapters/part-09-quantization/02-numerical-representations-and-tensor-contracts/examples/walk_groups.py
python docs/chapters/part-09-quantization/02-numerical-representations-and-tensor-contracts/examples/walk_groups.py --axis 1
```

默认 `axis=0,G=2`：scale shape `[3,3]`。改成 `axis=1,G=2`：scale shape `[5,2]`。先画出哪些元素属于一组，再查看输出中的 `scales` 和 `zero_points`。

代码中的 `axis` 是切组方向。不要直接用它替换 ONNX per-axis 算子的 `axis`，两个接口需要按参数广播关系转换。

## 三、真实零与整数零

```bash
python docs/chapters/part-09-quantization/02-numerical-representations-and-tensor-contracts/examples/walk_groups.py --asymmetric --pad
```

该示例默认使用 FP64 scale，便于手算；总实验的 `asymmetric_fp16_padded` 则使用 FP16 scale，两者会在非精确 scale 上有细小差异。

观察最后一个尾组：原始元素 `-0.5` 编码为 0，补入的实数零编码为 15。`dq` 的最后一行应全零。然后阅读测试里的错误 padding 案例：它故意把最后一个 code 改成 0，构造器应拒绝这个不满足契约的输入。

## 四、从位模式解释 FP8 / FP4

```bash
python docs/chapters/part-09-quantization/02-numerical-representations-and-tensor-contracts/examples/inspect_bits.py
python docs/chapters/part-09-quantization/02-numerical-representations-and-tensor-contracts/examples/inspect_bits.py --format e5m2 --code 0x38 --value 0.5
python docs/chapters/part-09-quantization/02-numerical-representations-and-tensor-contracts/examples/inspect_bits.py --format e2m1 --code 0x7 --value 5
python docs/chapters/part-09-quantization/02-numerical-representations-and-tensor-contracts/examples/inspect_bits.py --code 0x1 --ftz
```

依次观察：

1. E4M3FN `0x3D` 解为 1.625，再编码也是 `0x3D`。
2. E5M2 `0x38` 解为 0.5。
3. E2M1 `0x7` 解为 6；输入值 5 处于中点，编码为 `0x6`，还原为 4。
4. E4M3FN 的 `0x1` 在本课显式 FTZ 策略下解为 0；不加 `--ftz` 应是 $2^{-9}$。

`--ftz` 只影响第一步位模式解码，用于把策略和编码格式分开观察。有限输入编码器另采用饱和策略。

错误输入也要检查：

```bash
python docs/chapters/part-09-quantization/02-numerical-representations-and-tensor-contracts/examples/inspect_bits.py --format e2m1 --code 0x10
```

FP4 只有 4 位，`0x10` 超界；预期明确报错并退出 2。它不是实验失败，而是对非法契约的预期拒绝。

## 五、检查位数账单

```bash
python docs/chapters/part-09-quantization/02-numerical-representations-and-tensor-contracts/examples/count_bits.py
```

依次核对 `weight_bits=60`、`tensor_padding_bits=12`、`scale_bits=144`、`zero_point_bits=36`、`metadata_bits=32`、`byte_padding_bits=4`、`alignment_padding_bits=96`。总计 384 bit，除以 15 个真实权重等于 25.6。

可在交互式 Python 中把 `zero_point_bits=32` 传给 `calculate_effective_bpw()`，重新逐项计算。这里并没有写出一个量化 checkpoint，输出不能当成磁盘测量结果。

## 六、可选：独立 PyTorch 解码对照

本轮使用 PyTorch **2.7.1+cpu**。若还没安装，可以按 [PyTorch 历史版本安装页](https://pytorch.org/get-started/previous-versions/#v271) 选择适合当前系统的包；此项不是默认实验的前置条件。

```bash
python -m scripts.experiments.lesson0902 --torch-check
```

它对 E4M3FN、E5M2 分别枚举 256 个位模式，与 `torch.uint8 → view(float8) → float32` 的 CPU 结果比较，包括 NaN、Inf 和正负零。报告应有 `patterns=256`、空 `mismatch_codes`，两个格式都通过。

未安装 torch 时，显式传入 `--torch-check` 会报告失败，不会把未执行伪装成通过；默认运行始终不需要 torch。这个对照不涵盖 FP8 算术、FTZ 硬件策略或 NVFP4 指令。

## 七、单元测试与 Passport

```bash
python -m unittest tests.quantization.test_tensor_contract tests.quantization.test_minifloat tests.quantization.test_lesson0902 -v
```

共 39 项：22 项整数张量、14 项浮点、3 项课程入口/独立对照。没有 torch 时其中 1 项明确跳过；本轮本地环境执行了全部 39 项。

对称与非对称 Passport 分别记录 group axis、scale dtype、zero-point 矩阵、逻辑与物理 shape。已观察字段引用结果与课程源码，模型质量和性能字段保持 unknown。`not_applicable` 用于这个合成实验根本没有涉及的 checkpoint/kernel 字段。

## 八、提交实验记录时应包含什么

保存自己的命令、原始 JSON、两份 Passport，以及对轴/尾组/舍入规则的解释。时间、Python 版本和源码摘要会随环境变化；固定输入的 golden 数值与检查项应一致。

不要只发一张 PASS 截图：写出“我改了哪一个契约字段、哪些数字随之变化、为什么变化”。进入 9.3 后，这份记录会再加入校准协议与误差度量。
