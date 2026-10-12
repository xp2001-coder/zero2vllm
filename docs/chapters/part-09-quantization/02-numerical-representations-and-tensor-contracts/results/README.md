# 9.2 CPU 数值与张量契约运行记录

[正文](../README.md) · [复现命令](../lab.md) · [质量记录](../../../../../reports/quality/lesson-9.2-delivery/README.md)

记录时间（UTC）：`2026-09-20T09:47:41.977515+00:00`。本次在 Windows / Python 3.11.9 运行，独立 FP8 对照使用 PyTorch 2.7.1+cpu 的 CPU 解码。

| 文件 | 实际结果与作用 |
| --- | --- |
| [summary.json](summary.json) | 20 个固定案例全部通过；两种分组契约、Linear 输出、位数账单、FP8 位模式对照 |
| [symmetric-passport.json](symmetric-passport.json) | W[5,3]，G2，沿 K 分组，FP64 scale、z=0 |
| [asymmetric-passport.json](asymmetric-passport.json) | W[5,3] 补为物理 [6,3]，FP16 scale，非零 z，q_pad=z |
| [commands.json](commands.json) | 12 条命令符合预期退出码，含 1 条非法 FP4 code 的预期退出 2 |

## 数值观察

对称 q：`[[-7,0,1],[-2,7,7],[1,3,-7],[7,7,2],[7,7,-7]]`。权重最大绝对误差 0.25，MSE 为 1/120；这个 X 下输出最大误差为 0.25。

非对称 FP16 例末尾的 padding code 是 `[0,0,15]`，DQ 是 `[0,0,0]`。它与对称 Linear 例分别记录，不能共用一项模型质量结论。

按本课声明的缓冲区计划，真实权重 60 bit、张量 padding 12 bit、scale 144 bit、z 36 bit、头部 32 bit、字节补齐 4 bit、最终对齐 96 bit，总计 384 bit，15 个真实权重的 effective bpw 是 25.6。没有写出 packed checkpoint，也没有测 Python 对象占用或 GPU 显存。

E4M3FN、E5M2 各 256 个位模式与 PyTorch CPU 解码一致，含 NaN、Inf 与正负零；这不验证低精度 GEMM。

## 重现与快照

```bash
python -m scripts.experiments.lesson0902
python -m scripts.experiments.lesson0902 --torch-check
```

第一条默认离线、只用标准库；第二条显式要求已安装 PyTorch。输出默认进入 `.local/runs/part-09-quantization/02-tensor-contracts/`。固定数值和检查项应一致；时间、运行环境与源码摘要可不同。

源码摘要：`48be978fc5725f2d5f716f6d4da611d80634921c23218875a7b73ce887cad23d`。它覆盖课程代码、测试、章节示例和输入 JSON；图文另经本轮文档与视觉检查。`git_commit` 表示未提交工作区的基点，不能单独代替源码摘要。

本次没有真实 Qwen 量化、文件序列化、vLLM 集成、A100/C500 或服务性能实验。报告保持 `gpu_validated=false`、`packed=false`；Passport 中没有测量的质量和性能字段保持 unknown。
