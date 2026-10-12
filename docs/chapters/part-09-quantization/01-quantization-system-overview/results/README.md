# 9.1 CPU 与固定源码运行记录

[正文](../README.md) · [复现命令](../lab.md) · [质量检查](../../../../../reports/quality/lesson-9.1-delivery/README.md)

记录时间（UTC）：`2026-09-20T09:04:36.483841+00:00`。Python 3.11.9；本次在 Windows 的 CPU 环境执行课程程序。上游 vLLM 仅静态检查，没有导入或执行。

## 实际产物

| 文件 | 结果与范围 |
| --- | --- |
| [summary.json](summary.json) | 17 个 CPU 案例通过；六权重 Linear、非零 zero-point、Q/DQ/RQ、边界与 Passport；固定源码 4 文件/9 函数核验 |
| [cpu-passport.json](cpu-passport.json) | 对称 Linear：26 verified、21 not_applicable、9 unknown；不描述仿射例或 Qwen |
| [qwen-plan-passport.json](qwen-plan-passport.json) | 模型计划：5 declared、51 unknown、0 verified |
| [commands.json](commands.json) | 9 条命令均匹配预期退出码；其中 scale=0 的演示预期退出 2 |

## 两个手算例子的观察值

对称 Linear：`scale=0.5, zero_point=0`，q 为 `[[-7,-2],[0,2],[5,7]]`。原始输出 `[-5.5,-1.75]`，量化输出 `[-6,-2.5]`；权重最大误差 0.25、MSE 0.03125、输出最大误差 0.75。

仿射例：源 `(s1,z1)=(0.5,3)` 将 `[-1.5,0,0.5,2.5,6]` 编码为 `[0,3,4,8,15]`。目标 `(s2,z2)=(1,2)` 的重量化结果为 `[0,2,2,4,8]`，再反量化为 `[-2,0,0,2,6]`。两者都使用 `[0,15]`，没有物理 bit packing。

漏减 z、改变 round 顺序和双重舍入的差异分别保存在 `affine.wrong_dq_without_z`、`affine.rounding_order` 和 `affine.double_rounding`。这些是刻意构造的反例，不是正确模型的质量结论。

## 复现与范围

```bash
python -m scripts.experiments.lesson0901
python -m scripts.experiments.lesson0901 --inspect-source
```

第二条要求已缓存固定源码；首次显式使用 `--fetch-source`。默认路径为 `.local/runs/part-09-quantization/01-system-overview/`；完整步骤见[实验手册](../lab.md)。JSON 内的数值与检查项应一致，时间、Python 版本和源码摘要会随环境或代码改变。

源码摘要：`eeebc8752473d290b198c04f4ea9ec6852710bd59835f80865baa31ac2b50d6c`。它覆盖课程代码、测试、章节示例和输入 JSON；文档与 SVG/PNG 另经链接和视觉检查。`git_commit` 是未提交工作区的基点，不能单独代表这一轮的新代码。

这次没有：真实模型导出/加载、packed checkpoint、整数 RQ kernel、A100/C500 实测、模型质量或服务性能测量。`gpu_validated=false`、`upstream_executed=false`、`packed=false` 明确保留。源码中的类和调用可核查，但不能替代后端运行证据。
