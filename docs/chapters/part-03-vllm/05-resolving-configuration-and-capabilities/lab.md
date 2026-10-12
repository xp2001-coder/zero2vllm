# 3.5 实验：打印每个最终配置值的来源

[返回正文](README.md) · [练习与答案](exercises.md) · [源码路线图](source-map.md) · [实际记录](results/README.md)

## 1. 准备

从仓库根目录运行，使用 Python 3.11+ 标准库。Windows 可用 `.venv/Scripts/python.exe` 替换 `python`。不需要安装 vLLM、PyTorch 或 YAML 包，也不会读写进程的 `VLLM_USE_V1` 环境变量。

本实验输入的是课程 JSON 映射。上游 `vllm serve --config` 读取 YAML；两者不要混用。本节只模拟 v0.8.5 的部分 V1、CUDA、普通文本、TP=1 规则。

## 2. 运行十二个案例

```bash
python -m scripts.experiments.lesson0305
```

十二项应均为 `PASS`，详细结果写入 `.local/runs/part-03-vllm/05-config-resolution/summary.json`。默认不联网。

| 案例 | 预期观察 |
|---|---|
| `cli_over_file` | 文件 4096，被 CLI 3072 覆盖 |
| `false_is_explicit` | prefix cache 的 false 保留，未指定时默认 true |
| `v1_forces_chunking` | 该版本 V1 将分块 prefill 的 false 改为 true |
| `usage_and_memory_defaults` | 四种场景有不同的默认预算 |
| `engine_intent` | 未设置、显式 0、显式 1 与功能分类共同决定分支 |
| `forced_unsupported_errors` | 强制 V1 遇到不支持条件，报错 |
| `cross_field_validation` | 预算 8 小于序列上限 16，拒绝 |
| `model_limit` | 超过课程模型上限的长度被拒绝 |
| `chunk_budget` | 分块时预算 512 可小于模型长度 4096 |
| `eager_intent` | eager 意图与实际图启用状态分开记录 |
| `validation_stages` | 块大小 7 仅通过教学参数层，后端仍待验证 |
| `unverified_platform` | 不将这套 CUDA 规则用于断言 MACA 能力 |

报告中的预期错误由程序捕获并核对，因此整体可以成功结束。手工执行错误输入的 CLI 则会以非零状态退出，见后文。

## 3. 完成一次逐字段追踪

```bash
python docs/chapters/part-03-vllm/05-resolving-configuration-and-capabilities/examples/explain_config.py --max-num-batched-tokens 3072 --prefix false
```

默认读取 [demo-config.json](data/demo-config.json)。输出有三组关键字段：

- `values`：最终值。
- `origins`：每个字段最后赋值的阶段。
- `trace`：从初始声明到后续规则的完整赋值顺序。

先找到 token 预算对应的记录：`None → 4096 → 3072`，最终来源 `cli`。再找到 `enable_chunked_prefill`：`None → false → true`，最终来源以 `v1_policy` 开头。

prefix caching 是 `None → true → false`，最终保留 false。比较这两种布尔字段，比只记住“CLI 优先”更有用。

默认夹具同时写入了预算 4096 和序列上限 64。因此只改 `--memory-gib` 不会使这两个显式值自动变化。下一步比较默认值时，要换成空配置。

## 4. 比较四种默认上下文

[defaults.json](data/defaults.json) 内容是 `{}`，表示没有提供任何字段。

```bash
python docs/chapters/part-03-vllm/05-resolving-configuration-and-capabilities/examples/explain_config.py --config docs/chapters/part-03-vllm/05-resolving-configuration-and-capabilities/data/defaults.json --usage server --memory-gib 40
python docs/chapters/part-03-vllm/05-resolving-configuration-and-capabilities/examples/explain_config.py --config docs/chapters/part-03-vllm/05-resolving-configuration-and-capabilities/data/defaults.json --usage offline --memory-gib 40
python docs/chapters/part-03-vllm/05-resolving-configuration-and-capabilities/examples/explain_config.py --config docs/chapters/part-03-vllm/05-resolving-configuration-and-capabilities/data/defaults.json --usage server --memory-gib 80
python docs/chapters/part-03-vllm/05-resolving-configuration-and-capabilities/examples/explain_config.py --config docs/chapters/part-03-vllm/05-resolving-configuration-and-capabilities/data/defaults.json --usage offline --memory-gib 80
```

按顺序查看 `max_num_batched_tokens / max_num_seqs`：

```text
40 GiB / server  →  2048 / 256
40 GiB / offline →  8192 / 256
80 GiB / server  →  8192 / 1024
80 GiB / offline → 16384 / 1024
```

40 和 80 是人为输入的情景，不是本机检测值。再试 69 与 70，定位源码分支的门槛；不要将这个历史门槛写成所有硬件通用的调优法则。

## 5. 引擎意图：自动回退与强制报错

```bash
python docs/chapters/part-03-vllm/05-resolving-configuration-and-capabilities/examples/inspect_engine.py --intent auto --feature supported
python docs/chapters/part-03-vllm/05-resolving-configuration-and-capabilities/examples/inspect_engine.py --intent auto --feature experimental
python docs/chapters/part-03-vllm/05-resolving-configuration-and-capabilities/examples/inspect_engine.py --intent 1 --feature experimental
python docs/chapters/part-03-vllm/05-resolving-configuration-and-capabilities/examples/inspect_engine.py --intent 1 --feature unsupported
```

前三条依次输出 V1、V0、V1。第四条是**刻意的失败实验**：提示 `explicit V1 cannot satisfy unsupported feature`，退出状态为 2。

这里 `--feature` 是你提供的分类，并没有检测模型或平台。实际排障仍要定位 oracle 拒绝的是哪一个真实条件。课程命令不会修改全局或当前进程的上游环境变量。

## 6. 预算、模型长度与 eager

```bash
python docs/chapters/part-03-vllm/05-resolving-configuration-and-capabilities/examples/explain_config.py --max-num-batched-tokens 512 --max-num-seqs 16
python docs/chapters/part-03-vllm/05-resolving-configuration-and-capabilities/examples/explain_config.py --max-num-batched-tokens 8 --max-num-seqs 16
python docs/chapters/part-03-vllm/05-resolving-configuration-and-capabilities/examples/explain_config.py --enforce-eager
```

第一条通过：分块 prefill 开启，512 可以小于模型长度 4096。第二条退出 2，报 `token budget must be >= max_num_seqs`。第三条输出 `graph_intent: disabled`。

默认非 eager 输出 `eligible_for_configuration_not_proven_active`，只表示还可能进入后续配置流程，没有进行图捕获或执行。

独立观察“不分块”的交叉检查：

```bash
python -c "from zero2vllm.reference.config_resolution import verify_scheduler; verify_scheduler(512,16,4096,False)"
```

这条预期抛出 `ValueError` 并退出 1。它直接调用校验函数，避免 V1 强制分块规则先把 false 改成 true。

## 7. 查看固定源码

已有源码文本时：

```bash
python -m scripts.experiments.lesson0305 --inspect-source
```

首次明确联网获取时：

```bash
python -m scripts.experiments.lesson0305 --fetch-source
```

检查目录 `.local/sources/vllm-v0.8.5`，也可用 `--source-root` 指定版本子目录的父目录。8 个文件、17 个函数来自固定提交；工具检查摘要、符号与指定调用，不导入这些模块。

先从 [源码路线图](source-map.md) 读输入处理，再读引擎选择，最后读 `SchedulerConfig`、`VllmConfig` 和 CUDA 平台处理。不要只搜索某字段第一次出现的位置就下结论。

## 8. 测试与交付

```bash
python -m unittest tests.reference.test_config_resolution tests.maintenance.test_later_source_reading -v
```

本节 18 项配置测试，另有两节共用的 5 项源码阅读测试。两节合计新增 45 项测试，共用测试只计算一次。

验收时请提交：

1. 一次带覆盖的完整输出，说明三项字段的最终来源。
2. 四种默认情景对照，注明容量为假设值。
3. 两个预期失败输入及原因。
4. 参数层通过后仍需验证的模型、后端、缓存与图执行事项。

排查提示：`true/false/null` 应使用 JSON 类型，不能写成字符串；未知键或横线/下划线同名键会被拒绝；本解析器的正整数检查属于教学保护，不表示复制了上游所有类型转换；C500 上也可以运行本 CPU 实验，但不能据此填写 MACA 验证通过。
