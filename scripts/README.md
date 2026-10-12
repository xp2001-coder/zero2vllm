# 维护与运行工具

[目录规范](../docs/maintenance/repository-layout.md)

在仓库根目录用模块方式运行。这里按职责放工具；学生需要阅读的示例在各节 examples/。

| 目录 | 用途 | 入口 |
| --- | --- | --- |
| docs/ | 课程生成、翻译装配、链接与仓库布局检查 | `python -m scripts.docs.course --check` / `python -m scripts.docs.layout` |
| experiments/ | 章节实验批量运行 | `python -m scripts.experiments.part01` / `python -m scripts.experiments.lesson0201` / `python -m scripts.experiments.lesson0202` |
| environment/ | 设备环境信息采集 | `python -m scripts.environment.collect` |
| common.py | 共享根目录与源码指纹 | 内部复用，不单独运行 |

实验默认写 .local/runs/，质量检查写 .local/checks/。核验后按章节选择结果公开，不能把整个本地区上传。

2.2 首次准备真实权重需显式加 `--download`，随后默认离线运行。安装与复现步骤见[本节实验](../docs/chapters/part-02-engine/02-loading-real-model-weights/lab.md)。

2.3 使用 `python -m scripts.experiments.lesson0203`，默认标准库运行 11 个案例；`--real-checkpoint` 复用已有快照离线验证请求封装。见[本节实验](../docs/chapters/part-02-engine/03-request-and-cache-lifecycles/lab.md)。

2.4 使用 `python -m scripts.experiments.lesson0204 --trace`，标准库回放六组负载的两种策略；可分别覆盖三种上限，见[实验页](../docs/chapters/part-02-engine/04-from-static-to-continuous-batching/lab.md)。

2.5 使用 `python -m scripts.experiments.lesson0205`，CPU PyTorch 执行 11 个分页张量案例；`--case` 可单独检查映射、复用、分配失败、logits 或调度，见[实验页](../docs/chapters/part-02-engine/05-paged-caching/lab.md)。

2.6 使用 `python -m scripts.experiments.lesson0206`，执行 12 个 CPU 前缀共享案例；`--case` 分离冷暖、分叉、边界、碰撞、压力与调度，见[实验页](../docs/chapters/part-02-engine/06-prefix-caching/lab.md)。

2.7 使用 `python -m scripts.experiments.lesson0207`，运行 14 个 HTTP、受控背压和协议案例；`--include-prefix` 加上第 15 个真实 HTTP 前缀后端案例。真实网络需要 Uvicorn，前缀扩展还需 CPU PyTorch。`http_probe.py` 提供回环服务器与 ASGI 受控实验工具，见[实验页](../docs/chapters/part-02-engine/07-http-and-streaming-output/lab.md)。

3.1 使用 `python -m scripts.experiments.lesson0301 --fetch` 显式准备固定版本源码；之后默认离线，`--focus` 选择阅读阶段。工具只核验文本和语法树，不导入 vLLM，见[源码实验](../docs/chapters/part-03-vllm/01-a-request-s-journey-through-vllm/lab.md)。

3.2：`python -m scripts.experiments.lesson0302` 默认运行 10 个离线计数案例；`--inspect-source` 核验已有固定源码，`--fetch-source` 显式下载并核验。见[调度与 KV 实验](../docs/chapters/part-03-vllm/02-scheduling-and-kv-management-in-vllm/lab.md)。

3.3：`python -m scripts.experiments.lesson0303 --mode metadata` 只需标准库；默认运行 12 个案例，需要 CPU PyTorch。`--dtype float32` 切换数值精度；`--inspect-source` 静态核验现有源码，`--fetch-source` 显式准备固定官方文件。见[模型执行实验](../docs/chapters/part-03-vllm/03-model-execution-and-attention-backends/lab.md)。

3.4/3.5：`python -m scripts.experiments.lesson0304` 与 `python -m scripts.experiments.lesson0305` 分别运行 10/12 个标准库案例，默认离线。两者支持 `--inspect-source` 与显式联网的 `--fetch-source`；共享源码检查只筛选所需符号，仍拒绝目标歧义。见 [3.4 实验](../docs/chapters/part-03-vllm/04-v1-and-modelrunner-evolution/lab.md)与 [3.5 实验](../docs/chapters/part-03-vllm/05-resolving-configuration-and-capabilities/lab.md)。

9.1：`python -m scripts.experiments.lesson0901` 默认离线运行 17 个 Q/DQ/RQ、Linear 与 Passport 案例；可用 `--inspect-source` 静态检查缓存，`--fetch-source` 显式获取固定源码。见[实验页](../docs/chapters/part-09-quantization/01-quantization-system-overview/lab.md)。

9.2：`python -m scripts.experiments.lesson0902` 默认运行 20 个离线 CPU 案例；`--torch-check` 显式增加 512 个 FP8 位模式对照。见[实验手册](../docs/chapters/part-09-quantization/02-numerical-representations-and-tensor-contracts/lab.md)。

9.3：`python -m scripts.experiments.lesson0903` 运行 20 个离线校准与误差案例；`--torch-check` 显式对照 PyTorch Observer。见[实验手册](../docs/chapters/part-09-quantization/03-calibration-observers-and-error-measurement/lab.md)。

9.4：`python -m scripts.experiments.lesson0904` 运行 23 个离线算法案例；`--torch-check` 对照 GPTQ 补偿和 STE 的 PyTorch 数学表达。见[实验手册](../docs/chapters/part-09-quantization/04-quantization-algorithms-and-error-control/lab.md)。
