# CUDA / MACA 环境与配置

## 当前支持状态

| 路径 | 角色 | 当前验证范围 |
| --- | --- | --- |
| CPU 标准库 | 数学参考、基础学习、逻辑测试 | 已在本地 Windows/Python 3.11.9 执行 |
| CUDA / A100 | 上游 vLLM 与教学引擎 GPU 目标 | 配置模板已检查，尚无实机结果 |
| MACA / C500 | vLLM-MetaX 与教学引擎 GPU 目标 | 配置模板已检查，尚无实机结果 |

CPU 教学程序不调用正式 vLLM 的 CPU 后端。学生使用其他 NVIDIA GPU 时，以所选 vLLM 版本、模型和实际显存为准，当前尚未公布经过验证的最低显存要求。

## 两个运行环境

**NVIDIA：** 驱动、匹配的容器运行支持、CUDA/PyTorch/vLLM 镜像。

**沐曦：** 驱动、MXMACA SDK、适配版 PyTorch、vLLM、vLLM-MetaX 和对应 mcoplib。优先使用 C500 官方发布的配套镜像，不把通用 CUDA wheel 当作 MACA 的安装方案。[官方 MetaX 安装说明](https://vllm-metax.readthedocs.io/en/stable/getting_started/installation/maca.html)

先核对 C500 已有软件交付，再选择可在 A100 同时验证的 vLLM 版本。依赖和镜像分别固定，源码导读另记具体上游 commit。

## 填写配置

从以下模板各复制一份到 `configs/local/`：

- [A100 模板](../configs/cuda-a100.toml)
- [C500 模板](../configs/maca-c500.toml)

需要补充的字段：

| 分组 | 信息 |
| --- | --- |
| hardware | 实际 GPU 名称和显存 GiB、主机 OS、CPU 架构 |
| runtime | 完整镜像地址及 SHA256 摘要、Python、PyTorch、vLLM、驱动版本 |
| CUDA 特有 | 容器 CUDA runtime 版本；不要把驱动支持的最高 CUDA 版本当作此值 |
| MACA 特有 | MACA、vLLM-MetaX、mcoplib 的实际版本 |
| model | 模型 ID、固定 revision、容器内绝对路径、明确 dtype |

所有版本字段和显存字段均使用 TOML 字符串；显存填实际数值的字符串，例如将设备报告的 MiB 除以 1024 得到 GiB。显存值须为大于零的有限数，可带小数，不附带单位；未掌握的信息保持空字符串。

schema 1 只接受模板中的字段。拼错字段名、混入另一后端的字段，或增加尚未接入的配置项都会报错；工具不会把这些字段静默丢弃后输出启动参数。量化、多卡等功能实验参数写入对应实验记录，不能直接加在当前单卡基线模板中。

容器镜像摘要格式为 `实际仓库/实际镜像@sha256:64位十六进制摘要`，从自己的镜像清单获得。模型目录的实际权重要与记录的模型 revision 对应；配置检查器不会读取或校验权重。

```bash
python -m zero2vllm profile configs/local/cuda-a100.toml
python -m zero2vllm profile configs/local/maca-c500.toml --require-complete
```

检查分两层：

1. **配置检查**：字段白名单、必填项、单卡参数、基础类型、有效显存值、镜像摘要、插件声明版本一致。
2. **实机检查**：厂商版本组合兼容、GPU 可见、模型正确加载和生成、服务调用成功。

本工具只完成第 1 层；版本字符串填全也不代表第 2 层通过。

## 容器内采集环境

将工程源码放入容器后运行：

```bash
python scripts/collect_env.py --output runs/container-environment.json
```

该入口仅需要 Python 3.10+ 标准库，无需安装本项目或改变厂商镜像依赖。它记录安装包的元数据、操作系统与命令存在性，不导入 GPU 算子、不探测设备、不收集主机名和凭据。

还需要单独保存设备工具输出：CUDA 使用 `nvidia-smi`，MACA 使用其镜像提供的 `mx-smi`，并在目标容器中确认设备可见性。设备报告、镜像摘要和容器内包信息共同构成环境证据。

## 正式 vLLM 服务的起步约定

配置完整时，检查器输出 `command_argv`，表示**在已准备好的对应 GPU 容器内**运行的参数列表。启动命令以 [vLLM serve 文档](https://docs.vllm.ai/en/stable/cli/serve/) 为参考，最终对选定版本执行 `vllm serve --help` 核对。

- 只暴露一张完整 GPU；`tensor_parallel_size=1`。
- 候选小模型：Qwen3-0.6B，固定 revision。
- 基线 dtype 为 FP16，初始 `max_model_len=1024`、`max_num_seqs=4`。
- `gpu_memory_utilization=0.70` 是排查环境用的保守起点，尚未实测。
- 先使用 eager 模式；图执行作为后续独立实验。
- 服务绑定 `127.0.0.1`。最初的客户端在同一个容器或网络命名空间中运行。仅映射 Docker 端口并不能访问容器内回环地址上的服务。

本工程当前不生成 Docker 设备挂载命令。CUDA 和 MACA 容器如何暴露设备，以所选镜像和服务器的真实要求为准，拿到环境后补充经过验证的启动说明。

## 功能矩阵的记录方式

正式验证按“硬件 + 完整版本组合 + 模型 + dtype + 功能”记录。功能状态取 `not_run / passed / failed / unsupported`，同时附运行证据和错误原因。

首批只验证模型生成与服务；连续批处理、前缀缓存、图执行、量化、LoRA、推测解码分别加入矩阵。上游支持某功能不能直接推出两个硬件后端都支持。

## 完整课程的硬件分层

课程中包含并行切分、MoE、分离式部署和其他架构的优化，详见 [特性地图](vllm-feature-map.md)。当前两个 profile 仍服务于单卡基线，不能用它们启动多卡实验。

| 类别 | A100 / C500 当前状态 | 验证时额外记录 |
| --- | --- | --- |
| APC、Chunked Prefill、调度 | 两平台均 not_run | 缓存冷/热、共享前缀、预算与抢占 |
| Attention、编译、Graphs、融合 | 两平台均 not_run | 实际 kernel/backend、graph 模式、回退原因 |
| 权重/激活/KV 量化 | 两平台均 not_run | 格式、粒度、scale、量化工具/模型版本、质量 |
| Spec、LoRA、多模态、Pooling | 两平台均 not_run | 模型/task、draft/adapter、输入与组合限制 |
| 多卡并行、DBO/EP、PD/EC | 当前只有单卡验证资源 | 卡数、同类平台软件栈、拓扑、进程组与传输布局 |

`unsupported` 需要上游限制或真实失败分析作证据，不能把“没有设备”直接写成“不支持”。对 C500 不依据 A100、Ascend 或 ROCm 的结果推断能力。

每个优化记录采用 [功能验证模板](templates/feature-validation.json)。模板中的配置对象用于记录实际参数，不是现有 profile 工具的输入 schema。
