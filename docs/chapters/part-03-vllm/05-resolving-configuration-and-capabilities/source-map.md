# 3.5 固定源码路线图

[正文](README.md) · [实验步骤](lab.md)

每条链接指向公开仓库的不可变提交。阅读器核验文件摘要、函数范围和选定调用；没有执行这些模块，也没有验证 GPU 能力。行号从固定文本的语法树提取。

## v0.8.5 · `ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d`

范围：8 个文件（含许可证及必要背景文件）、17 个函数。

| 定位 | 阅读时回答的问题 |
| --- | --- |
| [FlexibleArgumentParser.parse_args](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/utils.py#L1334) | 参数归一化与最终解析 |
| [FlexibleArgumentParser._pull_args_from_config](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/utils.py#L1383) | 配置文件参数排在显式 CLI 前 |
| [FlexibleArgumentParser._load_config_file](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/utils.py#L1458) | YAML 转为参数序列 |
| [EngineArgs.from_cli_args](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/engine/arg_utils.py#L968) | Namespace 到 EngineArgs |
| [EngineArgs.create_model_config](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/engine/arg_utils.py#L975) | 模型配置构造 |
| [EngineArgs.create_engine_config](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/engine/arg_utils.py#L1078) | 能力判断、默认值与配置聚合 |
| [EngineArgs._is_v1_supported_oracle](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/engine/arg_utils.py#L1292) | V1 兼容性分支 |
| [EngineArgs._set_default_args_v1](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/engine/arg_utils.py#L1565) | 强制项、None 默认值与设备/用途默认 |
| [_raise_or_fallback](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/engine/arg_utils.py#L1649) | 不支持功能：显式报错或自动回退 |
| [_warn_or_fallback](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/engine/arg_utils.py#L1661) | 实验功能：显式尝试或自动回退 |
| [SchedulerConfig.__post_init__](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/config.py#L1954) | 补全调度字段与派生值 |
| [SchedulerConfig._verify_args](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/config.py#L2022) | 跨字段约束 |
| [VllmConfig.__post_init__](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/config.py#L3844) | 配置交叉验证及平台更新 |
| [ModelConfig._verify_cuda_graph](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/config.py#L835) | 图模式相关调整 |
| [CudaPlatformBase.check_and_update_config](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/platforms/cuda.py#L109) | CUDA 的 worker 与块长默认值 |
| [is_set](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/envs.py#L746) | 区分未设置和显式设置 |
| [set_vllm_use_v1](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/envs.py#L753) | 记录已作出的引擎选择 |

## 四次追踪

1. 输入：比较 `_pull_args_from_config` 拼接参数的顺序与 SDK 入口。
2. 引擎：追踪 `create_engine_config`、oracle、显式环境意图和回退。
3. 赋值：在 `_set_default_args_v1` 区分强制分块、仅补空值和设备/用途分支。
4. 验证：从 SchedulerConfig 到 VllmConfig，再到 CUDA 平台处理，列出尚未完成的后端和运行时验证。

课程 JSON 解析、严格类型限制、简单模型上限和追踪日志属于教学实现；没有实现完整 YAML、RoPE 覆盖、V0 默认值或其他平台能力探测。

## 清单与许可证

清单见 [data/](data/README.md)，含固定提交和 LF 归一化 SHA-256。源码原文只在显式下载时写入运行者的本地缓存；公开课程不打包上游源码。上游许可证作为固定文件一并校验，解释与原创图见本仓库许可。

若目标文件中存在与本节无关的 overload 或 property getter/setter 重名，只筛选本节所需符号；所需符号本身仍不得含歧义。3.1 原有默认严格模式不变。
