# 文档导航

课程按 **篇 → 节** 组织。先进入篇目录，再选择小节；每节的正文、实验和练习在同一目录中。129 个小节入口都已建立，状态区分完整课程与待填充骨架。

| 分区 | 内容 |
| --- | --- |
| [chapters/](chapters/README.md) | 19 篇、129 节；正文、实验、示例、输入数据和结果按节就近存放 |
| [platforms/](platforms/README.md) | CPU / NVIDIA A100 / 沐曦 C500 的环境与验证约定 |
| [engineering/](engineering/README.md) | 架构、实验规范、开发路线 |
| [references/](references/README.md) | 官方来源、特性地图、专题导读 |
| [maintenance/](maintenance/README.md) | 目录规范、更新课程、按文件提交 |
| [metadata/](../metadata/README.md) | 课程编排与英文大纲按篇拆分；特性和来源分别维护 |

## 开始学习

| 入口 | 内容 |
| --- | --- |
| [项目首页](../README.md) | 课程定位、快速开始、完整大纲和当前进度 |
| [English README](../README_en.md) | 英文项目介绍与完整 19 篇、129 章大纲；章节正文目前主要是中文 |
| [第一课：从 token 到下一步预测](chapters/part-01-inference/01-next-token/README.md) | 完整正文、二维 Attention 手算、CPU 生成实验与五组练习答案 |
| [Part 1 六节学习与命令总表](chapters/part-01-inference/README.md) | 完整正文、各节代码/参数/练习与 28 案例统一运行入口 |
| [2.1 最小 PyTorch 模型](chapters/part-02-engine/01-pytorch-model/README.md) | RMSNorm / RoPE / Attention / SwiGLU、逐层对齐、7 个案例与 6 组练习 |
| [2.2 加载真实模型权重](chapters/part-02-engine/02-loading-real-model-weights/README.md) | 固定 SmolLM2-135M、严格权重映射、4 张原创图、3 个真实输入与 8 组练习 |
| [2.3 请求与缓存生命周期](chapters/part-02-engine/03-request-and-cache-lifecycles/README.md) | 状态、独立缓存与回收、4 张原创图、11 个基础案例与真实模型扩展 |
| [2.4 静态与连续批处理](chapters/part-02-engine/04-from-static-to-continuous-batching/README.md) | 逐轮补位、三种预算、4 张原创图、6 组双策略回放与 8 组练习 |
| [2.5 分页缓存](chapters/part-02-engine/05-paged-caching/README.md) | 实际 CPU KV 张量、块表、有效前缀、4 张原创图、11 个案例与 8 组练习 |
| [2.6 前缀缓存](chapters/part-02-engine/06-prefix-caching/README.md) | 冷暖对照、真实 KV 共享、末块重算、引用计数、5 张中文图、12 个案例与 12 组练习 |
| [2.7 HTTP 与流式输出](chapters/part-02-engine/07-http-and-streaming-output/README.md) | HTTP/SSE 与异步基础、真实本机服务、背压、断连取消、6 张中文图、15 个案例与 14 组练习 |
| [3.1 vLLM 请求旅程](chapters/part-03-vllm/01-a-request-s-journey-through-vllm/README.md) | 固定源码、进程与消息、取消回收、5 张中文图、CPU 核验与 12 组练习 |
| [3.2 vLLM 调度与 KV 管理](chapters/part-03-vllm/02-scheduling-and-kv-management-in-vllm/README.md) | token 与块账本、抢占和前缀容量、6 张中文图、10 个计数案例与 14 组练习 |
| [3.3 模型执行与注意力后端](chapters/part-03-vllm/03-model-execution-and-attention-backends/README.md) | 权重布局、输入打包与数值对齐、7 张中文图、12 个 CPU 案例与 16 组练习 |
| [3.4 Runner 演进](chapters/part-03-vllm/04-v1-and-modelrunner-evolution/README.md) | 稳定槽位、增量写、拷贝寿命与图模式；6 张图、10 案例、14 组练习 |
| [3.5 配置解析](chapters/part-03-vllm/05-resolving-configuration-and-capabilities/README.md) | 字段来源、默认/强制规则与分层校验；6 张图、12 案例、14 组练习 |
| [9.1 量化系统全景](chapters/part-09-quantization/01-quantization-system-overview/README.md) | 含 zero-point 的 Q/DQ/RQ、repack 区别、6 张中文图、17 个案例、模型 Passport 与 14 组练习 |
| [9.2 数值表示与张量契约](chapters/part-09-quantization/02-numerical-representations-and-tensor-contracts/README.md) | 分组 Q/DQ、FP8/FP4、非零 z 补齐、有效位宽与 20 个 CPU 案例 |
| [9.3 校准与误差测量](chapters/part-09-quantization/03-calibration-observers-and-error-measurement/README.md) | 校准协议、Observer、受控对照、重构输入与 20 个 CPU 案例 |
| [9.4 量化算法与误差控制](chapters/part-09-quantization/04-quantization-algorithms-and-error-control/README.md) | 舍入、GPTQ 补偿、缩放/旋转、STE 与 23 个 CPU 案例 |
| [首次学习的短路线](chapters/learning-paths.md#先用现有材料走通一条短路线) | 从第一课继续完成缓存实验、调度手算、请求图和量化 Linear |
| [CPU KV Cache 实验](chapters/part-01-inference/03-kv-cache/lab.md) | 比较 KV Cache 前后的逐步 logits 与计算工作量 |
| [CPU INT4 Linear 实验](chapters/part-09-quantization/08-bit-packing-serialization-and-repack/lab.md) | 分组缩放、半字节打包、量化 Linear 与误差/存储分析 |
| [7 篇专题导读](references/topic-guides.md) | 请求、缓存、算子、编译、量化、并行、推测的具体学习入口 |
| [学习路线](chapters/learning-paths.md) | 学生、工程师、量化方向的阶段安排与章节分工 |
| [详细课程目录](chapters/README.md) | 各章前置、练习、设备条件和交付状态 |

## 深入工程与实验

- [工程设计](engineering/architecture.md)：教学引擎、正式 vLLM 实验与后端职责。
- [量化篇设计](chapters/part-09-quantization/design.md)：策略、张量契约、算子加速与端到端验证。
- [特性与优化地图](references/feature-map.md)：优化、并行术语、组合约束和注册项清单。
- [CUDA/MACA 环境](platforms/README.md)：A100/C500 配置填写与验证层级。
- [实验规范](engineering/experiments.md)：负载、性能指标、质量与原始证据。
- [开发路线](engineering/roadmap.md)：材料交付顺序及里程碑依赖。
- [参考来源](references/README.md)：上游文档、源码与查阅范围。
- [验证记录](../reports/README.md)：已执行的检查与尚待验证的部分。

## 参与建设

- [贡献说明](../CONTRIBUTING.md)与[更新记录](../CHANGELOG.md)。
- [章节模板](maintenance/templates/lesson.md)、[GPU 运行模板](../configs/templates/gpu-run.json)与[功能验证模板](../configs/templates/feature-validation.json)。
- [课程数据](../metadata/course/catalog.json)与[特性清单](../metadata/upstream/features.json)。
- [维护者上传说明](maintenance/publishing.md)。
