# 文档导航

课程按 **篇 → 节** 组织。先进入篇目录，再选择小节；每节的正文、实验和练习在同一目录中。126 个小节入口都已建立，状态区分完整课程与待填充骨架。

| 分区 | 内容 |
| --- | --- |
| [chapters/](chapters/README.md) | 19 篇、126 节；正文、实验、示例、输入数据和结果按节就近存放 |
| [platforms/](platforms/README.md) | CPU / NVIDIA A100 / 沐曦 C500 的环境与验证约定 |
| [engineering/](engineering/README.md) | 架构、实验规范、开发路线 |
| [references/](references/README.md) | 官方来源、特性地图、专题导读 |
| [maintenance/](maintenance/README.md) | 目录规范、更新课程、按文件提交 |
| [metadata/](../metadata/README.md) | 课程编排与英文大纲按篇拆分；特性和来源分别维护 |

## 开始学习

| 入口 | 内容 |
| --- | --- |
| [项目首页](../README.md) | 课程定位、快速开始、完整大纲和当前进度 |
| [English README](../README_en.md) | 英文项目介绍与完整 19 篇、126 章大纲；章节正文目前主要是中文 |
| [第一课：从 token 到下一步预测](chapters/part-01-inference/01-next-token/README.md) | 完整正文、二维 Attention 手算、CPU 生成实验与五组练习答案 |
| [Part 1 六节学习与命令总表](chapters/part-01-inference/README.md) | 完整正文、各节代码/参数/练习与 28 案例统一运行入口 |
| [2.1 最小 PyTorch 模型](chapters/part-02-engine/01-pytorch-model/README.md) | RMSNorm / RoPE / Attention / SwiGLU、逐层对齐、7 个案例与 6 组练习 |
| [首次学习的短路线](chapters/learning-paths.md#先用现有材料走通一条短路线) | 从第一课继续完成缓存实验、调度手算、请求图和量化 Linear |
| [CPU KV Cache 实验](chapters/part-01-inference/03-kv-cache/lab.md) | 比较 KV Cache 前后的逐步 logits 与计算工作量 |
| [CPU INT4 Linear 实验](chapters/part-09-quantization/04-writing-quantization-and-data-reordering-kernels/lab.md) | 分组缩放、半字节打包、量化 Linear 与误差/存储分析 |
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
