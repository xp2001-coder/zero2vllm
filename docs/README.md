# 文档导航

## 开始学习

| 入口 | 内容 |
| --- | --- |
| [项目首页](../README.md) | 课程定位、快速开始、完整大纲和当前进度 |
| [English README](../README_en.md) | 英文项目介绍与完整 19 篇、126 章大纲；章节正文目前主要是中文 |
| [首次学习的短路线](learning-paths.md#先用现有材料走通一条短路线) | 用现有材料完成缓存实验、调度手算、请求图和量化 Linear |
| [第一个 CPU 实验](labs/01-kv-cache.md) | 比较 KV Cache 前后的逐步 logits 与计算工作量 |
| [CPU INT4 Linear 实验](labs/02-int4-linear.md) | 分组缩放、半字节打包、量化 Linear 与误差/存储分析 |
| [7 篇专题导读与知识库映射](knowledge-base.md) | 请求、缓存、算子、编译、量化、并行、推测的具体学习入口 |
| [学习路线](learning-paths.md) | 学生、工程师、量化方向的阶段安排与章节分工 |
| [详细课程目录](curriculum.md) | 各章前置、练习、设备条件和交付状态 |

## 深入工程与实验

- [工程设计](engineering.md)：教学引擎、正式 vLLM 实验与后端职责。
- [量化篇设计](quantization-design.md)：策略、张量契约、算子加速与端到端验证。
- [特性与优化地图](vllm-feature-map.md)：优化、并行术语、组合约束和注册项清单。
- [CUDA/MACA 环境](backends.md)：A100/C500 配置填写与验证层级。
- [实验规范](experiments.md)：负载、性能指标、质量与原始证据。
- [开发路线](roadmap.md)：材料交付顺序及里程碑依赖。
- [参考来源](references.md)：上游文档、源码与查阅范围。
- [验证记录](../reports/validation.md)：已执行的检查与尚待验证的部分。

## 参与建设

- [贡献说明](../CONTRIBUTING.md)与[更新记录](../CHANGELOG.md)。
- [章节模板](templates/chapter.md)、[GPU 运行模板](templates/gpu-run.json)与[功能验证模板](templates/feature-validation.json)。
- [课程数据](curriculum-catalog.json)与[特性清单](feature-inventory.json)。
- [知识来源台账](knowledge-sources.json)与[本轮更新报告](../reports/knowledge-update.md)。
- [维护者上传说明](github-publishing.md)。
