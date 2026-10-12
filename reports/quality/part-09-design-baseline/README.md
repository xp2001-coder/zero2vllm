# Part 9 v1.0 设计基线检查

[返回质量记录](../README.md) · [Part 9 总设计](../../../docs/chapters/part-09-quantization/design.md)

本轮将确认的设计拆成四个 Module、14 个教学单元，建立 Passport 字段规范与 15 项写作模板。现有 CPU INT4 实验迁到 9.8；量化算法归入 9.4，vLLM 接入归入 9.13，端到端验证归入 9.14。

| 检查 | 结果 | 证据 |
| --- | --- | --- |
| 课程编排、中英文入口、公开引用与本地链接 | 19 篇、129 节；18 节已完成；Part 9 保持教学计划 | [课程检查](course.json) |
| 目录和公开实验引用 | 无空目录、无旧入口目录、无失效结果引用 | [目录检查](layout.json) |
| 文档维护、目录与 CPU 量化回归 | 37 项通过，0 失败、0 错误、0 跳过 | [运行摘要](verification.json) |
| CPU INT4 默认案例 | 验证通过；除新运行时间外与发布记录相同 | [运行摘要](verification.json) |
| 历史 INT4 原始结果迁移 | SHA-256 一致，原始字节保留 | [原始结果](../../../docs/chapters/part-09-quantization/08-bit-packing-serialization-and-repack/results/int4-reference.json) |

37 项测试针对本轮改动，不代表重新运行了全部 380 项历史测试。共享实现未改动；前三篇仅同步 2.2 中一处量化章节编号引用。历史检查报告保留原始章节数和测量值，只修正迁移后的 Markdown 链接。

本轮没有实现 9.1 正文或新的 GPU 算子；没有执行真实量化模型生产、A100/C500、多卡或远程 CI。所有此类内容继续按总设计逐节交付和验证。
