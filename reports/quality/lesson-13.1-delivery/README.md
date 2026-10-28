# 13.1 交付与验证

[课程正文](../../../docs/chapters/part-13-disaggregation/01-prefill-decode-disaggregation/README.md) · [实验手册](../../../docs/chapters/part-13-disaggregation/01-prefill-decode-disaggregation/lab.md)

2026-10-28：交付PD适用边界、同预算对照、首token归属与联合SLO，含2张原创中文Mermaid示意图、6组合成请求轨迹、标准库分析器、运行命令及9组练习答案。

- [回归记录](regression.json)：本节22项测试及第一篇相关参考测试；覆盖独立手算、数据完整性、窗口/资源边界与命令执行。测试选择及实际通过数量以记录为准，未声称执行全仓库回归。
- [课程检查](course.json)：中英文目录、链接、公开引用规则和数据格式。
- [结构检查](layout.json)：课程分层目录与公开结果链接。
- [默认结果](../../../docs/chapters/part-13-disaggregation/01-prefill-decode-disaggregation/results/baseline.json)与[收紧TTFT结果](../../../docs/chapters/part-13-disaggregation/01-prefill-decode-disaggregation/results/strict-ttft.json)来自同一合成输入，包含输入SHA256。其他阈值命令也按实验页执行核对。

本节完成状态仅覆盖正文与离线分析交付。未执行vLLM/Dynamo部署、GPU、网络传输、模型质量、扩缩容或多机性能实验。6个场景不是调度模拟输出，不能据其数字排名硬件或框架。
