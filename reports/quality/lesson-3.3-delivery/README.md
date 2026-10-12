# 3.3 交付时的仓库验证

[质量导航](../README.md) · [课程正文](../../../docs/chapters/part-03-vllm/03-model-execution-and-attention-backends/README.md)

2026-09-20，Windows / Python 3.11.9 / PyTorch 2.7.1+cpu。全仓库 **335 项测试通过，0 项跳过**，耗时约 236 秒；其中本节新增 30 项检查：13 项标准库输入与工具检查、17 项 CPU 张量检查。运行期间课程源码摘要没有变化。

- [完整测试输出](tests.txt)和[回归记录](regression.json)：既有模型、加载、调度、缓存、服务与文档检查继续通过。
- [课程检查](course.json)和[目录检查](layout.json)：中英文进度、生成区段、公开参考规则、本地链接及层级结构。
- [本节运行记录](../../../docs/chapters/part-03-vllm/03-model-execution-and-attention-backends/results/README.md)：float64/float32 各 12 个案例、纯标准库打包、固定源码静态核验与 5 组短示例。

## 数值与源码证据

数值夹具真实执行 Embedding、融合 QKV、RoPE、分页 GQA、输出投影、残差和词表头。对照路径使用各请求独立的完整前缀、分离投影、头重复与 dense 因果 Attention。float64/float32 的最大 logits 绝对误差分别约 2.22e-16 / 1.19e-7，各自低于 1e-11 / 2e-5 的容差。

两种 dtype 分别生成参数，不把跨 dtype 的不同随机权重输出当成误差对照。额外验证执行重排、未来 token、其他请求、未使用 NaN 槽位、填充缓冲、部分 Prefill 候选过滤，以及有手算值的方阵方向与 RoPE 位置。

metadata 模式、源码检查和两种顺序的打包短示例均用 Python `-S` 实际运行。源码检查复用并补齐固定官方 v0.8.5 提交，核验 15 个文件、27 个函数与 2 类结构；只检查文本身份、语法树和字段，没有导入上游模块。

## 材料与范围

7 张中文原创 PNG 已逐张查看，SVG 保留可编辑版本；手算表与真实运行的数组、候选和误差相符。16 组练习提供折叠答案。中英文首页与课程目录将 3.3 标为已完成，下一步指向 3.4。

CI 增加标准库打包、短示例与单独的 CPU PyTorch 检查。本轮没有执行远程 GitHub Actions，没有 Git 提交或推送。工作区包含此前课程改动，源码身份以回归记录和各实验共有的摘要为准。

本节没有运行完整预训练 Llama、上游 vLLM、CUDA Graph 或 FlashAttention GPU 内核。A100/C500 的软件栈、后端选择、数值结果与性能仍需实机验收；静态源码检查与 CPU 夹具通过不代替设备证据。
