# 3.1 交付时的仓库验证

[质量导航](../README.md) · [课程正文](../../../docs/chapters/part-03-vllm/01-a-request-s-journey-through-vllm/README.md)

2026-09-17，Windows / Python 3.11.9。全仓库 **280 项测试通过，0 项跳过**。其中新增 16 项离线工具测试，覆盖源码内容、调用作用域、结构字段、错误下载与默认离线行为。

- [完整测试输出](tests.txt)与[回归记录](regression.json)。
- [课程检查](course.json)与[目录检查](layout.json)：中英文生成区段、公开引用、链接与层级结构。
- [真实源码核验](../../../docs/chapters/part-03-vllm/01-a-request-s-journey-through-vllm/results/README.md)：空目录首次下载 25 文件，再离线检查 38 个函数和 4 类消息结构。
- [双请求示例](../../../docs/chapters/part-03-vllm/01-a-request-s-journey-through-vllm/results/iteration.txt)：位置、槽位与两种执行排列的结果归属检查。

源码准备与核验用 Python `-S` 实际运行，不依赖 vLLM 或 PyTorch 的安装。全仓库回归则使用当前课程虚拟环境，包含已有 CPU 模型与本机 HTTP 检查；这两类验证范围分别记录。

5 张中文 PNG 已逐张查看，SVG 可编辑。公式、示例数字与源码符号已对照；本节没有修改已有模型、调度、缓存或服务算法。CI 原有测试发现入口会包含新增测试，并增加双请求示例；本轮未运行远程 GitHub Actions。

静态调用表达式不证明运行时分支命中。A100/C500 未运行，未产生 GPU 速度、显存或服务性能结论。源码核验固定历史 v0.8.5，不能据此判断当前版本或 MACA 分支的兼容性。
