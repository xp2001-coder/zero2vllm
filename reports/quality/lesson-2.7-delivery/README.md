# 2.7 交付时的仓库验证

[质量导航](../README.md) · [课程正文](../../../docs/chapters/part-02-engine/07-http-and-streaming-output/README.md)

2026-09-17，Windows / Python 3.11.9。全仓库 **264 项测试全部通过，0 项跳过**，包含本节新增的 26 项协议、异步服务与 ASGI 检查，以及 3 项真实本机 HTTP 检查。HTTP 服务器使用 Uvicorn 0.35.0 / h11 0.16.0，前缀后端使用 PyTorch 2.7.1+cpu。

- [tests.txt](tests.txt)：完整测试输出。
- [regression.json](regression.json)：实际命令、环境、测试数量与源码指纹。
- [course.json](course.json) / [layout.json](layout.json)：中英文目录、公开引用、本地链接与仓库布局检查，均通过。
- [本节实测](../../../docs/chapters/part-02-engine/07-http-and-streaming-output/results/README.md)：15 个案例全部通过，分别为 7 个真实本机 HTTP、7 个受控 ASGI/服务边界和 1 个字节协议案例。

两个教学示例均执行通过：异步生成器提前退出时明确关闭请求；逐字节读取含中文的 SSE 帧时保持边界，并拒绝截断。

新增 29 项在 Python `-S` 下运行时，25 项标准库检查通过，3 项 Uvicorn 和 1 项 PyTorch 检查明确跳过。这确认基础路径不被强制依赖可选包；跳过不计入相应功能验证通过的证据。

本回归与本节公开基线的 `source_sha256` 相同。Git commit 指向已有提交，工作区尚未提交；2.2—2.6 的历史记录保留原始源码指纹。

六张白底中文 PNG 已逐张查看，事件顺序、背压队列、同步前向和回收边界与实现核对，SVG 保留可编辑原稿。课程同步、公开引用、本地链接和目录检查另行执行。工作流新增 Windows/Linux 的 HTTP 检查及 CPU PyTorch 前缀后端检查；远程 GitHub Actions 本轮未执行。

验证范围为单进程 CPU 教学服务与本机回环连接。受控背压通过阻塞 ASGI 发送制造，不代表测得真实网络拥塞速度；同步前向仍占用事件循环，没有运行 A100/C500，也没有测量 GPU 延迟、TTFT 或服务吞吐量。
