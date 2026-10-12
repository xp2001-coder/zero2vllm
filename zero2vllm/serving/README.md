# HTTP 与流式输出

[2.7 正文](../../docs/chapters/part-02-engine/07-http-and-streaming-output/README.md) · [实验命令](../../docs/chapters/part-02-engine/07-http-and-streaming-output/lab.md)

| 文件 | 责任 |
| --- | --- |
| `service.py` | 单事件循环驱动现有 LifecycleEngine；有界输出队列、按就绪状态推进、截止时间与所有权回收 |
| `app.py` | ASGI 请求体、输入校验、响应头、流式发送、断连监听与传输任务收尾 |
| `protocol.py` | JSON 数据的 SSE 编解码、增量拼帧、请求 ID/序号/输出数量核对 |
| `client.py` | 标准库 POST 客户端，增量读取，提前关闭与未完成流检测 |
| `server.py` | Uvicorn 单进程、本机监听入口；默认 Tiny，可选 2.6 前缀缓存后端 |

仅有 HTTP 启动需要可选 `uvicorn==0.35.0`；默认模型与 ASGI 应用使用标准库，前缀后端另需 CPU PyTorch。驱动器没有使用线程同时修改 engine。同步前向返回后才能处理网络取消；不同请求逐个执行，没有合并张量 batch。

有界队列负责应用待发事件；一条发送中的事件、单独的终态槽、操作系统缓冲区是不同的存储位置。教学引擎历史保留最近 32 条请求；PrefixKVStore 的详细调试事件沿用 2.6 记录方式，长期服务还应关闭或截断这些记录。
