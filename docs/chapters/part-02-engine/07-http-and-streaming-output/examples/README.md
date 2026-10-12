# 两个无网络短示例

[实验页](../lab.md)

- [walk_events.py](walk_events.py)：异步生成器、逐项读取、提前退出与 aclosing 收尾。
- [inspect_frames.py](inspect_frames.py)：每次读取一个字节，恢复 SSE 与中文 UTF-8，拒绝不完整事件。

只需标准库。真实 HTTP 入口在 `python -m zero2vllm.serving.server` 和 `python -m zero2vllm.serving.client`；固定命令见实验页。
