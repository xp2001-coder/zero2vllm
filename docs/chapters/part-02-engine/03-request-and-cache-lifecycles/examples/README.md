# 2.3 教学示例

[正文](../README.md) · [完整命令](../lab.md)

- [walk_requests.py](walk_requests.py)：两条请求在 6 个 token 的逻辑预算下依次运行，观察取消后的额度回收。
- [cancellation_boundary.py](cancellation_boundary.py)：同线程回调在计算后、结果提交前发出取消，观察延迟回收和 token 丢弃。此例不启动并发线程，也不模拟 GPU 执行时间。

两者仅需 Python 标准库，使用已有 TinyAttentionLM 的真实 K/V 列表；共享实现位于 `zero2vllm/engine/`。
