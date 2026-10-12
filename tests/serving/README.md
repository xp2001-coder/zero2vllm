# 流式服务检查

[2.7 实验](../../docs/chapters/part-02-engine/07-http-and-streaming-output/lab.md)

- `test_streaming.py`：26 项协议、异步服务和受控 ASGI 传输检查；其中一项前缀共享检查需要 CPU PyTorch。
- `test_http.py`：3 项真实本机 HTTP 检查，需要 Uvicorn；覆盖两客户端、断连和非法 JSON。

```bash
python -m unittest tests.serving.test_streaming tests.serving.test_http -v
```

可选依赖缺失时对应检查明确跳过。`scripts/experiments/http_probe.py` 提供本机随机端口和可控传输边界；判断依据是独立输出、事件顺序与资源状态。人工阻塞 ASGI send 用于确定性地制造背压，不声称测出了真实网络或 GPU 性能。
