# 缓存存储测试

`test_paged.py` 检查块映射、有效长度、陈旧数据隔离、容量不足、GQA 与张量字节数。运行 `python -m unittest tests.cache.test_paged -v`。

需要可选 CPU PyTorch；无依赖时明确跳过。与调度器连接的测试在 `tests/integration/test_paged_engine.py`，复现入口见[2.5 实验](../../docs/chapters/part-02-engine/05-paged-caching/lab.md)。

`test_prefix.py` 增加 18 项共享存储检查：完整前缀与最后输入边界、命名空间、碰撞保护、LRU、OOM 原子性、引用计数、重复冷启动、有效前缀与 reset。另有 11 项引擎检查在 `tests/integration/test_prefix_engine.py`，见[2.6 实验](../../docs/chapters/part-02-engine/06-prefix-caching/lab.md)。
