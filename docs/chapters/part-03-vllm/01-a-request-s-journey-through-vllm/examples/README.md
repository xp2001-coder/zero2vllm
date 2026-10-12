# 3.1 短示例

[正文](../README.md) · [实验](../lab.md)

`walk_iteration.py` 只使用 Python 标准库，展示两条请求的输入拼接、绝对位置、物理槽位与结果路由。它检查两种执行排列，演示按登记顺序取结果会出错。

```bash
python docs/chapters/part-03-vllm/01-a-request-s-journey-through-vllm/examples/walk_iteration.py
```

全部 token、块号、调度额度和候选输出是为手算设置的数据；未执行模型，不代表实际 Tokenizer 或 vLLM 调度策略。真实源码定位工具位于 [lesson0301.py](../../../../../scripts/experiments/lesson0301.py)。
