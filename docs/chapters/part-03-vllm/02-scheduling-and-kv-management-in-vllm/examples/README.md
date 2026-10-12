# 3.2 短示例

[walk_schedule.py](walk_schedule.py) 打印正文的混合调度三轮账本，参数可调整总 token 预算、单请求阈值和回放轮数。只有标准库依赖，不执行模型。

```bash
python docs/chapters/part-03-vllm/02-scheduling-and-kv-management-in-vllm/examples/walk_schedule.py
python docs/chapters/part-03-vllm/02-scheduling-and-kv-management-in-vllm/examples/walk_schedule.py --token-budget 8 --threshold 2 --rounds 1
```

完整 10 个案例见[实验页](../lab.md)。示例导入的是[计数教学模块](../../../../../zero2vllm/reference/scheduling.py)，不能替代第二篇的真实 CPU 模型实验。
