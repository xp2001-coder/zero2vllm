# 2.5 固定输入

[cases.json](cases.json)由[实验脚本](../../../../../scripts/experiments/lesson0205.py)读取。

- `model`：Tiny 模型的全部结构与种子参数；改变最大上下文也会影响随机权重生成，复现实验时一起固定。
- `generation`：单请求 logits 对照输入；默认两种 dtype、两种块大小。
- `mapping`：可手算的 q/K/V，插图与短示例使用相同数值。
- `engine_cases`：池大小、块大小、活跃上限、请求到达、取消和故障注入。`arrival`/`cancel_at` 是逻辑轮边界；`fail_after_write` 在完成实际分页写入后注入异常。
- `block_size_sweep`：三条请求的预留和有效长度；所有长度以 token 位置计。

这些是教学构造输入，不是真实用户文本或性能负载。正常试验通过 CLI 选择固定案例；修改 JSON 时同步更新图、预期行为、测试与公开结果。
