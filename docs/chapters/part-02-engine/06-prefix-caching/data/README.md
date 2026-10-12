# 2.6 固定教学输入

[cases.json](cases.json)由[本节运行器](../../../../../scripts/experiments/lesson0206.py)读取。

- `model` 固定参考模型结构与随机种子。完整结构共同影响权重生成。
- `generation` 固定冷暖两轮的 prompt 和输出数量。
- `branches` 是同时活跃、共享前四个 ID 的两条请求。
- `boundary_lengths/boundary_block_sizes` 生成 18 组完整命中边界。
- `scope_seed/scope_probes` 分别检查共同前缀、不同父历史、不同共享域和中间分叉。

token ID 是教学符号，不代表真实分词文本。LRU、OOM、reset 与重复发布使用脚本中的小型固定池，报告会保存操作前后的完整状态。CLI 的 dtype 和 block-size 覆盖仅作用于冷暖案例，避免无意改变其他手算案例的含义。
