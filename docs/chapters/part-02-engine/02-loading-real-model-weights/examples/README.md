# 2.2 示例

在仓库根目录、已安装本节依赖的虚拟环境中运行，完整步骤见[实验](../lab.md)。

| 示例 | 是否需要模型下载 | 观察内容 |
| --- | --- | --- |
| [layout_traps.py](layout_traps.py) | 否 | 方阵转置、RoPE 配对、GQA 头顺序：形状相同，计算不同 |
| [inspect_snapshot.py](inspect_snapshot.py) | 是；运行时只读本地文件 | 配置、权重形状、BF16 存储、共享权重与真实 token ID |

真实模型加载与对齐统一运行 `python -m scripts.experiments.lesson0202`。公共实现位于 [llama.py](../../../../../zero2vllm/modeling/llama.py) 与 [checkpoint.py](../../../../../zero2vllm/modeling/checkpoint.py)，示例不复制模型实现。
