# 本节调用示例

本目录的调用示例复用仓库中的共享模型，不复制模型实现。先按 [实验说明](../lab.md) 安装 CPU PyTorch 环境，命令在仓库根目录运行。

| 示例 | 输入 | 阅读目的 |
| --- | --- | --- |
| [attention_math.py](attention_math.py) | 手工指定的 Q、K、V | 复现正文三位置 Attention 手算，观察正确和错误 mask 的差别 |
| [rope_math.py](rope_math.py) | 两个头各自的四维向量 | 逐对核对旋转、长度保持、相对位置关系，演示单 token 位置重置错误 |
| [walk_forward.py](walk_forward.py) | `[[1,2,3]]` | 逐层运行模型，观察形状和最后位置的预测 |
| [run.py](run.py) | `[[1,2,3,4]]` | 查看完整 trace 与四步生成，使用默认实验输入 |

```bash
python docs/chapters/part-02-engine/01-pytorch-model/examples/attention_math.py
python docs/chapters/part-02-engine/01-pytorch-model/examples/rope_math.py
python docs/chapters/part-02-engine/01-pytorch-model/examples/walk_forward.py
python docs/chapters/part-02-engine/01-pytorch-model/examples/run.py
```

PowerShell 未激活环境时，用 `.\.venv\Scripts\python.exe` 替换 `python`。`attention_math.py` 与 `rope_math.py` 使用各自独立的手算数据；另外两个示例的 ID 来自随机教学权重。它们的输入不同，不应比较它们是否生成相同的 ID。

[返回正文](../README.md)
