# 2.6 两个短示例

| 文件 | 观察内容 |
| --- | --- |
| [walk_prefix.py](walk_prefix.py) | 同 prompt 冷暖对照，命中 0→4、计算 8→4，结束后完整缓存留存 |
| [share_and_release.py](share_and_release.py) | 两个活跃请求的实际物理共享；A 退出后 B 继续计算 |

在仓库根目录用 Python 运行，需要 CPU PyTorch，不下载权重。解释器、预期输出和逐行解读见[实验页](../lab.md)。两个示例都与独立完整重算的输出对照，并在 finally 中关闭上下文。
