# 2.5 可独立运行示例

从仓库根目录运行，依赖 CPU PyTorch。共享实现位于 `zero2vllm/cache/` 与 `zero2vllm/engine/`。

| 文件 | 观察内容 |
| --- | --- |
| [walk_pages.py](walk_pages.py) | A 的非连续物理块、slot 映射、实际 V 顺序、手算 Attention 和越界读取拒绝 |
| [paged_schedule.py](paged_schedule.py) | 每请求逻辑 R=5 向物理 8 slot 取整、队首等待、6 轮执行和最终回收 |

详细命令与输出解释见[实验页](../lab.md)。示例中的断言验证具体教学输入，不是可直接泛化到任意负载的常量。
