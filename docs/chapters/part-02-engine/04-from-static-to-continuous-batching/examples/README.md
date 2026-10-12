# 调度教学示例

- [walk_batches.py](walk_batches.py)：同一组四个请求逐轮对比静态与连续准入，检查输出一致。
- [inspect_budgets.py](inspect_budgets.py)：活跃、选中、持有额度的区别，以及取消等待队首后的变化。

在仓库根目录运行，命令见[实验页](../lab.md)。只依赖 Python 标准库；示例使用上下文管理器保证退出时收尾。
