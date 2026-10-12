# 3.2 实际运行记录

[正文](../README.md) · [复现实验](../lab.md)

2026-09-20，Windows / Python 3.11.9。使用 Python `-S` 运行本节工具和短示例，确认不依赖第三方 site-packages。

- [simulation.json](simulation.json)：10 个计数案例通过，覆盖混合调度、分块、序列上限、队首阻塞、抢占、前缀边界、容量、共享引用与逐出。
- [source-inspection.json](source-inspection.json)：同样运行 10 个案例，并离线核验固定提交的 6 个文件、11 个函数与 1 类 SchedulerOutput 字段。源码复用上一节已准备的官方快照。
- [walk-default.txt](walk-default.txt)：默认三轮，A/B 分别为 1/3、1/3、1/1。
- [walk-budget1.txt](walk-budget1.txt)：预算 1 时三轮均由 A 用完，B 尚未开始。
- [walk-threshold2.txt](walk-threshold2.txt)：总预算 8、阈值 2 时，A=1/B=2，剩余 5。
- [walk-budget3.txt](walk-budget3.txt)：练习第 3 题的四轮核对。

JSON 的 `course_source` 保存实际运行时的源码摘要。计数模型只保存数量、块 ID 与引用关系，反馈中的输出数量是明确设定的输入。所有记录均标记未执行上游代码、未执行模型、未验证 GPU；没有 A100/C500 性能数据。

全仓库回归见[3.2 交付记录](../../../../../reports/quality/lesson-3.2-delivery/README.md)。历史章节记录保留在对应章节，不能用本次结果冒充历史版本的重新执行。
