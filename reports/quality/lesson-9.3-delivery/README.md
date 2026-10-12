# 9.3 课程交付检查

[返回质量导航](../README.md) · [9.3 正文](../../../docs/chapters/part-09-quantization/03-calibration-observers-and-error-measurement/README.md)

本轮交付校准协议、Observer、分布统计、裁剪搜索与分层误差测量。实际回归时间（UTC）：`2026-09-20T12:42:28.085983+00:00`，Windows / Python 3.11.9。

| 检查 | 实际结果 | 记录 |
| --- | --- | --- |
| 全仓库回归 | 491项，退出0，跳过0项 | [regression.json](regression.json)、[完整日志](tests.txt) |
| 新增针对性测试 | 39项：协议12、Observer14、指标10、实验入口/对照3，已纳入回归 | [test_calibration.py](../../../tests/quantization/test_calibration.py) |
| 课程固定案例 | 20项全部通过 | [summary.json](../../../docs/chapters/part-09-quantization/03-calibration-observers-and-error-measurement/results/summary.json) |
| 文档命令 | 11条符合预期；含一条预算不足的预期退出2 | [commands.json](../../../docs/chapters/part-09-quantization/03-calibration-observers-and-error-measurement/results/commands.json) |
| 独立框架对照 | PyTorch 2.7.1+cpu CPU的MinMax/EMA两组affine [0,15]对照通过 | 同summary的torch_observers |
| 文档与目录 | 19篇129节，21节完成；核对链接与生成段落 | [course.json](course.json)、[layout.json](layout.json) |

## 教学与视觉检查

正文按15项结构展开，配实验手册、16组含答案练习、完整Calibration Protocol与CPU Passport。7张中文原创插图保留SVG与PNG，已逐张查看标签、公式、数值和边界；协议图修正文字宽度后重新检查。

正文区分校准、验证和最终测试，也区分权重、层、block、logits与任务误差。固定token预算的分布对照与固定数据池的样本量对照分别记录；数据为合成输入，不解释为真实模型质量。

## 验证范围

源码摘要：`d69b29a4a453bd60c64e315444ab8231868d3d3bbd96dbcf6f9ec17c0ed3c508`；实验、命令记录与回归使用同一代码/数据快照，测试期间摘要未改变。旧章节正文、实验与历史质量记录保持原样。

一次跨越任务中断的回归耗时约7063秒，旧服务的30ms截止时间案例只收到accepted/error，未收到预期的首个token。保留[该次结果](interrupted-regression.json)与[失败日志](interrupted-tests.txt)；随后在不修改服务代码/测试的情况下，[单项复跑](deadline-recheck.txt)通过，最终全量结果见上表。异常耗时与该测试对调度时间的依赖提示环境时序影响，但仅凭日志不能确定中断或系统暂停就是直接原因。

已执行CPU参考和固定版本PyTorch Observer对照。MQBench仅做公开源码阅读，未固定main分支commit、未运行模型；KL优化器与完整CNN支线待后续交付。A100/C500、真实Qwen和服务性能未测。本轮已补充CI入口，但没有提交、推送或执行GitHub远程CI。
