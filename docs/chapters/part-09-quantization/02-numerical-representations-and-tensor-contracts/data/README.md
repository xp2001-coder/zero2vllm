# 9.2 固定输入

[返回正文](../README.md)

[grouped-linear.json](grouped-linear.json) 是本课程自行设计的二维数值例。W[5,3] 非方阵、G=2 有尾组，包含零、正负数和恰好位于舍入中点的元素。输入均可精确表示为 FP32。

`symmetric` 保存独立手算的 q、DQ、Linear 输出与误差；`asymmetric_fp16_padded_storage` 保存声明存储计划的逐项 bit 账单。它不是模型权重、校准数据集或量化 checkpoint。
