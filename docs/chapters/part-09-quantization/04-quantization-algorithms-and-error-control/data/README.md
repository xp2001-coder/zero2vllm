# 9.4 固定输入

[正文](../README.md) · [linear.json](linear.json)

全部为原创float64合成矩阵，布局X[M,K]×W[K,N]，无tokenizer或真实模型权重。

- main：W[3,2]，4条校准输入、3条留出输入；固定范围比例、缩放alpha和damping。
- rounding：W[2,1]，固定s=1、z=0，展示四选一与误差补偿；另有方向不同的留出输入。
- scaling：W[2,1]，激活通道幅值为8和1；展示缩放不单调。文件保留held_out供扩展练习，当前缩放专题只搜索/展示calibration，主比较的留出评价使用main.held_out。

所有候选选择只读calibration。结果保存规范化输入JSON的SHA256；不声称随机语料抽样、长上下文或模型级质量。数据的两个通道/三维空间是教学结构，不代表实际模型维度。
