# Part 10 · Speculative Decoding 推测解码

[课程总目录](../README.md) · [学习路线](../learning-paths.md)

前置知识：Part 1、3、5、6。

## 本篇章节

<!-- BEGIN part-index -->

| 节 | 课程 | 验收产出 | 进度 | 正文 |
| --- | --- | --- | --- | --- |
| 10.1 | [Draft、Verify 与拒绝采样](01-drafting-verification-and-rejection-sampling/README.md) | 用小词表验证拒绝采样分布与边界 | 🧩 | 教学骨架 |
| 10.2 | [N-gram 与 Suffix](02-n-gram-and-suffix-methods/README.md) | 比较重复文本和开放生成的命中率 | 📋 | 教学骨架 |
| 10.3 | [独立 Draft Model](03-separate-draft-models/README.md) | 扫描草稿长度并记录接受率与端到端收益 | 📋 | 教学骨架 |
| 10.4 | [EAGLE 与 EAGLE3](04-eagle-and-eagle3/README.md) | 画出特征/草稿/验证调用关系并核对 checkpoint | 📋 | 教学骨架 |
| 10.5 | [MTP 与模型原生多 token 预测](05-mtp-and-native-multi-token-prediction/README.md) | 验证预测模块适配并观察逐位置接受率 | 📋 | 教学骨架 |
| 10.6 | [其他草稿与并行提案](06-other-draft-methods-and-parallel-proposals/README.md) | 区分文档方案、注册项与所选 runner 的可运行实现 | 📋 | 教学骨架 |
| 10.7 | [动态推测与自适应验证](07-dynamic-speculation-and-adaptive-verification/README.md) | 改变 QPS 并检查草稿长度、验证量与尾延迟 | 📋 | 教学骨架 |
| 10.8 | [推测的缓存与执行组合](08-speculation-caching-and-execution-combinations/README.md) | 为支持的组合设计错误恢复和正确性检查 | 🧩 | 教学骨架 |
| 10.9 | [接受率与真实收益](09-acceptance-rates-and-actual-gains/README.md) | 解释高接受率但没有加速的场景并交付消融报告 | 🧩 | 教学骨架 |

<!-- END part-index -->
