# 9.4 源码、论文与核验范围

[正文](README.md) · [实验手册](lab.md)

## 课程实现与实际执行

| 位置 | 内容 |
| --- | --- |
| [algorithms.py](../../../../zero2vllm/quantization/algorithms.py) | float64固定网格、裁剪、舍入枚举、GPTQ核心、变换与替代梯度 |
| [lesson0904.py](../../../../scripts/experiments/lesson0904.py) | 23个案例、统一对照、协议/Passport、PyTorch数学核验 |
| [test_algorithms.py](../../../../tests/quantization/test_algorithms.py) | 33项黄金数值、形状、边界与独立对照测试 |
| [输入](data/linear.json) | 固定X/W与搜索候选，无模型下载 |

本轮使用Python3.11.9，PyTorch2.7.1+cpu、float64。独立对照在每步重新计算剩余Hessian主子矩阵的逆，再求受约束最小二次代价；它与本课一次Cholesky后逐步补偿的路径不同。STE对照使用显式detach构造所声明的反向规则；没有执行完整QAT训练。

## GPTQ：固定论文与官方源码阅读

- [论文v2](https://arxiv.org/html/2210.17323v2)：第3节的受约束补偿，以及Algorithm1的Cholesky/块更新。
- [官方gptq.py](https://github.com/IST-DASLab/gptq/blob/main/gptq.py)：`add_batch`、`fasterquant`、dead通道处理、static_groups、actorder、damping与`Hinv`含义。

源码阅读日期2026-09-29，链接为可变main分支，没有取得固定commit，未导入执行上游包。本课的数学依据固定到论文v2；实现和输入由课程源码摘要固定。实际矩阵测试不等价于官方GPU程序验证。

阅读时先转置：官方权重为[N,K]，本课为[K,N]。官方输入统计的归一化与本课H=2XᵀX不同；damping和损失比较需同步对齐。本课不复制官方dead通道修复、动态分组、CUDA同步或模型遍历。

## AWQ与缩放

- [AWQ论文](https://arxiv.org/abs/2306.00978)：激活信息与显著通道缩放。
- [官方auto_scale.py](https://github.com/mit-han-lab/llm-awq/blob/main/awq/quantize/auto_scale.py)：`get_act_scale`、`_search_module_scale`、`scale_ln_fcs`等。
- [SmoothQuant](https://arxiv.org/abs/2211.10438)：W/A困难迁移。
- [OS+](https://aclanthology.org/2023.emnlp-main.102/)与[OmniQuant](https://arxiv.org/abs/2308.13137)：平移/缩放、学习裁剪/等价变换。

AWQ源码同样仅为2026-09-29 main公开阅读。本课参考了“激活幅值→候选因子→量化→输出误差”的职责划分，重新实现单Linear实验；没有复刻完整模块搜索、图折叠或裁剪导出。课程数据和图均为原创。

## 舍入、重构、旋转与QAT入口

| 资料 | 本节用途 |
| --- | --- |
| [AdaRound](https://proceedings.mlr.press/v119/nagel20a.html) | 学习舍入与软硬选择 |
| [AutoRound/SignRound](https://arxiv.org/abs/2309.05516) | 用带符号梯度优化舍入的路线；框架版本能力另核对 |
| [BRECQ](https://arxiv.org/abs/2102.05426) / [QDrop](https://arxiv.org/abs/2203.05740) | block重构与随机去掉激活量化影响 |
| [QuaRot](https://arxiv.org/abs/2404.00456) / [SpinQuant](https://arxiv.org/abs/2405.16406) / [OSTQuant](https://arxiv.org/abs/2501.13987) | 固定/学习正交变换及缩放组合 |
| [LSQ](https://arxiv.org/abs/1902.08153) / [LSQ+](https://arxiv.org/abs/2004.09576) / [PACT](https://arxiv.org/abs/1805.06085) | 学习步长、offset与裁剪阈值 |

除所声明的CPU/PyTorch数学核验外，本课没有运行这些论文的完整模型实验，不复述论文性能为课程结果。算法层理解与框架生产/导出在9.6继续衔接。
