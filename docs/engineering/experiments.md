# 实验记录与比较规范

## 记录种类

| record_type | 含义 |
| --- | --- |
| measured_reference | 实际执行的 CPU 数学参考；不可作为 GPU 性能数据 |
| analytical_estimate | 按明示结构、形状和假设计算的 FLOPs、字节或下界；不是计时或显存测量 |
| measured_gpu | 实际 GPU 运行的结果，必须附版本与设备证据 |
| simulation | 调度、逻辑块等模拟结果，明确模拟假设 |
| template | 尚未运行的模板；指标使用 null |

当前的 [KV Cache 实验](../chapters/part-01-inference/03-kv-cache/results/initial-baseline.json)和 [INT4 Linear 实验](../chapters/part-09-quantization/08-bit-packing-serialization-and-repack/results/int4-reference.json)属于 `measured_reference`。GPU [记录模板](../../configs/templates/gpu-run.json)尚未执行；其他设备或第三方的测量记录不计入本项目的单卡验证。

[Part 1](../chapters/part-01-inference/README.md)中，`phase-demo`、`kv-demo` 的外层为实际执行的 CPU 参考，嵌套容量/性能模型标为 `analytical_estimate`；`metrics-demo` 默认事件与统计结果均为 `simulation`。不能只看最外层记录类型就把其中所有数字当成设备测量。

## GPU 结果的最小字段

1. 运行 ID、UTC 时间、成功/失败状态、采集脚本版本或源码 commit。
2. GPU 完整型号、显存、可见卡数、是否为整卡、主机 OS/CPU/内存。
3. 镜像摘要、驱动、运行时、PyTorch、vLLM、插件和算子库版本。
4. 模型 ID/revision、Tokenizer revision、dtype、量化方式。
5. 完整启动参数、请求参数和启用的优化功能。
6. 数据集或生成器版本、随机种子、实际输入 token 数、实际输出 token 数。
7. 请求 ID、提交时间、首内容到达时间、后续内容到达时间、完成时间、错误与超时。
8. 预热策略、重复次数、负载模式、并发或请求到达速率。
9. 原始测量文件和汇总结果的相对路径。

未运行的字段保留 `null`，不填入预计吞吐或占位性能值。失败和超时请求必须计入请求总数，并单独报告。

## 从 smoke 到 W0—W7 负载矩阵

| 场景 | 输入目标 | 输出上限 | 目的 |
| --- | --- | --- | --- |
| W0 smoke | 128 token | 16 token | 并发 1，首次正确加载、生成与停止 |
| W1 常规对话 | 1024 token | 512 token | 吞吐与延迟折中 |
| W2 Prefill 压力 | 4096，后续 16384 token | 128 token | 首响应、计算预算、峰值激活 |
| W3 Decode 压力 | 128 token | 2048 token | 每 token 延迟、带宽、量化/推测 |
| W4 长上下文 | 16384，后续 32768 token | 512 token | KV 容量、长度边界与稳定性 |
| W5 共享前缀 | 4096 共享前缀 + 256 后缀 | 128 token | APC 冷/热/无共享对照 |
| W6 混合长度 | 80% W1 + 20% W2 | 按相应场景 | 短请求长尾、调度与公平性 |
| W7 业务回放 | 固定数据与到达序列 | 固定生成规则 | 真实分布下满足 SLO 的 Goodput |

这里是待实现的负载设计。目标输入长度要在 Tokenizer 和 chat template 处理后验证；输出上限不等于实际输出长度。若实验强制固定输出长度，必须记录忽略 EOS 等设置，不与自然停止的实验混合比较。

先用并发 1、4 建立基线，再根据实测显存扩大范围。保留客户端与服务端日志，分别做冷启动、预热后运行和缓存命中实验。

W2/W4 只有在输入加输出预算不超过实际模型和服务长度限制时执行；增加长度后重新检查容量，不能向短上下文基线直接发送 16K 请求。上述矩阵是本项目选用的实验设计，W0 保持短输出；不要求 C500 跑到某个固定高并发。

有限 request rate 用于考察到达负载，固定最大并发与尽快发送用于观察饱和行为。客户端并发限制、排队及到达分布会影响实际发送率，记录设定与实际值。

## 指标口径

- **输出吞吐**：测量窗口内成功请求的实际输出 token 总数 / 对应测量窗口秒数，同时报告成功率和窗口定义。
- **端到端延迟**：客户端提交请求至收到完成事件，包含排队和传输。
- **首内容延迟**：提交至第一个含内容的流式事件。若能可靠得到逐 token 事件，再按明确口径报告 TTFT。
- **ITL**：相邻输出 token 之间的间隔。网络流式 chunk 可能包含多个 token，chunk 间隔不能直接称为 ITL。
- **TPOT**：在逐 token 时间或等价可靠信息可用时，按 `(最后 token 时间 - 首 token 时间) / (输出 token 数 - 1)` 计算；不足两个 token 时为 null。
- **Goodput**：满足预先声明的延迟目标的成功请求数 / 时间，同时附目标和总成功率。

P50/P95/P99 使用同一种计算方法，并保存样本数。客户端测量不等价于算子内核时间，GPU profiling 的时间口径另行说明。

至少保留三轮独立结果，汇总可以报告中位数和范围；合并分位数必须从原始样本重算，不能平均三轮 P99。缓存命中和推测接受率使用同一窗口内计数器增量，并核对该版本的单位、分母和 bonus token 口径。

## 公平比较

先比较同一平台优化前后，再比较 A100 与 C500 的完整系统表现。固定相同模型、revision、dtype、输入 token、输出策略、负载模式、预热和重复次数；记录注意力后端与软件差异。

公开原始数据，让无 GPU 的学生能够重新计算指标、画图和讨论结论。一次成功生成仅表示该场景可运行，不能推广成全模型、全功能支持。

## 优化专题的补充字段

使用 [功能验证模板](../../configs/templates/feature-validation.json)，将具体章节、baseline、单项与组合实验对应起来。

- 缓存：冷/热、共享前缀 token 数、命中/未命中、block size、匹配粒度、KV dtype 和 offload。
- 编译与算子：实际选中的 Attention/Linear/MoE 后端、融合是否发生、graph 模式、预热/编译缓存与 workspace。
- 量化：原模型/量化模型 revision、W/A/KV 格式、粒度和 scale 来源、校准集与独立评测集、质量损失。
- 量化算子：packed layout、符号/舍入/饱和、累加 dtype、shape/stride/group size、独立/融合 Q/DQ、实际 kernel、预打包与在线量化耗时；分别记录实现误差与策略质量，见 [量化篇验收](../chapters/part-09-quantization/design.md)。
- 推测：target/draft 模型、方法、草稿长度、接受率口径、草稿/验证成本、回滚与失败。
- 并行：可见卡数、TP/PP/DP/EP/PCP/DCP/SP、rank 映射、KV 布局、通信后端、节点拓扑。
- MoE/分离式部署：专家负载、EPLB/DBO、KV/EC Connector、传输字节、P/D 版本与布局、错误恢复。

首先比较单变量，再选有意义的组合。固定场景报告“支持/失败”之外，还要分开回答正确性、质量和性能；优化开关启用不代表内核实际生效或一定加速。完整待执行矩阵见 [特性地图](../references/feature-map.md)。

## 按原理逐项验收

| 专题 | 最小对照 | 正确性/生效证据 | 进一步组合 |
| --- | --- | --- | --- |
| APC | 关闭、冷、热、无共享、块边界 | 实际命中、执行长度、输出 | Chunked Prefill、缓存逐出 |
| 调度 | 相同到达序列，不同 token 预算 | 每轮推进、排队、抢占和回收 | 短长请求混合、异步 |
| Attention | 相同 shape/dtype/layout 的两条实现 | 数值对齐、实际 kernel | 分页、KV 量化、Graph |
| 编译/Graph | 编译和图模式分别消融 | guards/缓存、实际模式、回退 | runner、量化、推测 |
| 量化 | 浮点、显式 DQ、相同权重的融合实现 | 两种误差、真实布局、层/模型质量 | 图执行、不同 batch、KV/MoE |
| 推测 | 相同 target 与负载，开/关提案 | 接受/回滚、有效 token、停止 | 图执行、APC、取消 |
| TP/CP/EP/PD | 设备与版本满足条件后单独安排 | 张量/组/布局、通信、回收 | 混合拓扑与故障恢复 |

性能测量时关闭 profiler，另取同负载区间采集 trace。按算子语义、shape、dtype、次数对齐 A100/C500；不同平台名称相似的 kernel 不一定相同。累计 kernel 时间、单次时间、重叠区间和墙钟时间分别报告。具体练习见[编译与 Profiling 导读](../chapters/part-08-compilation/guide.md)。
