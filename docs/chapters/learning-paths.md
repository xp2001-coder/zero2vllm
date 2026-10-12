# 学习路线与章节分工

完整目录负责覆盖技术范围，学习路线负责选择进入顺序，开发里程碑负责安排材料交付。三者分别见 [课程目录](README.md)、本文和 [开发路线](../engineering/roadmap.md)。

**[Part 1 的 1.1—1.6](part-01-inference/README.md)已提供完整正文、CPU 代码、操作命令和练习答案；初学者从 1.1 顺序学习。** 还可直接运行 [CPU KV Cache 引导实验](part-01-inference/03-kv-cache/lab.md)、[CPU INT4 Linear 实验](part-09-quantization/08-bit-packing-serialization-and-repack/lab.md)、逻辑块模拟及配置/环境检查。[7 篇专题导读](../references/topic-guides.md)可配合下表学习。所有 GPU 实验仍需独立验证。

## 先用现有材料走通一条短路线

这条路线只使用已经提供的教程、代码和手算题。先从一次生成走到缓存、调度，再完成一个量化 Linear；有部署经验的读者可以从第 5 步进入。

| 步骤 | 读与做 | 交付一个小结果 | 类型 |
| --- | --- | --- | --- |
| 1. 看懂一次生成 | [第一课 1.1](part-01-inference/01-next-token/README.md)，运行与手算 | 张量维度表、最后一行 logits、五道练习解释 | 完整正文与 CPU 实验 |
| 2. 复用历史计算 | [KV Cache 实验](part-01-inference/03-kv-cache/lab.md)，运行默认命令 | 解释 4 个输入、8 个输出为什么缓存路径计算 11 个位置 | CPU 实验 |
| 3. 管理多个请求 | [调度与缓存](part-05-kv-cache/guide.md)，手算三轮预算和块寻址 | 调度表、block/slot 映射、KV 容量估算 | 手算；2.3—2.6 另有 CPU 生命周期、调度、张量分页与前缀共享实验 |
| 4. 对应到真实框架 | [请求、状态与张量](part-03-vllm/guide.md) | 标明 request ID、调度输出、模型输入和输出归属的职责图 | 3.1—3.5 已提供固定源码、调度、张量、状态与配置实验 |
| 5. 做出量化 Linear | [INT4 实验](part-09-quantization/08-bit-packing-serialization-and-repack/lab.md)，比较 G=4 与 G=8 | 两份结果，说明 scale 数量、实现误差与量化误差 | CPU 实验 |
| 6. 设计一次优化 | [量化导读](part-09-quantization/guide.md)和[Profiling 导读](part-08-compilation/guide.md) | 浮点、显式 DQ、融合路径的对照表，标明哪些还未实现 | 实验设计；GPU 测量待完成 |

### 第一步：运行后先看什么

在包含 `pyproject.toml` 的工程根目录执行：

```bash
python -m zero2vllm next-token-demo --output .local/runs/part-01-inference/01-next-token/default.json
```

先看 `input.token_ids`、`trace.next_token_logits` 和 `generation.steps`，解释 `[0, 1]` 如何预测出 ID `2`。`verification.passed` 应为 `true`；完成第一课的手算、维度表和练习后，再进入缓存实验。

第 2 步运行 `python -m zero2vllm demo --output .local/runs/part-01-inference/03-kv-cache/first-kv-cache.json`。检查 `verification.passed` 和 `same_token_ids` 为 `true`，`max_abs_logit_error` 不超过 `1e-12`。再解释投影位置数 `4+7=11`；如果还说不清最后一个输出为什么不用再计算，回看 KV 实验的“默认结果如何解释”。

第 5 步用同一个随机种子比较：

```bash
python -m zero2vllm quant-demo --group-size 4 --seed 7 --output .local/runs/part-09-quantization/08-bit-packing-serialization-and-repack/int4-g4.json
python -m zero2vllm quant-demo --group-size 8 --seed 7 --output .local/runs/part-09-quantization/08-bit-packing-serialization-and-repack/int4-g8.json
```

先比较 `storage.scale_bytes` 和 `verification` 下的两种误差。更小分组会增加本例的 scale 数量；量化误差的变化以运行结果为准。它们都是 CPU 数学实验，不能根据输出中的字节数推断 GPU 加速。

### 完成短路线以后

- 学生补齐下面 S0—S2 所列基础，再进入真实模型与 PyTorch 教学引擎。
- 工程师用 E0 锁定基线，再把第 6 步的设计落实到一种平台。
- 量化方向先完成 [9.1](part-09-quantization/01-quantization-system-overview/README.md) 的 Q/DQ/RQ、zero-point 与 Passport，接着完成 [9.2](part-09-quantization/02-numerical-representations-and-tensor-contracts/README.md) 的分组轴、数值表示与尾组契约，完成 [9.3](part-09-quantization/03-calibration-observers-and-error-measurement/README.md) 的校准协议、Observer 和独立误差对照后，继续 [9.4](part-09-quantization/04-quantization-algorithms-and-error-control/README.md) 的误差控制推导、GPTQ 与缩放对照，再沿 Part 9 v1.0 的 W4A16 主线追踪格式和执行，最后扩展 KV/MoE。

下面各表给出完整学习目标，其中仍有待建设内容。当前手头没有 GPU，可以完整学习 Part 1 并运行全部 CPU 案例，再选择量化实验、调度手算和职责图。

## 1. 学生路线：先完成一个能解释的推理闭环

起点：基础 Python。矩阵运算按需补充；进入 2.1 前学习基础 PyTorch。没有 GPU 时，可以先完成 CPU 阶段，0.2 的真实模型生成体验移到 GPU 阶段。

| 阶段 | 建议顺序 | 过关时应能交付 | 设备与当前入口 |
| --- | --- | --- | --- |
| S0 · 跑通与观察 | 0.1/0.3 作课前准备，1.1 → 1.2 → 1.3 | 解释 Prefill/Decode，比较缓存前后的 logits 和工作量 | CPU；1.1—1.3 正文与实验已提供 |
| S1 · 指标、结构与正确性 | 1.4 → 1.5 → 1.6 → 2.1 → 2.2 | 重算性能指标，比较缓存对象；构建 PyTorch 模型，加载真实权重并逐层对齐 | CPU；正文、代码与实验均已提供 |
| S2 · 管理多个请求 | 2.3 → 2.4 → 2.5 → 2.6 | 回放请求加入、结束、取消、抢占、前缀复用与回收 | 2.3—2.6 已有 CPU 请求管理、迭代调度、张量分页与完整前缀共享；抢占重计算后续建设 |
| S3 · 进入真实框架 | 3.1 请求路径 → 3.2 调度与 KV → 3.3 模型执行 → 3.4 Runner → 3.5 配置 → 4.1 或 4.2 → 0.2 | 保存一次 vLLM 小模型生成记录，并将教学模块映射到 vLLM | 3.1—3.5 已可用 CPU 核验源码、回放调度、对齐张量并追踪状态与配置；设备运行再选择一种 GPU |
| S4 · 完成一个优化专题 | 5.1 → 5.2 → 6.1 → 6.2 → 7.1；再选 8、9 或 10 篇 | 用固定输入解释某项优化的原理、正确性和收益条件 | 数学与模拟可用 CPU；性能结论需要 GPU |
| S5 · 服务与结课 | 2.7 → 15.1 → 17.1 → 17.2 | 提交小型引擎、流式取消检查和一份原始实验报告 | 2.7 已提供 CPU 本机 HTTP 服务与取消实验；15.1、17.1—17.2 的 GPU 服务实验继续建设 |

S0/S1 中的 Part 1、2.1 和 2.2 已提供完整 CPU 学习材料；S2 的 2.3 请求生命周期、2.4 迭代调度、2.5 张量分页和 2.6 前缀共享已完成 CPU 验证；S5 的 2.7 可在 CPU 上完成真实 HTTP、流式读取、背压与断连实验。真实模型增量缓存、GPU 算子和共享感知准入继续建设。2.2 使用真实模型权重，但不要求 GPU。Part 11—13 可在理解基本缓存和调度后学习切分推导与源码，实际通信性能实验需要额外设备。

## 2. 工程师路线：从负载和测量进入优化

起点：已有 PyTorch、Transformer 或部署经验。按问题补读 Part 1—2，不要求重新做完所有入门练习。

| 阶段 | 建议章节 | 交付物 |
| --- | --- | --- |
| E0 · 固定基线 | 0.3、1.4、4.1 或 4.2、17.1—17.3 | 版本记录、固定负载、正确性基线、计时方法与初始性能记录 |
| E1 · 找到瓶颈 | 3.1—3.5、5.1、6.1—6.3 | 请求与张量调用链；说明时间主要花在计算、访存、调度还是通信 |
| E2 · 单项优化 | 按瓶颈选择 Part 5—10 | 一次只改变一个因素，保存 kernel 命中、回退、质量和延迟结果 |
| E3 · 检查组合 | 10.8、14.6、15.1、17.5—17.6 及相关专题 | 对功能组合、并发、长上下文和取消/抢占做回归 |
| E4 · 平台与分布式扩展 | 4.3、Part 11—13、18.1—18.2 | 两平台独立证据，或并行拓扑/通信推导与有条件的实机结果 |

性能分析的基础方法提前使用，Part 17 保留完整工具与综合项目。这样能在改算子或调度之前，先定义负载、指标和验收条件。

## 3. 量化方向：算法 → 格式 → 算子 → vLLM → 服务

按 [Part 9 v1.0 总设计](part-09-quantization/design.md)学习。Qwen3-8B W4A16 G128 为贯穿案例，先用 CPU 小矩阵建立契约，再进入少量层与完整模型。GPU 前完成 E0，并补读 3.3、7.6—7.7；经典 CNN 支线不构成 LLM 的额外前置。

| 阶段 | 建议顺序 | 核心产出 |
| --- | --- | --- |
| A · 基础与校准 | 9.1 → 9.2 → 9.3 | Passport、数值契约、Calibration Protocol、CPU 误差参考 |
| B · 算法与生产 | 9.4 → 9.5 → 9.6 | 算法对照、精度策略、两条独立模型生产路径 |
| C · 格式与算子 | 9.7 → 9.8 → 9.9 → 9.10 | 格式分析、packing/repack、W4A16 调优与 W/A 分项消融 |
| D · 运行时与收益 | 9.13 → 9.14；9.11/9.12 按需扩展 | 实际执行证据、模型质量、独立平台服务报告、KV/MoE 契约 |

算法集中在 9.4；9.5 讲精度策略，9.10 讲激活量化执行开销。工程师可先沿 W4A16 完整链路阅读，正式算法对照仍须具备最小校准协议与独立评测。A100/C500 分别验证，不能根据模型名称推断 kernel 或原生计算精度。

## 4. 交叉专题由哪一章负责

| 交叉内容 | 主讲位置 | 其他章节怎样复用 |
| --- | --- | --- |
| 教学缓存与正式实现 | Part 2 实现最小机制；Part 5 分析正式 vLLM 的缓存管理 | Part 3 负责源码映射，不重复写一套缓存教程 |
| APC、Attention、KV 量化 | Part 5 讲缓存命中与生命周期；Part 7 讲计算；9.11 讲低精度表示与 Q/DQ | 组合实验引用三者的约定，检查命中、质量和 Attention 成本 |
| 手写融合与编译融合 | 7.6—7.7 讲算子与选择机制；Part 8 讲编译变换；9.9/9.10 讲量化内核调优 | 分别记录生成代码与手写 kernel，避免只用“融合”概括效果 |
| DBO 与 MoE 重叠 | 6.5 讲机制及与 TBO 的术语对照 | 12.6 负责 MoE 负载下的通信/计算实验 |
| KV 分层与跨进程传输 | 5.7 讲缓存容量与层级；Part 13 讲传输协议和生命周期 | Part 16 的权重 offload 单独记录对象、预算与开销 |
| CP、PCP、DCP | Part 11 统一讲切分、进程组与通信 | Part 5/7 仅引用对应布局，Part 13 复用拓扑条件 |
| 量化算法、格式与后端 | 9.4 统一讲算法；9.2/9.5 定义契约与策略；9.7/9.8 讲格式与重排；9.13 讲接入 | 具体格式和库名称维护在特性清单，避免按工具重复拆章 |
| 正确性与性能口径 | 1.4/1.6 给出基础约定，Part 17 系统展开 | 每个专题记录自己的容差、负载与原始证据 |

每章写作时链接已解释过的概念，只展开本章新增的机制、代码与实验。目录中的“前置”表示推荐背景；工程师可用已有经验满足，不要求按篇号顺序读完。

## 5. 建设时的优先级

1. **可用入口**：完善 CPU 实验正文和基本检查，让学生在普通电脑上开始学习。
2. **真实基线**：分别锁定单卡 A100、C500 的环境、模型和生成结果。
3. **共享基础**：完成 PyTorch 对等实现、请求生命周期及缓存/调度主线。
4. **首个深入专题**：沿量化张量契约、Q/DQ、Linear 内核、vLLM 接入逐步交付；与单卡基线配合。
5. **持续扩展**：按瓶颈与依赖完善其他优化、模型服务和分布式专题，并维护上游特性清单。

维护者负责逐步发布两平台的真实实验记录；学生可以选择一种平台。无 GPU 时提交数学、模拟或数据分析成果，并标明验证层级。
