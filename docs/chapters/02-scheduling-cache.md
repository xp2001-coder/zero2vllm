# 调度与缓存：一次推进多少 token，放在哪里

对应 3.2、5.1—5.4、6.1—6.3。前置：[请求链路](01-request-lifecycle.md)、[CPU KV Cache 实验](../labs/01-kv-cache.md)。状态：**导读、手算与 GPU 实验设计已提供；完整调度模拟和 GPU 实测待完成**。

本篇吸收 vLLM 第 4、5 讲与测试手册，结合 SGLang 第 4、6、7 讲的对照问题。

## 1. 三种预算不能互相替代

| 预算 | 控制什么 | 不代表什么 |
| --- | --- | --- |
| 请求/序列数量 | 本轮或运行集合容纳多少条序列 | 每条都能放进最大上下文 |
| 计算 token 数 | 一轮安排多少新位置参与计算 | GPU 所有缓存 token 的总量 |
| KV 容量 | 历史和新位置可占多少缓存空间 | 本轮可执行的计算量 |

另有最大序列长度、encoder 工作量、LoRA/模型约束、推测预留等限制。当前源码还可能区分 batched-token 与 scheduled-token 上限；阅读时应查有效配置字段的取值与回退关系。[Scheduler 源码](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py)

## 2. 手算一个调度轮次

教学假设：FCFS、允许分块 prefill、无推测/多模态/远端缓存，KV 足够。总 token 预算 4096，已有 B/C 各需一个 decode token，新请求 A 有 8192 个 prompt token。

| 轮次 | B/C 消耗 | A 本轮计算 | A 累计计算 | A 是否完成 prefill |
| --- | --- | --- | --- | --- |
| 1 | 2 | 4094 | 4094 | 否 |
| 2 | 2 | 4094 | 8188 | 否 |
| 3 | 2 | 4 | 8192 | 是 |

4094 是简化假设下的可用上限。真实 scheduler 还可能受对齐、可用块、长请求策略、异步依赖等限制；表格不是逐行复刻任何版本的调度器。

改变条件再推导：第二轮 B 完成，第三轮新请求 D 到达，或空闲块不足。必须同时更新队列、预算和块所有权；只画 waiting/running 两个盒子不够。

Chunked Prefill 把长输入分到多个调度轮次，让短请求有机会更早推进；chunk 变小也会改变批量效率。它与跨设备切分上下文的 CP 是两个维度，见[并行导读](06-parallel-moe-disagg.md)。

## 3. PagedAttention 先解决寻址

假设每块放 4 个 token，某请求的逻辑块映射为 `[7, 2, 9]`。其位置 6 落在逻辑块 `6 // 4 = 1`，块内偏移为 `6 % 4 = 2`，对应物理块 2 的偏移 2。若把教学缓存视为一维数组，slot 为 `2 × 4 + 2 = 10`。

实际 KV 有层、K/V、头、head dimension、dtype 和后端布局；这个 slot 例子只说明索引关系。读取 [BlockPool](../../zero2vllm/reference/blocks.py)能验证逻辑分配，当前实现不分配设备 KV Tensor。

普通 dense/GQA 模型的一个粗略预算为：

```text
KV 字节 ≈ token 数 × 层数 × 2 × KV heads × head dimension × 每元素字节
```

例如 32 层、8 个 KV heads、head dimension 128、每元素 2 字节，每 token 约 128 KiB，8192 token 约 1 GiB。这个手算不含页尾浪费、scales、workspace、图内存，也不适用于直接套算 MLA/SSM、混合层和带 KV 复制的并行布局。

## 4. Prefix Cache 复用的是已计算前缀

分页决定 KV 怎么放，APC 决定已有 KV 能否被另一个请求复用。vLLM 的哈希块设计把父块哈希、当前块 token 和额外标识连接起来，完整块才进入这一缓存匹配机制；LoRA、多模态输入和隔离 salt 等会影响缓存身份。[APC 设计](https://docs.vllm.ai/en/latest/design/prefix_caching/)

假设块大小 4，两个请求共享前 9 个 token。按完整块计算，前 8 个 token 构成两个可复用块，剩下 1 个仍在不完整块中。真实执行还要区分匹配出的缓存长度与为生成 logits 实际跳过的长度；完全相同的 prompt 也不能简单断言零次 forward。

APC 可以减少重复 prefill 工作；后续 decode 对历史 KV 的读取仍存在。SGLang 的 RadixAttention 是有用的对照，类名、粒度和逐出细节仍应在各自框架内核对。

## 5. 块的生命周期比命中率更重要

逐步追踪：申请 → 写入 → 满块可缓存 → 被其他请求引用 → 引用减少 → 空闲且可回收 → 淘汰/复用。一个不再被活跃请求引用的块可以保留可复用内容，也可能在容量不足时被覆盖。

抢占重算保留请求的逻辑上下文，释放部分执行资源，之后重新计算 KV；它会消耗计算和增加延迟。普通重算、CPU offload、远端 KV 传输有不同对象和流程，不要统一称为 swap。

读异步路径时分别核对“请求逻辑结束”和“内存可以安全复用”。当前 scheduler 对异步依赖、远端 KV 等有专门状态，不能从 waiting 队列名称推断所有资源状态。[Scheduler 实现](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py)

## 6. 一组有解释力的 APC 实验

所有 GPU 行目前为待执行。固定模型/Tokenizer revision、实际 token 前缀、采样、并发和计时窗口，分开比较：

| 用例 | 控制方法 | 应观察什么 |
| --- | --- | --- |
| 关闭 APC | 同一组请求，缓存策略为关闭 | prefill 工作量基线 |
| 冷缓存 | 新实例或确认重置完成 | 第一次请求的计算和缓存建立 |
| 热缓存 | 先建立前缀，再发送相同前缀的新请求 | 命中 token 与 TTFT，不只看总吞吐 |
| 无共享 | 长度相同但 token 前缀不同 | 排除长度、批量和预热差异 |
| 块边界 | 共享长度设为 B−1、B、B+1 | 解释命中粒度与执行长度 |
| 隔离与逐出 | 不同 salt；再用无关请求施加容量压力 | 隔离是否有效，淘汰后能否正确重算 |

APC × Chunked Prefill 先各做一组单变量，再做组合。热缓存和冷缓存不能混成一个没有标签的吞吐数字。

## 7. 验收与思考

**学生**：提交三轮调度表、位置 6 的块寻址、8192 token 的 KV 预算；解释为什么“并发翻倍”不等于“token 预算翻倍”。

**工程师**：保存有效配置、缓存指标增量、每请求延迟、抢占次数和实际后端；解释命中增加却没有端到端加速的情况。可能的原因包括输入处理、排队和 decode 占比，但结论需要时间线支持。

继续阅读[Attention 与算子](03-attention-kernels.md)；对模型特有缓存，回到 Part 5 的 Hybrid、Mamba、KV Offload 专题，不能把本篇的普通 KV 假设直接推广。
