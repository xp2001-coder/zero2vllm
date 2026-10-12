# 2.3 练习与答案

[正文](README.md) · [实验](lab.md)

## 1. 把状态与资源一起填完整

总容量 6，A、B 均为 P=3、N=4。依次执行：提交两条请求、准入 A、尝试准入 B、A 生成一项、取消 A、准入 B、B 完成。写出每一步的状态、已用额度与实际 K/V 长度。

<details>
<summary>答案</summary>

初始二者 WAITING，额度与 K/V 都为 0。A 准入后 RUNNING，预留 6，但首次前向前 K/V 仍为 0；B 的准入被暂缓。A 首步后 K/V=3、输出一项。取消 A 后保留输出，额度与 K/V 回到 0。B 再准入并完成四项，缓存峰值 6；终态中活动 K/V 和额度均为 0。

不要把预留 6 误写成“准入时已经算出六组 K/V”。

</details>

## 2. 生成与缓存之间为什么差一项

P=5，最多生成 N=3。三次 token 提交后的缓存长度分别是多少？第 3 项提交后，为什么还可以马上释放缓存？

<details>
<summary>答案</summary>

分别为 5、6、7，预留上限为 `5+3-1=7`。每次预测出的新 token 在下一次前向才写入缓存。最后一项只需交给调用者，没有后续前向需要使用它的 K/V。

第三次提交后同一 step 会进入 FINISHED，公开 view 中的缓存长度已经为 0；可从提交事件或 peak 字段看见历史长度 7。

</details>

## 3. 以下三个数字为什么不能混用

解释 `required_tokens`、`reserved_tokens` 和 `cached_positions`。一个终态请求的三个字段能否分别为 6、0、0？

<details>
<summary>答案</summary>

required 是这条请求在当前策略下需要的最大输入位置数；reserved 是当前仍占着的逻辑额度；cached 是实际尚存的 K/V 位置数。6、0、0 完全合理：请求历史需求为 6，现在已经收尾。

完整前缀 Llama 路径没有增量 KV，所以即使已预留 6，cached 也始终为 0。逻辑额度不是实际物理内存分配记录。

</details>

## 4. 收到取消后，为什么不能立即清空缓存

模型已经读取缓存并开始本次计算。取消回调到达时，先清空 keys/values，再等 forward 返回，有什么问题？本节应该留下什么事件？

<details>
<summary>答案</summary>

正在执行的代码仍可能使用这些对象；提前清理会破坏它的输入或允许资源被过早复用。先记录 cancel_requested，保留上下文与额度；forward 返回后检查标记，不提交本次 token，再 finalized。

应出现 `forward_started → cancel_requested → forward_returned → finalized`，其中没有 token_committed。这只是同步边界的顺序验证；真实异步设备还需同步在途任务。

</details>

## 5. 两种终止条件同时满足

如果最后一项既是 EOS，又达到 max_new_tokens，本节保存什么原因？如果已经收到取消，而 forward 又抛出了异常呢？

<details>
<summary>答案</summary>

前者为 FINISHED/eos，EOS ID 保留在原始输出中。后者为 FAILED/model_error，同时保留 cancel_requested=True，防止错误被取消信号掩盖。两种情况都统一回收。

不同引擎可以有不同约定，但实现、文档和测试必须一致。

</details>

## 6. 找出“状态正确、资源泄漏”的实现

有人把取消实现为 `request.status = CANCELLED`，不做其他动作。小规模演示里输出已经停止，看起来没有问题。设计两个能发现缺陷的检查。

<details>
<summary>答案</summary>

第一，用容量恰好只够一条请求的 A/B 实验。取消 A 后 B 应能准入，否则额度没有归还。第二，保留测试中的缓存对象引用，确认取消后 keys/values 实际为空，同时检查请求已撤销 context 引用。

还可检查 finalized 恰好一次、重复 cancel 不改变账本。仅检查 status 或进程 RSS 都不充分。

</details>

## 7. 取消 A 是否影响 B

让 A 使用 `[1,2,3]`、B 使用 `[4,2]`。手动交错推进后取消 A，让 B 完成。怎样证明 B 没有误用 A 的缓存？为什么 KVCache.owner 检查还不够？

<details>
<summary>答案</summary>

检查上下文与 KV 对象不同，然后将 B 的完整输出与独立 full-prefix 参考比较。只检查不同地址不能证明索引和输入没有混淆。

owner 检查的是模型实例，A/B 共用模型时两者 owner 本来相同。请求归属还依赖 Engine 对 request_id 与独立 context 的管理。

</details>

## 8. 为下一节写出调度器的最小职责

本节调用者手动选择 admit/step 顺序。下一节若自动轮流推进 A/B，哪些事情属于调度策略，哪些生命周期约束不能改变？

<details>
<summary>答案</summary>

策略决定本轮准入谁、推进谁、是否等待以及活跃队列顺序。它不能让未准入请求运行，不能将终态继续放入计算，不能提前释放计算中使用的资源，也不能重复回收或把 A 的可变缓存交给 B。

本节按最坏长度一次性预留额度。下一节若改成增量分配，还要处理增长失败、等待与可能的死锁；不能只把一次 reserve 拆成多次调用就认为完成了调度。

</details>
