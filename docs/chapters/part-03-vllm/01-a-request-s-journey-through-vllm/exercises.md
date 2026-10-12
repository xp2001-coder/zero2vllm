# 3.1 练习与参考答案

[正文](README.md) · [实验](lab.md) · [源码定位表](source-map.md)

先写自己的判断，再展开答案。涉及实现的结论限定于本节固定提交及选定运行条件。

## 1. 一个组件是否对应一个进程

把 AsyncLLM、EngineCore、Executor、Worker、ModelRunner 按本节路径放进进程框。哪些还需要标明线程或异步任务？

<details>
<summary>参考答案</summary>

AsyncLLM 在前端，EngineCoreProc 在核心进程；单卡 uni 路径下执行器、Worker 与 ModelRunner 在核心进程内协作。核心输入/输出收发各有线程，主循环处理请求和执行；前端通过异步任务接收核心输出，并由各请求协程消费。不能根据类名数量推断进程数，也不能据此排除其他配置的辅助进程。

</details>

## 2. generator 创建时请求一定已经入队吗

聊天服务调用 `engine_client.generate(...)` 后，能否立即断言 Scheduler.waiting 中已经有该请求？

<details>
<summary>参考答案</summary>

不能。它返回异步生成器，函数体随迭代推进；之后还有输入处理、前端登记、消息发送、核心输入队列和主循环分派。阅读时区分对象创建、开始迭代、成功发送与核心登记。

</details>

## 3. 为什么同一请求需要不同对象

前端 RequestState 与核心 Request 共享什么？是否可以用前端直接改一个字段的方式取消核心请求？

<details>
<summary>参考答案</summary>

它们共享逻辑关联 ID，不共享普通 Python 对象内存。前端保存输出处理状态，核心保存调度状态；取消需通过消息通知核心。仅删除前端对象不会自动删除核心状态。

</details>

## 4. 工作单中是否需要携带权重

区分 EngineCoreRequest 和 SchedulerOutput。`scheduled_cached_reqs` 是否证明前缀命中？

<details>
<summary>参考答案</summary>

前者描述新请求，后者描述一次迭代的工作。已加载的模型权重不需要随每次请求重发。cached 在此指执行侧已经保留请求信息，不能单独用来判断 APC 命中。

</details>

## 5. 手算 A 的槽位

块大小 4，A 的块表 `[8,3]`，本轮处理位置 2、3、4。计算槽位；若执行顺序变成 A、B，正文的完整输入数组应怎样变化？

<details>
<summary>参考答案</summary>

A 的槽位为 34、35、12。A、B 顺序下 input_ids 为 `[12,13,14,23]`，positions 为 `[2,3,4,3]`，槽位为 `[34,35,12,23]`，片段边界 `[0,3,4]`。按 ID 归属的最终输出仍为 A 空、B `[24]`。

</details>

## 6. 为什么输入四个位置，只输出一个 token

A 计算了三个位置，为什么没有三个新输出？B 输出 24 后，24 的 KV 是否已写入本轮槽位？

<details>
<summary>参考答案</summary>

A 正在处理 prompt 的一部分，尚未计算完有效前缀；计算输入位置数不等于对外生成数。B 本轮读入 23 并假设采样出 24，24 尚未作为输入计算 KV。模型在部分 prefill 上可能产生后续会丢弃的候选，不能把对外无输出理解为实现绝对没做采样。

</details>

## 7. 输出为什么不能直接 zip

请求登记顺序 A、B，runner 顺序 B、A，输出 `[[24],[]]`。写出错误分配与正确分配。

<details>
<summary>参考答案</summary>

直接按登记顺序 zip 得到 A `[24]`、B `[]`，归属错误。使用 `req_id_to_index={"B":0,"A":1}` 得到 B `[24]`、A `[]`。同理，执行排列改变后不能沿用旧行号。

</details>

## 8. 两次内部输出为什么可能只读到一次

DELTA 模式下消费者尚未读取，连续产生两份输出。RequestOutputCollector 会怎样处理？这是否证明慢消费者会暂停模型？

<details>
<summary>参考答案</summary>

它可把未消费结果合并，下一次消费获得多份增量。对象合并不等于对调度器施加有界队列背压；也不能由合并推断 GPU 已暂停。SSE 还涉及角色、文本和结束事件，所以事件数不能直接当作 token 数。

</details>

## 9. 正常停止为什么发 ABORT

前端发现文本 stop string，核心还没判定结束，为什么发送 ABORT？用户看到的结束原因一定变成 abort 吗？

<details>
<summary>参考答案</summary>

前端反分词后才能发现一些跨 token 字符串条件，需要通知核心停止后续工作。ABORT 是内部控制消息，输出层已经记录的 stop 语义与核心控制动作可以不同，不能直接把控制消息当作用户完成原因。

</details>

## 10. 取消后显存没降，是否一定泄漏

写出从 AsyncLLM.abort 到块池的路径，并说明“ABORT 发送成功”“引用归还”“设备内存下降”的区别。

<details>
<summary>参考答案</summary>

前端撤销 → 核心客户端发送 → 输入线程/队列 → 主循环分派 → EngineCore.abort_requests → Scheduler.finish_requests → _free_request → KVCacheManager.free → BlockPool.free_blocks。发送完成不确认远端回收；引用归还后仍需考虑其他请求引用和前缀缓存；预分配设备池仍存在，不能仅由显存数字判断泄漏。

</details>

## 11. 迟到结果和 runner 状态怎么办

前端先取消，后收到旧输出；核心已经释放，runner 还留着请求。分别在哪些地方处理？

<details>
<summary>参考答案</summary>

OutputProcessor 查不到请求状态时跳过迟到输出。核心将完成 ID 放到 finished_req_ids，后续 SchedulerOutput 携带给 runner，_update_states 清理请求及 batch 状态。逻辑上有多个回收阶段，不能只在前端删除一个字典项就认为全部完成。

</details>

## 12. 工具 PASS 能证明什么

源码扫描通过，能否证明某次请求实际走了这些函数，或者 C500 支持这个版本？如果只有注释写着 `self.worker.run()`，它应算调用检查通过吗？

<details>
<summary>参考答案</summary>

只能证明清单中的固定文件与指定语法结构一致。运行时分支、动态派发和硬件支持需要实际环境验证；源码定位不能代替追踪日志。注释和字符串不属于 AST 的调用节点，不应算作函数调用。还要区分当前函数体、装饰器与嵌套函数体。

</details>
