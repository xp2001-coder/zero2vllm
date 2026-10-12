# 3.1 固定版本源码定位表

[正文](README.md) · [实验](lab.md) · [核验记录](results/README.md)

版本：v0.8.5；提交：`ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d`。

这是源码的静态定位记录。函数中的调用表达式不等于运行时调用轨迹；分支、继承和消息分派仍需按正文阅读。

| 阶段 | 阅读目的 | 符号与位置 |
| --- | --- | --- |
| input | HTTP 路由选择聊天处理器 | [`create_chat_completion`](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/entrypoints/openai/api_server.py#L470) |
| input | 模板、参数与引擎生成器 | [`OpenAIServingChat.create_chat_completion`](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/entrypoints/openai/serving_chat.py#L121) |
| input | 输入转换为跨进程请求 | [`Processor.process_inputs`](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/v1/engine/processor.py#L195) |
| input | 请求协程等待并交出输出 | [`AsyncLLM.generate`](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/v1/engine/async_llm.py#L259) |
| input | 创建收集器并处理输入 | [`AsyncLLM.add_request`](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/v1/engine/async_llm.py#L197) |
| input | 先登记输出，再提交核心请求 | [`AsyncLLM._add_request`](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/v1/engine/async_llm.py#L239) |
| input | 辅助对照：客户端工厂的分支（不是本路径的直接调用） | [`EngineCoreClient.make_client`](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/v1/engine/core_client.py#L52) |
| input | 发送 ADD 消息 | [`AsyncMPClient.add_request_async`](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/v1/engine/core_client.py#L771) |
| input | 请求类型与序列化载荷 | [`AsyncMPClient._send_input`](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/v1/engine/core_client.py#L719) |
| input | 带引擎身份的 ZMQ 发送 | [`AsyncMPClient._send_input_message`](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/v1/engine/core_client.py#L730) |
| input | 输入线程解码并放入进程内队列 | [`EngineCoreProc.process_input_socket`](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/v1/engine/core.py#L496) |
| input | 主循环分派 ADD 与 ABORT | [`EngineCoreProc._handle_client_request`](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/v1/engine/core.py#L443) |
| input | 构造核心 Request 并进入等待队列 | [`EngineCore.add_request`](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/v1/engine/core.py#L163) |
| core | 生成本轮工作与缓存决策 | [`Scheduler.schedule`](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/v1/core/sched/scheduler.py#L137) |
| core | 调度、执行、更新结果 | [`EngineCore.step`](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/v1/engine/core.py#L192) |
| core | 按接口提交模型执行 | [`Executor.execute_model`](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/v1/executor/abstract.py#L82) |
| core | 单进程后端在本地调用 Worker | [`UniProcExecutor.collective_rpc`](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/executor/uniproc_executor.py#L49) |
| core | 转交模型运行器 | [`Worker.execute_model`](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/v1/worker/gpu_worker.py#L264) |
| core | 更新状态、准备输入、前向与采样 | [`GPUModelRunner.execute_model`](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/v1/worker/gpu_model_runner.py#L1003) |
| core | 组织位置、槽位与注意力元数据 | [`GPUModelRunner._prepare_inputs`](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/v1/worker/gpu_model_runner.py#L488) |
| core | 按请求取采样结果并检查结束 | [`Scheduler.update_from_output`](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/v1/core/sched/scheduler.py#L625) |
| output | 输出线程发送核心结果 | [`EngineCoreProc.process_output_socket`](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/v1/engine/core.py#L526) |
| output | 后台协程接收结果并入队 | [`AsyncMPClient._ensure_output_queue_task.process_outputs_socket`](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/v1/engine/core_client.py#L681) |
| output | 持续处理核心输出与字符串停止 | [`AsyncLLM._run_output_handler.output_handler`](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/v1/engine/async_llm.py#L353) |
| output | 按 ID 查状态、反分词并路由输出 | [`OutputProcessor.process_outputs`](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/v1/engine/output_processor.py#L293) |
| output | 合并未被消费者取走的结果 | [`RequestOutputCollector.put`](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/v1/engine/output_processor.py#L34) |
| output | 生成聊天协议的 SSE 数据 | [`OpenAIServingChat.chat_completion_stream_generator`](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/entrypoints/openai/serving_chat.py#L395) |
| cancel | 清理前端并通知核心 | [`AsyncLLM.abort`](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/v1/engine/async_llm.py#L404) |
| cancel | 发送 ABORT 消息 | [`AsyncMPClient.abort_requests_async`](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/v1/engine/core_client.py#L775) |
| cancel | 请求调度器结束指定 ID | [`EngineCore.abort_requests`](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/v1/engine/core.py#L183) |
| cancel | 移出队列并标记终态 | [`Scheduler.finish_requests`](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/v1/core/sched/scheduler.py#L762) |
| cancel | 释放逻辑所有权并记录已完成 ID | [`Scheduler._free_request`](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/v1/core/sched/scheduler.py#L791) |
| cancel | 归还请求持有的块引用 | [`KVCacheManager.free`](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/v1/core/kv_cache_manager.py#L296) |
| cancel | 引用归零后回到可复用队列 | [`BlockPool.free_blocks`](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/v1/core/block_pool.py#L227) |
| cancel | 接收完成 ID 并清理执行侧状态 | [`GPUModelRunner._update_states`](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/v1/worker/gpu_model_runner.py#L285) |
| input | 直接选择异步核心客户端 | [`AsyncLLM.__init__`](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/v1/engine/async_llm.py#L44) |
| core | 在迭代边界处理输入与执行 | [`EngineCoreProc.run_busy_loop`](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/v1/engine/core.py#L405) |
| output | 长度、EOS 与 token ID 停止 | [`check_stop`](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/v1/core/sched/utils.py#L5) |
