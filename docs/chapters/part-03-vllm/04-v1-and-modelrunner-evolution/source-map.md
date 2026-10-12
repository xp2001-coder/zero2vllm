# 3.4 固定源码路线图

[正文](README.md) · [实验步骤](lab.md)

每条链接指向公开仓库的不可变提交。阅读器核验文件摘要、函数范围和选定调用；没有执行这些模块，也没有验证 GPU 能力。行号从固定文本的语法树提取。

## v0.8.5 · `ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d`

范围：3 个文件（含许可证及必要背景文件）、4 个函数。

| 定位 | 阅读时回答的问题 |
| --- | --- |
| [GPUModelRunner._update_states](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/v1/worker/gpu_model_runner.py#L285) | 按请求 ID 与批次排列维护状态 |
| [GPUModelRunner._prepare_inputs](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/v1/worker/gpu_model_runner.py#L488) | 早期 V1 输入准备 |
| [GPUModelRunner.execute_model](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/v1/worker/gpu_model_runner.py#L1003) | 模型执行及输出整理 |
| [InputBatch.condense](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/v1/worker/gpu_input_batch.py#L451) | 压紧持久 batch 的空行 |

## v0.17.1 · `95c0f928cdeeaa21c4906e73cee6a156e1b3b995`

范围：12 个文件（含许可证及必要背景文件）、22 个函数。

| 定位 | 阅读时回答的问题 |
| --- | --- |
| [Worker.__init__](https://github.com/vllm-project/vllm/blob/95c0f928cdeeaa21c4906e73cee6a156e1b3b995/vllm/v1/worker/gpu_worker.py#L104) | 读取 MRV2 开关 |
| [Worker.init_device](https://github.com/vllm-project/vllm/blob/95c0f928cdeeaa21c4906e73cee6a156e1b3b995/vllm/v1/worker/gpu_worker.py#L221) | 在 V1 Worker 内选择两种 runner |
| [GPUModelRunner.finish_requests](https://github.com/vllm-project/vllm/blob/95c0f928cdeeaa21c4906e73cee6a156e1b3b995/vllm/v1/worker/gpu/model_runner.py#L534) | 释放完成和抢占的执行侧状态 |
| [GPUModelRunner.add_requests](https://github.com/vllm-project/vllm/blob/95c0f928cdeeaa21c4906e73cee6a156e1b3b995/vllm/v1/worker/gpu/model_runner.py#L551) | 新增或恢复状态 |
| [GPUModelRunner.prepare_inputs](https://github.com/vllm-project/vllm/blob/95c0f928cdeeaa21c4906e73cee6a156e1b3b995/vllm/v1/worker/gpu/model_runner.py#L597) | 按本轮执行顺序构造映射与张量 |
| [GPUModelRunner.execute_model](https://github.com/vllm-project/vllm/blob/95c0f928cdeeaa21c4906e73cee6a156e1b3b995/vllm/v1/worker/gpu/model_runner.py#L839) | 实际图模式与模型执行 |
| [GPUModelRunner.capture_model](https://github.com/vllm-project/vllm/blob/95c0f928cdeeaa21c4906e73cee6a156e1b3b995/vllm/v1/worker/gpu/model_runner.py#L481) | 单独图捕获入口及 PP 限制 |
| [RequestState.add_request](https://github.com/vllm-project/vllm/blob/95c0f928cdeeaa21c4906e73cee6a156e1b3b995/vllm/v1/worker/gpu/states.py#L81) | 分配稳定状态槽与写入初值 |
| [RequestState.remove_request](https://github.com/vllm-project/vllm/blob/95c0f928cdeeaa21c4906e73cee6a156e1b3b995/vllm/v1/worker/gpu/states.py#L111) | 归还槽位 |
| [async_copy_to_gpu](https://github.com/vllm-project/vllm/blob/95c0f928cdeeaa21c4906e73cee6a156e1b3b995/vllm/v1/worker/gpu/buffer_utils.py#L17) | 独立临时副本与异步拷贝 |
| [StagedWriteTensor.stage_write](https://github.com/vllm-project/vllm/blob/95c0f928cdeeaa21c4906e73cee6a156e1b3b995/vllm/v1/worker/gpu/buffer_utils.py#L143) | 记录变长增量 |
| [StagedWriteTensor.apply_write](https://github.com/vllm-project/vllm/blob/95c0f928cdeeaa21c4906e73cee6a156e1b3b995/vllm/v1/worker/gpu/buffer_utils.py#L162) | 打包增量并分发写入 |
| [_apply_write_kernel](https://github.com/vllm-project/vllm/blob/95c0f928cdeeaa21c4906e73cee6a156e1b3b995/vllm/v1/worker/gpu/buffer_utils.py#L197) | 按段写入目标表 |
| [prepare_pos_seq_lens](https://github.com/vllm-project/vllm/blob/95c0f928cdeeaa21c4906e73cee6a156e1b3b995/vllm/v1/worker/gpu/input_batch.py#L244) | GPU 元数据入口 |
| [_prepare_pos_seq_lens_kernel](https://github.com/vllm-project/vllm/blob/95c0f928cdeeaa21c4906e73cee6a156e1b3b995/vllm/v1/worker/gpu/input_batch.py#L208) | 按状态映射读取已计算进度 |
| [expand_idx_mapping](https://github.com/vllm-project/vllm/blob/95c0f928cdeeaa21c4906e73cee6a156e1b3b995/vllm/v1/worker/gpu/input_batch.py#L552) | 为多 logits 行展开状态索引 |
| [CudaGraphManager.capture_graph](https://github.com/vllm-project/vllm/blob/95c0f928cdeeaa21c4906e73cee6a156e1b3b995/vllm/v1/worker/gpu/cudagraph_utils.py#L82) | 分段图或整图的捕获分派 |
| [CudaGraphManager.get_cudagraph_runtime_mode](https://github.com/vllm-project/vllm/blob/95c0f928cdeeaa21c4906e73cee6a156e1b3b995/vllm/v1/worker/gpu/cudagraph_utils.py#L288) | 每轮图模式及未捕获回退 |
| [CudaGraphManager.run_fullgraph](https://github.com/vllm-project/vllm/blob/95c0f928cdeeaa21c4906e73cee6a156e1b3b995/vllm/v1/worker/gpu/cudagraph_utils.py#L314) | 整图重放 |
| [get_cudagraph_sizes](https://github.com/vllm-project/vllm/blob/95c0f928cdeeaa21c4906e73cee6a156e1b3b995/vllm/v1/worker/gpu/cudagraph_utils.py#L333) | 预设形状映射 |
| [Sampler.sample](https://github.com/vllm-project/vllm/blob/95c0f928cdeeaa21c4906e73cee6a156e1b3b995/vllm/v1/worker/gpu/sample/sampler.py#L101) | 处理 logits 并进入 Gumbel 采样 |
| [Sampler.__call__](https://github.com/vllm-project/vllm/blob/95c0f928cdeeaa21c4906e73cee6a156e1b3b995/vllm/v1/worker/gpu/sample/sampler.py#L56) | 可选 top-k logprobs 与输出 |

## 建议顺序

1. 先看旧版 `_update_states` 与 `condense`，区分请求对象和当前 batch。
2. 在新版 Worker 核对开关，再从 `RequestState` 追到 `prepare_inputs`。
3. 分别标出 CPU 排序/累计和、状态映射、拷贝、设备端元数据准备。
4. 阅读增量写入时，区分暂存、内容拷贝、UVA 元数据和目标写入。
5. 阅读采样器调用顺序；本课 Gumbel 与 logprob 代码仅作独立数学演示。
6. 读图管理的捕获与运行时分派；设计文档的能力说明与实现不一致时，保留差异并以固定实现和实测核验。

旧版与新版共核验 26 个函数；两份清单分别计数，不把不同版本的同路径合并为一个文件。

本课状态池分配最低空行，额外增加 generation 防护并拒绝重叠写；它不是上游 RequestState 的等价替换。

## 清单与许可证

清单见 [data/](data/README.md)，含固定提交和 LF 归一化 SHA-256。源码原文只在显式下载时写入运行者的本地缓存；公开课程不打包上游源码。上游许可证作为固定文件一并校验，解释与原创图见本仓库许可。

若目标文件中存在与本节无关的 overload 或 property getter/setter 重名，只筛选本节所需符号；所需符号本身仍不得含歧义。3.1 原有默认严格模式不变。
