# 测试分组

| 目录 | 范围 |
| --- | --- |
| reference/ | 数学参考、缓存、第一篇与量化 |
| quantization/ | 量化模型 Passport 的声明、观察、证据引用与序列化 |
| cache/ | CPU 分页 K/V 张量、有效前缀、GQA、句柄、前缀共享、引用计数与逐出 |
| modeling/ | 可选 PyTorch 模型、检查点加载与独立参考对照 |
| integration/ | 命令行、平台配置、请求生命周期与迭代调度 |
| serving/ | SSE 字节拆分、异步服务、队列背压、ASGI 取消与真实 HTTP 客户端 |
| maintenance/ | 课程生成、数据分片、目录规范 |

全套检查：`python -m unittest discover -s tests -v`。
2.1：`python -m unittest tests.reference.test_decoder tests.modeling.test_decoder -v`。
2.2：`python -m unittest tests.modeling.test_checkpoint -v`；固定依赖见[实验页](../docs/chapters/part-02-engine/02-loading-real-model-weights/lab.md)。该组在临时目录生成随机小检查点，不访问网络。

2.3：`python -m unittest tests.integration.test_lifecycle -v`；24 项生命周期测试只需标准库，真实模型扩展见[本节实验](../docs/chapters/part-02-engine/03-request-and-cache-lifecycles/lab.md)。

没有安装可选依赖时相关测试明确跳过；这不能作为相应模型验证通过的证据。2.2 的真实 SmolLM2 结果另见[本节实测](../docs/chapters/part-02-engine/02-loading-real-model-weights/results/README.md)。

2.4：`python -m unittest tests.integration.test_scheduler -v`；23 项检查覆盖补位、队首阻塞、预算、公平轮转、取消/失败与独立输出，另含小型参数网格，见[实验页](../docs/chapters/part-02-engine/04-from-static-to-continuous-batching/lab.md)。

2.5：`python -m unittest tests.cache.test_paged tests.integration.test_paged_engine -v`；26 项检查覆盖张量分页与引擎连接，依赖 CPU PyTorch，见[实验页](../docs/chapters/part-02-engine/05-paged-caching/lab.md)。

2.6：`python -m unittest tests.cache.test_prefix tests.integration.test_prefix_engine -v`；29 项检查覆盖完整前缀匹配、隔离、引用回收、碰撞、OOM、冷暖 logits 与引擎连接，依赖 CPU PyTorch，见[实验页](../docs/chapters/part-02-engine/06-prefix-caching/lab.md)。

2.7：`python -m unittest tests.serving.test_streaming tests.serving.test_http -v`；29 项检查中 25 项只需标准库，3 项真实 HTTP 检查依赖 Uvicorn，1 项共享引用检查依赖 CPU PyTorch。未安装相应依赖时明确跳过，见[实验页](../docs/chapters/part-02-engine/07-http-and-streaming-output/lab.md)。

3.1：`python -m unittest tests.maintenance.test_source_journey -v`；16 项离线检查验证源码摘要、限定名、调用作用域、消息字段及下载校验。真实上游快照核验单独运行 [3.1 实验](../docs/chapters/part-03-vllm/01-a-request-s-journey-through-vllm/lab.md)。

3.2：`python -m unittest tests.reference.test_scheduling -v`；25 项标准库检查覆盖预算与队列、抢占后历史保留、反馈边界、命中容量、共享引用、随机容量守恒及默认离线。它们验证教学计数程序，不执行上游 vLLM。

3.3：`python -m unittest tests.reference.test_execution tests.modeling.test_packed_attention -v`；13 项标准库检查覆盖输入契约与离线 CLI，17 项 CPU PyTorch 检查覆盖融合权重、带偏移的因果性、GQA、跨请求隔离、填充、重排和输出过滤。不执行上游 vLLM；缺少 PyTorch 时后 17 项明确跳过。

3.4/3.5：`python -m unittest tests.reference.test_runner_state tests.reference.test_config_resolution tests.maintenance.test_later_source_reading -v`；22 项状态/数学测试、18 项配置测试与 5 项共用源码/离线入口测试，共 45 项，只需标准库。覆盖状态归属、复用、整批校验、增量写、采样数学、图分派和字段来源；不执行 GPU 或上游 vLLM。

9.1：`python -m unittest tests.quantization.test_passport tests.reference.test_quantization_overview -v`；33 项检查覆盖记录结构、非零 zero-point、Q/DQ/RQ、舍入顺序、饱和、FP32 输入和 Linear 数值。只需标准库。

9.2：`python -m unittest tests.quantization.test_tensor_contract tests.quantization.test_minifloat tests.quantization.test_lesson0902 -v`；39 项检查覆盖分组轴、尾组、非零 z、scale 精度、浮点特殊值与 bpw。仅 PyTorch oracle 1 项需要可选 torch，其余使用标准库。

9.3：`python -m unittest tests.quantization.test_calibration -v`；39 项检查覆盖数据协议、Observer、裁剪目标、误差与 NLL。1 项 PyTorch 对照为可选，其余只需标准库。

9.4：`python -m unittest tests.quantization.test_algorithms -v`；33 项检查覆盖舍入、补偿、处理顺序、等价变换与替代梯度。1 项 PyTorch 数学核验为可选，其余只需标准库。
