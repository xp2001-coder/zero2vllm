# 请求管理与迭代调度

[2.3 课程](../../docs/chapters/part-02-engine/03-request-and-cache-lifecycles/README.md)

- `lifecycle.py`：状态、准入额度、取消、统一收尾和不可变结果快照。
- `runners.py`：复用已有 TinyAttentionLM 的真实 K/V 列表，以及 2.2 Llama 的完整前缀路径。
- `scheduler.py`：[2.4 课程](../../docs/chapters/part-02-engine/04-from-static-to-continuous-batching/README.md)的静态组/连续准入、FIFO、轮转选择、冻结计划与逻辑边界记录。

当前为单线程同步教学实现。2.3 可显式选择 `admit` 和 `step`；2.4 可把新建 engine 交给调度器专用，之后仅通过 scheduler 提交、取消和推进。计划内请求仍串行执行，未合成张量 batch。

调度器接受已验证的增量缓存运行器：Tiny 列表、PagedTinyRunner 与 PrefixTinyRunner。前两者首步处理 P 个输入；PrefixTinyRunner 跳过已命中前缀，后续三者均每步处理一个输入。完整前缀重算的 Llama 尚不支持。逻辑预留、物理分页和设备存储是不同层次；2.5 已接入 CPU 张量分页，2.7 已在 [serving/](../serving/README.md)接入 HTTP，异步服务单独驱动 LifecycleEngine。没有线程安全或 GPU 调度性能的实测声明。

- `paged_runner.py`：[2.5 课程](../../docs/chapters/part-02-engine/05-paged-caching/README.md)的 Tiny 投影 + CPU 分页 K/V 与 Attention。独占池由一个 engine 管理，准入额度向整块取整。物理布局见 [cache/](../cache/README.md)。

- `prefix_runner.py`：[2.6 课程](../../docs/chapters/part-02-engine/06-prefix-caching/README.md)的固定 Tiny 权重快照、实际完整前缀复用与计算计数。首步绑定块表；engine 和 scheduler 仍按完整请求预留与输入预算收费，共享收益尚未增加准入名额。
