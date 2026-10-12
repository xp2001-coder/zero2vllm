# KV 缓存存储

[2.5 分页缓存课程](../../docs/chapters/part-02-engine/05-paged-caching/README.md)

`paged.py` 使用可选的 CPU PyTorch，复用 `reference/blocks.py` 的分配器，把块表连接到真正的 K/V 张量。每个存储对象代表一层，K/V 分别采用 `[num_blocks, block_size, kv_heads, head_dim]` 布局。

- `create / reserve / release`：独立句柄、可增长预留与归还；释放后旧句柄失效。
- `locate / slot_mapping`：预留范围内的地址映射，不意味着该地址已写入。
- `append / gather`：提交有效长度前完成 K、V 两次写入；读取只暴露已写逻辑前缀。
- `attention`：按逻辑顺序 gather 后的单查询参考计算，可检查 GQA 头映射。
- `snapshot / physical_snapshot`：教学观察；后者返回包含无效/陈旧内容的整池副本。

包入口不导入 PyTorch，基础课程仍可仅用标准库。两个存储实现固定 CPU float32/float64，没有高性能分页内核、异步设备写入或线程安全保证。

## 2.6：完整前缀共享

`prefix.py` 是独立的教学存储，保留 2.5 的独占实现便于对照；布局中的 `kv_heads=1`。见[2.6 正文](../../docs/chapters/part-02-engine/06-prefix-caching/README.md)。

- `acquire`：在提示词最后一个输入之前匹配完整块，固定命中引用，再分配私有尾部。
- `append`：真实写入 K/V 后提交有效长度，发布完整块；部分尾块保持私有。
- `gather / attention`：读取有效逻辑前缀，执行 CPU 单查询参考 Attention。
- `release`：减少引用；完整已索引块在零引用时仍保留，可被后续命中或 LRU 逐出。
- `reset`：没有活跃句柄时清除索引；不把清除索引解释为擦除底层字节。

模型与 dtype 指纹、显式缓存域参与身份；摘要查找另用完整 token 前缀精确核对，支持碰撞实验。元数据扫描和完整前缀记录用于讲解，未做生产级时间/空间优化。不同请求共享已完成且不可变的完整块，未实现部分块的写时复制。
