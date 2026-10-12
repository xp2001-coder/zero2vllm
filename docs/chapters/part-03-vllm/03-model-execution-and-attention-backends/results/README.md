# 3.3 实际运行记录

[正文](../README.md) · [实验命令](../lab.md) · [仓库质量记录](../../../../../reports/quality/lesson-3.3-delivery/README.md)

2026-09-20，Windows / Python 3.11.9 / PyTorch 2.7.1+cpu。本目录来自实际命令运行，未手工填造 PASS；可复跑命令和退出码见 [commands.json](commands.json)。两个数值模式均通过 12 个案例。

| 记录 | 结果与边界 |
| --- | --- |
| [metadata.json](metadata.json) | 使用 `python -S` 跑通 2 个打包案例；没有加载第三方依赖 |
| [tensors-float64.json](tensors-float64.json) | 全部 12 案例；分页与 dense 最大绝对误差 2.22e-16，容差 1e-11 |
| [tensors-float32.json](tensors-float32.json) | 全部 12 案例；分页与 dense 最大绝对误差 1.19e-7，容差 2e-5 |
| [source-inspection.json](source-inspection.json) | 使用 `python -S` 离线核验固定官方源码：15 文件、27 函数、2 类结构 |
| [inputs-ba.txt](inputs-ba.txt)、[inputs-ab.txt](inputs-ab.txt) | 两种顺序的标准库短示例，保留完整输入坐标 |
| [tensor-ba.txt](tensor-ba.txt)、[tensor-ab.txt](tensor-ab.txt) | 两种执行顺序下的真实 CPU 数值结果；按请求归属保持一致 |
| [padding.txt](padding.txt) | 真实输入 4、投影缓冲 8；只有 4 个写入地址和有效输出行 |

## 怎样解读数值

默认 float64 下，B/A 的候选分别为 24/16，保留 `{B:[24], A:[]}`；float32 下是 15/4，保留 `{B:[15], A:[]}`。这不是浮点舍入把同一模型的输出改变了：两份实验在不同 dtype 下分别生成随机参数，同一 seed 不保证参数逐元素相同。正确性判断是在每一份模型内部，比较分页与独立 dense 路径。

改变 A 后面的已调度 token，不改变 A 前两个 Query 的结果，而被修改位置的结果确实改变；改变 B 的全部历史，也不改变 A 的结果。填充缓冲只增加投影行数，不增加写入位置。未使用 KV 槽保持 NaN，用于暴露越界读取。

## 源码与复现边界

官方源码固定到 v0.8.5 提交 `ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d`。本轮补齐模型加载、Linear、Attention 等官方文件，再复用已有下载目录离线核验。本表的 `downloaded=[]` 表示记录的是后续静态检查，不表示下载从未发生；完整身份依据在清单与文件摘要中。

所有记录具有相同的 `course_source.source_sha256`。记录生成时工作区尚未提交，因此 `git_commit` 是当时基点，内容身份以源码摘要为准；未来课程变更后摘要自然会变化，历史记录不会自动代表新版本通过。

没有执行上游 vLLM、完整预训练 Llama、FlashAttention GPU 内核或 CUDA Graph；没有 A100/C500 测量与加速比。只验证本节单轮数值夹具和固定源码的静态位置。
