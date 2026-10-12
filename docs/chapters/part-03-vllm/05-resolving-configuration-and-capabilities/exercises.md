# 3.5 练习与参考答案

[正文](README.md) · [实验手册](lab.md)

以下题目中的上游规则均固定在 v0.8.5；容量输入和模型限制是课程夹具。

## 1. 一个值的四个阶段

终端打印了 `max_num_batched_tokens=3072`。为什么仅凭这一行不能确定模型已经按这个值运行？

<details><summary>参考答案</summary>

需要知道打印发生在输入、解析、配置构建还是运行时。后续可能补值、改写、校验失败；模型或后端可能初始化失败。应追踪最后赋值和实际运行记录，而不是把输入回显当作执行证据。

</details>

## 2. 普通单值参数的输入覆盖

字段默认未指定，配置文件写 4096，CLI 写 3072。输入合并后是多少？对应记录至少应该保存什么？

<details><summary>参考答案</summary>

输入合并后 3072。记录字段名、原值、新值、来源和阶段；过程为 `None → 4096（file）→ 3072（cli）`。这个结论适用于本题的普通单值参数，不应扩大到所有 argparse action 或后续引擎规则。

</details>

## 3. CLI 与 SDK 是否走同一条前半段路径

Python 中调用 `LLM(...)`，是否会自动加载 CLI 的 YAML 文件？本课的 JSON 又是什么？

<details><summary>参考答案</summary>

SDK 将 Python 参数交给 EngineArgs，不自动经过 CLI/YAML 路径。课程 JSON 只是教学解析器的输入载体，已经解码成映射后再演示覆盖与默认规则；不是 vLLM YAML 格式的重新定义。

</details>

## 4. 修复一个布尔默认值错误

代码 `if not prefix: prefix = True` 为什么可能有错？在“未指定才默认开启”的约定下如何写？

<details><summary>参考答案</summary>

`None` 和 `False` 都进入该分支，显式关闭会被覆盖。应使用 `if prefix is None`。也要先确认上游字段是否采用这种三态约定，不能无差别地改所有 bool 判断。

</details>

## 5. 两个 false，为何结局不同

固定 V1 路径同时输入 `enable_prefix_caching=False` 和 `enable_chunked_prefill=False`。最终值分别是什么？

<details><summary>参考答案</summary>

prefix cache 保留 false，因为只在 None 时补 true；分块 prefill 被设为 true，因为该历史版本 V1 无条件开启。后者属于策略改写；不能从函数名 `_set_default_args_v1` 推断它只补空值，也不能推广为全部版本都如此。

</details>

## 6. 四种默认组合

空配置、模型限制 4096，依次输入 40/server、40/offline、80/server、80/offline。写出 token 预算与序列上限。

<details><summary>参考答案</summary>

分别是 `2048/256`、`8192/256`、`8192/1024`、`16384/1024`。容量单位 GiB，源码分支门槛是 70 GiB。它看报告的设备总容量，而非“A100”字符串；课程输入并不检测本机硬件。

</details>

## 7. 改容量却没有改变预算

使用默认 demo 文件，把 `--memory-gib 40` 改成 80，预算仍是 4096、序列数仍是 64。这是 bug 吗？

<details><summary>参考答案</summary>

不是。demo 已显式指定两个字段，容量默认规则只补 None。要观察容量分支，应使用空的 defaults.json；要主动覆盖则传对应 CLI 参数。这说明实验输入应先检查是否已经固定了你想研究的变量。

</details>

## 8. 自动选择与显式选择

功能被分类为实验性，环境变量未设置时怎样处理？显式 1 呢？若改为不支持且显式 1 呢？

<details><summary>参考答案</summary>

实验性且未设置时回退 V0；显式 1 时警告并选择 V1；不支持且显式 1 时抛错。课程分类是已经给定的输入，完整 oracle 仍需检查实际模型/功能条件。不能因为显式设置环境变量就跳过能力检查。

</details>

## 9. 长度与预算的三个例子

判断 `(tokens,seqs,model_len,chunked)`：`(512,16,4096,True)`、`(512,16,4096,False)`、`(8,16,4096,True)`。

<details><summary>参考答案</summary>

第一组通过提取出的两个约束；第二组非分块情况下无法容纳模型长度范围内的长提示，失败；第三组 token 预算小于序列上限，失败。通过这两个约束不等于显存足够或完整上游配置合法。

</details>

## 10. 为什么 max_model_len 不等于显存预算

模型长度和单轮 token 预算都为 4096，是否意味着一定能分配 KV cache？还缺哪些信息？

<details><summary>参考答案</summary>

不意味着。还需层数、KV 头数、每头维度、KV dtype、同时保留的历史长度与并发，以及权重、激活、工作区和可用于缓存的内存。调度上限限制计算安排，不能代替真实内存规划与分配结果。

</details>

## 11. block_size=7 被接收了

教学 resolver 保留 7，是否能据此判断 FlashAttention 或 C500 后端支持它？应怎样描述结果？

<details><summary>参考答案</summary>

不能。这里只做部分正整数与配置规则检查，后端可能有额外的对齐、dtype、尺寸或平台限制。应描述“值在教学配置阶段被保留，后端和运行时未验证”，并列出待检查事项。也不能把 CUDA 默认块大小直接抄成 MACA 验证结果。

</details>

## 12. enforce_eager=False 的证据范围

日志写 false，是否足以填报“CUDA Graph 已启用”？应再记录什么？

<details><summary>参考答案</summary>

不足。要记录最终配置、后端/模型约束、实际捕获形状、捕获是否成功以及本轮是否命中。在图桶外或无已捕获整图时仍可能回退。课程 false 只输出具备继续配置资格，不声称已经捕获或执行。

</details>

## 13. 进程里临时改环境变量

引擎对象已经建立，再修改 `os.environ`，为什么不能期待现有 worker 自动切换实现？

<details><summary>参考答案</summary>

已有对象可能早已读取配置、选择类并分配状态，某些读取还可能被缓存或写回环境。环境字典变化不自动重建对象或重新协商进程配置。应在可控启动环境中重新创建所需进程，并记录其真正收到的设置，而不是仅观察调用方当前环境。

</details>

## 14. 写一份可复核的跨平台记录

准备在单卡 A100 与单卡 C500 上跑相同负载。除 CLI 字符串外还应记录哪些内容？一个配置 hash 是否够用？

<details><summary>参考答案</summary>

需要准确代码提交、模型/tokenizer revision、输入与生成策略、驱动/运行时/PyTorch/扩展版本、实际 dtype 与后端、块大小和缓存容量、图模式与命中证据、正确性标准和计时范围。配置 hash 可能只覆盖特定用途的字段，不能替代模型、代码和环境记录。两个平台分别填写实际结果；未运行的条目保留未验证。

</details>
