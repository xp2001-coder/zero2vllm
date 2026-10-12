# 3.3 练习与参考答案

[正文](README.md) · [实验](lab.md) · [源码定位](source-map.md)

默认使用正文的固定历史版本和 TP=1 浮点主路径。先手算、画对应表或定位代码，再展开答案。

## 1. 模型能实例化，就算加载完成了吗

注册表找到模型类，所有参数 shape 都能打印出来。还缺哪些证据？为什么 eval() 不能代替它们？

<details><summary>参考答案</summary>

还需确认 checkpoint 参数已正确写入，参数名、布局、融合/分片和后处理满足约定，并用可信参考核对实际数值。eval() 切换模块的训练/推理行为，不验证参数来源或数值。

</details>

## 2. 三种权重布局

输入 X 为 [5,8]，数学权重 W 为 [8,12]。普通 F.linear 的 weight 应是什么形状？若 checkpoint 已是这种形状，是否还要转置？

<details><summary>参考答案</summary>

普通 weight 为 [12,8]，因为 F.linear 计算 X @ weight.T。checkpoint 已符合这个存储契约时，不应再无条件转置。数学、checkpoint 和运行内核布局可能不同，需要逐项解释转换。

</details>

## 3. 方阵为什么会骗过 shape 检查

X=[[1,2]]，普通 Linear 存储权重为 [[1,3],[2,4]]。正确输出是多少？错误地转置后是多少？

<details><summary>参考答案</summary>

正确为 [[7,10]]；转置权重后变成 [[5,11]]。两份权重都是 [2,2]，shape 检查发现不了方向错误；一个手算数值测试就能区分。

</details>

## 4. GQA 的 QKV 切分

H=8、Hq=4、Hkv=2、D=2，T=4。Wq/Wk/Wv、融合权重、融合输出与拆开后的 Q/K/V 各是什么形状？

<details><summary>参考答案</summary>

普通存储 Wq=[8,8]，Wk/Wv=[4,8]；融合权重=[16,8]，输出=[4,16]。按最后维 [8,4,4] 拆开，reshape 后 Q=[4,4,2]，K/V 各=[4,2,2]。不能把 16 平均分成三份。

</details>

## 5. 融合参数名

文件里有 q_proj、k_proj、v_proj，运行模块只有 qkv_proj。这是否一定是漏加载？应该检查什么？

<details><summary>参考答案</summary>

不一定。应检查模型的参数名映射、每项对应的 shard_id、目标区域、形状和实际写入。普通 TP=1 夹具可用沿输出行拼接解释，但量化或并行格式还要检查专门 loader 与后处理。

</details>

## 6. 重新手算默认输入

正文 A/B 输入不变，执行顺序为 [B,A]。写出 input_ids、positions、slot_mapping、query_start_loc、seq_lens、logits_indices。

<details><summary>参考答案</summary>

依次为 [23,12,13,14]、[3,2,3,4]、[23,34,35,12]、[0,1,4]、[4,5]、[0,3]。其中 seq_lens 包括旧 KV；positions 是请求内位置，不是扁平行号。

</details>

## 7. 换顺序，不换请求

执行顺序改为 [A,B]，哪些数组变了？默认候选 [24,16] 是按 B/A 排列，如果错误地按 A/B 登记顺序分发会怎样？

<details><summary>参考答案</summary>

输入变成 [12,13,14,23]，位置 [2,3,4,3]，slot [34,35,12,23]，边界 [0,3,4]，seq_lens [5,4]，logits 行 [2,3]。候选必须按对应执行排列解释。若直接按旧登记顺序 zip，会把 B 的候选还给 A；应使用本轮 req_id_to_index。

</details>

## 8. 调度进度为何要保留旧值

A 原来 C=2，本轮 q=3。调度器返回后自己已经把 C 改成 5。Runner 应用哪个起点？

<details><summary>参考答案</summary>

应使用本轮调度消息携带的旧进度 2，读取位置 2、3、4。实时 C=5 是调度器登记的推进结果；误用它会从 5 开始，跳过本轮该算的位置，甚至越过已知序列。

</details>

## 9. 写地址与读规则

为什么 slot_mapping=[23,34,35,12] 还不足以告诉 Attention 应该读哪些历史？

<details><summary>参考答案</summary>

它表达本轮新 K/V 的写位置，没有独自表达每个请求的完整历史、分段边界和每条 Query 的因果可见性。读取还需要本请求块表、有效 KV 长度、Query 边界及对应因果规则。

</details>

## 10. 带历史前缀的矩形掩码

一条请求 C=3，本轮 q=2，总 KV 长度为 5。两条 Query 分别可看哪些 Key 位置？为什么 2×5 的左上三角形不对？

<details><summary>参考答案</summary>

Query 的绝对位置是 3、4。第一条可看 0—3，第二条可看 0—4。左上三角形会只给它们 1、2 个 Key，漏掉了此前 3 个位置的历史。因果边界必须带上 C 的偏移。

</details>

## 11. KV head 共享与前缀共享

Hq=8、Hkv=2 时，每个 Query head 使用哪个 KV head？它与 A/B 共用一块前缀有什么区别？

<details><summary>参考答案</summary>

每组有 4 个 Query heads，映射为 [0,0,0,0,1,1,1,1]。这是每条请求内部的 GQA 头映射；跨请求前缀共享还涉及内容身份与物理块引用。本节夹具禁止跨请求块表重叠，不把它当作已验证的前缀缓存。

</details>

## 12. 补齐到八行

T=4，缓冲长度补到 8。input_ids、positions、slot_mapping 和输出分别应怎样处理？可否给补齐 token 填词表外 ID？

<details><summary>参考答案</summary>

输入和位置缓冲可以有 8 行，但实际工作仍是 4；本节只有 4 个 KV 写地址，Query 边界最后仍是 4，真实结果也只来自这 4 行。额外投影行不能写入或成为请求输出。若经过 Embedding，padding ID 仍须合法，不能依赖后续丢弃来允许越界索引。

</details>

## 13. 两个候选，只返回一个

本轮 A 的 C+q=5，已知长度 N=6；B 的 C+q=N=4。两条都得到候选时，应如何形成输出？为什么测试不能要求绝不计算 A 的候选？

<details><summary>参考答案</summary>

A 仍是部分 Prefill，其候选列表清空；B 保留候选。固定快照为简化批处理会先产生候选再过滤，所以“绝不计算 A 的候选”不是这个版本的实现约束。用户可见边界应是 A 没有有效新输出；随机生成器状态还需要按照对应实现处理。本节只验证 greedy 路径。

</details>

## 14. CPU 玩具尺寸能直接用 FlashAttention 吗

本节 B=4、D=2，CPU 案例通过。能否据此启动所选 vLLM FlashAttention 后端，或声称 C500 兼容？

<details><summary>参考答案</summary>

不能。该固定后端要求块长是 16 的倍数，支持的 head size 有独立列表；玩具尺寸不满足它。CPU 对照也没有验证扩展构建、设备、MACA 适配或 GPU 内核。平台实验必须固定对应软件栈，确认实际后端和运行结果。

</details>

## 15. 只看 token 相同够不够

优化前后 greedy token 一样，是否说明输出正确？默认缓存 [2,10,4,2,2]、float64 占多少字节？这能代表整个模型显存吗？

<details><summary>参考答案</summary>

token 一样只是 argmax 没变，不保证 logits 或中间结果正确，应比较数值误差和必要的不变量。缓存共有 2×10×4×2×2=320 个元素，float64 共 2560 字节。它只是单层玩具 KV 存储，不包含完整模型权重、多层 KV、临时张量和运行时资源，不能代表整个模型显存。

</details>

## 16. 设计一个能抓住索引错误的测试

任选一类错误：请求串扰、未来泄漏、padding 写错、方阵权重转置、执行排列错配。给出有区分力的输入和断言。

<details><summary>参考答案</summary>

可分别使用：改变 B 全部 token 而断言 A logits 不变；只改变 A 最后一个已安排 token 而断言之前 Query 不变，同时断言被改行确实变化；补齐到 8 行并改变占位内容而断言实际结果和写槽集合不变；用非对称方阵加手算期望；交换 A/B 执行排列并按请求 ID 比较全部 logits。不能只断言两个函数返回相同 shape，也不能只检查源码里出现某个函数名。

</details>
