# 9.1 练习与参考答案

[正文](README.md) · [实验](lab.md)

共 14 组。建议先隐藏答案，写出中间过程；第 1 组含 20 个分类条目，不另计为 20 组。

## 1. 术语分类

打开 [terms.json](data/terms.json)，只看 `term`，按其上下文分类。允许先使用“方法、数值、存储、执行、系统、证据”六个大类，再细分。

<details>
<summary>参考答案：为什么上下文不能省略</summary>

| # | 术语 | 分类与理由 |
| --- | --- | --- |
| 1 | GPTQ 论文中的误差补偿方法 | algorithm：描述怎样依据输入统计求量化参数。 |
| 2 | W4A16 G128 | scheme：声明 W/A 位宽与分组大小；没有说明分组轴或执行算子。 |
| 3 | 有符号 INT4 的整数值域 | representation：规定可表示的整数；本课使用其中 [-7,7]。 |
| 4 | 本例 scale=0.5、zero_point=0 | parameters：把具体整数与还原实数连接起来。 |
| 5 | 本例 ties-to-even 舍入规则 | numerical_contract：定义恰在两个整数中间时如何取值。 |
| 6 | AutoGPTQ-compatible 导出格式 | checkpoint_format：描述文件中的张量与元数据如何被解释；还需版本。 |
| 7 | 两个 INT4 合成一字节且第一个放低位 | packing_layout：规定整数值怎样占据字节中的位置。 |
| 8 | 为某个 Marlin kernel 准备的权重排列 | compute_layout：规定执行时按哪个位置取权重。 |
| 9 | 某个实际执行的 Marlin GEMM kernel | kernel：执行加载、解码、计算与输出的程序。 |
| 10 | vLLM 请求调度、加载和执行系统 | runtime：组织模型运行与服务，而非仅定义一条公式。 |
| 11 | NVIDIA A100 GPU | hardware：执行程序的物理设备；能力与软件路径还要核对。 |
| 12 | 沐曦 C500 GPU | hardware：另一硬件平台，需要独立 MACA 软件栈记录。 |
| 13 | safetensors 文件容器 | container：承载张量与元数据，不独自定义量化语义。 |
| 14 | checkpoint 加载后的 repack 操作 | layout_conversion：转换权重排列；正确转换要保持逻辑值及其元数据对应。 |
| 15 | 本例用 Python int 列表存放 q | storage_type：宿主实际存储类型；不意味着每个值只占 4 bit。 |
| 16 | C4 作为某次实验的校准输入 | calibration_data：它是求统计量或优化参数的数据，不是算法名称。 |
| 17 | FP32 作为某次 GEMM 的累加类型 | compute_type：说明部分和如何计算，与权重文件类型分开。 |
| 18 | AutoRound Python 工具包 | producer_framework：生产量化产物的工具；其中的算法与导出格式要另记。 |
| 19 | MACA 编译器与设备软件栈 | device_software_stack：支撑设备程序编译和运行，不等同于上层推理引擎。 |
| 20 | 固定负载下测得 TTFT=40ms | measurement：带口径和证据的测量；本题数字仅为分类示例。 |

“GPTQ”单独出现时可能指论文算法，也可能指 checkpoint 兼容格式，不能不看上下文就分到唯一一类。TTFT 的示例数值不是本课测量结果。

</details>

## 2. 一个模型名字能填写几项

拿到 `Qwen3-8B-W4A16-G128`，能否填写算法、weight dtype、activation dtype、group axis、kernel 和 accumulator？

<details>
<summary>参考答案</summary>

可以暂记模型标签、W/A 位宽和组大小，但这些首先只是名称声明。`W4` 不单独确定 INT4/FP4、signed/unsigned；`A16` 不单独区分 FP16/BF16。算法、分组轴、文件格式和执行类型不能猜测。目标记录可另外声明计划使用 vLLM，仍然没有实际执行证据。

</details>

## 3. 非零 zero-point 手算

固定 `s=0.5,z=3,q∈[0,15]`，计算 `x=[-1.5,0,0.5,2.5,6]` 的 q 和 DQ。为什么编码全为非负，却能还原负数？

<details>
<summary>参考答案</summary>

除以 scale 得 `[-3,0,1,5,12]`；取整后加 3，得到 `[0,3,4,8,15]`。DQ 先减 3 再乘 0.5，恢复本例全部原值。实数符号由 `q−z` 决定，不能由 q 是否为非负来判断。

</details>

## 4. 漏减 zero-point 和移动 round 括号

沿用上题参数：若 DQ 写成 `s*q`，每个值偏多少？再计算 `x=0.25` 时，`round(x/s)+z` 与 `round(x/s+z)` 是否相同。

<details>
<summary>参考答案</summary>

漏减导致每个值偏大 `s*z=1.5`，这是实现错误。正确舍入顺序得到 `round(0.5)+3=3`，改动顺序得到 `round(3.5)=4`。中点取偶数时，奇数 zero-point 会影响平移后的中点选择，所以必须固定顺序。

</details>

## 5. 从一个量化网格转换到另一个

将第 3 题的表示重量化为 `s2=1,z2=2,q2∈[0,15]`。逐步计算 `q2` 和 `DQ2`。两边都是 4 bit，这还是重量化吗？

<details>
<summary>参考答案</summary>

先 DQ1，得到 `[-1.5,0,0.5,2.5,6]`；除以 1、取偶数，再加 2，得到 `[0,2,2,4,8]`。DQ2 结果为 `[-2,0,0,2,6]`。这是重量化，因为源和目标数值网格不同，与是否改变位宽无关。

实数零仍精确表示，只是编码从 3 变成 2。0.5 和 2.5 在目标网格上产生新舍入损失。

</details>

## 6. 目标饱和与重复舍入

第一问：上题目标范围改成 `[0,3]` 后，原值 6 最终还原成多少？第二问：原值 0.625 经源参数再 RQ，是否等于直接用目标参数 Q？

<details>
<summary>参考答案</summary>

原值 6 对应目标未饱和编码 8；饱和为 3，再 DQ 得 `1×(3−2)=1`，误差 5。

0.625 经 Q1：`round(1.25)+3=4`，DQ1 为 0.5；再 Q2 得 2，DQ2 为 0。直接 Q2：`round(0.625)+2=3`，DQ2 为 1。两条路径不等价，第一次 Q 丢失了 0.625 与 0.5 的差别。

</details>

## 7. RQ、DQ 与 repack 分别改什么

某程序把 q 从行优先存放改为分块排列，同时按索引搬运 scales。另一个程序把 `(s,z)` 从 `(0.5,3)` 改为 `(1,2)` 并重算 q。怎样分别验证？

<details>
<summary>参考答案</summary>

前者是布局变换：将目标布局按定义解回逻辑坐标后，q 和参数对应关系应该保持，DQ 值不变。后者是 RQ：按 `Q2(DQ1(q1))` 计算参考编码，并检查目标饱和与新误差。仅改元数据而不重算 q，会改变解释出的值，却未完成正确的 RQ。DQ 单独做的事是用原参数解释已有编码。

</details>

## 8. 完成六个权重的 Linear 手算

按 [linear.json](data/linear.json)，算 scale、q、还原权重和两组输出。解释为什么使用一个非方阵。

<details>
<summary>参考答案</summary>

`amax=3.5,s=0.5,z=0`，q 为 `[[-7,-2],[0,2],[5,7]]`；还原为 `[[-3.5,-1],[0,1],[2.5,3.5]]`。

`X=[1,2,-1]`：原始输出 `[-5.5,-1.75]`，量化输出 `[-6,-2.5]`。权重 shape `[3,2]` 明确区分 K 与 N；若是方阵，转置错误有时仍能通过 shape 检查。

</details>

## 9. 半格误差界能用到哪里

计算第 8 题权重的最大绝对误差和 MSE，再解释输出误差为什么更大。`s=0.5,z=0` 时输入 4，又发生了什么？

<details>
<summary>参考答案</summary>

六项中三项误差绝对值为 0.25，其余为零，因此最大值 0.25，MSE 为 `3×0.25²/6=0.03125`。输出误差由输入加权求和；本例最大为 0.75。

输入 4 超出实数范围 `[-3.5,3.5]`，被饱和为 3.5，误差 0.5。无裁剪的 `s/2=0.25` 界不适用。它也不是模型质量或输出误差的统一界。

</details>

## 10. 全零与实际 dtype

全零矩阵的 `amax/7` 是多少，为什么不能直接使用？程序写着“FP32 输入”，其乘加和 scale 是否一定也是 FP32？

<details>
<summary>参考答案</summary>

`amax/7=0` 会导致除零，本课约定 scale=1，q 和 DQ 仍为零。其他实现可以有不同的正 scale 约定，需要记录。

本课先用二进制单精度转换输入，随后使用 Python float64 做 scale、DQ 和乘加。比如 0.1 进入计算时已变为 `0.10000000149011612`。`FP32 input` 不等于全流程 FP32；q 的 Python int 列表也不等于 INT4 物理存储。

</details>

## 11. 位宽、payload 和文件大小

六个 FP32 权重的理想 payload 多大？假设六个 4-bit 编码紧密打包，再加一个 64-bit scale，共多大？可以据此报告本节实际压缩率吗？

<details>
<summary>参考答案</summary>

FP32 权重 payload 为 `6×32/8=24 byte`。假设打包的 q 为 `6×4/8=3 byte`，再加 8 byte scale，总计 11 byte。这里假定 z=0 由约定给出，且不计 shape、格式头和 padding；只是估算。

本节没有生成这个 11-byte buffer，而是使用 Python 对象和 JSON，所以不能把该比值当作实测文件压缩率，更不能当作 GPU 加速比。

</details>

## 12. 写一条可信的 Passport 记录

已知计划 group size 为 128，但还没有模型文件。能否写 `observed=128,status=verified`？如果源码里有 Marlin 类，能否把实际 kernel 写为 Marlin？

<details>
<summary>参考答案</summary>

只写 `expected=128,status=declared`。观察值需要检查产物并给出证据。类存在只证明有这个源码对象；要知道实际 kernel，还需核对后端选择和运行结果。

尝试把没有 evidence 的字段写为 verified，程序应抛出 ValueError。补一个任意字符串让结构通过，也不会让证据自动真实；仍需人工核查。

</details>

## 13. 追一次固定源码的委托关系

打开 [源码定位表](source-map.md)，回答 `GPTQMarlinLinearMethod` 在哪个方法选择 kernel，`apply()` 调用了谁。可以认为这就是 Qwen3-8B 的完整运行路径吗？

<details>
<summary>参考答案</summary>

固定 v0.8.5 在 `create_weights()` 中调用 `choose_mp_linear_kernel()`，构建 `self.kernel`；`apply()` 调用其 `apply_weights()`。加载后处理也委托给这个对象。

这里只阅读接口例子，没有加载 Qwen，不能断言实际模型都走示例 Linear。该历史切片也不是当前版本的兼容性承诺。

</details>

## 14. 设计下一步验证

你已让本节 17 个案例全部 PASS。现在有人问“A100 上 W4A16 比 BF16 快多少，C500 能否得到同样加速”，怎样回答并安排下一步？

<details>
<summary>参考答案</summary>

目前没有这个性能结论。先分别固定两平台的软件栈、模型 revision、格式和执行后端；确认 Q/DQ 与加载正确，检查模型质量，再测固定 M/N/K 的 kernel 和层，最后定义输入/输出长度、并发、预热与统计口径测服务。

要把 weight-only 的解码开销、加载时 repack 成本和稳态前向分开。不能把 Python 参考的运行时间当作 GPU kernel 延迟，也不能把一个平台的日志复制成另一平台的结论。

</details>
