# 2.1 构建最小 PyTorch 模型

[本篇目录](../README.md) · [课程总目录](../../README.md) · [上一节 1.6](../../part-01-inference/06-sampling-correctness/README.md) · [下一节 2.2](../02-loading-real-model-weights/README.md)

> ✅ 完整课程：正文、代码、实验、练习与答案齐备。验证范围是 CPU 上的 PyTorch 与纯 Python 数学参考；A100/C500 尚未实测。

**本节要解决的问题：怎样把第一篇的数学计算变成一个可检查、可扩展的张量模型？**

学习顺序：[读懂模型](#1-本节交付什么) → [运行实验](lab.md) → [完成练习](exercises.md) → [核对运行记录](results/baseline/summary.json)。建议预留 2–3 小时读课与运行，再用 1–2 小时做练习。

## 1. 本节交付什么

第一篇 `TinyAttentionLM` 是单头 Attention，使用可学习的绝对位置向量，没有 Norm 和 MLP。本节另外实现一个**多层、多头、pre-norm 的稠密 Decoder**，使用 RMSNorm、RoPE 和 SwiGLU。两者架构不同，不要求彼此输出相同。本节建立新的纯 Python 参考模型，再将同一份权重加载到 PyTorch 模型中逐层比较。

模型没有训练，输出只是 token ID；没有真实 Tokenizer，也不会生成有意义的自然语言。权重加载在 2.2，请求管理在 2.3，批处理调度在 2.4，分页缓存在 2.5。当前每一步重新计算整个前缀，方便建立正确性基线。

| 交付物 | 入口 | 作用 |
| --- | --- | --- |
| 纯 Python 参考 | [reference/decoder.py](../../../../zero2vllm/reference/decoder.py) | 标量循环、同一份权重、逐层中间值 |
| PyTorch 模型 | [modeling/torch_decoder.py](../../../../zero2vllm/modeling/torch_decoder.py) | nn.Module、多头 Attention、生成与误差报告 |
| 最小调用示例 | [run.py](examples/run.py) | 构造模型、打印形状、生成 ID |
| 实验说明 | [lab.md](lab.md) | 环境、命令、参数、结果、排错 |
| 练习与答案 | [exercises.md](exercises.md) | 手算、代码修改、因果性与边界 |
| 自动验证 | [参考测试](../../../../tests/reference/test_decoder.py)、[PyTorch 测试](../../../../tests/modeling/test_decoder.py) | 数值、因果性、批处理、输入与上下文边界 |

前置：能阅读 Python 函数，理解矩阵乘法；完成 1.1、1.2、1.5、1.6。没有 GPU 也能完成本节。

## 2. 先确定模型契约

默认配置在 `DecoderConfig` 中。先固定很小的形状，再讨论性能优化。

| 符号 | 默认值 | 含义 |
| --- | --- | --- |
| B | 1 | 本次输入的请求数，模型支持同长度 batch |
| T | 4 | 默认 prompt `[1,2,3,4]` 的长度 |
| V | 16 | 词表大小，合法 ID 为 0–15 |
| D | 8 | 隐藏维度 |
| H | 2 | Q/K/V 头数相同，即 MHA |
| Dh | 4 | 每个头的维度 D/H，必须是偶数 |
| F | 16 | MLP 中间维度 |
| L | 2 | Decoder 层数 |
| max_context | 32 | 一次前向允许的最大输入长度 |
| eps / rope_base | 1e-6 / 10000 | RMSNorm 稳定项 / RoPE 频率基数 |

`forward(tokens)` 接收 **CPU、torch.long、[B,T]**，返回 **[B,T,V]** logits。输入不允许空 batch、空序列、负 ID 或越界 ID。本节不支持 padding 和不同有效长度混在同一个张量中；补零会被当成真实的 token 0。

`generate(tokens, max_new_tokens)` 接收单个请求的 Python ID 列表，返回新增 ID 列表，不修改原列表。默认贪心解码；没有 EOS、stop string 和采样配置。第一篇这些概念继续有效，后面再接入请求层。

## 3. 一个 Decoder 的完整前向

```text
token IDs [B,T]
  → Embedding [B,T,D]
  → 重复 L 次：
       RMSNorm → Q/K/V → 分头 → RoPE(Q,K)
       → causal Attention → 合头 → O 投影 → 加回输入
       → RMSNorm → SwiGLU MLP → 加回 Attention 后的状态
  → final RMSNorm
  → LM Head [B,T,V]
```

采用 pre-norm：先归一化，再执行子层，最后加残差。

```text
u = x + Attention(RMSNorm1(x))
y = u + MLP(RMSNorm2(u))
logits = RMSNorm_final(y_last_layer) @ W_head
```

读代码时先看 `TorchDecoder.forward` 的主链路，再看 `DecoderBlock.forward`；最后回到参考模型，确认两条实现的数学操作一致。

### 3.1 Embedding：ID 查表

权重形状为 `[V,D]`。ID 2 的向量就是第 2 行，不是把数字 2 与权重相乘。本节没有把绝对位置向量加到 embedding 上，位置由 Attention 内的 RoPE 引入。

### 3.2 RMSNorm：稳定尺度，不减均值

对最后一个维度独立计算：

```text
rms(x) = sqrt(mean(x²) + eps)
y_i = x_i / rms(x) * gamma_i
```

`mean(..., dim=-1, keepdim=True)` 的结果是 `[B,T,1]`，可以广播回 `[B,T,D]`。gamma 是可学习参数 `[D]`，用 `nn.Parameter` 注册。本节显式设置 eps，避免随 dtype 的默认值变化。RMSNorm 没有 LayerNorm 的减均值步骤。[PyTorch RMSNorm 文档](https://docs.pytorch.org/docs/2.7/generated/torch.nn.RMSNorm.html)

手算：`x=[3,4]`、`eps=3.5`、`gamma=[2,1]`，分母为 `sqrt(12.5+3.5)=4`，输出 `[1.5,1]`。这个较大的 eps 只用于手算测试，实际模型使用 `1e-6`。

本节始终用 float64 或 float32 计算。将来引入 BF16/FP16 时，必须重新定义归一化和累加精度，不能直接认定当前实现适合低精度部署。

### 3.3 Linear：特别注意权重方向

参考模型统一使用 `y = x @ W`，权重为 `[in_features,out_features]`。PyTorch `nn.Linear` 保存 `[out_features,in_features]`，前向等价于 `x @ weight.T`。因此加载时要转置。

以 LM Head 为例：参考权重 `[8,16]`，PyTorch 参数 `[16,8]`。Embedding 保持 `[16,8]`，Norm 的一维参数也不转置。代码遍历 `named_parameters()` 逐项复制，避免两边各自初始化而误以为“种子相同就一定相同”。

默认共有 1,576 个参数：

```text
Embedding: V*D = 128
每层: 2*D + 4*D*D + 3*D*F = 656
两层: 1312
final Norm: D = 8
LM Head: D*V = 128
总计: 128 + 1312 + 8 + 128 = 1576
```

所有 Linear 都无 bias；Embedding 与 LM Head 不共享权重。`named_parameters()`、`state_dict()` 可以检查参数是否正确注册。

### 3.4 分头与 RoPE

Q/K/V 投影后为 `[B,T,D]`。先变为 `[B,T,H,Dh]`，交换时间与头维度，得到 `[B,H,T,Dh]`。

每个头中相邻两个分量组成一对。位置为 p、成对维度起点为 2i 时：

```text
angle = p / base^(2i / Dh)
q_even' = q_even*cos(angle) - q_odd*sin(angle)
q_odd'  = q_even*sin(angle) + q_odd*cos(angle)
```

Q 和 K 使用相同规则，V 不旋转。位置 0 的旋转是恒等变换；旋转保持每对向量的平方和。代码在每个头内部旋转，不能把两个头边界上的分量拼成一对。

这是**相邻配对**的教学约定。其他模型可能使用前后半区配对、不同频率缩放或权重排列；它们需要与 checkpoint 的训练约定匹配，不能只凭“都是 RoPE”就互换。

### 3.5 因果 Attention

```text
scores = Q @ K.transpose(-2,-1) / sqrt(Dh)   # [B,H,T,T]
scores[i,j] = -inf, if j > i
probs = softmax(scores, dim=-1)
context = probs @ V                         # [B,H,T,Dh]
```

缩放用 `sqrt(Dh)`。遮罩只挡住未来，保留对角线。`[T,T]` 遮罩广播给所有请求和头；softmax 最后一个维度表示“当前 query 在哪些 key 之间分配概率”。

PyTorch 路径显式构造矩阵和遮罩；参考路径只遍历 `j <= i`，不接触未来位置。这样，因果性检查不是两边重复使用同一份遮罩代码。

Attention 后转回 `[B,T,H,Dh]`，通过 `contiguous().reshape(...)` 合并成 `[B,T,D]`，再执行 O 投影。`transpose` 改变 stride，直接把非连续张量交给 `view` 可能失败。

本节显式分配 `T×T` Attention 矩阵，目的是观察计算。FlashAttention 的分块与 IO 优化放在第七篇，不能把当前实现当作高性能算子。

### 3.6 SwiGLU 与残差

```text
gate = x @ W_gate          # [B,T,F]
up = x @ W_up              # [B,T,F]
gated = SiLU(gate) * up    # 逐元素乘法
down = gated @ W_down      # [B,T,D]
SiLU(z) = z * sigmoid(z)
```

gate 和 up 是独立投影。中间维度扩张到 F，再由 down 回到 D，才能与残差相加。参考实现通过标量公式计算 SiLU，张量实现调用 `F.silu`。

## 4. 从前向到生成

```python
sequence = list(prompt)
for _ in range(max_new_tokens):
    logits = model(torch.tensor([sequence], dtype=torch.long))
    next_id = int(logits[0, -1].argmax())
    sequence.append(next_id)
```

`logits[:, -1, :]` 才是最后一个输入位置对下一个 token 的预测。其他位置也有预测，但不能把整段 logits 展平后取全局最大值。

prompt 长度 P=4、生成 O=4 时，前向长度依次为 4、5、6、7。最后预测出的 token 没有再次送进模型，因此容量检查是 `P + max(O-1,0) <= max_context`。输入已占满 32 个位置时仍能预测 1 个新 token；再预测第 2 个时需要输入长度 33，必须拒绝。输出总长度和已参与计算的上下文长度要区分。

代码调用 `model.eval()`，并在生成/实验入口使用 `torch.inference_mode()`。`eval()` 切换模块的训练行为，本身不关闭梯度记录。本模型没有 Dropout，仍保留这两个明确的推理约定。

## 5. 如何知道模型写对了

仅看“没有报错”或“生成 ID 一样”都不够。我们同时做三层检查：

1. **独立小例子**：RMSNorm 手算、RoPE 保范数、权重转置。
2. **逐层比较**：固定权重、输入、位置，比较 embedding、Q/K/V、旋转、Attention 合头结果、残差、MLP、final Norm 与 logits。默认两层共 31 个比较点。
3. **行为约束**：改变未来 token 不改变历史 logits；同长度 batch 与逐请求运行一致；检查空输入、非法 ID、容量边界与零输出。

每个元素必须满足：

```text
abs(torch_value - reference_value) <= atol + rtol * abs(reference_value)
```

| 路径 | atol | rtol | 比较方式 |
| --- | --- | --- | --- |
| float64 | 1e-12 | 1e-10 | PyTorch 双精度对 Python float |
| float32 | 1e-5 | 1e-4 | 单精度结果转双精度后对照原始参考 |

float32 比较包含参数转换的舍入误差。报告还要求值有限、贪心 ID 一致，并记录配置、权重哈希、环境与种子。这不是所有模型、低精度或硬件的通用阈值。

定位差异时找第一个失败的中间层：Embedding 就不同，查 ID 和权重；Q 开始不同，查转置；RoPE 开始不同，查位置与配对；context 开始不同，查遮罩、softmax 轴和缩放。不要先放宽最终 logits 的阈值。

## 6. 完成标准与下一节

- [ ] 能解释 1,576 个参数的组成。
- [ ] 能说明分头与合头的形状变化。
- [ ] 跑通 float64/float32 两条路径，读懂 31 个中间比较结果。
- [ ] 完成因果性、转置错误与上下文边界练习。
- [ ] 能区分随机权重对齐、真实模型质量验证和 GPU 性能测量。

下一节 2.2 加入真实模型配置、Tokenizer 与 safetensors 权重映射。先固定 revision 和张量布局，再谈“成功加载”。

## 参考资料

- [PyTorch 历史版本安装页](https://pytorch.org/get-started/previous-versions/)：本节固定 2.7.1 CPU wheel，供复现，不表示最新版本。
- [PyTorch RMSNorm](https://docs.pytorch.org/docs/2.7/generated/torch.nn.RMSNorm.html)：公式与参数语义。
- [PyTorch Linear 源码 v2.7.1](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/nn/modules/linear.py)：参数布局与前向约定。
- [第一篇模型结构](../../part-01-inference/05-model-structures/README.md)：位置编码与结构边界。

本节代码为独立教学实现，不构成对任何具体模型 checkpoint 的兼容承诺。
