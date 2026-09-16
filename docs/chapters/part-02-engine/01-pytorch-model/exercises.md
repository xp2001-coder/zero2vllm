# 2.1 练习与参考答案

[正文](README.md) · [实验](lab.md)

先独立解答，再展开答案。第 1–4 题为基础，第 5–6 题为工程进阶。

## 1. 形状与参数预算

默认模型的 Q、Attention 分数、gate、logits 是什么形状？为什么 MLP 要投影回 D？参数量是多少？从两层改为三层多出多少参数？

<details>
<summary>参考答案</summary>

Q 为 `[1,4,8]`，分头后为 `[1,2,4,4]`；Attention 分数为 `[1,2,4,4]`，最后两个 4 分别是 query/key 位置。gate 与 logits 都为 `[1,4,16]`，但 gate 的 16 是中间维度，logits 的 16 是词表大小。

残差为 `[1,4,8]`，MLP 需 down 投影回 8。默认参数量 `128+2*656+8+128=1576`。增加一层多 656，总计 2232。

```bash
python -m zero2vllm torch-demo --layers 3 --output .local/runs/part-02-engine/01-pytorch-model/three-layers.json
```

</details>

## 2. RMSNorm 与 RoPE 手算

计算 x=[3,4]、eps=3.5、gamma=[2,1] 的 RMSNorm。将 [1,0] 按角度 π/2 旋转，平方和是否改变？位置 0 呢？

<details>
<summary>参考答案</summary>

归一化分母为 4，输出 [1.5,1]。旋转输出 [0,1]（浮点可能有接近零的残差），平方和保持为 1。位置 0 的角度都是 0，旋转是恒等变换。RMSNorm 的 gamma 可以改变尺度，RoPE 本身保持每对向量长度。

</details>

## 3. “种子一样”为什么还不够

两边分别用 random.uniform 和 torch.randn、都设 seed=7，能否算同一个模型？为什么 Q 的转置错误比 LM Head 更难发现？

<details>
<summary>参考答案</summary>

不能。随机算法、分布、生成顺序与 dtype 都可能不同。本节只在参考侧用局部 random.Random 生成一次，再逐项复制；权重哈希用于核对。

Q 为 `[8,8]`，转置不改变形状，但会改变数值。Head 是非方阵，方向错通常立即产生形状冲突。必须比较中间结果，不能只检查形状。

</details>

## 4. 生成位置与边界

prompt 长度 4，预测 4 个 token，前向长度依次是多少？为什么满上下文仍可预测一个 token？max_new_tokens=0 呢？

<details>
<summary>参考答案</summary>

前向长度为 4、5、6、7。最后新增的 token 没有再次送进模型。容量限制作用于前向输入，因此满上下文仍能预测一个 token，但不能消费它继续预测。

零输出返回空列表，不执行生成前向，但仍检查 prompt 合法性。实验入口会另外对 prompt 做一次前向，所以零输出报告仍有 31 个比较点。

</details>

## 5. 因果性和 batch 隔离

设计测试证明未来不影响过去、batch 中请求互不影响。同一个 batch 中有效长度不同，能否补 token 0？

<details>
<summary>参考答案与可运行代码</summary>

在仓库根目录启动装好 torch 的 Python：

```python
import torch
from zero2vllm.reference.decoder import ScalarDecoder
from zero2vllm.modeling.torch_decoder import TorchDecoder

model = TorchDecoder(ScalarDecoder())
with torch.inference_mode():
    a = model(torch.tensor([[1, 2, 3]]))
    b = model(torch.tensor([[1, 2, 15, 9]]))
    batch = model(torch.tensor([[1, 2, 3], [3, 2, 1]]))
torch.testing.assert_close(a[:, :2], b[:, :2], atol=1e-12, rtol=1e-10)
torch.testing.assert_close(batch[0], a[0], atol=1e-12, rtol=1e-10)
```

补 0 不行：0 是合法 token，会参与 Attention 和位置计算。变长 batch 需要有效长度、padding mask、位置与输出索引的契约。

</details>

## 6. 误差检查能发现故障吗

清零第一层 Q 权重，首个偏离参考的阶段是什么？只比较生成 ID 会漏掉什么？能否用单精度误差判断模型质量？

<details>
<summary>参考答案</summary>

复制权重后执行 `with torch.no_grad(): model.layers[0].q.weight.zero_()`。Embedding、norm1 不变，首个偏离点是 layers.0.q，下游可能继续偏离。`test_mutated_projection_fails_oracle_comparison` 确认最终 logits 会超出双精度阈值。

贪心只看最大值索引，logits 明显改变后仍可能有相同 argmax。应同时比较中间值、最终 logits 和生成 ID。单精度误差只是当前输入的数值差异，质量需要真实模型与评测任务。

做完恢复代码，运行测试与七案例运行器，保存故障定位与修复说明。

</details>
