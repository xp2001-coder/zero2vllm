# CPU 实验：INT4 打包、分组缩放与 Linear

对应 9.1、9.3—9.5 的基础练习。Python 3.11+，无需 GPU、模型下载或第三方包。状态：**可运行的 CPU 数学参考**；GPU 算子与正式量化章节仍在建设。

## 1. 运行并观察

在仓库根目录执行：

```bash
python -m zero2vllm quant-demo
python -m zero2vllm quant-demo --output .local/runs/part-09-quantization/04-quantization-kernels/cpu-int4-linear.json
python -m zero2vllm quant-demo --group-size 4 --seed 7
```

固定输入为 `X[3,17] × W[17,5]`，默认沿 K 每 8 个权重为一组，每个输出列独立计算 scale。K=17 让最后一组只有 1 行；总共 85 个权重让 packed 数据出现奇数尾部。

实现见[quantization.py](../../../../zero2vllm/reference/quantization.py)，验证见[测试](../../../../tests/reference/test_quantization.py)，默认运行证据见[实验 JSON](results/int4-reference.json)。

## 2. 本实验的完整张量契约

| 项目 | 定义 |
| --- | --- |
| 逻辑权重 | W[K,N]，row-major 展平，索引 `i=k*N+n` |
| 分组 | 沿 K，G=8；scales 形状 `[ceil(K/G),N]` |
| scale | 每组每列 `max(abs(w))/7`，全零组取 1 |
| 量化 | `q=clip(round_even(w/scale),-7,7)`，zero-point=0 |
| 存储 | 有符号二进制补码 INT4；第一个值在低半字节 |
| 尾部 | 不完整 group 只处理有效值；最后未用高半字节为 0 |
| 反量化 | `w_hat=q*scale` |
| 计算 | Python float64 权重/scale/乘法/求和，未模拟 FP16 |
| 输入限制 | 非空矩形有限数值；正整数 G；scale 下溢时明确报错 |

INT4 存储可表示 `[-8,7]`，而本次对称量化选 `[-7,7]`；`-8` 是存储练习可用值，但教学量化权重中保留不用。这种约定必须明确，不能只写“INT4”。

### 半字节例子

`[-8,-1,0,7,-7]` 对应补码半字节 `[8,f,0,7,9]`。低位优先组合成 `f8 70 09`，最后的高半字节补 0。反向读取时，半字节值大于等于 8 就减 16，得到有符号数。

本实验不会读取真实 AWQ/GPTQ 模型；这些格式的分组、打包和重排需要各自适配。

## 3. 两种误差分别回答什么

程序计算三份输出：

1. `unquantized = X @ W`。
2. `explicit_dequant = X @ dequantize(qW)`。
3. `packed_reference` 在求和时直接读取对应半字节并乘 scale。

第 2 与第 3 路之差是**实现误差**：同一份量化权重被两种读取方式解释，应该一致。第 1 与第 3 路之差是**量化后的输出误差**：量化改变了权重，因此不要求为零。随机矩阵的误差不代表真实模型任务质量。

对每组采用 amax/7 且没有范围外输入时，就近舍入误差应满足 `abs(w-w_hat) <= scale/2`，浮点检查额外留 `1e-12` 容差。使用裁剪策略后，范围外误差不再受这个半 scale 界限制。

## 4. 默认结果怎样检查

| 检查 | 预期 |
| --- | --- |
| verification.passed | true |
| implementation_max_abs_error | 0.0 |
| weight_half_scale_bound_passed | true |
| packed_bytes | ceil(85/2)=43 |
| scale_bytes | 3 组 × 5 列 × 8 字节=120 |
| logical_payload_bytes | 163 |

同样 85 个权重若按 FP16 存放是 170 字节；本例 scale 用 float64，且矩阵很小，所以远达不到“4 倍压缩”。这里比较的是假设格式的逻辑 payload，**没有测量 Python 对象内存或 GPU 显存**。

直接从 packed 数据读取不创建完整 dense 权重中间矩阵，但 Python 循环还会反复解包。这是以后写融合 kernel 的数值参考，不是加速实现；程序因此不输出速度比。

## 5. 练习

**学生必做**：

1. 先预测 G=1、4、8、32 时 scale 的数量，再运行比较存储和误差。G≥17 时只有一组。
2. 手算权重 `[-7,0,7,2]`、G=3、输入 `[1,2,3,4]`，Linear 输出应为 22。
3. 说明 ties-to-even 对 0.5、1.5、2.5 的结果；区分饱和和整数溢出。
4. 阅读全零列、奇数尾部和非法参数测试，解释每一项保护的契约。

**工程进阶**：

1. 扩展一种带 zero-point 的格式，补齐布局、范围和补偿公式，与现有格式分开命名。
2. 设计一种 tile 重排，验证重排/反重排和 Linear；增加非对齐形状。
3. 将 scale 存储改成明确的 FP32/FP16 编解码，重新制定误差界与 payload 计算，不能只改报告中的字节数。
4. 再迁移到 GPU 的独立 Q/DQ 与融合 GEMM，按[量化导读](../guide.md)进行三基线对照。

## 6. 复现检查

```bash
python -m unittest discover -s tests -p test_quantization.py -v
```

数学输入、scales、packed 字节、三份输出和 seed 都写入 JSON，可独立重算。本实验的成功只覆盖这里的数学与数据格式，不验证 A100/C500 内核或 vLLM 集成。
