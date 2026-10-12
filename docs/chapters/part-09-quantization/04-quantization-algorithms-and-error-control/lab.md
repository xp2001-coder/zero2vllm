# 9.4 实验：先算出量化结果，再比较误差

[正文](README.md) · [练习](exercises.md) · [结果](results/README.md)

在仓库根目录运行，默认Python3.11+标准库即可。Windows中文输出可加`-X utf8`。不需要GPU、模型下载或网络。

## 1. 全部案例

```bash
python -m scripts.experiments.lesson0904
python -m scripts.experiments.lesson0904 --output-dir .local/runs/part-09-quantization/04-algorithms/replay
```

预期23个PASS、退出0。默认输出位于`.local/runs/part-09-quantization/04-algorithms/`，有summary.json和cpu-passport.json。replay参数保留独立记录；不会覆盖本节公开results。

## 2. 看舍入组合

```bash
python docs/chapters/part-09-quantization/04-quantization-algorithms-and-error-control/examples/walk_rounding.py
python docs/chapters/part-09-quantization/04-quantization-algorithms-and-error-control/examples/walk_rounding.py --max-variables 1
```

第一条比较四个候选，RTN为[0,0]、层目标最优为[0,1]，输出SSE分别为1.2005/.2405。第二条预期退出2：两个自由变量超过你给的上限，不会偷偷跳过搜索或改成近似算法。

## 3. 逐步观察GPTQ

```bash
python docs/chapters/part-09-quantization/04-quantization-algorithms-and-error-control/examples/walk_gptq.py
python docs/chapters/part-09-quantization/04-quantization-algorithms-and-error-control/examples/walk_gptq.py --block-size 2
python docs/chapters/part-09-quantization/04-quantization-algorithms-and-error-control/examples/walk_gptq.py --act-order
python docs/chapters/part-09-quantization/04-quantization-algorithms-and-error-control/examples/walk_gptq.py --block-size 0
```

前三条使用主矩阵W[3,2]，打印order、upper、trace和最终code。block_size=1/2只改变更新组织，主案例code一致。act-order可能改变选择；输出已经还原到原始K顺序。最后一条预期退出2。

手算例的第二个权重.49→.98可在完整结果`rounding_toy.gptq.trace`看到。主矩阵trace的数值与手算例不同，不能混读。

## 4. 观察变换

```bash
python docs/chapters/part-09-quantization/04-quantization-algorithms-and-error-control/examples/walk_transforms.py
```

缩放例α=0/.25/.5/.75/1的校准MSE约为.010580/.101821/.029647/.006237/.001600。正交例把[8,0]变为[5.656854,5.656854]，与配套变换后的权重相乘仍得到2。

## 5. 独立PyTorch核验

```bash
python -m scripts.experiments.lesson0904 --torch-check
```

实际使用PyTorch2.7.1+cpu，float64。GPTQ核心与逐步直接求剩余Hessian逆矩阵的解比较，关闭/启用act-order各一组；STE用显式detach构造的autograd规则检查四个内部/饱和位置。没有调用官方GPTQ、AWQ模型包。

缺少PyTorch而显式要求核验时，命令失败，不把未运行写成通过。安装参考[PyTorch历史版本页](https://pytorch.org/get-started/previous-versions/#v271)。

## 6. 测试与结果解释

```bash
python -m unittest tests.quantization.test_algorithms -v
```

33项测试覆盖黄金数值、非方阵、zero-point、指数搜索上限、奇异H、补偿符号、分块、坐标还原和替代梯度。PyTorch对照是可选的1项；本轮全部运行。

主比较表由固定输入计算，范围搜索选r=1、缩放选α=.75。校准误差降低但留出误差上升的行是有效观察，不应删掉或改成漂亮数字。所有候选按校准目标选择，不依据留出数据重选。

## 7. 扩展实验

复制固定输入到自己的实验目录：只改变校准X的相关性，或只改变damping。记录改动前后的输入摘要、网格、候选预算和误差。不要覆盖公开结果，也不要将小矩阵CPU耗时当作A100/C500量化或推理性能。
