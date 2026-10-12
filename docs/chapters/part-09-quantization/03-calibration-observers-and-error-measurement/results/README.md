# 9.3 CPU 校准与误差运行记录

[正文](../README.md) · [复现命令](../lab.md) · [质量记录](../../../../../reports/quality/lesson-9.3-delivery/README.md)

记录时间（UTC）：`2026-09-20T10:34:45.288502+00:00`。本次在 Windows / Python 3.11.9 运行，框架对照为 PyTorch 2.7.1+cpu CPU。输入是原创合成 token 与两个通道的固定特征，不是真实语言模型。

| 文件 | 实际结果与作用 |
| --- | --- |
| [summary.json](summary.json) | 20 个固定案例全部通过；含两组受控对照、裁剪损失与三组误差排名反例 |
| [protocol.json](protocol.json) | general 的8条校准样本、50个有效token；保存实际ID、特殊token、mask与数据摘要 |
| [cpu-passport.json](cpu-passport.json) | 静态 per-tensor INT4 激活校准的参数和证据；模型质量/性能保持 unknown |
| [commands.json](commands.json) | 11条命令符合预期退出码，包括一次预算不足的预期退出2 |

## 两组受控对照

| 实验 | 校准有效token | 冻结scale | 相同test上的元素MSE |
| --- | --- | --- | --- |
| general，2条 | 11 | 1/7 | 3.850552721 |
| general，4条 | 24 | 1.5/7 | 3.013286565 |
| general，8条 | 50 | 1 | .09375 |
| 固定预算，general | 16 | 1.5/7 | 3.013286565 |
| 固定预算，code | 16 | 1 | .09375 |

均采用 MinMax、signed INT4 [-7,7]、z=0、FP64 scale；参数冻结后在同一4条test序列的24个有效token、48个特征元素上评测。第一组固定嵌套样本顺序，第二组固定有效token预算。后两行都请求4条，实际使用3条并截断最后一条；各自完整协议保存在summary的studies中。

这些数字不支持“真实代码数据一定更好”或“更多样本一定更好”的结论。本次没有运行模型PPL。

## 关键数值

- EMA的新批权重为.25：A→B得[-1.75,2.75]，B→A得[-3.25,6.25]；合为一次首批得到[-4,8]。
- 给定裁剪候选[.5,1,2,4,7]，校准MSE最低的T=4约为.039946；在另给出的验证分布上约为4.502551，T=7约为.125。阈值按校准目标选择，没有据验证结果回调。
- 权重反例A/B的权重MSE为.0078125/.03125，层输出MSE为1/0。
- 重构反例使用FP输入时A/B的MSE为0/1，量化上游输入时为1/.0625。
- 合成logits例A的MSE更小，却因跨过决策边界而具有更低准确率。

MinMax和MovingAverageMinMax各用两批输入，与PyTorch CPU的affine [0,15]对照：极值、zero-point一致，scale在1e-6相对容差内。只覆盖两组明确的输入与设置。

## 重现与快照

```bash
python -m scripts.experiments.lesson0903
python -m scripts.experiments.lesson0903 --torch-check
```

默认离线、只用标准库；第二条显式要求PyTorch。输出进入 `.local/runs/part-09-quantization/03-calibration/`。固定数值和检查项应一致；时间、运行环境与源码摘要可不同。

源码摘要：`d69b29a4a453bd60c64e315444ab8231868d3d3bbd96dbcf6f9ec17c0ed3c508`。它覆盖课程代码、测试、章节示例和输入JSON；图文另经本轮文档与视觉检查。git_commit表示未提交工作区的基点，不能代替源码摘要。

本次未执行MQBench模型、真实Qwen量化、checkpoint生产、GPU kernel、A100/C500或服务性能实验。相关标记为false，未测指标不填推测值。
