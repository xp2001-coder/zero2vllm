# 9.3 源码与资料定位

[正文](README.md) · [实验手册](lab.md)

## 已运行：课程代码与 PyTorch 2.7.1+cpu

| 入口 | 本节核对内容 |
| --- | --- |
| [calibration.py](../../../../zero2vllm/quantization/calibration.py) `prepare_samples` | ID、split、截断、预算、mask与摘要 |
| 同文件 `RangeObserver` | MinMax / EMA、新批系数、冻结/重置 |
| 同文件 `FixedHistogram`、`mse_clip_search` | 分箱边界、超界计数、对原始值算损失 |
| 同文件 `error_metrics`、`classification_metrics` | shape、零范数、有效目标分母与稳定NLL |
| [实验入口](../../../../scripts/experiments/lesson0903.py) | 受控对照、反例、协议与Passport |
| [PyTorch v2.7.1 observer.py](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/ao/quantization/observer.py) | `MinMaxObserver.forward`、`MovingAverageMinMaxObserver.forward`、`_calculate_qparams` |

本轮实际读取安装版函数，并执行两组PyTorch CPU对照。MinMax/EMA极值与zero-point一致，scale在1e-6相对容差内。Observer的输入统一为FP32，课程参数计算为float64。只覆盖固定affine [0,15]案例，不是全部dtype/qscheme等价认证。

另静态阅读同版本 `HistogramObserver._non_linear_param_search`，确认其搜索近似L2误差；没有把它称为KL校准。本课FixedHistogram不复刻其流式重分箱或优化过程。

## 公开阅读：MQBench 经典支线

[MQBench observer.py 的 main 分支](https://github.com/ModelTC/MQBench/blob/main/mqbench/observer.py)，阅读日期2026-09-20。该链接会变化，本轮没有取得可固定的commit，也没有导入或运行MQBench。

读取范围为 `MinMaxObserver`、`EMAMinMaxObserver` 和 `MSEObserver`：分别关注批次极值合并、旧状态权重 `ema_ratio`，以及候选范围 / fake-quant / loss 的组合。可对照概念分工，不作为版本固定的执行证据。

经典CNN支线的模型级调用将单独固定环境与commit再交付；本节的完成范围是CPU协议与Observer基础。

## 文档与后续论文

- [MovingAverageMinMaxObserver 2.7](https://docs.pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.MovingAverageMinMaxObserver.html)：EMA更新式和新批次系数。
- [FakeQuantize 2.7](https://docs.pytorch.org/docs/2.7/generated/torch.ao.quantization.fake_quantize.FakeQuantize.html)：观察与假量化的开关职责。
- [Transformers v4.51.3 PPL](https://huggingface.co/docs/transformers/v4.51.3/perplexity)：目标计数、窗口与tokenizer对度量的影响。本课未运行文档中的GPT-2示例。
- [GPTQ v2](https://arxiv.org/abs/2210.17323v2)、[BRECQ v2](https://arxiv.org/abs/2102.05426v2)：后续算法阅读入口；不将论文性能写成本课结果。

所有中文图与反例均由本课程重新设计。资料仅用于规范和方法核对。
