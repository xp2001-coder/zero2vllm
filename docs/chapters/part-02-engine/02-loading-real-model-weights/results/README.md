# 2.2 CPU 实测记录

[返回正文](../README.md) · [复现命令](../lab.md)

## 固定条件

- 实测日期：2026-09-17；Windows、Python 3.11.9、PyTorch 2.7.1+cpu。
- 模型：HuggingFaceTB/SmolLM2-135M，revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`。
- 参考：Transformers 4.51.3；CPU float32、2 个线程、eager Attention、无 KV Cache。
- 三个输入分别编码，不加特殊 token、不套聊天模板；两套模型共用输入 ID。
- 最多八个新 token，贪心生成，EOS=0；参考生成由 Transformers 的 `generate` 独立执行。
- A100/C500 均未实测；本记录不提供 GPU 支持或性能结论。

## 文件

- [baseline/summary.json](baseline/summary.json)：模型九个文件的摘要、运行环境、源码指纹、每个输入的 33 个阶段比较和生成 ID。
- [baseline/weights.json](baseline/weights.json)：272 个源权重到教学模块的映射、形状与 dtype；参数量 134,515,008。

模型原始权重不随报告提交。报告中 Git commit 是执行时已提交的父版本，`working_tree_dirty=true` 表示本节源码尚未提交；同时记录 `source_sha256` 区分当时的实际代码。它不声称父 commit 已包含本节全部实现。

## 观察结果

三个输入的全部阶段比较均通过，当前环境下最大绝对误差均为 0.0；各生成八个 ID，与参考完全相同。验收仍使用 `atol=1e-4, rtol=1e-4`，不外推逐位一致性。

| 输入 | 输入 ID | 本次续写 |
| --- | --- | --- |
| `The capital of France is` | `[504,3575,282,4649,314]` | ` the capital of the country.` 后接两个换行 |
| `Once upon a time` | `[6403,1980,253,655]` | `, there was a little girl named Lily` |
| `1 + 1 =` | `[33,1232,216,33,446]` | ` 1000000` |

最后一条结果保留原样，说明计算对齐与模型质量不同。没有挑选更好看的回答替换原始结果。

## 验证范围

真实权重实测使用本模型的单文件检查点。分片、不同 GQA 头数、共享/独立输出层和错误注入由 [18 项离线测试](../../../../../tests/modeling/test_checkpoint.py)覆盖，使用临时随机小模型，不冒充第二个真实模型实验。

按照实验文档，可先复跑三个真实输入，再单独运行测试。后续修改代码时生成自己的 `.local/runs/` 报告；不要覆盖这份基线后仍沿用旧日期和说明。
