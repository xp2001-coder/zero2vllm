# 3.3 短示例

[实验步骤](../lab.md) · [正文](../README.md)

在仓库根目录运行：

```bash
python docs/chapters/part-03-vllm/03-model-execution-and-attention-backends/examples/walk_inputs.py
python docs/chapters/part-03-vllm/03-model-execution-and-attention-backends/examples/walk_inputs.py --order AB
python docs/chapters/part-03-vllm/03-model-execution-and-attention-backends/examples/walk_inputs.py --tensors
python docs/chapters/part-03-vllm/03-model-execution-and-attention-backends/examples/walk_inputs.py --tensors --pad-to 8
```

默认及 `--order AB` 只用标准库，输出输入数组、有效长度、写入槽位和行号。`--tensors` 才需要 CPU PyTorch，并实际运行本节的单层数值夹具；输出按请求对齐误差、候选与过滤结果。`--pad-to 8` 只模拟缓冲填充，不捕获 CUDA Graph。

[walk_inputs.py](walk_inputs.py) 与完整实验共用本节的打包实现和输入文件，便于逐项比对。
