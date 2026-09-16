# 9.3 量化策略与精度分配

[本篇目录](../README.md) · [全部课程](../../README.md) · [上一节 9.2](../02-quantization-algorithms/README.md) · [下一节 9.4](../04-writing-quantization-and-data-reordering-kernels/README.md)

> 🧩 已有相关材料；本节正文与配套实验待完善。目录存在不表示内容已经完成。

## 学习目标

W/A/KV、weight-only/低精度计算、静态/动态、在线/离线、分层/分组/专家混合精度、负载感知。

## 前置知识与设备

- 前置：Part 1、4、7；内核调优结合 Part 8，MoE 量化补读 Part 12.1—12.3。
- 设备：CPU/GPU。具体版本和执行命令随本节实验一起固定。
- A100/C500 的支持情况以实机记录为准，未执行的路径保持待验证。

## 正文安排

<!-- BEGIN lesson-plan -->

1. W/A/KV：补充概念、推导、例子及实现边界。
2. weight-only/低精度计算：补充概念、推导、例子及实现边界。
3. 静态/动态：补充概念、推导、例子及实现边界。
4. 在线/离线：补充概念、推导、例子及实现边界。
5. 分层/分组/专家混合精度：补充概念、推导、例子及实现边界。
6. 负载感知：补充概念、推导、例子及实现边界。

<!-- END lesson-plan -->

## 代码与实验

- [ ] 确定最小输入、预期输出与张量/请求契约。
- [ ] 补充可独立运行的实现、命令、参数解释和结果示例。
- [ ] 覆盖正常输入、边界输入和失败排查。
- [ ] 保存可复核结果；涉及性能时注明设备、版本、负载和计时范围。

## 练习与验收

<!-- BEGIN lesson-exercise -->

目标产出：**在质量、显存和延迟预算下设计策略，并形成敏感度与消融表**。

- [ ] 提供输入、命令、结果与解释。
- [ ] 补充参考答案和至少一个边界案例。

<!-- END lesson-exercise -->

完成正文、代码、命令和必要验证后，才能把课程数据中的状态改为已完成。填充方法见[章节模板](../../../maintenance/templates/lesson.md)。

## 已有材料

先读[相关专题导读](../guide.md)。它覆盖多个小节，不能替代本节的完整实现和验收。

## 参考资料

- [量化与硬件矩阵](https://docs.vllm.ai/en/stable/features/quantization/)
- [vLLM 量化方法与配置接口](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/base_config.py)
- [Linear/MoE Kernel 配置](https://github.com/vllm-project/vllm/blob/main/vllm/config/kernel.py)
- [Triton 矩阵乘教程](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
- [Marlin 量化内核实现](https://github.com/IST-DASLab/marlin)
- [Quantized KV Cache](https://docs.vllm.ai/en/stable/features/quantization/quantized_kvcache/)
- [量化方法注册表](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/__init__.py)
- [Online Quantization](https://docs.vllm.ai/en/stable/features/quantization/online/)
- [LLM Compressor](https://docs.vllm.ai/en/stable/features/quantization/llm_compressor/)
