# 4.1 CUDA 路径与 A100 基线

[本篇目录](../README.md) · [全部课程](../../README.md) · [上一节 3.5](../../part-03-vllm/05-resolving-configuration-and-capabilities/README.md) · [下一节 4.2](../02-the-maca-path-and-c500-baseline/README.md)

> 🧩 已有相关材料；本节正文与配套实验待完善。目录存在不表示内容已经完成。

## 学习目标

上游 vLLM、CUDA/PyTorch、显存规格、eager 起步、离线/在线。

## 前置知识与设备

- 前置：Part 0、1、3。
- 设备：GPU。具体版本和执行命令随本节实验一起固定。
- A100/C500 的支持情况以实机记录为准，未执行的路径保持待验证。

## 正文安排

<!-- BEGIN lesson-plan -->

1. 上游 vLLM：补充概念、推导、例子及实现边界。
2. CUDA/PyTorch：补充概念、推导、例子及实现边界。
3. 显存规格：补充概念、推导、例子及实现边界。
4. eager 起步：补充概念、推导、例子及实现边界。
5. 离线/在线：补充概念、推导、例子及实现边界。

<!-- END lesson-plan -->

## 代码与实验

- [ ] 确定最小输入、预期输出与张量/请求契约。
- [ ] 补充可独立运行的实现、命令、参数解释和结果示例。
- [ ] 覆盖正常输入、边界输入和失败排查。
- [ ] 保存可复核结果；涉及性能时注明设备、版本、负载和计时范围。

## 练习与验收

<!-- BEGIN lesson-exercise -->

目标产出：**固定环境并保存单卡 A100 的正确性与性能基线**。

- [ ] 提供输入、命令、结果与解释。
- [ ] 补充参考答案和至少一个边界案例。

<!-- END lesson-exercise -->

完成正文、代码、命令和必要验证后，才能把课程数据中的状态改为已完成。填充方法见[章节模板](../../../maintenance/templates/lesson.md)。

## 参考资料

- [GPU 安装与设备条件](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)
- [vLLM-MetaX](https://github.com/MetaX-MACA/vLLM-metax)
- [MetaX 安装](https://vllm-metax.readthedocs.io/en/stable/getting_started/installation/maca.html)
- [MetaX Quickstart](https://vllm-metax.readthedocs.io/en/stable/getting_started/quickstart.html)
