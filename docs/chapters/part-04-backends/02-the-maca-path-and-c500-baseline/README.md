# 4.2 MACA 路径与 C500 基线

[本篇目录](../README.md) · [全部课程](../../README.md) · [上一节 4.1](../01-the-cuda-path-and-a100-baseline/README.md) · [下一节 4.3](../03-the-same-experiment-on-two-platforms/README.md)

> 🧩 已有相关材料；本节正文与配套实验待完善。目录存在不表示内容已经完成。

## 学习目标

vLLM-MetaX、MACA、mcoplib、官方镜像、版本关系。

## 前置知识与设备

- 前置：Part 0、1、3。
- 设备：GPU。具体版本和执行命令随本节实验一起固定。
- A100/C500 的支持情况以实机记录为准，未执行的路径保持待验证。

## 正文安排

<!-- BEGIN lesson-plan -->

1. vLLM-MetaX：补充概念、推导、例子及实现边界。
2. MACA：补充概念、推导、例子及实现边界。
3. mcoplib：补充概念、推导、例子及实现边界。
4. 官方镜像：补充概念、推导、例子及实现边界。
5. 版本关系：补充概念、推导、例子及实现边界。

<!-- END lesson-plan -->

## 代码与实验

- [ ] 确定最小输入、预期输出与张量/请求契约。
- [ ] 补充可独立运行的实现、命令、参数解释和结果示例。
- [ ] 覆盖正常输入、边界输入和失败排查。
- [ ] 保存可复核结果；涉及性能时注明设备、版本、负载和计时范围。

## 练习与验收

<!-- BEGIN lesson-exercise -->

目标产出：**固定配套环境并保存单卡 C500 的独立验证记录**。

- [ ] 提供输入、命令、结果与解释。
- [ ] 补充参考答案和至少一个边界案例。

<!-- END lesson-exercise -->

完成正文、代码、命令和必要验证后，才能把课程数据中的状态改为已完成。填充方法见[章节模板](../../../maintenance/templates/lesson.md)。

## 参考资料

- [GPU 安装与设备条件](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)
- [vLLM-MetaX](https://github.com/MetaX-MACA/vLLM-metax)
- [MetaX 安装](https://vllm-metax.readthedocs.io/en/stable/getting_started/installation/maca.html)
- [MetaX Quickstart](https://vllm-metax.readthedocs.io/en/stable/getting_started/quickstart.html)
