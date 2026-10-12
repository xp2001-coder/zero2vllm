# 2.2 交付时的仓库验证

[质量记录](../README.md) · [课程正文](../../../docs/chapters/part-02-engine/02-loading-real-model-weights/README.md)

记录日期：2026-09-17。环境为 Windows、Python 3.11.9 与 CPU PyTorch 2.7.1；完整依赖和源码指纹见 [regression.json](regression.json)。

## 全仓库回归

`python -m unittest discover -s tests -v`：**133 项通过，0 项跳过**。原始输出见 [tests.txt](tests.txt)，包含新增的 18 项权重加载与独立数值对照测试。

课程同步、公开参考规则、本地链接和目录布局另由 `python -m scripts.docs.course --check` 与 `python -m scripts.docs.layout` 检查。它们不验证外部网址或 GPU 支持。

目录维护测试使用临时副本构造教学骨架，避免依赖某一节永久处于未完成状态。

## 真实模型实验

固定 SmolLM2-135M 的三组输入、每组 33 个阶段和八步贪心生成均通过，记录在[本节 results](../../../docs/chapters/part-02-engine/02-loading-real-model-weights/results/README.md)。本回归记录与真实模型报告中的 `source_sha256` 相同。

另外实际运行了只检查权重、零生成和四步生成的命令。只检查权重的报告为 `verification=not_run`、`passed=null`；其余两条路径与参考结果一致。

这些是本地 CPU 验证。远程 GitHub Actions、A100 和 C500 本轮均未运行；Git commit 为工作区改动的父版本，不能单独用该 commit 代表本节源码。
