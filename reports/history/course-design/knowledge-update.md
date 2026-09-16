# 知识库驱动的课程更新

日期：2026-09-15。范围：本地课程更新；GitHub 发布前供维护者 review。

## 建议审阅顺序

1. [README](../../../README.md)：学习入口、课程状态和完整范围。
2. [专题导读](../../../docs/references/topic-guides.md)：各主题的学习与练习入口。原备课映射已归档到维护者本地区，公开课程不再展示内部来源编号。
3. [量化导读](../../../docs/chapters/part-09-quantization/guide.md)和 [CPU INT4 实验](../../../docs/chapters/part-09-quantization/04-writing-quantization-and-data-reordering-kernels/lab.md)：算法统一成章后的具体教学深度。
4. [编译与 Profiling](../../../docs/chapters/part-08-compilation/guide.md)和[实验规范](../../../docs/engineering/experiments.md)：优化如何测、怎样避免错误归因。
5. [验证记录](../validation-before-restructure.md)：实际检查范围和未完成项。

## 本轮形成的成果

| 内容 | 更新结果 |
| --- | --- |
| 本地工程 | 将此前课程工程同步到当前项目根目录，保留原有 `vllm/` 资料 |
| 来源盘点 | 33 ZIP 内 Markdown + 5 个独立 Markdown，共 38 份；包括 20 讲 vLLM、15 讲 SGLang 和 3 份总结/手册 |
| 课程组织 | 保留 19 篇、126 章；把已有正文与练习链接回目录，导读相关章标为部分材料 |
| 实际正文 | 7 篇专题导读，包含请求/张量、调度/缓存、Attention、编译/性能、量化、并行/MoE/PD、推测/服务 |
| 量化主线 | 算法集中 9.2；增加数据契约、加载后重排、三基线、shape 扫描与回本次数分析 |
| 可运行代码 | 新增 `quant-demo`，验证 group-wise INT4、pack/unpack、Linear、误差和元数据成本 |
| 实验规范 | W0—W7、单项/组合对照、无 profiler 性能测量、同 shape trace、失败和分位数口径 |
| 维护工具 | 校验知识来源映射和正文链接；生成知识库表格；公开课程不依赖本地参考库 |

## 主要修正与保留边界

- Scheduler 的请求数、token 预算、已计算 token 和 KV 容量分别解释；调度输出与 GPU 输入张量分开追踪。
- vLLM V1 与 MRV2 分开，runner × graph 分别消融；不引用笔记中的性能百分比作为我们的数据。
- 量化算法、文件格式、运行时布局和真实 kernel 分开；缩放补偿必须能对应到部署计算等式。
- DBO/TBO、APC/Radix、Chunked Prefill/CP 分开说明，SGLang 保留为对照材料。
- C600/多卡和旧 PD 记录用于提出排查问题，不转换为单卡 C500 支持或性能声明。

## 尚待完成

7 篇导读没有取代 126 个完整章节。本轮提供可读的主线、手算和实验设计；固定 tag/commit 的逐行源码实验、PyTorch 教学引擎、GPU kernels、真实模型质量、A100/C500 服务和分布式性能仍待完成。远程 CI 未执行。

原资料内容与附件保持原样；只读取和整理 Markdown，未执行其中的部署/运维命令，未逐张验证图片。`vllm/` 和 `work/` 默认不提交，整理后的公共正文不含原笔记中的内部镜像、主机或模型目录。

## 验证入口

本次测试结果以[验证记录](../validation-before-restructure.md)、[测试输出](../part1-delivery/reference-tests.txt)、[课程检查](../lesson-2.1-delivery/curriculum-checks.json)和 [INT4 原始记录](../../../docs/chapters/part-09-quantization/04-writing-quantization-and-data-reordering-kernels/results/int4-reference.json)为准。所有 GPU 状态继续为 `not_run`。本轮没有生成发布压缩包、创建提交或推送 GitHub。
