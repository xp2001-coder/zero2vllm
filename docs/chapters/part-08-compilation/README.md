# Part 8 · 编译、CUDA Graph 与执行优化

[课程总目录](../README.md) · [学习路线](../learning-paths.md)

前置知识：Part 3、6、7。

## 本篇章节

<!-- BEGIN part-index -->

| 节 | 课程 | 验收产出 | 进度 | 正文 |
| --- | --- | --- | --- | --- |
| 8.1 | [torch.compile 集成](01-torch-compile-integration/README.md) | 区分两层编译链，追踪冷/热编译、specialization 和一个自定义算子的 lowering | 🧩 | 教学骨架 |
| 8.2 | [CUDA Graph 执行模式](02-cuda-graph-execution-modes/README.md) | 分别记录 eager、分段图、完整图的启动开销与显存 | 🧩 | 教学骨架 |
| 8.3 | [形状特化与编译缓存](03-shape-specialization-and-compilation-caches/README.md) | 比较冷启动与稳态，定位重复编译 | 🧩 | 教学骨架 |
| 8.4 | [计算融合 Pass](04-compute-fusion-passes/README.md) | 对照融合前后 kernel 数和误差，不只看开关值 | 🧩 | 教学骨架 |
| 8.5 | [通信融合与 Async TP](05-communication-fusion-and-async-tp/README.md) | 画出依赖关系，测量通信隐藏比例 | 📋 | 教学骨架 |
| 8.6 | [优化级别、CustomOp 与 vLLM IR](06-optimization-levels-customop-and-vllm-ir/README.md) | 追踪一个算子从模型代码到后端实现的过程 | 📋 | 教学骨架 |
| 8.7 | [多模态图与编译排错](07-multimodal-graphs-and-compilation-debugging/README.md) | 对齐图执行结果并保存最小失败场景 | 📋 | 教学骨架 |

<!-- END part-index -->
