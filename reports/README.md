# 仓库验证与历史记录

9.4 已完成量化算法与误差控制的 CPU 课程；本轮524项测试通过，见[质量记录](quality/lesson-9.4-delivery/README.md)。

9.3 已完成校准协议、Observer与误差测量的CPU课程；本轮491项测试通过，见[质量记录](quality/lesson-9.3-delivery/README.md)。

9.2 已完成数值表示与张量契约的 CPU 课程；本轮 452 项测试通过，见[质量记录](quality/lesson-9.2-delivery/README.md)。

9.1 已完成量化系统全景、含 zero-point 的 Q/DQ/RQ、模型 Passport 与 6 张中文图；本轮 413 项测试通过，见[质量记录](quality/lesson-9.1-delivery/README.md)。GPU 与真实模型仍单独验收。

Part 9 v1.0 设计已整理为四个 Module、14 节，全课程同步为 19 篇 129 节；本轮 37 项相关测试通过，已有 CPU INT4 结果完整保留，见[设计基线检查](quality/part-09-design-baseline/README.md)。正文与 GPU 实验继续分开验收。

3.4—3.5 已完成 Runner 演进与配置解析：22 个标准库案例、12 张中文图、28 组练习；本轮全仓库 380 项测试通过、0 跳过，见[第三篇收尾质量记录](quality/part-03-completion/README.md)。第三篇五节已完成 CPU/源码层面的课程交付；GPU 实测继续单独验收。

3.3 已完成模型执行与注意力后端课程、CPU 张量实验和固定源码核验；交付时全仓库 335 项测试通过、0 项跳过，见[3.3 质量记录](quality/lesson-3.3-delivery/README.md)。完整预训练模型与 GPU 实测另行验证。

3.2 已完成调度与 KV 计数实验和固定源码核验，交付时全仓库 305 项测试通过，见[3.2 质量记录](quality/lesson-3.2-delivery/README.md)。本节是 CPU 教学模拟与源码阅读，GPU 实测另行验证。

3.1 已完成固定 vLLM 源码阅读与静态核验，交付时全仓库 280 项测试通过，见[3.1 质量记录](quality/lesson-3.1-delivery/README.md)。源码文件与函数核验不代表 GPU 执行。

这里仅放跨章节的仓库检查、环境记录和编辑历史。单节实验数据已移到 docs/chapters/ 对应小节的 results/。

| 分区 | 用途 |
| --- | --- |
| [quality/](quality/README.md) | 本轮测试、课程同步和目录检查；由实际运行生成 |
| [platforms/cpu/initial-environment.json](platforms/cpu/initial-environment.json) | 初始本地 CPU 环境记录 |
| [history/](history/README.md) | 迁移前的检查与课程建设记录 |

第一篇教学修缮已通过 27 项专项测试、28 个已有案例和 6 个新增短示例，见[修订质量记录](quality/part-01-revision/README.md)。课程与目录检查通过，未运行 GPU 或远程 CI。

2.7 交付时的全仓库 264 项测试全部通过，见[2.7 质量记录](quality/lesson-2.7-delivery/README.md)。15 个服务案例见[2.7 CPU 与 HTTP 实测](../docs/chapters/part-02-engine/07-http-and-streaming-output/results/README.md)。A100/C500 与远程 CI 未在该轮验证。

2.6 的 235 项测试保留在[2.6 质量记录](quality/lesson-2.6-delivery/README.md)。12 个共享前缀案例见[2.6 CPU 实测](../docs/chapters/part-02-engine/06-prefix-caching/results/README.md)。A100/C500 与远程 CI 未在本轮验证。

2.5 的 206 项测试保留在[2.5 质量记录](quality/lesson-2.5-delivery/README.md)。11 个分页张量案例见[2.5 CPU 实测](../docs/chapters/part-02-engine/05-paged-caching/results/README.md)。A100/C500 与远程 CI 未在本轮验证。

2.4 的 180 项测试保留在[2.4 质量记录](quality/lesson-2.4-delivery/README.md)。六组负载、两种调度策略的十二次回放见[2.4 CPU 实测](../docs/chapters/part-02-engine/04-from-static-to-continuous-batching/results/README.md)。A100/C500 与远程 CI 未在本轮验证。

2.3 的 157 项测试保留在[2.3 质量记录](quality/lesson-2.3-delivery/README.md)，11 个基础案例与两个真实模型生命周期案例见[2.3 CPU 实测](../docs/chapters/part-02-engine/03-request-and-cache-lifecycles/results/README.md)。

2.2 交付时的 133 项测试记录保留在[2.2 质量记录](quality/lesson-2.2-delivery/README.md)，真实 SmolLM2 逐层误差与生成 ID 见[2.2 CPU 实测](../docs/chapters/part-02-engine/02-loading-real-model-weights/results/README.md)。

此前目录重构时记录了 108 项测试、第一篇 28 案例和 2.1 的 7 案例，详见[重构验证](quality/repository-refactor/README.md)。历史测试数量与源码哈希属于各自运行，不代表当前源码。
