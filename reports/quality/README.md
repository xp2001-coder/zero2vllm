# 跨章节质量检查

[返回验证导航](../README.md)

| 记录 | 范围 |
| --- | --- |
| [Part 13 规划扩充](part-13-design/README.md) | 12节规划、19篇135节目录；27项维护检查经复跑通过、1项系统条件跳过，未执行GPU或集群实验 |
| [9.4 课程交付](lesson-9.4-delivery/README.md) | 524项回归通过、0跳过；33项新增检查、23个CPU案例、6组数学核验与8张中文图 |
| [9.3 课程交付](lesson-9.3-delivery/README.md) | 491项回归通过、0跳过；39项新增检查、20个CPU案例、2组PyTorch Observer对照与7张中文图 |
| [9.2 课程交付](lesson-9.2-delivery/README.md) | 452 项回归通过、0 跳过；39 项新增检查、20 个 CPU 案例、512 个 FP8 位模式对照与 7 张中文图 |
| [9.1 课程交付](lesson-9.1-delivery/README.md) | 413 项回归通过、0 跳过；33 项新增检查、17 个 CPU 案例、Q/DQ/RQ 中文图与固定源码核验 |
| [Part 9 v1.0 设计基线](part-09-design-baseline/README.md) | 四个 Module、14 节；37 项相关测试通过，19 篇 129 节目录同步，CPU INT4 历史结果保留 |
| [第三篇收尾：3.4—3.5](part-03-completion/README.md) | 380 项测试通过、0 跳过；22 个标准库案例、25 条命令、45 项新增检查和固定源码核验 |
| [3.3 课程交付](lesson-3.3-delivery/README.md) | 335 项测试通过；两种精度各 12 个 CPU 案例、30 项新增检查及固定源码 15 文件/27 函数/2 类结构核验 |
| [3.2 课程交付](lesson-3.2-delivery/README.md) | 305 项测试通过；10 个计数案例、25 项新增检查及固定源码 6 文件/11 函数/1 类调度输出核验 |
| [3.1 课程交付](lesson-3.1-delivery/README.md) | 280 项测试通过；固定源码 25 文件/38 函数/4 类消息核验，16 项新增工具检查 |
| [第一篇教学修缮](part-01-revision/README.md) | 27 项第一篇测试、28 个已有案例、6 个新增短示例通过；补充 10 张中文图和后半篇 32 组练习 |
| [2.7 课程交付](lesson-2.7-delivery/README.md) | 264 项测试全部通过；15 个服务案例，29 项新增协议、异步与真实 HTTP 检查 |
| [2.6 课程交付](lesson-2.6-delivery/README.md) | 235 项测试全部通过；12 个前缀共享案例，29 项新增存储与引擎检查 |
| [2.5 课程交付](lesson-2.5-delivery/README.md) | 206 项测试全部通过；11 个分页张量案例，26 项新增存储与引擎检查 |
| [2.4 课程交付](lesson-2.4-delivery/README.md) | 180 项测试全部通过；六组负载各两种策略，23 项新调度检查 |
| [2.3 课程交付](lesson-2.3-delivery/README.md) | 157 项测试全部通过；11 个基础案例与两个真实模型生命周期案例 |
| [2.2 课程交付](lesson-2.2-delivery/README.md) | 133 项测试全部通过；真实权重对照与源码指纹关联 |
| [仓库目录重构](repository-refactor/README.md) | 108 项测试、第一篇 28 案例、2.1 的 7 案例及课程/目录检查 |

单节公开实验继续保存在对应章节的 results/。这里每轮仓库维护使用独立目录，不与历史课程设计记录混放。

## Part 13 工程定位更新

[检查记录](part-13-engineering/README.md)：Dynamo/llm-d控制面、分层KV与生产运维规划；目录/链接检查，未执行集群实验。
