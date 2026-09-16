# 维护与运行工具

[目录规范](../docs/maintenance/repository-layout.md)

在仓库根目录用模块方式运行。这里按职责放工具；学生需要阅读的示例在各节 examples/。

| 目录 | 用途 | 入口 |
| --- | --- | --- |
| docs/ | 课程生成、翻译装配、链接与仓库布局检查 | `python -m scripts.docs.course --check` / `python -m scripts.docs.layout` |
| experiments/ | 章节实验批量运行 | `python -m scripts.experiments.part01` / `python -m scripts.experiments.lesson0201` |
| environment/ | 设备环境信息采集 | `python -m scripts.environment.collect` |
| common.py | 共享根目录与源码指纹 | 内部复用，不单独运行 |

实验默认写 .local/runs/，质量检查写 .local/checks/。核验后按章节选择结果公开，不能把整个本地区上传。
