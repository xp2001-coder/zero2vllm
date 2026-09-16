# 贡献说明

## 一次改动的最小闭环

1. 说明学习问题或运行问题。
2. 更新代码或对应章节。
3. 运行相关检查，保留真实结果和环境。
4. 在说明中区分已实现、已测试和计划内容。

## 本地检查

```bash
python -m unittest discover -s tests -v
python -m zero2vllm demo --output .local/runs/part-01-inference/03-kv-cache/cpu-kv-cache.json
python -m scripts.docs.course --check
python -m scripts.docs.layout
```

文档改动检查链接和命令。涉及数学、调度、缓存或错误处理的变更应提供能发现行为错误的测试；硬件相关改动应附对应平台实测证据。

Part 1 改动可运行 `python -m scripts.experiments.part01`，一次保存 28 个案例、全套测试和课程检查。案例、命令参数与预期解释见 [Part 1 总表](docs/chapters/part-01-inference/README.md)。如已独立运行测试，可加 `--skip-tests`，报告会明确标记跳过状态。

2.1 改动使用独立虚拟环境安装 CPU PyTorch，再运行 `python -m scripts.experiments.lesson0201`；详见[本节实验](docs/chapters/part-02-engine/01-pytorch-model/lab.md)。

## 课程目录的维护入口

- [中文课程数据](metadata/course/README.md)：`parts/` 下每篇一个 JSON，维护章节标题、知识点、练习、设备、状态、前置和篇内说明；`catalog.json` 只维护篇文件顺序与公共来源。
- [英文课程数据](metadata/translations/en/README.md)：同样每篇一个 JSON，维护篇名、章节标题和知识点；`index.json` 维护篇文件顺序与来源标题。编号与中文一一对应，状态、阅读链接和来源网址共用中文课程数据。
- [features.json](metadata/upstream/features.json)：命名条目、来源和章节映射。它是特性盘点，不是设备支持矩阵。
- [learning-paths.md](docs/chapters/learning-paths.md)：学习顺序与章节分工；[roadmap.md](docs/engineering/roadmap.md)：材料交付顺序。

修改课程、特性或[知识来源](metadata/knowledge/sources.json)清单后，在工程根目录运行：

```bash
python -m scripts.docs.course --write
python -m scripts.docs.course --check
```

工具更新中英文 README、课程总表、19 篇目录、未完成小节的教学安排/验收目标、来源总表、特性附录和知识库映射中的 `BEGIN/END` 标记区段，保留其余正文；请把 JSON 和生成的 Markdown 一起提交。`--check` 默认只读，内容不同步时返回非零状态，CI 同样执行此检查。不要手动编辑生成区段。

中英文首页分别是 [README.md](README.md) 和 [README_en.md](README_en.md)，共用 `docs/assets/zero2vllm-banner.webp`。横幅的生成提示词与导出参数见 [banner-design.json](docs/assets/banner-design.json)，原 SVG 草图保留供参考。修改章节内容时同步维护英文译文；调整首页介绍、命令或进度时同步检查两份 README 的人工正文。工具会检查翻译编号是否齐全，但不判断译文含义是否过期。英文首页不代表章节正文已全部翻译，翻译范围说明应与实际交付一致。

检查涵盖重复章节/注册项、未知引用、章节状态、英文翻译编号、生成内容、Markdown 及 HTML 的本地链接和图片、本项目标题锚点、表格及 JSON/TOML 语法。它不访问远程网址，不核实上游功能，不执行 GPU。需要留存检查报告时加 `--report .local/checks/course.json`；只在显式指定时写报告。新增章节后也要检查人工维护的学习路线是否仍合理。`python -m scripts.docs.layout` 另行检查空目录、旧目录残留、未归类文件及公开结果的文件引用。

## 代码约定

- 保持学习入口小而清楚，公共依赖中不混入两种 GPU 的运行库。
- CPU 参考用于正确性教学，GPU 加速路径单独声明能力与版本。
- 不在公共示例中写入账户凭据、私有镜像认证信息或实际模型权重。
- 结果保存为结构化文件，测量值、模拟值和未运行模板分别标记。
- 单卡主线的改变同步更新课程说明和配置检查。

## 章节约定

按[分级目录规范](docs/maintenance/repository-layout.md)进入对应小节，完成后同步 `content=lesson` 与 `status=complete`，并保留验证证据。

使用 [章节模板](docs/maintenance/templates/lesson.md)，沿 [课程目录](docs/chapters/README.md) 递进。每章有基础和工程进阶内容，平台差异在相关位置说明。引用上游源码时固定版本，注明文件和符号。

## GitHub 协作

只暂存本次明确涉及的文件，Review `git diff --cached` 后提交。目录迁移需同时选择旧路径和新路径，不能漏掉删除项；不要把无关改动一并暂存。

Issue 提供运行问题、课程建议与平台实测三种表单。提交 PR 时说明最终行为、实际检查和课程/平台影响，采用仓库提供的 PR 模板。

平台证据只包含可公开的版本、输入、配置和结果；本地环境填写在 `.local/configs/`，个人运行输出保存在 `.local/runs/`。单节公开实验放入对应小节的 `results/`，跨章节检查放入 `reports/quality/`，平台环境记录放入 `reports/platforms/`。示例、输入数据也随所属小节存放，只有需要时才创建资源目录。

## 发布与许可证

项目原创内容采用 [CC BY-NC-SA 4.0](LICENSE)：署名、非商业性使用、相同方式共享。贡献者须有权按该许可提供所提交内容；第三方材料保留各自的许可与出处。

本地审阅与上传步骤见[GitHub 上传说明](docs/maintenance/publishing.md)。项目不宣称与 vLLM、Datawhale 或沐曦存在官方合作关系。
