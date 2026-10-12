# 上传 GitHub 与仓库设置

直接使用本地 `zero2vllm/` 工程目录。`README.md` 与 `.github/`、`docs/`、`zero2vllm/`、`pyproject.toml` 同级；里面的 `zero2vllm/` 子目录是 Python 包。

仓库已连接 GitHub。日常更新先完成本地 review，再按文件提交和推送；创建仓库的说明仅供首次建立或迁移时参考。

本地资料、个人输出与临时文件统一放入已忽略的 `.local/`；原整理资料保存在 `.local/archive/editorial/`。可发布正文及单节资源位于 `docs/chapters/` 对应的篇与节目录，运行课程无需原始资料 ZIP。上传前优先 review [目录规范](repository-layout.md)与[验证入口](../../reports/README.md)。

## 1. 创建仓库

在自己的 GitHub 账号或组织下新建仓库：

| 设置 | 建议填写 |
| --- | --- |
| Repository name | `zero2vllm` |
| Description | 从零学习 vLLM：推理原理、量化算子与系统优化，面向单卡 NVIDIA A100 和沐曦 C500 的双后端课程。 |
| Visibility | 面向社区教学时选择 Public；准备阶段可使用 Private |
| README / .gitignore / License 初始化 | 创建空仓库；上述文件均已包含在工程中，上传项目自己的 LICENSE |
| 默认分支 | `main` |

当前版本号为 `0.1.0.dev0`。首批材料仍处于建设阶段，实际检查与 GPU 状态见[验证入口](../../reports/README.md)。发布说明采用 [CHANGELOG](../../CHANGELOG.md) 的 Unreleased 内容。

## 2. 本地 review

先检查[中文 README](../../README.md)、[English README](../../README_en.md)、[课程目录](../chapters/README.md)、[学习路线](../chapters/learning-paths.md)和 [LICENSE](../../LICENSE)，确认课程范围、当前进度和许可说明。中英文切换和横幅均已使用本地相对链接，英文首页注明章节正文的翻译范围。

在工程根目录检查：

```bash
python -m unittest discover -s tests -v
python -m scripts.docs.course --check
python -m scripts.docs.layout
```

## 3. Review 后按文件提交

本项目已初始化并连接到 GitHub，日常更新无需重新创建仓库或重新设置远端。在 Git Bash 中先看变更：

```bash
git status --short
git diff --stat
git diff
```

只暂存这次要发布的明确文件。例如仅改了 2.1 正文和实验页时：

```bash
git add -- docs/chapters/part-02-engine/01-pytorch-model/README.md docs/chapters/part-02-engine/01-pytorch-model/lab.md
git diff --cached --stat
git diff --cached
```

上面只是两个文件的示例，不是本次整个目录迁移的提交清单。若更新了课程状态，还应逐个选入课程数据及生成的中英文首页、课程总表、篇目录；若更新代码，还应选入对应实现、测试和公开证据。

移动文件要同时暂存**旧路径删除与新路径新增**，Git 会根据内容识别重命名。通过 `git status --short` 核对实际旧、新路径，逐项传给 `git add --`。不要对仓库整体执行暂存，也不要遗漏迁移的删除项。个人 `.local/`、虚拟环境 `.venv/` 和模型文件保持忽略。

确认暂存区只包含本次预期内容后，提交并推送：

```bash
git commit -m "docs: update reviewed course materials"
git push origin main
```

提交说明应按实际改动填写。本轮目录重组建议独立成一次易于审阅的提交，后续章节按节提交。远端若已有更新，先检查并正常同步，不使用强制推送覆盖他人内容。

## 4. 上传后确认

- 仓库首页正常显示课程 README，文档导航和相对链接可打开。
- Actions 中出现 `CPU tests and course checks`；工作流运行成功后，再将结果标为远程 CI 通过。
- 新建 Issue 时出现运行问题、课程建议、平台实测三种表单；它们位于[模板目录](../../.github/ISSUE_TEMPLATE)。
- PR 默认正文来自[PR 模板](../../.github/pull_request_template.md)。
- `.local/`、`.venv/`、`models/` 和 `.env` 文件不在提交中；单节公开实验在该节 `results/`，跨章节验证在 `reports/quality/`。

Issue 表单与 PR 模板采用 GitHub 支持的目录约定；合入默认分支后供贡献者使用。[Issue 表单规范](https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests/syntax-for-issue-forms) · [PR 模板规范](https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests/creating-a-pull-request-template-for-your-repository)

## 5. About 与后续维护

About 可使用本页建议的 Description；Topics 可选：`vllm`、`llm-inference`、`quantization`、`cuda`、`metax`、`gpu`、`tutorial`、`chinese`。当前没有独立文档站，Website 可留空。

课程导航使用仓库内相对链接；工作流没有绑定个人账号。两份 README 的 Stars、Forks 和项目入口已指向 [xp2001-coder/zero2vllm](https://github.com/xp2001-coder/zero2vllm)。仓库转移或改名时同步更新这些徽章地址。当前不显示尚未开通的在线文档站入口。

章节与特性清单修改后，按[贡献说明](../../CONTRIBUTING.md)生成并检查文档。核心命令为：

```bash
python -m unittest discover -s tests -v
python -m zero2vllm demo
python -m scripts.docs.course --check
python -m scripts.docs.layout
```

## 6. 许可证状态

采用 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)，许可正文保存在 [LICENSE](../../LICENSE)，包元数据标识为 `CC-BY-NC-SA-4.0`。上传时保留许可文件。

遵守署名、非商业和相同方式共享条件即可转载及改编，无需逐次申请同意。第三方材料保留各自的来源与许可。
