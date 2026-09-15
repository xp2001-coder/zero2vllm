# 上传 GitHub 与仓库设置

直接使用本地 `zero2vllm/` 工程目录。`README.md` 与 `.github/`、`docs/`、`zero2vllm/`、`pyproject.toml` 同级；里面的 `zero2vllm/` 子目录是 Python 包。

先完成本地 review，再创建远程仓库和推送。

当前根目录的 `vllm/` 是本地参考知识库，默认由 `.gitignore` 排除；`work/` 也不提交。可发布的课程正文已整理到 `docs/chapters/` 和 `docs/labs/`，不需要这些 ZIP 才能阅读或运行 CPU 实验。上传前优先 review [本轮更新](../reports/knowledge-update.md)。

## 1. 创建仓库

在自己的 GitHub 账号或组织下新建仓库：

| 设置 | 建议填写 |
| --- | --- |
| Repository name | `zero2vllm` |
| Description | 从零学习 vLLM：推理原理、量化算子与系统优化，面向单卡 NVIDIA A100 和沐曦 C500 的双后端课程。 |
| Visibility | 面向社区教学时选择 Public；准备阶段可使用 Private |
| README / .gitignore / License 初始化 | 创建空仓库；上述文件均已包含在工程中，上传项目自己的 LICENSE |
| 默认分支 | `main` |

当前版本号为 `0.1.0.dev0`。首批材料仍处于建设阶段，GPU 状态见[验证记录](../reports/validation.md)。发布说明采用 [CHANGELOG](../CHANGELOG.md) 的 Unreleased 内容。

## 2. 本地 review

先检查[中文 README](../README.md)、[English README](../README_en.md)、[课程目录](curriculum.md)、[学习路线](learning-paths.md)和 [LICENSE](../LICENSE)，确认课程范围、当前进度和许可说明。中英文切换和横幅均已使用本地相对链接，英文首页注明章节正文的翻译范围。

在工程根目录检查：

```bash
python -m unittest discover -s tests -v
python scripts/course_docs.py --check
```

## 3. 审阅后使用 Git 推送

适用于新建的空仓库。先安装 Git 并配置自己的提交身份、完成 GitHub 身份验证；在本地含 `README.md` 的工程根目录执行：

```bash
git init -b main
git add .
git status --short
git diff --cached
git commit -m "chore: initialize zero2vllm course repository"
```

从 GitHub 仓库页面复制实际远程地址，将下一条命令中的 `REMOTE-URL` 替换为该地址：

```bash
git remote add origin REMOTE-URL
git remote -v
git push -u origin main
```

这些命令只适用于本地尚未初始化 Git、远程尚无提交的首次导入。若已有远程内容，先克隆该仓库并把工程文件合入，再正常提交。[GitHub 官方本地仓库导入说明](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github)

## 4. 上传后确认

- 仓库首页正常显示课程 README，文档导航和相对链接可打开。
- Actions 中出现 `CPU tests and course checks`；工作流运行成功后，再将结果标为远程 CI 通过。
- 新建 Issue 时出现运行问题、课程建议、平台实测三种表单；它们位于[模板目录](../.github/ISSUE_TEMPLATE)。
- PR 默认正文来自[PR 模板](../.github/pull_request_template.md)。
- `vllm/`、`work/`、`models/`、`runs/`、`configs/local/` 和 `.env` 文件不在提交中；`reports/` 保留经过整理的公开实验记录。

Issue 表单与 PR 模板采用 GitHub 支持的目录约定；合入默认分支后供贡献者使用。[Issue 表单规范](https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests/syntax-for-issue-forms) · [PR 模板规范](https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests/creating-a-pull-request-template-for-your-repository)

## 5. About 与后续维护

About 可使用本页建议的 Description；Topics 可选：`vllm`、`llm-inference`、`quantization`、`cuda`、`metax`、`gpu`、`tutorial`、`chinese`。当前没有独立文档站，Website 可留空。

课程导航使用仓库内相对链接；工作流没有绑定个人账号。两份 README 的 Stars、Forks 和项目入口已指向 [xpxpxp2001xpxpxp/zero2vllm](https://github.com/xpxpxp2001xpxpxp/zero2vllm)。仓库转移或改名时同步更新这些徽章地址。当前不显示尚未开通的在线文档站入口。

章节与特性清单修改后，按[贡献说明](../CONTRIBUTING.md)生成并检查文档。核心命令为：

```bash
python -m unittest discover -s tests -v
python -m zero2vllm demo
python scripts/course_docs.py --check
```

## 6. 许可证状态

采用 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)，许可正文保存在 [LICENSE](../LICENSE)，包元数据标识为 `CC-BY-NC-SA-4.0`。上传时保留许可文件。

遵守署名、非商业和相同方式共享条件即可转载及改编，无需逐次申请同意。第三方材料保留各自的来源与许可。
