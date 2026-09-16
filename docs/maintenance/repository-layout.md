# 仓库目录规范

[项目首页](../../README.md) · [文档导航](../README.md)

## 一条归属规则

学习一节时，该节正文、实验、练习、示例、输入数据和公开结果在同一个目录。共享代码按功能复用，维护工具按用途分组，个人运行文件集中到本地区。

```text
zero2vllm/
├── docs/
│   ├── chapters/                 19 篇、126 节
│   │   └── part-02-engine/
│   │       └── 01-pytorch-model/
│   │           ├── README.md     正文
│   │           ├── lab.md        实验命令
│   │           ├── exercises.md  练习答案
│   │           ├── examples/     本节示例
│   │           └── results/      已验证结果
│   ├── platforms/                平台说明
│   ├── engineering/              架构与实验规范
│   ├── references/               参考资料入口
│   ├── maintenance/              维护说明和写作模板
│   └── assets/                   共享图片
├── metadata/
│   ├── course/parts/             按篇拆分的中文课程数据
│   ├── translations/en/parts/    对应英文大纲
│   ├── upstream/                 功能清单
│   └── knowledge/                来源台账
├── zero2vllm/                    共享 Python 实现
├── tests/{reference,modeling,integration,maintenance}/
├── scripts/{docs,experiments,environment}/
├── configs/{platforms,templates}/
├── reports/{quality,platforms,history}/
└── .local/{runs,checks,archive}/  本地区，不上传
```

有实际内容才创建 examples/、data/、results/；尚未完成的小节只保留明确的教学骨架，避免复制 126 套空文件夹。普通课文中直接包含练习时，不强行拆文件。

## 文件放哪里

| 文件类型 | 位置 |
| --- | --- |
| 小节正文、实验说明、答案 | docs/chapters/对应篇/对应节/ |
| 教学调用示例、固定输入 | 同节 examples/、data/ |
| 经过审阅的实验结果 | 同节 results/；整篇汇总可放篇目录 results/ |
| 中文标题、目标、进度 | metadata/course/parts/对应篇.json |
| 英文大纲 | metadata/translations/en/parts/对应篇.json |
| 可复用的数学/模型逻辑 | zero2vllm/reference/、zero2vllm/modeling/ |
| 自动验证 | tests/对应功能组/ |
| 环境模板、记录模板 | configs/platforms/、configs/templates/ |
| 个人输出、调试文件 | .local/runs/、.local/checks/ |
| 全仓库检查、历史编辑记录 | reports/quality/、reports/history/ |

## 更新一节

1. 在本节目录补正文、代码入口、命令、预期结果和练习。
2. 必要实现和测试放进功能模块，不复制共享模型。
3. 只修改该篇中文/英文 JSON；完整验证后标记 content=lesson 与 status=complete。
4. 生成导航，检查课程与目录，再跑相关测试。
5. Review 后明确选择对应文件暂存；移动文件同时选择旧路径删除和新路径新增。

```bash
python -m scripts.docs.course --write
python -m scripts.docs.course --check
python -m scripts.docs.layout
python -m unittest discover -s tests -v
```

## 运行产物

第一篇运行器默认按六节写入 .local/runs/part-01-inference/；2.1 默认写入 .local/runs/part-02-engine/01-pytorch-model/，内部区分 cases/ 与 checks/。公开 baseline 结果保留历史测量及原始源码指纹。

## 迁移与保护

本轮迁移保留了所有已有文件，原始字节先校验，再更新文本链接。历史整批结果中的文件引用随位置调整，测量值没有重算冒充历史。原 work/、runs/ 已迁入 .local/，旧空目录实际删除。目录检查会拒绝旧入口、根目录散落文件、空目录和断开的结果引用。
