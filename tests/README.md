# 测试分组

| 目录 | 范围 |
| --- | --- |
| reference/ | 数学参考、缓存、第一篇与量化 |
| modeling/ | 可选 PyTorch 模型 |
| integration/ | 命令行与平台配置 |
| maintenance/ | 课程生成、数据分片、目录规范 |

全套检查：`python -m unittest discover -s tests -v`。
2.1：`python -m unittest tests.reference.test_decoder tests.modeling.test_decoder -v`。
没有安装 PyTorch 时相关测试明确跳过；这不能作为 PyTorch 验证通过的证据。
