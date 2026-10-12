# 2.2 实验配方

[model.json](model.json) 固定公开模型仓库、完整 revision、下载文件清单、三个输入和对齐容差。模型 revision 是 Hugging Face 仓库的 commit，与本课程 Git commit、Transformers 版本分别记录。

模型权重和 Tokenizer 文件下载到 `.local/models/smollm2-135m/`，不随课程提交。原模型采用 Apache-2.0；课程许可不改变模型许可。

本节生成的 `zero2vllm-snapshot.json` 记录下载文件的 SHA-256，用于离线复跑时检测修改；它不是第三方数字签名。公开[实测记录](../results/README.md)还保存了一份本轮实际下载的文件摘要。
