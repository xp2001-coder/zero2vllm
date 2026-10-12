# 9.3 固定数据与协议

[正文](../README.md)

[corpus.json](corpus.json)含24条原创合成token序列：calibration16、validation4、test4；每个split包含general/code两种合成分布。它们不是真实文本语料，特征来自显式二维lookup，不经过Transformer。

保留ID：PAD=0、BOS=1、EOS=2，正文ID从3开始。PAD特征为[999,-999]用于检测mask错误；正常统计必须排除。真实数据可能有非零padding中间激活，这里只是放大现象。

默认规则：固定seed打乱候选，取样本前缀，为BOS/EOS预留空间后右截断正文；精确token预算对最后一条取前缀，可能截掉EOS；然后右padding。预算包含保留下来的BOS/EOS，不含PAD。Observer每次接收一条文档，统计粒度per-tensor。

`prepare_samples()`保存实际ID、各类计数与处理后token/mask摘要；实验入口再记录特征表摘要与前向路径。重复ID或跨split完全相同内容会被拒绝，但这不等于真实语料的近重复检测。

裁剪、重构和logits手算例在[实验入口](../../../../../scripts/experiments/lesson0903.py)中显式定义；它们与corpus的受控样本对照是不同实验，不能混用数量与质量结论。
