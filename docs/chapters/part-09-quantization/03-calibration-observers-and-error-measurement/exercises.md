# 9.3 练习与参考答案

[正文](README.md) · [实验手册](lab.md)

## 1. 128 条样本足够描述校准吗

甲使用 128×512 个有效 token，乙使用 128×2048，但乙有 25% padding。比较两者有效 token 数；再指出至少四个仍未描述的变量。

<details><summary>参考答案</summary>

甲 65,536，乙 196,608（这里的2048包含padding）。乙是甲的3倍。还需数据ID/领域、tokenizer revision、特殊token、截断、拼接、seed、Observer、输入主干、统计mask等。相同样本数不等于相同预算。

</details>

## 2. token budget 与 EOS

一条已处理序列 `[BOS,a,b,c,EOS]`，剩余预算为3。本课会保留什么？若改成“永远保留EOS”，属于同一协议吗？

<details><summary>参考答案</summary>

保留 `[BOS,a,b]`。若改为 `[BOS,a,EOS]`，正文覆盖和终止符都变了，应另记策略。长度截断保留EOS与预算前缀截断是两个不同步骤。

</details>

## 3. padding 的 mask

真实激活 `[1,2]`，padding位置为999。只给模型传attention mask，却把全部中间激活送Observer，MinMax会得到什么？

<details><summary>参考答案</summary>

会看到最大值999。应在统计前按有效位置选取，得到最大值2。attention mask不必保证所有中间padding激活为零；统计mask和模型mask应分别核对。

</details>

## 4. 手算 EMA

新批权重α=.25，先观察[-1,1]，再[-4,8]，再[0,2]。求最终极值。若接口用“旧状态权重β”，应传多少？

<details><summary>参考答案</summary>

第二步[-1.75,2.75]；第三步min=.75×(-1.75)+.25×0=-1.3125，max=.75×2.75+.25×2=2.5625。β=.75。不能把α=.25直接复制给旧权重接口。

</details>

## 5. EMA 是否按 token 数加权

两个批次分别有1和100个token，普通EMA每批更新一次。第二批是否自动拥有100倍统计权重？

<details><summary>参考答案</summary>

不会。更新式混合的是每批极值，每次新批权重仍为α。更换batch大小和切分位置会改变结果；若需要元素数加权，必须定义不同更新法。

</details>

## 6. Histogram 不等于 KL

范围[-2,2]、4个bin，输入[-3,-2,-1,0,1,2,3]。写计数、超界数；为什么有直方图仍不能确定scale？

<details><summary>参考答案</summary>

左闭右开、末bin含右端点时计数[1,1,1,2]，下溢/上溢各1。直方图是分布表示，仍需选择裁剪目标、候选范围、量化范围和参数规则；可以服务MSE、分位数、KL等。

</details>

## 7. 裁剪误差不能丢

只有一个输入7，阈值T=.5，DQ输出.5。对原始target的平方误差是多少？若先clip target会怎样？

<details><summary>参考答案</summary>

(7-.5)²=42.25。先把target也裁成.5会报告0，漏掉全部裁剪损失。MSE搜索必须与原始有效值比较。

</details>

## 8. 分位数约定

绝对值[1,2,3,100]，nearest-rank p=.75的阈值？它与所有库的quantile结果一定一致吗？

<details><summary>参考答案</summary>

ceil(.75×4)=3，阈值3。插值分位数可能不同；也要区分绝对值单阈值与两侧分位点，不能只记录百分数。

</details>

## 9. 零向量指标

参考和候选都是零向量，MSE、NMSE、cosine如何报告？若参考[1,2]、候选[2,4]呢？

<details><summary>参考答案</summary>

全零：MSE=0，NMSE和cosine未定义，本课用null。后者MSE=2.5、NMSE=1、cosine=1。cosine不能判断整体幅度是否正确。

</details>

## 10. 换输入就换误差排名

正文W与候选A/B不变，把X从[8,0]改为[0,8]。比较输出误差。

<details><summary>参考答案</summary>

教师输出8；A输出8，平方误差0；B输出6，平方误差4。排名反转。权重误差未变，输入激发方向变了。

</details>

## 11. XTX 的形状

X有T个token、K个输入通道，W[K,N]。XTX是什么形状？若另一份实现写H=XXT，是否必然错误？

<details><summary>参考答案</summary>

本课X[T,K]，XTX为[K,K]，用于描述输入通道方向上的二阶结构。若另一实现把X存成[K,T]，XXT也是[K,K]。先核对记号；Hessian还可能有2和归一化因子，不能仅凭名字认同全部公式。

</details>

## 12. 重构输入与目标

h_FP=[4,1]、h_Q=[3,1]、教师输出5。候选A=[1,1]ᵀ、B=[1.25,1]ᵀ。计算两条输入路径的MSE，并说明缓存需要标注什么。

<details><summary>参考答案</summary>

FP输入：输出5/6，MSE0/1；量化上游输入：4/4.75，MSE1/.0625。应标缓存模块与位置、输入来自哪个主干、target来自哪个主干、dtype/mask以及是否包含累积误差。

</details>

## 13. 为什么 logits MSE 小仍会答错

参考二分类logits=[1,.875]，候选=[.875,1]。误差不大，为什么argmax改变？这说明应废弃MSE吗？

<details><summary>参考答案</summary>

样本在决策边界附近，两个分量交换就足以改变最大项。MSE仍是有用的局部诊断，但应与完整模型质量、任务指标一起报告，不承担它没测量的结论。

</details>

## 14. 聚合 NLL

batch A有2个有效目标、NLL总和2，B有8个有效目标、NLL总和16。总体PPL怎样求？平均两个batch的平均NLL为什么不同？

<details><summary>参考答案</summary>

总体平均NLL=(2+16)/(2+8)=1.8，PPL=exp(1.8)。直接平均batch均值会得到(1+2)/2=1.5，给短batch过大权重；平均PPL还额外受指数非线性影响。

</details>

## 15. 设计两个受控对照

准备比较128/512/1024条校准样本，以及WikiText/C4/领域数据。怎样避免把数据量和分布的影响混在一起？

<details><summary>参考答案</summary>

第一组固定一个数据池、预处理与seed，用嵌套样本前缀改变样本数，报告实际有效token。第二组固定有效token预算与量化配置，更换数据源，保持同一独立评测。再用多个预先规定seed评估波动；策略选择使用validation，最终test不用于持续调参。

</details>

## 16. 接入真实模型前还缺什么

本节20个案例都PASS，能否宣布Qwen3-8B量化质量通过？列出至少五项下一阶段证据。

<details><summary>参考答案</summary>

不能。还需模型/tokenizer revision、真实校准数据与协议、浮点质量基线、真实量化产物与参数、独立PPL/任务评测、目标后端加载和实际kernel证据。GPU速度、内存和服务指标还要分别实测。合成特征结果只证明本节参考流程与数值案例。

</details>
