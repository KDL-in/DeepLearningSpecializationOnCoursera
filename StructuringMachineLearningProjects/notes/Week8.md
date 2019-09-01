#  实践一

## 1  摘要
本周主要是在构建机器学习模型的过程中的处理问题的思考方式和构建整个项目内容的思考方式，如dev/test集，策略，算法调整。相对于实践理论，从整体上对项目进行设置和调整的思考方法。

这一门课就是炼丹参考书，出问题来翻阅。

## 2  正交思想（Orthogonalization）

这是所有理论的基础，大致上就是那个点出问题就调哪个点的开关，尽可能不要影响到其他东西。即，让处理对象之间正交，这对于巨大的系统是很有意义的。

## 3  模型分析
### 3.1  从train/dev/test/real world的表现来调整
| 数据集合（表现不佳）   | 处理方式       |
| :--------------------- | -------------- |
| train set              | Bigger network |
|                        | Adam等优化算法 |
| dev set                | Regulation     |
|                        | Big train set  |
| test set               | Big dev        |
| real world（实际表现） | change dev set |
|                        | cost function  |

这个表格将会贯穿整一周的内容。非常重要，下面讲了之后，可以不时回过头来看看，问问处理方式，是为什么。

### 3.2  Bias、Variance分析
这个主要是train、dev、test出问题的时候的处理方式老容易混淆，搞不清哪跟哪。其实很简单，只要记住，Bias一般出现在train中，一般指训练结果和实际情况的对比，也就是低拟合。Variance出现在其他数据集，指过拟合。

低拟合怎么办？查看表格，一般使用Bigger network能很好提高拟合，Adam等优化算法可以减少振荡，使得在有限迭代次数内尽可能找到最小值，提高拟合。

高拟合怎么办？一般只有两种方式，dev set中表现不好，出现高拟合，那么可以优先考虑Regulation，原因是train set 对于数据的拟合太过了，用Regulation可以降低它的拟合程度。另一种方式是通用的，增大训练数据，因为原来的数据它训练过度，所以dev表现不好，说明train过拟合，增大train的数据，test表现不好，说明dev过拟合，增大dev的数据。

### 3.3  human-level 人类基本性能表现
到底什么表现才是天花板？一般定义为贝叶斯偏差，指的就是数据本身出现的问题，这是最低最低的偏差。但是这个偏差无法衡量。所以一般来说，很常用用人类实际表现来近似估算贝叶斯偏差——人类在识别能力上可谓非常优秀。

所以上表所说的真实世界的表现，一般就是人类表现。这是一个重要的衡量方法。

## 4  最佳实践
这里主要是很多的实践经验，很有用。

### 4.1  train/test/dev 的设置
**不变的目标，真实的反映**：

很多时候，train的过程非常漫长，可以几个星期几个月，所以像这种前期准备之前，必须保证两件事。第一是，你训练的目标、验证集的拟合目标和最终测试的目标是一样的，否则几个月训练等同白费。第二是，最终应用需要什么数据，就收集什么数据。

**test/dev的分布应该一致**

数据分布一致，确保你优化的目标和最终测试的目标一致，保证最终你能进行bias、variance分析，这个也是非常非常重要的。

### 4.2  单一评估标准
有很多个指标怎么办？如果都很重要，那么我们需要变成单一指标，具体而言，可以求平均，可以加权等等。

### 4.3  满足策略和优化策略
一个问题可能会有很多的策略需要追求，但是如同上面转化为单一策略一样，太多策略对于模型构建是个大问题。于是这里的方法就是，选择一个最重要的作为优化策略，这个来做最小优化，其他策略选出来作为满足策略，考虑只要满足一定条件即可。

### 4.4  模型临时变更
很多时候，会出现新需求或者测试之后发现新问题。比如课程中举例，分类猫的同时出现了很多不雅图片。那么调整办法是，重新设计评估标准——即损失函数，不用去改数据集，只需把这些图片找出来，加标签，损失函数中给他们一个权重，就可以了。这就是最开始表格最后一项，改变cost function指的内容。

变更数据集是一件大事，尽量不要做。但什么时候该做？可能只有发现实际目标变更，和数据集中的不同的时候吧，这时候就完全没办法了，只能重新调整重新训练。