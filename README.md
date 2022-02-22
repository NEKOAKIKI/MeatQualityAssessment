# MeatQualityAssessment - 基于ResNet50的红肉新鲜度检测
> Meat Quality Assessment based on PaddlePaddle-ResNet50.
## 一、项目背景介绍
1. 选择[新鲜和过期红肉数据集](https://aistudio.baidu.com/aistudio/datasetdetail/84831)，完成**基于ResNet50的红肉新鲜度检测**项目。
2. 项目背景/意义：
食用过期肉类危害很大。肉类变质后微生物大量繁殖并产生代谢物，破坏人体的神经，导致肌肉僵硬麻痹，甚至引起急性中毒，出现恶心、呕吐、腹泻、腹痛、眩晕等症状，严重的还会昏迷，甚至会因为心力衰竭而死亡。如果长期食用，有毒物质会逐渐积累，从而造成慢性中毒。对大量肉品逐一进行化学物质检测耗时长、操作复杂，而通过计算机视觉的方法对肉品新鲜度进行检测，简化了检测流程，可以预防过期肉品流入市场引起消费者的健康问题。计算能力的提高、存储设备的发展，使得传统视觉技术中存在的问题逐渐得到改善或解决。我们可以将模型部署在移动设备或相关的硬件上来实现落地应用。
3. 项目拟使用的方法：
检测肉品新鲜或过期属于一个二分类问题。使用ResNet50对新鲜和过期红肉数据集进行训练，检测时输入一张大小为224 x 224 x 3的图片，输出一个list类型的变量，其每个元素为输入图片的预测结果。预测结果为dict类型，key为该图片分类结果label，value为该label（肉品新鲜/过期）对应的概率。
## 二、数据介绍
- 数据集名称：新鲜和过期红肉数据集
- 来源：[JavaRoom](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/89263)
- 数据集下载地址：[https://aistudio.baidu.com/aistudio/datasetdetail/84831](https://aistudio.baidu.com/aistudio/datasetdetail/84831)
- 图片数量：新鲜/过期两类各948张，共计1896张
- 图片分辨率：1280x720像素
新鲜：![新鲜](https://ai-studio-static-online.cdn.bcebos.com/1af15d5ec7ab43ebb31144ab7630864786a87e80335342e8a20b430b051ab31c)  
过期：![过期](https://ai-studio-static-online.cdn.bcebos.com/1c78609634e747d28f0e290db370ad543f046ad433904fb5aa587ac9cde3f4b1)  
## 三、模型介绍
### ResNet残差网络
**ResNet**是2015年ImageNet比赛的冠军，将识别错误率降低到了3.6%，这个结果甚至超出了正常人眼识别的精度。
为了解决经典神经网络层数增加后，训练误差不减反增的问题，Kaiming He等人提出了残差网络ResNet。其基本思想如下图所示：  
![](https://ai-studio-static-online.cdn.bcebos.com/e10f22f054704daabf4261ab46719629a36749631db74eb0a368499de3e5d3d6)  
- 图中(a)表示经典模型在增加网络的时候，将$x$映射成$y=F(x)$输出。
- 图中(b)表示ResNet对(a)进行了改进，输出$y=F(x)+x$。这时不是直接学习输出特征y的表示，而是学习$y-x$。
  - 如果想学习出原模型的表示，只需将$F(x)$的参数全部设置为0，则$y=x$是恒等映射。
  - $F(x)=y-x$也叫残差项，如果$x\to y$的映射接近恒等映射，(b)中通过学习残差项也比图(a)学习完整映射形式更加容易。  

上图(b)的结构是残差网络的基础，这种结构也叫做残差块（Residual block）。输入x通过跨层连接，能更快的向前传播数据，或者向后传播梯度。由于ResNet每层都存在直连的旁路，相当于每一层都和最终的损失有“直接对话”的机会，自然可以更好的解决梯度弥散的问题。残差块的具体设计方案如下图所示，这种设计方案也常称作瓶颈结构（BottleNeck）。1\*1的卷积核可以非常方便的调整中间层的通道数，在进入3\*3的卷积层之前减少通道数（256->64），经过该卷积层后再恢复通道数(64->256)，可以显著减少网络的参数量。这个结构（256->64->256）像一个中间细，两头粗的瓶颈，所以被称为“BottleNeck”。  
![322b26358d43401ba81546dd134a310cfb11ecafb3314aab88b5885ff642870b](https://user-images.githubusercontent.com/43382657/155121823-62a1c72a-db44-42d6-b6a1-bc7e83e04d8d.png)  
下图表示出了ResNet-50的结构，一共包含49层卷积和1层全连接，所以被称为ResNet-50。  
![8f42b3b5b7b34e45847a9c61580f1f8239a80ca6fa67448e8baeeb0209a2d556](https://user-images.githubusercontent.com/43382657/155123061-f5a2a3a5-c85b-4a0a-9447-41181c32b102.jpg)  
### Simple Momentum优化器
该优化器含有牛顿动量标志，更新公式如下：
```python
velocity=mu∗velocity+gradient
if(use_nesterov):
  param=param−(gradient+mu∗velocity)∗learning_rate
else:
  param=param−learning_rate∗velocity
```
