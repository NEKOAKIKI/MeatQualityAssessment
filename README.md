# MeatQualityAssessment - 基于ResNet50的红肉新鲜度检测
> Meat Quality Assessment based on PaddlePaddle-ResNet50.
## 一、项目背景介绍
1. 选择[新鲜和过期红肉数据集](https://aistudio.baidu.com/aistudio/datasetdetail/84831)，完成**基于ResNet50的红肉新鲜度检测**项目。
2. 项目背景/意义：
食用过期肉类危害很大。肉类变质后微生物大量繁殖并产生代谢物，破坏人体的神经，导致肌肉僵硬麻痹，甚至引起急性中毒，出现恶心、呕吐、腹泻、腹痛、眩晕等症状，严重的还会昏迷，甚至会因为心力衰竭而死亡。如果长期食用，有毒物质会逐渐积累，从而造成慢性中毒。对大量肉品逐一进行化学物质检测耗时长、操作复杂，而通过计算机视觉的方法对肉品新鲜度进行检测，简化了检测流程，可以预防过期肉品流入市场引起消费者的健康问题。计算能力的提高、存储设备的发展，使得传统视觉技术中存在的问题逐渐得到改善或解决。我们可以将模型部署在相关的硬件上来实现落地应用。
3. 项目拟使用的方法：
检测肉品新鲜或过期属于一个二分类问题。使用ResNet50对新鲜和过期红肉数据集进行训练，检测时输入一张大小为224 x 224 x 3的图片，输出一个list类型的变量，其每个元素为输入图片的预测结果。预测结果为dict类型，key为该图片分类结果label，value为该label（肉品新鲜/过期）对应的概率。
## 二、数据介绍
- 数据集名称：新鲜和过期红肉数据集
- 来源：[JavaRoom](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/89263)
- 数据集下载地址：[https://aistudio.baidu.com/aistudio/datasetdetail/84831](https://aistudio.baidu.com/aistudio/datasetdetail/84831)
- 图片数量：新鲜/过期两类各948张，共计1896张
- 图片分辨率：1280x720像素
![新鲜](https://ai-studio-static-online.cdn.bcebos.com/1af15d5ec7ab43ebb31144ab7630864786a87e80335342e8a20b430b051ab31c)
![过期](https://ai-studio-static-online.cdn.bcebos.com/1c78609634e747d28f0e290db370ad543f046ad433904fb5aa587ac9cde3f4b1)
