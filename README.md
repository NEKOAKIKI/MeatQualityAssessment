# 【AI达人创造营第二期】MeatQualityAssessment - 基于ResNet50的红肉新鲜度检测

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

> References: 
> - [PaddlePaddle教程 - 零基础实践深度学习/第三章：计算机视觉（下）/ResNet](https://www.paddlepaddle.org.cn/tutorials/projectdetail/3106582);
> - [API文档 - paddle.optimizer/Momentum](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/optimizer/Momentum_cn.html)

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
velocity = mu * velocity + gradient
if(use_nesterov):
  param = param − (gradient + mu * velocity) * learning_rate
else:
  param = param − learning_rate * velocity
```

## 四、模型训练

### 1. 处理数据

#### 解压数据集并查看目录结构

```bash
# 在同级目录下解压所挂载的数据集
! unzip -oq data/data84831/Meat_Quality_Assessment_Dataset.zip -d data/data84831
# 删除压缩包
! rm -r data/data84831/Meat_Quality_Assessment_Dataset.zip
! rm -r data/data84831/README.txt
! rm -r data/data84831/license.txt
# 查看数据集的目录结构
! tree data/data84831 -d
```

输出：

```
data/data84831
├── Fresh
└── Spoiled

2 directories
```

#### 输入数据集路径，生成记录训练集和测试集的txt文件

```python
# 输入数据集路径，生成记录训练集和测试集的txt文件
file_dir = "data"
data_list = []

if(os.path.exists('data/train.txt')):  # 判断有误文件
    os.remove('data/train.txt')  # 删除文件
if(os.path.exists('data/validation.txt')):
    os.remove('data/validation.txt')

# 遍历数据集图片路径
for i in os.listdir(file_dir):
    class_id = 0
    path = os.path.join(file_dir, i)
    if os.path.isdir(path): 
        for j in os.listdir(path):
            class_id += 1
            for k in os.listdir(os.path.join(path, j)):
                s = os.path.join(path, j, k) + " " + str(class_id - 1)
                data_list.append(s)

random.shuffle(data_list)  # 乱序

data_len = len(data_list)
count = 0

# 生成记录训练集和测试集的txt文件
for data in data_list:
    if count <= data_len*0.2:
        with open('data/validation.txt', 'a')as f:  # 20%写入测试集
            f.write(data + '\n')
            count += 1
    else:
        with open('data/train.txt', 'a')as tf:  # 80%写入训练集
            tf.write(data + '\n')
            count += 1
```

格式：

```python
data/data84831/Spoiled/test_20171018_120721D.jpg 1  # 图片路径 分类
```

#### 语义分割数据集

```python
# 读取数据集中的图片
image_path_list = []
f = open(f'data/train.txt')
for line in f:
    temp = line.split(' ')
    image_path_list.append(temp[0])
```

#### 计算图像数据整体均值和方差

```python
def get_mean_std(image_path_list):
    print('Total images:', len(image_path_list))
    max_val, min_val = np.zeros(3), np.ones(3) * 255
    mean, std = np.zeros(3), np.zeros(3)
    for image_path in image_path_list:
        image = cv2.imread(image_path)
        for c in range(3):
            mean[c] += image[:, :, c].mean()
            std[c] += image[:, :, c].std()
            max_val[c] = max(max_val[c], image[:, :, c].max())
            min_val[c] = min(min_val[c], image[:, :, c].min())

    mean /= len(image_path_list)
    std /= len(image_path_list)

    mean /= max_val - min_val
    std /= max_val - min_val

    return mean, std


mean, std = get_mean_std(image_path_list)
print('mean:', mean)
print('std:', std)
```

输出：

```python
Total images: 1516
mean: [0.4498888  0.41782444 0.57211235]
std: [0.20592395 0.21688003 0.16523887]
```

#### 数据集的定义

```python
class MeatDataset(paddle.io.Dataset):
    def __init__(self, mode='train'):
        """
        初始化函数
        """
        self.data = []
        with open(f'data/{mode}.txt') as f:
            for line in f.readlines():
                info = line.strip().split(' ')
                if len(info) > 0:
                    self.data.append([info[0].strip(), info[1].strip()])
        self.transform = T.Compose([
            T.Resize(size=(224, 224)),
            T.ToTensor(),
            T.Normalize(mean=127.5, std=127.5)])


    def __getitem__(self, index):
        """
        读取图片，对图片进行归一化处理，返回图片和 标签
        """
        image_file, label = self.data[index]  # 获取数据
        img = Image.open(image_file)  # 读取图片
        img = img.convert('RGB')
        img = img.resize((224, 224), Image.ANTIALIAS)  # 图片大小样式归一化
        img = np.array(img).astype('float32')  # 转换成数组类型浮点型32位
        img = img.transpose((2, 0, 1))     #读出来的图像是rgb,rgb,rbg..., 转置为 rrr...,ggg...,bbb...
        img = img / 255.0  # 数据缩放到0-1的范围
        # label = np.random.randint(low=0, high=self.num_classes, size=(1,))
        return img, np.array(label, dtype='int64')


    def __len__(self):
        """
        获取样本总数
        """
        return len(self.data)
```

#### 数据集类的测试

```python
# 实例化训练集、测试集，并输出图片数量
train_dataset = MeatDataset(mode='train')
val_dataset = MeatDataset(mode='validation')
print(len(train_dataset))
print(len(val_dataset))

image, label = train_dataset[0]
print(image.shape, label.shape)

for image, label in train_dataset:
    print(image.shape, label.shape)
    break
```

```python
# 加载训练集
train_dataloader = paddle.io.DataLoader(
    train_dataset,
    batch_size = 128,
    shuffle = True,
    drop_last = False)

for step, data in enumerate(train_dataloader):
    image, label = data
    print(step, image.shape, label.shape)
```

### 2. 训练模型

#### 模型准备与可视化

```python
# 模型准备
from paddle.vision.models import resnet50
from paddle.metric import Accuracy

# 实例化模型
model = resnet50()
paddle.summary(model, (1,3, 224, 224))  # 可视化

model = paddle.Model(model)
optim = paddle.optimizer.Momentum(
    learning_rate=0.001, 
    momentum=0.9, 
    parameters=model.parameters(), 
    weight_decay=0.001)

model.prepare(
    optimizer=optim,
    loss=paddle.nn.CrossEntropyLoss(),
    metrics=Accuracy()
    )
```

输出过长，这里只展示部分图片：

![](https://ai-studio-static-online.cdn.bcebos.com/65bd3e44e19140a68e3dd88dd02e6e5277a347f7b1a74822b17389bea3072e32)

#### 模型训练

```python
# train prepare
model.fit(
    train_dataset,
    epochs=10,
    batch_size=128,
    verbose=1
    )
```

输出：

![](https://ai-studio-static-online.cdn.bcebos.com/6e5c4b3179b54a9fa5cc984a410e1caeae366e1b7ea94cffa648eb52301cc389)

如图所示，经过10个epochs的训练，loss可以达到0.02左右，acc可以达到99.54%。

## 五、模型评估

```python
# 使用测试集对模型进行评估
model.evaluate(val_dataset, batch_size=128, verbose=1)
```

输出：

```python
Eval begin...
step 3/3 [==============================] - loss: 0.0331 - acc: 0.9658 - 3s/step
Eval samples: 380
{'loss': [0.033065766], 'acc': 0.9657894736842105}
```

经过测试，训练后模型精确度可达96%。

## 六、总结与升华

- 亮点：
  - 使用了ResNet50网络进行训练，训练后的模型精确度较高；
  - 调用飞桨API库，开发周期较短；
  - 本项目可以部署在带有摄像头的固定硬件上，流水线；也可以部署在移动端app或小程序上，作为轻便的便民检测工具。
- 不足：
  - 本项目为本人初次独立完成的深度学习相关的项目，创新性方面有所欠缺，日后会继续学习并对本项目进行优化与完善；
  - 时间有限，未完成本项目的部署工作。

## 七、个人总结

- 作者：储氢合金M.H.
- 山西农业大学本科生
- 兴趣方向：计算机视觉，学习视觉SLAM与深度学习结合ing
- 个人主页：
  - GitHub: https://github.com/NEKOAKIKI
  - 飞桨AI Studio: https://aistudio.baidu.com/aistudio/personalcenter/thirdview/771061

