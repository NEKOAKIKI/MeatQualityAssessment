# 在同级目录下解压所挂载的数据集
get_ipython().system(' unzip -oq data/data84831/Meat_Quality_Assessment_Dataset.zip -d data/data84831')
# 删除压缩包
get_ipython().system(' rm -r data/data84831/Meat_Quality_Assessment_Dataset.zip')
get_ipython().system(' rm -r data/data84831/README.txt')
get_ipython().system(' rm -r data/data84831/license.txt')
# 查看数据集的目录结构
get_ipython().system(' tree data/data84831 -d')


import os
import random
import cv2
import matplotlib.pyplot as plt
import glob
import numpy as np
import paddle
import paddle.vision.transforms as T
from paddle.vision.models import resnet50
from paddle.metric import Accuracy
from PIL import Image


### 处理数据
# 输入数据集路径，生成记录训练集和测试集的txt文件
file_dir = "data"
data_list = []

if(os.path.exists('data/train.txt')):  # 判断有误文件
    os.remove('data/train.txt')  # 删除文件
if(os.path.exists('data/validation.txt')):
    os.remove('data/validation.txt')

for i in os.listdir(file_dir):
    class_id = 0
    path = os.path.join(file_dir, i)
    if os.path.isdir(path): 
        for j in os.listdir(path):
            class_id += 1
            for k in os.listdir(os.path.join(path, j)):
                s = os.path.join(path, j, k) + " " + str(class_id - 1)
                # print(s)
                data_list.append(s)

random.shuffle(data_list)
print(data_list[0])

data_len = len(data_list)
count = 0

for data in data_list:
    if count <= data_len*0.2:
        with open('data/validation.txt', 'a')as f:
            f.write(data + '\n')
            count += 1
    else:
        with open('data/train.txt', 'a')as tf:  # 80%写入训练集
            tf.write(data + '\n')
            count += 1

# 语义分割数据集抽样可视化
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
get_ipython().run_line_magic('matplotlib', 'inline')

# 读取数据集中的图片
image_path_list = []
f = open(f'data/train.txt')
for line in f:
    temp = line.split(' ')
    image_path_list.append(temp[0])

test_list = ['data/data84831/Fresh/test_20171016_120921D.jpg', 'data/data84831/Spoiled/test_20171017_210921D.jpg']

plt.figure(figsize=(8, 8))
for i in range(len(test_list)):
    plt.subplot(len(test_list), 2, i + 1)
    plt.title(test_list[i])
    plt.imshow(cv2.imread(test_list[i])[:, :, ::-1])
    
plt.show()


# 计算图像数据整体均值和方差
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


# 数据集类的定义
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


# 数据集类的测试
train_dataset = MeatDataset(mode='train')
val_dataset = MeatDataset(mode='validation')
print(len(train_dataset))
print(len(val_dataset))

image, label = train_dataset[0]
print(image.shape, label.shape)

for image, label in train_dataset:
    print(image.shape, label.shape)
    break

train_dataloader = paddle.io.DataLoader(
    train_dataset,
    batch_size = 128,
    shuffle = True,
    drop_last = False)

for step, data in enumerate(train_dataloader):
    image, label = data
    print(step, image.shape, label.shape)


### 模型准备与可视化
# build model and visualize
model = resnet50()
paddle.summary(model, (1,3, 224, 224))

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


### 模型训练
# train prepare
model.fit(
    train_dataset,
    epochs=10,
    batch_size=128,
    verbose=1
    )


### 测试模型
# test prepare
model.evaluate(val_dataset, batch_size=128, verbose=1)

