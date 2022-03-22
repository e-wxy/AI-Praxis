import torch
from torch import nn
import matplotlib.pyplot as plt

from utils import CAM, draw_cam


feature = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1),
    nn.BatchNorm2d(256),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(256, 512, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(512),
    nn.ReLU()
)


# CAM的适用情况：模型最后必须是一个全局平均池化层，接一个全连接层做分类
net = nn.Sequential(
    feature,
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(512, 3)
)

# X = torch.rand((1, 3, 224, 224))
# for layer in net:
#     X = layer(X)
#     print(layer.__class__.__name__, "\t\t output shape:\t", X.shape)

input_batch = torch.rand((10, 3, 224, 224))

# 取全局池化前的特征图输出
feature_output = net[0](input_batch).detach()

# 取最后一层全连接层的权重weight
weights = net[-1].weight.detach()

# 设置每个样本的类别，应该是一个一维向量，每个分量对应batch中的每张图的分类标签
class_idx = 1

cams = CAM(feature_output, weights, class_idx)

print(len(cams), cams[0].shape)
draw_cam(plt, cams[0])
plt.show()