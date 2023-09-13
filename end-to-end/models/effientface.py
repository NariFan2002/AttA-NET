
# -*- coding: utf-8 -*-

"""
这个文件中包含两个类：LocalFeatureExtractor局部特征提取器 和 InvertedResidual信道空间调制器 因此，该网络能够识别局部和全局的显著面部特征。
This code is modified from https://github.com/zengqunzhao/EfficientFace/blob/master/models/EfficientFace.py

本文提出了一种高效鲁棒的面部表情识别(FER)网络，命名为EfficientFace，它包含的参数更少，但在野外更准确和鲁棒。
首先，为了提高轻量级网络的鲁棒性，设计了局部特征提取器和信道空间调制器，其中采用了深度卷积;因此，该网络能够识别局部和全局的显著面部特征。
然后，考虑到大多数情绪以基本情绪的组合、混合或复合形式出现，我们引入了一种简单但有效的标签分布学习(LDL)方法作为一种新的训练策略。
在真实遮挡和位姿变化数据集上进行的实验表明，所提出的高效人脸在遮挡和位姿变化条件下具有鲁棒性。
此外，该方法在RAF-DB、CAER-S和AffectNet-7数据集上的准确率分别为88.36%、85.87%和63.70%，在AffectNet-8数据集上的准确率为59.89%。
RAF-DB:Real-world Affective Faces Database 是一个大规模的人脸表情数据库，包含从互联网上下载的约30000多种多样的人脸图像。基于众包标注，每幅图像都由约40名标注者独立标注。
        该数据库中的图像在受试者的年龄、性别和种族、头部姿势、光照条件、遮挡(例如眼镜、面部毛发或自遮挡)、后处理操作(例如各种滤镜和特效)等方面具有很大的差异性。RAF-DB种类繁多，数量庞大，注释丰富，
CAER:Context-Aware Emotion Recognition 老友记视频，上下文感知情感识别，包含超过13000个带注释的视频。可以使用CAER基准来训练用于情感识别的深度卷积神经网络。这些视频带有7种情感类别的扩展列表。
CAER-S：CAER- s数据集是通过从CAER (Lee et al. 2019)数据集中选择具有65983张图像的静态帧创建的，并被分为两个集:训练集(44996个样本)和测试集(20987个样本)。每张图片都有7种基本表情。
AffectNet：野外面部表情数据库AffectNet。AffectNet包含了从互联网上收集的100多万张面部图像，通过查询三大搜索引擎，使用六种不同语言的1250个情感相关关键词。
    AffectNet数据集包含大约45万张图像，这些图像由11个表情类别手动标注。AffectNet-7表示的7个表达式类别包含7个基本表达式，而AffectNet-8表示的8个表达式类别中增加了鄙视表达式。
    对于AffectNet-7，有283,901张图像作为训练数据，3500张图像作为测试数据;对于AffectNet-8，有287,568张图像作为训练数据，4000张图像作为测试数据。
代码和训练日志可以在https://github.com/zengqunzhao/EfficientFace上找到。
"""

import torch
import torch.nn as nn

def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
    return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x


class LocalFeatureExtractor(nn.Module):

    def __init__(self, inplanes, planes, index):
        super(LocalFeatureExtractor, self).__init__()
        self.index = index

        norm_layer = nn.BatchNorm2d
        self.relu = nn.ReLU()

        self.conv1_1 = depthwise_conv(inplanes, planes, kernel_size=3, stride=2, padding=1)
        self.bn1_1 = norm_layer(planes)
        self.conv1_2 = depthwise_conv(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn1_2 = norm_layer(planes)

        self.conv2_1 = depthwise_conv(inplanes, planes, kernel_size=3, stride=2, padding=1)
        self.bn2_1 = norm_layer(planes)
        self.conv2_2 = depthwise_conv(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn2_2 = norm_layer(planes)

        self.conv3_1 = depthwise_conv(inplanes, planes, kernel_size=3, stride=2, padding=1)
        self.bn3_1 = norm_layer(planes)
        self.conv3_2 = depthwise_conv(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn3_2 = norm_layer(planes)

        self.conv4_1 = depthwise_conv(inplanes, planes, kernel_size=3, stride=2, padding=1)
        self.bn4_1 = norm_layer(planes)
        self.conv4_2 = depthwise_conv(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn4_2 = norm_layer(planes)

    def forward(self, x):

        patch_11 = x[:, :, 0:28, 0:28]
        patch_21 = x[:, :, 28:56, 0:28]
        patch_12 = x[:, :, 0:28, 28:56]
        patch_22 = x[:, :, 28:56, 28:56]

        out_1 = self.conv1_1(patch_11)
        out_1 = self.bn1_1(out_1)
        out_1 = self.relu(out_1)
        out_1 = self.conv1_2(out_1)
        out_1 = self.bn1_2(out_1)
        out_1 = self.relu(out_1)

        out_2 = self.conv2_1(patch_21)
        out_2 = self.bn2_1(out_2)
        out_2 = self.relu(out_2)
        out_2 = self.conv2_2(out_2)
        out_2 = self.bn2_2(out_2)
        out_2 = self.relu(out_2)

        out_3 = self.conv3_1(patch_12)
        out_3 = self.bn3_1(out_3)
        out_3 = self.relu(out_3)
        out_3 = self.conv3_2(out_3)
        out_3 = self.bn3_2(out_3)
        out_3 = self.relu(out_3)

        out_4 = self.conv4_1(patch_22)
        out_4 = self.bn4_1(out_4)
        out_4 = self.relu(out_4)
        out_4 = self.conv4_2(out_4)
        out_4 = self.bn4_2(out_4)
        out_4 = self.relu(out_4)

        out1 = torch.cat([out_1, out_2], dim=2)
        out2 = torch.cat([out_3, out_4], dim=2)
        out = torch.cat([out1, out2], dim=3)

        return out


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride):
        super(InvertedResidual, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True))

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True))

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out
