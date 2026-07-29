# -*- coding: utf-8 -*-
"""
轻量化孪生卷积神经网络变化检测模型定义

【项目背景】
这是变化检测(Change Detection)项目的核心模型文件。
使用孪生网络(Siamese Network)架构来对比两张图片并检测变化区域。

【孪生网络】
孪生网络是一种特殊的神经网络架构,它包含两个或多个结构完全相同、权重共享的子网络。
在本项目中:
- 两个子网络分别处理变化前图像(A)和变化后图像(B)
- 两个子网络共享权重,即使用同一组参数
- 通过对比两个子网络的输出特征来检测变化

【孪生网络作用】
1. 权重共享:减少模型参数量,提高训练效率
2. 特征对齐:确保两张图像提取的特征在同一个语义空间
3. 自然融合:通过特征相减的方式,自动得到差异信息
4. 效果更好:相比分别独立处理,孪生网络更适合对比任务

【模型架构详解】
本模型包含四个主要模块:

1. **共享编码器(Siamese Encoder)**
   - 两个完全相同的编码器,分别处理 A 和 B 图像
   - 每个编码器包含 4 层卷积块: Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d
   - 通道数变化: 3 -> 32 -> 64 -> 128 -> 256
   - 空间尺寸变化: 1024 -> 512 -> 256 -> 128 -> 64
   - 作用:提取图像的深层特征表示

2. **差异融合模块(Fusion Module)**
   - 计算两个编码器输出的差异: diff = f_A - f_B
   - 通过 2 层 Conv-BN-ReLU 进一步融合差异特征
   - 作用:将差异信息转换为更抽象的特征表示

3. **转置卷积解码器(Decoder)**
   - 4 层转置卷积(ConvTranspose2d),逐步还原空间分辨率
   - 空间尺寸变化: 64 -> 128 -> 256 -> 512 -> 1024
   - 通道数变化: 256 -> 128 -> 64 -> 32 -> 16
   - 作用:将抽象特征解码为像素级的变化概率图

4. **输出层(Output Head)**
   - 1x1 卷积,输出单通道 logit(未经过 sigmoid 的原始输出)
   - 训练时使用 BCEWithLogitsLoss(内部包含 sigmoid)
   - 推理时需要手动 sigmoid 得到 0-1 之间的概率

【设计约束】
仅使用基础层:
- Conv2d: 卷积层
- BatchNorm2d: 批归一化层
- ReLU: 激活函数
- MaxPool2d: 最大池化层
- ConvTranspose2d: 转置卷积层

【输入输出】
- 输入: 两张 1024x1024 RGB 图像,形状 (B, 3, 1024, 1024)
- 输出: 单通道 logit,形状 (B, 1, 1024, 1024)
- 推理时 sigmoid 后得到 0-1 概率,阈值化后得到 0/1 二值掩码

"""
import torch
import torch.nn as nn


class ConvBNReLU(nn.Module):
    """
    卷积+批归一化+ReLU 基础单元

    这是一个常用的卷积神经网络基础模块,将三个常用层封装在一起。
    结构: Conv2d(3x3) -> BatchNorm2d -> ReLU

    【功能说明】
    1. Conv2d(3x3, padding=1): 3x3 卷积,padding=1 保持特征图尺寸不变
    2. BatchNorm2d: 批归一化,加速训练,提高稳定性
    3. ReLU: 激活函数,引入非线性

    【参数说明】
    in_ch (int): 输入通道数
    out_ch (int): 输出通道数

    【使用场景】
    - 编码器中的特征提取
    - 解码器中的特征融合
    - 差异融合模块中的特征处理
    """

    def __init__(self, in_ch, out_ch):
        """
        初始化 ConvBNReLU 模块

        【参数说明】
        in_ch (int): 输入特征图的通道数
        out_ch (int): 输出特征图的通道数(卷积核数量)

        【结构说明】
        1. Conv2d:
           - kernel_size=3: 3x3 卷积核
           - padding=1: 边缘填充 1 像素,保持特征图尺寸不变
           - 无 bias: 后接 BatchNorm,可省略 bias
        2. BatchNorm2d:
           - 对输出通道进行批归一化
           - 训练时使用 batch 统计量,推理时使用全局统计量
        3. ReLU:
           - inplace=True: 节省内存,直接在原张量上修改
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),  # 3x3 卷积
            nn.BatchNorm2d(out_ch),                               # 批归一化
            nn.ReLU(inplace=True),                                # ReLU 激活
        )

    def forward(self, x):
        """
        前向传播

        【参数说明】
        x (torch.Tensor): 输入特征图,形状 (B, in_ch, H, W)

        【返回值】
        torch.Tensor: 输出特征图,形状 (B, out_ch, H, W)
                       注意:空间尺寸 HxW 保持不变(因为 padding=1)
        """
        return self.block(x)


class SiameseChangeNet(nn.Module):
    """
    轻量化 Siamese 孪生卷积变化检测网络

    【核心思想】
    使用孪生网络架构,通过共享权重的编码器分别提取两张图像的特征,
    然后通过特征相减得到差异信息,最后解码为像素级的变化掩码。

    【架构详解】
    1. **孪生编码器**: 两个权重共享的编码器,分别处理图像 A 和 B
       - 输入: (B, 3, 1024, 1024)
       - 4 个编码块,每个包含: Conv-BN-ReLU-MaxPool
       - 输出: (B, 256, 64, 64)

    2. **差异融合**: 计算特征差异并进一步提取
       - diff = f_A - f_B (逐通道相减)
       - 2 层 Conv-BN-ReLU 融合差异特征
       - 输出: (B, 256, 64, 64)

    3. **转置卷积解码器**: 逐步还原空间分辨率
       - 4 层转置卷积,每层 stride=2,尺寸翻倍
       - 通道: 256 -> 128 -> 64 -> 32 -> 16
       - 空间: 64 -> 128 -> 256 -> 512 -> 1024

    4. **输出层**: 1x1 卷积输出单通道 logit
       - 输出: (B, 1, 1024, 1024)
       - 训练时使用 BCEWithLogitsLoss
       - 推理时需要 sigmoid 得到概率
    """

    def __init__(self):
        """
        初始化 SiameseChangeNet 模型

        【初始化流程】
        1. 创建编码器: 4 层卷积块(含池化)
        2. 创建差异融合模块: 2 层 ConvBNReLU
        3. 创建解码器: 4 层转置卷积
        4. 创建输出层: 1x1 卷积
        """
        super().__init__()

        # ===== 1. 共享特征提取编码器 =====
        # 两个编码器共享权重,分别处理图像 A 和 B
        # 结构: Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d
        # 通道数变化: 3 -> 32 -> 64 -> 128 -> 256
        # 空间尺寸变化: 1024 -> 512 -> 256 -> 128 -> 64

        channels = [32, 64, 128, 256]  # 每层的输出通道数
        enc = []
        in_ch = 3  # 初始输入通道数为 3(RGB)

        for c in channels:
            # 每个编码块包含:
            # 1. Conv2d: 3x3 卷积,padding=1 保持尺寸不变
            # 2. BatchNorm2d: 批归一化
            # 3. ReLU: 激活函数
            # 4. MaxPool2d: 2x2 池化,步长 2,尺寸减半
            enc.append(nn.Sequential(
                nn.Conv2d(in_ch, c, kernel_size=3, padding=1),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ))
            in_ch = c  # 下一层的输入通道数 = 当前层的输出通道数

        # 将所有编码块组合成一个序列
        self.encoder = nn.Sequential(*enc)

        # ===== 2. 差异特征融合模块 =====
        # 计算两个编码器输出的差异: diff = f_A - f_B
        # 然后通过 2 层 ConvBNReLU 进一步融合差异特征
        # 输入输出形状: (B, 256, 64, 64) -> (B, 256, 64, 64)

        self.fusion = nn.Sequential(
            ConvBNReLU(256, 256),  # 第一层融合
            ConvBNReLU(256, 256),  # 第二层融合
        )

        # ===== 3. 转置卷积解码器 =====
        # 逐步还原空间分辨率: 64 -> 128 -> 256 -> 512 -> 1024
        # 每层使用 ConvTranspose2d(stride=2),尺寸翻倍
        # 同时减少通道数: 256 -> 128 -> 64 -> 32 -> 16

        # 第一层上采样: 64 -> 128
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),  # 转置卷积
            nn.BatchNorm2d(128),                                     # 批归一化
            nn.ReLU(inplace=True),                                   # 激活函数
        )

        # 第二层上采样: 128 -> 256
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # 第三层上采样: 256 -> 512
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # 第四层上采样: 512 -> 1024
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        # ===== 4. 输出层 =====
        # 1x1 卷积,将 16 通道转换为 1 通道 logit
        # 输出形状: (B, 16, 1024, 1024) -> (B, 1, 1024, 1024)
        self.head = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x_A, x_B):
        """
        前向传播:计算变化检测掩码

        【参数说明】
        x_A (torch.Tensor): 变化前图像,形状 (B, 3, 1024, 1024)
        x_B (torch.Tensor): 变化后图像,形状 (B, 3, 1024, 1024)

        【返回值】
        torch.Tensor: 变化掩码的 logit 值,形状 (B, 1, 1024, 1024)
                      注意:这是未经 sigmoid 的原始输出
                      训练时配合 BCEWithLogitsLoss 使用
                      推理时需要 torch.sigmoid() 得到概率

        【处理流程】
        1. **孪生编码**: 用同一个编码器处理两张图像
           - x_A -> f_A: (B, 3, 1024, 1024) -> (B, 256, 64, 64)
           - x_B -> f_B: (B, 3, 1024, 1024) -> (B, 256, 64, 64)
           - 关键:两个编码器共享权重

        2. **差异计算**: 逐通道相减得到差异特征
           - diff = f_A - f_B: (B, 256, 64, 64)
           - 正值表示 A 比 B 特征强
           - 负值表示 B 比 A 特征强
           - 绝对值大表示差异大

        3. **差异融合**: 进一步提取差异特征
           - fused = fusion(diff): (B, 256, 64, 64)
           - 通过卷积学习更抽象的差异表示

        4. **解码还原**: 逐步上采样到原始分辨率
           - up1: 64 -> 128 (B, 128, 128, 128)
           - up2: 128 -> 256 (B, 64, 256, 256)
           - up3: 256 -> 512 (B, 32, 512, 512)
           - up4: 512 -> 1024 (B, 16, 1024, 1024)

        5. **输出**: 1x1 卷积得到单通道 logit
           - logits: (B, 1, 1024, 1024)
        """
        # 1. 孪生共享权重提取特征
        # 使用同一个编码器处理两张图像,权重共享
        f_A = self.encoder(x_A)  # (B, 256, 64, 64)
        f_B = self.encoder(x_B)  # (B, 256, 64, 64)

        # 2. 逐通道相减生成差异特征图
        # 这是孪生网络的核心思想:通过特征差异来检测变化
        diff = f_A - f_B  # (B, 256, 64, 64)

        # 3. 融合差异特征
        # 通过卷积进一步提取和融合差异信息
        fused = self.fusion(diff)  # (B, 256, 64, 64)

        # 4. 转置卷积逐步还原至 1024x1024
        # 每层上采样尺寸翻倍,通道数减半
        x = self.up1(fused)   # 64  -> 128, 通道 256 -> 128
        x = self.up2(x)       # 128 -> 256, 通道 128 -> 64
        x = self.up3(x)       # 256 -> 512, 通道 64 -> 32
        x = self.up4(x)       # 512 -> 1024, 通道 32 -> 16

        # 5. 输出单通道 logit
        logits = self.head(x)  # (B, 1, 1024, 1024)

        return logits  # 训练时配合 BCEWithLogitsLoss,推理时需 sigmoid

    @torch.no_grad()
    def predict(self, x_A, x_B, threshold=0.5):
        """
        推理方法:直接输出二值变化掩码

        【参数说明】
        x_A (torch.Tensor): 变化前图像,形状 (B, 3, 1024, 1024)
        x_B (torch.Tensor): 变化后图像,形状 (B, 3, 1024, 1024)
        threshold (float): 二值化阈值,范围 [0, 1],默认 0.5
                          - 大于阈值认为是变化区域
                          - 小于等于阈值认为无变化

        【返回值】
        torch.Tensor: 二值变化掩码,形状 (B, 1, 1024, 1024)
                      - 255: 变化区域
                      - 0: 无变化区域
                      - dtype: torch.uint8

        【处理流程】
        1. forward 得到 logit
        2. sigmoid 得到概率 prob (0-1 之间)
        3. 阈值化得到布尔掩码
        4. 转换为 0/255 的 uint8 格式

        【装饰器说明】
        @torch.no_grad():
        - 禁用梯度计算,节省内存和计算
        - 推理时不需要反向传播,可以关闭梯度
        - 加速推理过程

        【使用场景】
        - 模型训练完成后进行预测
        - 对新图像进行变化检测
        - 可视化和评估时使用

        【使用示例】
        >>> model.eval()  # 切换到评估模式
        >>> mask = model.predict(img_A, img_B, threshold=0.5)
        >>> # mask 中 255 表示变化, 0 表示无变化
        """
        # 1. 前向传播得到 logit
        logits = self.forward(x_A, x_B)

        # 2. sigmoid 得到概率(0-1 之间)
        prob = torch.sigmoid(logits)

        # 3. 阈值化并转换为 0/255 的 uint8 格式
        # prob > threshold 得到布尔张量
        # to(torch.uint8) 转换为 0/1 的 uint8 类型
        # * 255 将 0/1 映射为 0/255
        mask = (prob > threshold).to(torch.uint8) * 255

        return mask


if __name__ == '__main__':
    """
    模型自检脚本

    【功能说明】
    当直接运行此文件时,会创建模型并测试前向传播,
    验证模型结构是否正确,输出形状是否符合预期。

    【运行方式】
    python model.py

    【预期输出示例】
    输出尺寸: (2, 1, 1024, 1024)  参数量: 1.823M

    【作用】
    1. 验证模型是否可以正常构建
    2. 验证前向传播是否正常
    3. 检查输出形状是否符合预期
    4. 统计模型参数量,评估模型复杂度
    """
    # 创建模型实例
    net = SiameseChangeNet()

    # 创建两个随机输入张量,模拟两张 1024x1024 的 RGB 图像
    # batch_size=2, channels=3, height=1024, width=1024
    a = torch.randn(2, 3, 1024, 1024)
    b = torch.randn(2, 3, 1024, 1024)

    # 前向传播,得到输出
    out = net(a, b)

    # 统计模型参数量
    # p.numel(): 计算每个参数张量的元素数量
    # sum(): 对所有参数求和
    # /1e6: 转换为百万(Million)单位
    n_params = sum(p.numel() for p in net.parameters())

    # 打印输出形状和参数量
    print(f'输出尺寸: {tuple(out.shape)}  参数量: {n_params/1e6:.3f}M')
