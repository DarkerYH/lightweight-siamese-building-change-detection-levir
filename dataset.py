# -*- coding: utf-8 -*-
"""
LEVIR-CD 变化检测数据集加载脚本

【项目背景】
这是一个变化检测(Change Detection)项目的数据加载模块。
变化检测任务:给定两张不同时间拍摄的同一地区遥感图像,检测出发生变化的区域。
典型应用:建筑物变化检测、城市扩张监测、灾害评估等。

【数据集介绍】
LEVIR-CD 是一个建筑物变化检测数据集,包含:
- A图像:变化前的RGB遥感图像
- B图像:变化后的RGB遥感图像
- label:变化掩码(真值),白色表示变化区域,黑色表示无变化区域

【目录结构规范】
    root/
      train/          # 训练集
        A/            # 变化前 RGB 图像 (如: train_1.png)
        B/            # 变化后 RGB 图像 (如: train_1.png)
        label/        # 单通道变化掩码真值图 (如: train_1.png)
      val/            # 验证集(结构同上)
      test/           # 测试集(结构同上)

【数据预处理】
- A/B图像:RGB三通道,归一化至 [0,1] 范围
- label:单通道灰度图,归一化至 [0,1] 并强制二值化为 0 或 1

【数据增强】(仅训练时)
- 随机水平翻转(50%概率)
- 随机垂直翻转(50%概率)
- 注意:A、B、label三张图必须同步翻转,保持空间对齐

【不执行的操作】
- 不裁剪/缩放,保持原始 1024x1024 分辨率
- 不做色彩增强,保持原始色彩信息

【核心类】
LEVIRCDDataset:继承自 torch.utils.data.Dataset,实现数据加载逻辑

【工作流程】
1. 初始化时扫描数据目录,匹配 A、B、label 三元组
2. __getitem__ 时读取图像、预处理、数据增强
3. 返回 img_A, img_B, label 三个张量
"""
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class LEVIRCDDataset(Dataset):
    """
    LEVIR-CD 变化检测数据集加载器

    继承自 PyTorch 的 Dataset 类,实现标准的 __len__ 和 __getitem__ 方法,
    可直接用于 DataLoader。

    【功能说明】
    1. 自动扫描数据目录,匹配 A、B、label 三元组
    2. 自动过滤残缺样本(如 A存在但 B或 label 不存在)
    3. 提供数据增强功能(随机翻转)
    4. 返回预处理后的 PyTorch 张量
    """

    def __init__(self, root, subset='train', augment=False):
        """
        初始化数据集

        【参数说明】
        root (str): 数据集根目录路径,例如 'data/LEVIR-CD'
        subset (str): 数据子集,可选 'train'、'val'、'test',默认 'train'
        augment (bool): 是否应用数据增强,默认 False
                        训练时通常设为 True,验证/测试时设为 False

        【初始化流程】
        1. 保存参数
        2. 构建三个子目录路径: A、B、label
        3. 扫描 A 目录下的所有 .png 文件
        4. 检查对应的 B 和 label 是否存在
        5. 将完整的三元组路径保存到 self.samples 列表
        """
        self.root = root
        self.subset = subset
        self.augment = augment

        # 构建三个子目录的完整路径
        self.dir_A = os.path.join(root, subset, 'A')      # 变化前图像目录
        self.dir_B = os.path.join(root, subset, 'B')      # 变化后图像目录
        self.dir_L = os.path.join(root, subset, 'label')  # 变化掩码目录

        # 遍历 A 文件夹,匹配 B 与 label,自动过滤残缺样本
        self.samples = []
        # 获取 A 目录下所有 .png 文件(不区分大小写),并排序保证可复现性
        a_files = sorted(f for f in os.listdir(self.dir_A) if f.lower().endswith('.png'))

        # 对每个 A 文件,检查对应的 B 和 label 是否存在
        for fn in a_files:
            path_A = os.path.join(self.dir_A, fn)
            path_B = os.path.join(self.dir_B, fn)
            path_L = os.path.join(self.dir_L, fn)

            # 只有当三个文件都存在时,才添加到样本列表
            # 这样可以自动过滤掉残缺样本,避免训练时出错
            if os.path.exists(path_B) and os.path.exists(path_L):
                self.samples.append((path_A, path_B, path_L))

    def __len__(self):
        """
        返回数据集样本总数

        【返回值】
        int: 可用样本的数量(已过滤残缺样本)

        【说明】
        DataLoader 会调用此方法来确定数据集大小,
        用于计算每个 epoch 的迭代次数。
        """
        return len(self.samples)

    def _read_rgb(self, path):
        """
        读取 RGB 图像并归一化

        【参数说明】
        path (str): 图像文件路径

        【返回值】
        numpy.ndarray: 形状为 (H, W, 3),数据类型 float32,值域 [0, 1]
                       H=高度, W=宽度, 3=RGB三通道

        【处理流程】
        1. 使用 PIL.Image.open 读取图像
        2. convert('RGB') 确保是三通道彩色图像
        3. 转换为 numpy 数组,float32 类型
        4. 归一化: 原始像素值 0-255 -> 浮点值 0.0-1.0

        【归一化】
        - 神经网络训练时,输入数据范围一致有助于收敛
        - 避免数值过大导致梯度爆炸
        - 配合 BatchNorm 等层使用
        """
        img = Image.open(path).convert('RGB')
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return arr

    def _read_label(self, path):
        """
        读取单通道灰度 label,归一化并二值化

        【参数说明】
        path (str): label 文件路径

        【返回值】
        numpy.ndarray: 形状为 (H, W, 1),数据类型 float32,值域 {0.0, 1.0}
                       H=高度, W=宽度, 1=单通道

        【处理流程】
        1. 使用 PIL.Image.open 读取图像
        2. convert('L') 转换为单通道灰度图
        3. 转换为 numpy 数组,float32 类型
        4. 归一化: 原始像素值 0-255 -> 浮点值 0.0-1.0
        5. 二值化: 大于 0.5 的像素设为 1.0(变化),小于等于 0.5 的设为 0.0(无变化)
        6. 扩展维度: (H, W) -> (H, W, 1),方便后续处理

        【二值化】
        - 变化检测是二分类任务(变化/无变化)
        - 确保标签只有 0 和 1 两个值,避免中间值干扰训练
        - 方便计算二元交叉熵损失
        """
        img = Image.open(path).convert('L')
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr > 0.5).astype(np.float32)
        return arr[:, :, None]  # 扩展维度: (H,W) -> (H,W,1)

    def __getitem__(self, idx):
        """
        获取指定索引的样本

        【参数说明】
        idx (int): 样本索引,范围 [0, len(self.samples)-1]

        【返回值】
        tuple: (img_A, img_B, label)
            img_A (torch.Tensor): 形状 (3, H, W),变化前RGB图像
            img_B (torch.Tensor): 形状 (3, H, W),变化后RGB图像
            label (torch.Tensor): 形状 (1, H, W),变化掩码

        【处理流程】
        1. 从 self.samples 获取第 idx 个样本的三个文件路径
        2. 分别读取并预处理三张图像
        3. 如果启用数据增强,对三张图同步应用相同的随机变换
        4. 将 numpy 数组转换为 PyTorch 张量
        5. 调整维度顺序: HWC (高度,宽度,通道) -> CHW (通道,高度,宽度)

        【数据增强详解】
        - 随机水平翻转: 50%概率,左右镜像翻转
        - 随机垂直翻转: 50%概率,上下镜像翻转
        - 关键:A、B、label三张图必须同步翻转,保持空间对齐
        - 作用:增加训练数据多样性,提高模型泛化能力
        - 注意:只在训练时启用,验证/测试时不启用

        【维度顺序说明】
        - PIL/numpy 默认格式: HWC (高度,宽度,通道)
        - PyTorch 要求格式: CHW (通道,高度,宽度)
        - permute(2, 0, 1) 实现维度重排: (H,W,C) -> (C,H,W)
        - ascontiguous() 确保内存连续,避免潜在错误

        【数据增强】
        - 增加训练样本的多样性
        - 提高模型的泛化能力,避免过拟合
        - 让模型学习到更本质的特征,而非记忆训练数据
        - 对于遥感图像,翻转是常用的增强方式(不会改变语义)
        """
        # 1. 获取第 idx 个样本的三个文件路径
        path_A, path_B, path_L = self.samples[idx]

        # 2. 读取并预处理三张图像
        # 此时格式为 HWC: 高度x宽度x通道
        img_A = self._read_rgb(path_A)      # H W 3 (RGB三通道)
        img_B = self._read_rgb(path_B)      # H W 3 (RGB三通道)
        label = self._read_label(path_L)    # H W 1 (单通道)

        # 3. 数据增强(仅训练时启用)
        # 必须对三张图同步应用相同的变换,保持空间对齐
        if self.augment:
            # 随机水平翻转(50%概率)
            if np.random.rand() < 0.5:
                img_A = np.fliplr(img_A).copy()  # 左右镜像
                img_B = np.fliplr(img_B).copy()  # 左右镜像
                label = np.fliplr(label).copy()  # 左右镜像

            # 随机垂直翻转(50%概率)
            if np.random.rand() < 0.5:
                img_A = np.flipud(img_A).copy()  # 上下镜像
                img_B = np.flipud(img_B).copy()  # 上下镜像
                label = np.flipud(label).copy()  # 上下镜像

        # 4. 维度转换: HWC -> CHW
        # PyTorch 的卷积层要求输入格式为 (通道, 高度, 宽度)
        # np.ascontiguousarray: 确保数组在内存中连续存储
        # permute(2, 0, 1): 重排维度 (H,W,C) -> (C,H,W)
        # contiguous(): 确保张量在内存中连续存储
        img_A = torch.from_numpy(np.ascontiguousarray(img_A)).permute(2, 0, 1).contiguous()
        img_B = torch.from_numpy(np.ascontiguousarray(img_B)).permute(2, 0, 1).contiguous()
        label = torch.from_numpy(np.ascontiguousarray(label)).permute(2, 0, 1).contiguous()

        return img_A, img_B, label


if __name__ == '__main__':
    """
    数据集自检脚本

    【功能说明】
    当直接运行此文件时,会扫描数据目录并打印各子集的样本数量。
    用于验证数据集是否正确加载。

    【运行方式】
    python dataset.py

    【预期输出示例】
    train: 193 个匹配样本
    val: 0 个匹配样本
    test: 128 个匹配样本

    【作用】
    1. 验证数据目录结构是否正确
    2. 检查是否有文件缺失或路径错误
    3. 了解各子集的样本数量
    """
    # 获取数据集根目录: 当前文件所在目录/data/LEVIR-CD
    root = os.path.join(os.path.dirname(__file__), 'data', 'LEVIR-CD')

    # 遍历三个子集,分别创建数据集对象并打印样本数
    for s in ['train', 'val', 'test']:
        ds = LEVIRCDDataset(root, subset=s, augment=False)
        print(f'{s}: {len(ds)} 个匹配样本')
