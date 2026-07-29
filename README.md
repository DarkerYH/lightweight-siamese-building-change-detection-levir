# lightweight-siamese-building-change-detection-levir
Lightweight Siamese Network for Remote Sensing Building Change Detection | 轻量化孪生网络遥感建筑物变化检测（LEVIR-CD 数据集）

# 项目简介

本项目基于**权重共享Siamese孪生卷积神经网络**，针对遥感建筑物变化检测任务开发，适配LEVIR-CD公开遥感数据集。网络仅使用基础卷积、池化、转置卷积模块，无Transformer、自注意力等高阶结构，实现轻量化设计，在保障检测精度的同时降低参数量与推理成本，可在普通NVIDIA GPU环境快速训练部署。

任务逻辑：输入配准后的前后两时相遥感影像，输出单通道二值变化掩码，区分图像建筑物变化区域与不变区域；完整覆盖数据集加载、训练、验证、测试、指标评估、结果可视化全流程。

# 数据集与权重说明

## LEVIR-CD 数据集

本项目使用 **LEVIR-CD 建筑物遥感变化检测数据集**，数据集体积较大，未上传至仓库，请自行下载解压至 `./data/` 目录。

### 官方主页

https://justchenhao.github.io/LEVIR/

### 论文引用

https://www.mdpi.com/2072-4292/12/10/1662

## 训练输出文件

训练产生的模型权重 `.pth`、运行日志全部保存在 `./outputs/`，本地生成，不纳入版本管理。

# 网络结构设计

**孪生共享特征提取分支**
两条权重完全一致的卷积支路，每组包含4层卷积单元（Conv3×3 + BN + ReLU + MaxPool2d），通道数逐层 32→64→128→256，输入1024×1024影像下采样至64×64特征图。

**差异特征融合**
对双分支特征逐通道相减得到差异特征，两层卷积单元降噪融合，过滤无效背景噪声。

**转置卷积上采样解码**
使用TransposedConv2d完成上采样，逐步将64×64特征还原至原图1024×1024尺寸；末尾单通道卷积+Sigmoid输出0~1概率掩码，阈值0.5划分变化/非变化像素。

# 评估指标

验证集、测试集统一计算四项变化检测量化指标：

Precision（精确率）、Recall（召回率）、F1-Score、IoU（交并比）

# 快速运行教程

1.下载数据集并放置至 `./data/LEVIR-CD/`

2.安装项目依赖

```
pip install -r requirements.txt
```

3.启动模型训练

```
python train.py
```

4.加载最优权重执行测试推理，输出指标与可视化对比图

```
python test.py
```

## 输出文件说明

训练日志：记录每轮损失、每 5 轮验证集完整指标

模型权重：`outputs/` 下保存 IoU 最优 `.pth` 文件

实验结果：测试集指标表格、四拼接可视化图（原图 A / 原图 B / 真值 label / 预测掩码）

## 许可证

MIT License
