# -*- coding: utf-8 -*-
"""
训练主程序:轻量化孪生 CNN 变化检测模型

【项目背景】
这是变化检测项目的训练脚本,负责模型的训练、验证和权重保存。

【训练流程】
1. 加载训练集和验证集
2. 创建模型、优化器、损失函数
3. 训练循环:
   - 前向传播计算损失
   - 反向传播更新权重
   - 定期验证并保存最优模型
4. 记录训练日志

【超参数设置】
- 损失函数: BCEWithLogitsLoss(二元交叉熵,内部包含 sigmoid)
- 优化器: Adam(lr=1e-4)
- batch_size: 1(可根据显存调整)
- epochs: 50(训练轮数)
- 验证频率: 每 5 轮验证一次
- 保存策略: 验证集 IoU 最优时保存权重

【输出文件】
- outputs/best_model.pth: 最优模型权重
- outputs/training_log.txt: 训练日志

【注意事项】
1. Windows 系统建议 NUM_WORKERS=0,避免多进程问题
2. GPU 训练速度远快于 CPU,推荐使用 GPU
3. 可根据显存大小调整 batch_size
4. 训练过程中会自动创建 outputs 目录

【训练指标说明】
- Loss: 训练损失值,越小越好
- IoU(Intersection over Union): 交并比,越大越好
- Precision: 精确率(预测为变化中真实变化的比例)
- Recall: 召回率(真实变化中被预测为变化的比例)
- F1: 精确率和召回率的调和平均
"""
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import LEVIRCDDataset
from model import SiameseChangeNet


# ===== 固定超参数配置 =====
# 项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 数据集根目录
DATA_ROOT = os.path.join(BASE_DIR, 'data', 'LEVIR-CD')
# 输出目录(保存模型权重和训练日志)
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')

# ===== 训练超参数 =====
BATCH_SIZE = 1          # 批量大小:每次训练处理的样本数
EPOCHS = 50             # 训练总轮数:整个数据集重复训练的次数
LR = 1e-4               # 学习率:控制梯度下降的步长
VAL_EVERY = 5           # 验证频率:每训练多少轮进行一次验证
THRESHOLD = 0.5         # 二值化阈值:将概率转换为 0/1 的阈值

# ===== Windows 兼容性设置 =====
NUM_WORKERS = 0         # DataLoader 的进程数
                        # - Windows 建议设为 0,避免多进程问题
                        # - Linux/Mac 可设为 4-8,加速数据加载
PIN_MEMORY = False      # 是否使用锁页内存
                        # - True: 加速 CPU 到 GPU 的数据传输,但占用更多内存
                        # - False: 节省内存,Windows 推荐 False

def compute_metrics(tp, fp, fn):
    """
    计算评价指标:精确率、召回率、F1、IoU

    【参数说明】
    tp (int): True Positive - 真阳性数(预测为变化且实际为变化)
    fp (int): False Positive - 假阳性数(预测为变化但实际无变化)
    fn (int): False Negative - 假阴性数(预测无变化但实际为变化)

    【返回值】
    tuple: (precision, recall, f1, iou)
        precision (float): 精确率,预测为变化中真实变化的比例
        recall (float): 召回率,真实变化中被预测为变化的比例
        f1 (float): F1 分数,精确率和召回率的调和平均
        iou (float): 交并比,预测和真实的交集/并集

    【指标计算公式】
    - Precision(精确率) = TP / (TP + FP)
      含义:预测为变化的像素中,真正变化的比例
      应用场景:关注预测的准确性,减少误报

    - Recall(召回率) = TP / (TP + FN)
      含义:真实变化的像素中,被正确预测的比例
      应用场景:关注不漏检,减少漏报

    - F1 = 2 * Precision * Recall / (Precision + Recall)
      含义:精确率和召回率的平衡指标
      应用场景:综合考虑准确性和完整性

    - IoU(交并比) = TP / (TP + FP + FN)
      含义:预测和真实的交集面积 / 预测和真实的并集面积
      应用场景:分割任务的核心指标,范围 0-1,越大越好

    【需要 eps】
    - 避免除零错误(当 TP+FP=0 或 TP+FN=0 时)
    - eps=1e-8 是一个极小值,不影响结果精度

    """
    eps = 1e-8  # 极小值,防止除零

    # 计算精确率:预测为变化中真实变化的比例
    precision = tp / (tp + fp + eps)

    # 计算召回率:真实变化中被预测为变化的比例
    recall = tp / (tp + fn + eps)

    # 计算 F1 分数:精确率和召回率的调和平均
    f1 = 2 * precision * recall / (precision + recall + eps)

    # 计算 IoU:交并比,分割任务的核心指标
    iou = tp / (tp + fp + fn + eps)

    return precision, recall, f1, iou


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
    """
    在验证集上评估模型性能

    【参数说明】
    model (nn.Module): 待评估的模型
    loader (DataLoader): 验证集数据加载器
    device (torch.device): 计算设备(cpu 或 cuda)
    threshold (float): 二值化阈值,默认 0.5

    【返回值】
    tuple: (precision, recall, f1, iou) - 四个评价指标

    【处理流程】
    1. 遍历验证集所有样本
    2. 对每个样本进行预测
    3. 累计 TP、FP、FN
    4. 计算并返回指标

    【装饰器说明】
    @torch.no_grad():
    - 禁用梯度计算,节省内存和计算
    - 验证时不需要反向传播
    - 加速推理过程

    【注意事项】
    - 验证前需要调用 model.eval() 切换到评估模式
    - 评估模式会关闭 Dropout、BatchNorm 等层的训练行为
    """
    model.eval()  # 切换到评估模式

    # 初始化统计变量
    tp = fp = fn = 0

    # 遍历验证集
    for img_A, img_B, label in loader:
        # 将数据移动到指定设备(GPU/CPU)
        # non_blocking=True: 异步传输,提高效率
        img_A = img_A.to(device, non_blocking=True)
        img_B = img_B.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        # 前向传播得到 logit
        logits = model(img_A, img_B)

        # sigmoid 得到概率(0-1 之间)
        prob = torch.sigmoid(logits)

        # 二值化:概率 > threshold 为 1,否则为 0
        pred = (prob > threshold).float()

        # 同样对 label 进行二值化
        gt = (label > threshold).float()

        # 累计 TP、FP、FN
        # TP: 预测为1 且 真实为1
        tp += ((pred == 1) & (gt == 1)).sum().item()
        # FP: 预测为1 但 真实为0
        fp += ((pred == 1) & (gt == 0)).sum().item()
        # FN: 预测为0 但 真实为1
        fn += ((pred == 0) & (gt == 1)).sum().item()

    # 计算并返回指标
    return compute_metrics(tp, fp, fn)


def main():
    """
    主函数:执行完整的训练流程

    【训练流程】
    1. 初始化:
       - 创建输出目录
       - 设置日志文件
       - 检测计算设备(GPU/CPU)

    2. 数据准备:
       - 创建训练集和验证集 Dataset
       - 创建 DataLoader

    3. 模型准备:
       - 创建模型实例
       - 定义损失函数
       - 定义优化器

    4. 训练循环:
       - 遍历 epoch
       - 前向传播、反向传播、参数更新
       - 定期验证并保存最优模型

    5. 训练结束:
       - 打印最终结果
       - 关闭日志文件

    【关键步骤详解】
    - 数据加载:使用 DataLoader 批量加载数据
    - 前向传播:模型计算预测值
    - 损失计算:计算预测值与真实值的差异
    - 反向传播:计算梯度
    - 参数更新:根据梯度更新模型参数
    - 验证评估:在验证集上测试模型性能
    - 模型保存:保存最优模型权重
    """
    # ===== 1. 初始化 =====
    # 创建输出目录(如果不存在)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 创建日志文件
    log_path = os.path.join(OUTPUT_DIR, 'training_log.txt')
    log_file = open(log_path, 'w', encoding='utf-8')

    # 定义日志函数:同时打印到控制台和写入文件
    def log(msg):
        """
        日志记录函数

        【参数说明】
        msg (str): 日志消息

        【功能】
        1. 添加时间戳
        2. 打印到控制台
        3. 写入日志文件
        4. 立即刷新缓冲区
        """
        line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
        print(line)                      # 打印到控制台
        log_file.write(line + '\n')      # 写入文件
        log_file.flush()                 # 立即刷新缓冲区

    # 检测并设置计算设备
    # 如果有 GPU 则使用 GPU,否则使用 CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f'使用设备: {device}')

    # ===== 2. 数据准备 =====
    # 创建训练集 Dataset
    # augment=True: 启用数据增强(随机翻转)
    train_set = LEVIRCDDataset(DATA_ROOT, subset='train', augment=True)

    # 创建验证集 Dataset
    # augment=False: 不启用数据增强
    val_set = LEVIRCDDataset(DATA_ROOT, subset='val', augment=False)

    log(f'训练集样本数: {len(train_set)}')
    log(f'验证集样本数: {len(val_set)}')

    # 创建训练集 DataLoader
    # shuffle=True: 每个 epoch 打乱数据顺序,提高训练效果
    # drop_last=True: 丢弃最后一个不完整的 batch
    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=True
    )

    # 创建验证集 DataLoader
    # shuffle=False: 验证时不需要打乱顺序
    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    # ===== 3. 模型准备 =====
    # 创建模型实例并移动到指定设备
    model = SiameseChangeNet().to(device)

    # 定义损失函数
    # BCEWithLogitsLoss: 二元交叉熵损失(内部包含 sigmoid)
    # 优点: 数值稳定,避免 log(0) 错误
    criterion = nn.BCEWithLogitsLoss()

    # 定义优化器
    # Adam: 自适应矩估计优化器
    # 特点: 自动调整学习率,收敛快,对初始学习率不敏感
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # 打印训练配置信息
    log(f'超参数: batch_size={BATCH_SIZE}, epochs={EPOCHS}, lr={LR}, val_every={VAL_EVERY}, threshold={THRESHOLD}')
    n_params = sum(p.numel() for p in model.parameters())
    log(f'模型参数量: {n_params/1e6:.3f}M')

    # 初始化最优 IoU 和模型保存路径
    best_iou = 0.0
    best_path = os.path.join(OUTPUT_DIR, 'best_model.pth')

    # ===== 4. 训练循环 =====
    # 遍历所有 epoch
    for epoch in range(1, EPOCHS + 1):
        # 切换到训练模式
        # 训练模式会启用 Dropout、BatchNorm 等层的训练行为
        model.train()

        # 初始化本轮的累计损失和 batch 数量
        running_loss = 0.0
        n_batches = 0
        t0 = time.time()  # 记录开始时间

        # 遍历训练集的所有 batch
        for img_A, img_B, label in train_loader:
            # 将数据移动到指定设备(GPU/CPU)
            img_A = img_A.to(device, non_blocking=True)
            img_B = img_B.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            # ===== 训练步骤:前向传播 -> 计算损失 -> 反向传播 -> 参数更新 =====

            # 1. 清零梯度
            # PyTorch 默认会累积梯度,需要手动清零
            optimizer.zero_grad()

            # 2. 前向传播:计算预测值
            logits = model(img_A, img_B)

            # 3. 计算损失:预测值与真实值的差异
            loss = criterion(logits, label)

            # 4. 反向传播:计算梯度
            loss.backward()

            # 5. 参数更新:根据梯度更新模型参数
            optimizer.step()

            # 累计损失和 batch 数量
            running_loss += loss.item()
            n_batches += 1

        # 计算本轮的平均损失
        avg_loss = running_loss / max(n_batches, 1)

        # 计算本轮训练耗时
        elapsed = time.time() - t0

        # 打印训练日志
        log(f'[Epoch {epoch:02d}/{EPOCHS}] 训练损失: {avg_loss:.6f}  耗时: {elapsed:.1f}s')

        # ===== 5. 验证评估 =====
        # 每 VAL_EVERY 轮或最后一轮进行验证
        if epoch % VAL_EVERY == 0 or epoch == EPOCHS:
            # 在验证集上评估模型
            precision, recall, f1, iou = evaluate(model, val_loader, device, THRESHOLD)

            # 打印验证结果
            log(f'  -> 验证集 Precision={precision:.4f} Recall={recall:.4f} F1={f1:.4f} IoU={iou:.4f}')

            # 如果当前 IoU 优于历史最优,保存模型
            if iou > best_iou:
                best_iou = iou
                # 保存模型权重
                # model.state_dict(): 获取模型的所有参数
                # torch.save(): 保存为 .pth 文件
                torch.save(model.state_dict(), best_path)
                log(f'  -> 新的最优模型已保存 (IoU={iou:.4f}) -> {best_path}')

    # ===== 6. 训练结束 =====
    log(f'训练结束. 最优验证集 IoU: {best_iou:.4f}')
    log_file.close()


if __name__ == '__main__':
    """
    程序入口

    【执行流程】
    当直接运行此文件时,调用 main() 函数开始训练。

    【预期输出】
    1. 控制台打印训练日志
    2. outputs/training_log.txt 记录详细日志
    3. outputs/best_model.pth 保存最优模型权重
    """
    main()
