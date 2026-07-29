# -*- coding: utf-8 -*-
"""
测试推理脚本:加载最优权重,对测试集批量推理

【项目背景】
这是变化检测项目的测试脚本,负责模型评估和结果可视化。

【测试流程】
1. 加载训练好的模型权重
2. 遍历测试集所有样本
3. 对每个样本进行预测
4. 计算整体评价指标
5. 生成可视化结果

【输出文件】
- outputs/test_metrics.csv: 测试集量化指标汇总表格
- outputs/visualizations/: 四宫格拼接图(A | B | label | 预测掩码)
  每张图包含:
  - A: 变化前图像
  - B: 变化后图像
  - label: 真实变化掩码
  - 预测: 模型预测的变化掩码

【运行前提】
必须先运行 train.py 完成模型训练,生成 best_model.pth 文件

【指标说明】
- Precision: 精确率,预测准确率
- Recall: 召回率,检出率
- F1: 综合指标
- IoU: 交并比,分割核心指标
"""
import os
import csv
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

from dataset import LEVIRCDDataset
from model import SiameseChangeNet


# ===== 配置参数 =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))       # 项目根目录
DATA_ROOT = os.path.join(BASE_DIR, 'data', 'LEVIR-CD')      # 数据集根目录
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')              # 输出目录
VIS_DIR = os.path.join(OUTPUT_DIR, 'visualizations')        # 可视化结果目录

BATCH_SIZE = 1          # 批量大小,测试时通常设为 1
NUM_WORKERS = 0         # DataLoader 进程数,Windows 建议设为 0
PIN_MEMORY = False      # 是否使用锁页内存
THRESHOLD = 0.5         # 二值化阈值

# ===== 可视化配置 =====
LABEL_HEIGHT = 50       # 标签条高度(像素)
LABEL_BG_COLOR = (255, 255, 255)  # 标签条背景色(白色)
LABEL_TEXT_COLOR = (0, 0, 0)       # 标签文字颜色(黑色)
LABEL_FONT_SIZE = 28   # 标签字体大小
LABELS = ['A (变化前)', 'B (变化后)', 'label (真值)', 'pred (预测)']  # 四张图的标签名称


def add_label_to_image(img_np, label_text):
    """
    在图片上方添加文字标签条

    【参数说明】
    img_np (numpy.ndarray): 输入图片,形状 (H, W, 3),值域 0-255
    label_text (str): 要显示的标签文字

    【返回值】
    numpy.ndarray: 添加了标签条的图片,形状 (H + LABEL_HEIGHT, W, 3)

    【实现流程】
    1. 创建白色背景的标签条
    2. 在标签条上居中绘制文字
    3. 将标签条和原图在垂直方向拼接
    """
    H, W, _ = img_np.shape

    # 1. 创建标签条:白色背景
    label_bar = np.full((LABEL_HEIGHT, W, 3), LABEL_BG_COLOR, dtype=np.uint8)

    # 2. 将 numpy 数组转换为 PIL Image 用于绘图
    label_img = Image.fromarray(label_bar)
    draw = ImageDraw.Draw(label_img)

    # 3. 尝试加载字体,失败则使用默认字体
    try:
        # 尝试使用系统字体,不同系统路径可能不同
        font = None
        # Windows 常见字体路径
        font_paths = [
            "C:/Windows/Fonts/msyh.ttc",      # 微软雅黑
            "C:/Windows/Fonts/simhei.ttf",    # 黑体
            "C:/Windows/Fonts/simsun.ttc",    # 宋体
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",  # Linux 文泉驿
            "/System/Library/Fonts/PingFang.ttc",            # Mac 苹方
        ]
        for fp in font_paths:
            if os.path.exists(fp):
                font = ImageFont.truetype(fp, LABEL_FONT_SIZE)
                break
        if font is None:
            font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    # 4. 计算文字位置,使其居中
    # 获取文字的边界框: left, top, right, bottom
    text_bbox = draw.textbbox((0, 0), label_text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # 水平和垂直居中
    text_x = (W - text_width) // 2
    text_y = (LABEL_HEIGHT - text_height) // 2 - text_bbox[1]  # 减去基线偏移

    # 5. 绘制文字
    draw.text((text_x, text_y), label_text, fill=LABEL_TEXT_COLOR, font=font)

    # 6. 将 PIL Image 转回 numpy 数组
    label_bar_with_text = np.array(label_img)

    # 7. 将标签条和原图在垂直方向拼接 (axis=0 表示垂直拼接)
    result = np.concatenate([label_bar_with_text, img_np], axis=0)

    return result


def compute_metrics(tp, fp, fn):
    """
    计算评价指标:精确率、召回率、F1、IoU

    【参数说明】
    tp (int): True Positive - 真阳性数(预测为变化且实际为变化)
    fp (int): False Positive - 假阳性数(预测为变化但实际无变化)
    fn (int): False Negative - 假阴性数(预测无变化但实际为变化)

    【返回值】
    tuple: (precision, recall, f1, iou)
        precision (float): 精确率
        recall (float): 召回率
        f1 (float): F1 分数
        iou (float): 交并比

    【说明】
    此函数与 train.py 中的函数相同,用于计算测试集指标。
    详细说明请参考 train.py 中的注释。
    """
    eps = 1e-8
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    return precision, recall, f1, iou


@torch.no_grad()
def main():
    """
    主函数:执行测试流程

    【测试流程】
    1. 初始化:
       - 创建输出目录
       - 检测计算设备

    2. 数据准备:
       - 加载测试集 Dataset
       - 创建 DataLoader

    3. 模型加载:
       - 创建模型实例
       - 加载训练好的权重
       - 切换到评估模式

    4. 批量推理:
       - 遍历测试集
       - 对每个样本进行预测
       - 累计 TP、FP、FN
       - 生成可视化结果

    5. 指标计算:
       - 计算整体指标
       - 保存为 CSV 文件
       - 打印结果

    【装饰器说明】
    @torch.no_grad():
    - 禁用梯度计算
    - 节省内存和计算
    - 推理时必须使用

    【输出说明】
    - 控制台打印汇总指标
    - outputs/test_metrics.csv: CSV 格式的指标表格
    - outputs/visualizations/: 可视化结果图片
    """
    # ===== 1. 初始化 =====
    # 创建输出目录(如果不存在)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(VIS_DIR, exist_ok=True)

    # 检测并设置计算设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    # ===== 2. 数据准备 =====
    # 创建测试集 Dataset
    # augment=False: 测试时不使用数据增强
    test_set = LEVIRCDDataset(DATA_ROOT, subset='test', augment=False)
    print(f'测试集样本数: {len(test_set)}')

    # 检查测试集是否为空
    if len(test_set) == 0:
        print('错误: 测试集无匹配样本,请检查 data/LEVIR-CD/test 目录。')
        return

    # 创建测试集 DataLoader
    # shuffle=False: 测试时不需要打乱顺序
    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    # ===== 3. 模型加载 =====
    # 创建模型实例并移动到指定设备
    model = SiameseChangeNet().to(device)

    # 构建模型权重文件路径
    weight_path = os.path.join(OUTPUT_DIR, 'best_model.pth')

    # 检查权重文件是否存在
    if not os.path.exists(weight_path):
        print(f'错误: 未找到模型权重 {weight_path},请先运行 train.py。')
        return

    # 加载模型权重
    # map_location: 将权重加载到指定设备(GPU/CPU)
    model.load_state_dict(torch.load(weight_path, map_location=device))

    # 切换到评估模式
    # 评估模式会关闭 Dropout、BatchNorm 等层的训练行为
    model.eval()
    print(f'已加载权重: {weight_path}')

    # ===== 4. 批量推理 =====
    # 初始化统计变量
    tp = fp = fn = 0
    sample_idx = 0  # 样本索引,用于生成可视化结果

    # 遍历测试集进行推理
    for img_A, img_B, label in test_loader:
        # 将数据移动到指定设备
        img_A = img_A.to(device, non_blocking=True)
        img_B = img_B.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        # 前向传播得到 logit
        logits = model(img_A, img_B)

        # sigmoid 得到概率(0-1 之间)
        prob = torch.sigmoid(logits)

        # 二值化:概率 > threshold 为 1,否则为 0
        pred = (prob > THRESHOLD).float()

        # 对 label 也进行二值化
        gt = (label > THRESHOLD).float()

        # ===== 累计指标 =====
        # TP: 预测为1 且 真实为1
        tp += ((pred == 1) & (gt == 1)).sum().item()
        # FP: 预测为1 但 真实为0
        fp += ((pred == 1) & (gt == 0)).sum().item()
        # FN: 预测为0 但 真实为1
        fn += ((pred == 0) & (gt == 1)).sum().item()

        # ===== 生成可视化四宫格图 =====
        # batch size 通常为 1,但代码支持任意 batch size
        b = img_A.size(0)
        for i in range(b):
            # 提取第 i 个样本的数据
            # 从 GPU 移回 CPU,转换为 numpy 数组

            # 提取图像 A (RGB)
            # img_A[i]: (3, H, W)
            # .cpu().numpy(): 转换为 numpy 数组
            # .transpose(1, 2, 0): (C, H, W) -> (H, W, C)
            # * 255: 归一化到 0-255
            # .astype(np.uint8): 转换为 uint8 类型
            A_img = (img_A[i].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

            # 提取图像 B (RGB)
            B_img = (img_B[i].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

            # 提取真实标签 (单通道)
            # gt[i, 0]: (H, W)
            L_img = (gt[i, 0].cpu().numpy() * 255).astype(np.uint8)

            # 提取预测结果 (单通道)
            # pred[i, 0]: (H, W)
            P_img = (pred[i, 0].cpu().numpy() * 255).astype(np.uint8)

            # 单通道转三通道,方便拼接
            # np.stack: 沿最后一个维度堆叠,生成 (H, W, 3)
            L_3 = np.stack([L_img, L_img, L_img], axis=-1)
            P_3 = np.stack([P_img, P_img, P_img], axis=-1)

            # ===== 在每张图片上方添加文字标签 =====
            # LABELS = ['A (变化前)', 'B (变化后)', 'label (真值)', 'pred (预测)']
            A_img_labeled = add_label_to_image(A_img, LABELS[0])
            B_img_labeled = add_label_to_image(B_img, LABELS[1])
            L_3_labeled = add_label_to_image(L_3, LABELS[2])
            P_3_labeled = add_label_to_image(P_3, LABELS[3])

            # 拼接为四宫格
            # axis=1: 沿宽度方向拼接
            # 结果: A | B | label | pred (每张图上方都带有标签)
            vis = np.concatenate([A_img_labeled, B_img_labeled, L_3_labeled, P_3_labeled], axis=1)

            # 保存可视化结果
            # 使用原始文件名作为文件名
            name = os.path.basename(test_set.samples[sample_idx][0])
            Image.fromarray(vis).save(os.path.join(VIS_DIR, name))

            # 更新样本索引
            sample_idx += 1

    # ===== 5. 指标计算与输出 =====
    # 计算整体指标
    precision, recall, f1, iou = compute_metrics(tp, fp, fn)

    # 写入 CSV 指标表格
    csv_path = os.path.join(OUTPUT_DIR, 'test_metrics.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 写入表头
        writer.writerow(['指标', '数值'])
        # 写入各项指标
        writer.writerow(['Precision', f'{precision:.6f}'])
        writer.writerow(['Recall', f'{recall:.6f}'])
        writer.writerow(['F1', f'{f1:.6f}'])
        writer.writerow(['IoU', f'{iou:.6f}'])
        writer.writerow(['样本数', f'{len(test_set)}'])

    # 控制台打印汇总表格
    print('\n========== 测试集量化指标汇总 ==========')
    print(f'样本数     : {len(test_set)}')
    print(f'Precision  : {precision:.6f}')
    print(f'Recall     : {recall:.6f}')
    print(f'F1         : {f1:.6f}')
    print(f'IoU        : {iou:.6f}')
    print('======================================')
    print(f'指标表格已保存: {csv_path}')
    print(f'可视化结果已保存: {VIS_DIR}')


if __name__ == '__main__':
    """
    程序入口

    【执行流程】
    当直接运行此文件时,调用 main() 函数开始测试。

    【运行前提】
    1. 必须先运行 train.py 完成模型训练
    2. 确保 outputs/best_model.pth 文件存在
    3. 确保测试集数据完整

    【预期输出】
    1. 控制台打印测试结果
    2. outputs/test_metrics.csv: 指标表格
    3. outputs/visualizations/: 可视化结果
    """
    main()
