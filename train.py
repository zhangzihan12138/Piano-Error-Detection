"""训练脚本：CNN + Bi-LSTM 钢琴错误检测模型。

流程：
1. 按 8:2 划分训练/测试样本（固定随机种子，保证 evaluate.py 可复现同一划分）
2. 用训练集构建 DataLoader（batch_size=8）
3. 用 BCELoss 和 Adam 训练 20 个 epoch，终端打印每个 epoch 的平均 Loss
4. 用 matplotlib 画出真实的 Loss 下降曲线，保存为 loss_curve.png
5. 保存模型权重 piano_model.pth 和测试集 ID，供 evaluate.py 评估使用
"""

import json
import os
import random

import matplotlib

# 无图形界面环境下只做文件输出，避免 plt.show() 阻塞
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import PianoErrorModel
from piano_dataset import PianoDataset

# ---------------- 超参数 ----------------
BATCH_SIZE = 8   # 批大小
EPOCHS = 20      # 训练轮数
LR = 1e-3        # Adam 学习率

# ---------------- 路径 ----------------
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(_HERE, "data", "features")
CURVE_PATH = os.path.join(_HERE, "loss_curve.png")
MODEL_PATH = os.path.join(_HERE, "piano_model.pth")
IDS_PATH = os.path.join(_HERE, "test_ids.json")


def split_ids(n_total: int, seed: int = 42, train_ratio: float = 0.8) -> tuple[list[int], list[int]]:
    """按固定种子打乱样本 ID 并划分为训练/测试两份，保证与 evaluate.py 复现一致。"""
    ids = list(range(n_total))
    random.Random(seed).shuffle(ids)
    k = int(n_total * train_ratio)
    return ids[:k], ids[k:]


def main() -> None:
    # CPU 多线程加速（16 线程），显著缩短每轮耗时
    torch.set_num_threads(16)

    # 0. 划分训练/测试集
    all_ids_dataset = PianoDataset(DATA_ROOT)
    train_ids, test_ids = split_ids(len(all_ids_dataset))
    dataset = PianoDataset(DATA_ROOT, ids=train_ids)
    # 先落盘划分，保证中途中断时模型权重与测试集划分始终成对
    with open(IDS_PATH, "w", encoding="utf-8") as f:
        json.dump({"test_ids": test_ids, "train_ids": train_ids}, f)

    # 1. DataLoader（batch_size=8）
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"总样本: {len(all_ids_dataset)} | 训练: {len(train_ids)} | 测试: {len(test_ids)} | 每轮 batch 数: {len(loader)}")

    # 2. 模型 + BCELoss + Adam
    model = PianoErrorModel()
    criterion = nn.BCELoss()  # 模型输出已过 Sigmoid，直接用 BCELoss
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    epoch_losses: list[float] = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        seen = 0

        for mel, label in loader:
            # mel: [B, 1, 128, 512]   label: [B, 512, 88]
            optimizer.zero_grad()
            pred = model(mel)                 # [B, 512, 88]，Sigmoid 概率
            loss = criterion(pred, label)     # 逐元素 BCE
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * mel.size(0)
            seen += mel.size(0)

        avg_loss = running_loss / max(seen, 1)
        epoch_losses.append(avg_loss)
        # 终端打印 Loss
        print(f"Epoch {epoch:>2}/{EPOCHS}  Loss: {avg_loss:.6f}", flush=True)
        # 每轮落盘一次，机器意外关机时也能保留已完成轮次的权重
        torch.save(model.state_dict(), MODEL_PATH)

    # 3. 绘制真实的 Loss 下降曲线
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, EPOCHS + 1), epoch_losses, marker="o", linewidth=1.5, label="Train Loss")
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("BCELoss")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(CURVE_PATH, dpi=150)
    print(f"Loss 曲线已保存: {CURVE_PATH}")

    # 4. 保存模型权重（测试集划分已在训练开始时落盘），供 evaluate.py 使用
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"模型权重已保存: {MODEL_PATH}")
    print(f"测试集划分已保存: {IDS_PATH}")


if __name__ == "__main__":
    main()
