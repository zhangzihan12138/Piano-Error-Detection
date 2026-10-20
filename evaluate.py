"""评估与诊断脚本：在测试集上逐帧评估音符检测性能，并诊断模型输出质量。

流程：
1. 数据集健全性检查：统计原始 mel/label 的帧数分布，确认样本是"截断"还是"补零"，
   以及标签中正样本（有音符帧）的占比 —— 用于排查补零帧是否污染训练信号
2. 模型输出诊断：统计帧级 88 维输出 > 阈值的比例，确认模型是否"饱和"（把大量无音符帧判成有音符）
3. 多阈值指标对比：在 0.5 / 0.6 / 0.7 三个阈值下分别计算
   准确率 / 精确率 / 召回率 / F1，判断阈值是否应上调
4. 绘制混淆矩阵保存为 confusion_matrix.png，并输出评估报告 evaluation_report.txt

运行方式（需先完成 train.py 训练）：
    python evaluate.py
"""

import glob
import json
import os

import matplotlib

# 无图形界面环境下只做文件输出
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score)
from torch.utils.data import DataLoader

from model import PianoErrorModel
from piano_dataset import PianoDataset

# ---------------- 配置 ----------------
THRESHOLD = 0.5                    # 主阈值（用于混淆矩阵与报告）
CANDIDATE_THRESHOLDS = [0.5, 0.6, 0.7]  # 待对比的阈值
BATCH_SIZE = 8
TARGET_LEN = 512
PAD_RATIO_WARN = 0.5                # 补零样本占比超过此值则提示风险

# ---------------- 路径 ----------------
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(_HERE, "data", "features")
MODEL_PATH = os.path.join(_HERE, "piano_model.pth")
IDS_PATH = os.path.join(_HERE, "test_ids.json")
CM_PATH = os.path.join(_HERE, "confusion_matrix.png")
REPORT_PATH = os.path.join(_HERE, "evaluation_report.txt")

# Windows 下显示中文
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


def check_dataset(raw_ids: list[int]) -> dict:
    """数据集健全性检查：帧数分布（截断 vs 补零）与标签正样本占比。"""
    lengths = []
    pos_ratio = []
    for sid in raw_ids:
        mel = np.load(os.path.join(DATA_ROOT, f"mel_{sid}.npy"), mmap_mode="r")
        lbl = np.load(os.path.join(DATA_ROOT, f"label_{sid}.npy"), mmap_mode="r")
        lengths.append(int(mel.shape[0]))
        pos_ratio.append(float(np.asarray(lbl, dtype=np.float32).mean()))

    lengths = np.array(lengths)
    n_pad = int((lengths < TARGET_LEN).sum())      # 短于 512 帧 -> 会被补零
    n_trunc = int((lengths >= TARGET_LEN).sum())   # 长于等于 512 帧 -> 会被截断
    print("=== 1. 数据集健全性检查 ===")
    print(f"  样本总数           : {len(lengths)}")
    print(f"  原始帧数范围       : {lengths.min()} ~ {lengths.max()}（均值 {lengths.mean():.0f}）")
    print(f"  需补零样本数(<512) : {n_pad}  ({n_pad / len(lengths):.1%})")
    print(f"  需截断样本数(>=512): {n_trunc} ({n_trunc / len(lengths):.1%})")
    print(f"  标签正样本占比     : {np.mean(pos_ratio):.4f}"
          f"（负:正 ≈ {1 / max(np.mean(pos_ratio), 1e-9):.1f}:1）")
    if n_pad / len(lengths) > PAD_RATIO_WARN:
        print(f"  ⚠️ 补零样本占比超过 {PAD_RATIO_WARN:.0%}，补零帧的零标签会稀释训练信号，"
              f"建议改用 loss masking 或随机裁剪")
    else:
        print("  ✅ 样本以截断为主，补零帧占比低，标签未被补零帧污染")
    print()
    return {"lengths": lengths, "n_pad": n_pad, "n_trunc": n_trunc,
            "pos_ratio": float(np.mean(pos_ratio))}


def main() -> None:
    with open(IDS_PATH, "r", encoding="utf-8") as f:
        split = json.load(f)
    test_ids = split["test_ids"]

    # ---------- 1. 数据集健全性检查（用原始文件，避免被 Dataset 的 pad 掩盖）----------
    raw_ids = sorted(int(os.path.splitext(os.path.basename(p))[0].split("_")[-1])
                     for p in glob.glob(os.path.join(DATA_ROOT, "mel_*.npy")))
    ds_info = check_dataset(raw_ids)

    # ---------- 2. 推理：收集全部逐帧概率 ----------
    test_set = PianoDataset(DATA_ROOT, ids=test_ids)
    loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
    model = PianoErrorModel()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    prob_list, label_list = [], []
    with torch.no_grad():
        for mel, label in loader:
            prob = model(mel)                     # [B, 512, 88] 概率
            prob_list.append(prob.numpy().reshape(-1))
            label_list.append(label.numpy().reshape(-1))
    prob = np.concatenate(prob_list)
    y_true = np.concatenate(label_list)
    total = prob.size

    print("=== 2. 模型输出诊断 ===")
    print(f"  测试样本数        : {len(test_set)}")
    print(f"  逐帧判定点总数    : {total:,}（样本数 x 512 帧 x 88 键）")
    print(f"  输出概率均值      : {prob.mean():.4f}")
    print(f"  输出概率分位数    : P50={np.percentile(prob, 50):.3f} "
          f"P90={np.percentile(prob, 90):.3f} P99={np.percentile(prob, 99):.3f}")
    for t in CANDIDATE_THRESHOLDS:
        ratio = float((prob > t).mean())
        print(f"  >{t} 的比例        : {ratio:.2%}" + ("   ← 饱和！" if ratio > 0.5 else ""))
    sat = float((prob > THRESHOLD).mean())
    if sat > 0.5:
        print(f"  ⚠️ 超过一半的帧被判为有音符，模型输出饱和（根因见上文诊断结论）")
    else:
        print("  ✅ 输出未饱和，模型能区分有音符/无音符帧")
    print()

    # ---------- 3. 多阈值指标对比 ----------
    print("=== 3. 多阈值指标对比 ===")
    header = f"  {'阈值':<6}{'准确率':>10}{'精确率':>10}{'召回率':>10}{'F1':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    best = None
    for t in CANDIDATE_THRESHOLDS:
        p = (prob > t)
        acc = accuracy_score(y_true, p)
        pre = precision_score(y_true, p, zero_division=0)
        rec = recall_score(y_true, p, zero_division=0)
        f1 = f1_score(y_true, p, zero_division=0)
        print(f"  {t:<6}{acc:>10.4f}{pre:>10.4f}{rec:>10.4f}{f1:>10.4f}")
        if best is None or f1 > best[1]:
            best = (t, f1, acc, pre, rec)
    print(f"  → F1 最优阈值: {best[0]}（F1={best[1]:.4f}）")

    # ---------- 4. 混淆矩阵（主阈值）----------
    p_main = (prob > THRESHOLD)
    cm = confusion_matrix(y_true, p_main, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    print(f"\n=== 4. 混淆矩阵（阈值 {THRESHOLD}，行=真实，列=预测）===")
    print(f"  TP={tp:,}  FP={fp:,}  FN={fn:,}  TN={tn:,}")
    if fp > tp:
        print(f"  ⚠️ FP({fp:,}) > TP({tp:,})：误报多于命中，主要来自模型判别的无音符帧")

    # ---------- 5. 报告与图片 ----------
    lines = [
        "=" * 60,
        "钢琴错误检测模型 - 测试集评估与诊断报告",
        "=" * 60,
        f"测试样本数          : {len(test_set)}",
        f"逐帧判定点总数      : {total:,}（样本数 x 512 帧 x 88 键）",
        "-" * 60,
        "[数据集健全性]",
        f"  原始帧数范围      : {ds_info['lengths'].min()} ~ {ds_info['lengths'].max()}"
        f"（均值 {ds_info['lengths'].mean():.0f}）",
        f"  需补零样本(<512)  : {ds_info['n_pad']}",
        f"  需截断样本(>=512) : {ds_info['n_trunc']}",
        f"  标签正样本占比    : {ds_info['pos_ratio']:.4f}",
        "-" * 60,
        "[模型输出诊断]",
        f"  概率均值          : {prob.mean():.4f}",
        f"  >{THRESHOLD} 的比例       : {sat:.2%}",
        "-" * 60,
        "[多阈值指标]",
        f"  {'阈值':<6}{'准确率':>10}{'精确率':>10}{'召回率':>10}{'F1':>10}",
    ]
    for t in CANDIDATE_THRESHOLDS:
        p = (prob > t)
        lines.append(f"  {t:<6}{accuracy_score(y_true, p):>10.4f}"
                     f"{precision_score(y_true, p, zero_division=0):>10.4f}"
                     f"{recall_score(y_true, p, zero_division=0):>10.4f}"
                     f"{f1_score(y_true, p, zero_division=0):>10.4f}")
    lines += [
        f"  → F1 最优阈值      : {best[0]}",
        "-" * 60,
        f"[混淆矩阵 @ 阈值 {THRESHOLD}]（行=真实，列=预测）",
        f"                    预测:无音符   预测:有音符",
        f"  真实:无音符(0)     {tn:>10,}   {fp:>10,}",
        f"  真实:有音符(1)     {fn:>10,}   {tp:>10,}",
        "=" * 60,
    ]
    report = "\n".join(lines)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\n评估报告已保存: {REPORT_PATH}")

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["无音符(0)", "有音符(1)"])
    disp.plot(cmap="Blues", values_format=",")
    plt.title(f"混淆矩阵（测试集, 阈值={THRESHOLD}）")
    plt.tight_layout()
    plt.savefig(CM_PATH, dpi=150)
    plt.close()
    print(f"混淆矩阵已保存: {CM_PATH}")


if __name__ == "__main__":
    main()
