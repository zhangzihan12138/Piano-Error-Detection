import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset

TARGET_LEN = 512


def _pad_or_truncate(arr: np.ndarray, target_len: int) -> np.ndarray:
    """沿第 0 维截断或 zero-pad 到 target_len。

    补齐策略：在时间维末尾补 0（zero-padding）。
    - 对 mel 频谱而言，0 值近似静音（log-mel 能量最低），不会引入伪信号。
    - 对 one-hot / multi-hot 标签而言，0 表示"无音符触发"，语义上等价于静音帧。
    - 若后续模型使用 attention 或 loss masking，可额外返回一个 length 字段，
      让模型忽略 padding 帧；当前实现仅做 zero-pad，保持接口简洁。
    """
    T = arr.shape[0]
    if T >= target_len:
        return arr[:target_len]
    pad_width = [(0, target_len - T)] + [(0, 0)] * (arr.ndim - 1)
    return np.pad(arr, pad_width, mode="constant", constant_values=0)


class PianoDataset(Dataset):
    """
    数据约定：
        mel_*.npy   -> [T, 128]  log-mel 频谱
        label_*.npy -> [T, 88]   每个时刻 88 键的激活标签

    返回：
        mel   : [1, 128, 512]  (batch, freq, time) 的图像式输入
        label : [512, 88]      时间维在前，便于逐帧计算 loss
    """

    def __init__(self, root: str, ids: list[int] | None = None):
        """
        Args:
            root: 存放 .npy 文件的目录。
            ids:  显式指定样本 ID 列表（如 [0, 1, 3]）。
                  若为 None，则自动扫描 root 下所有 mel_*.npy 并提取 ID。
        """
        self.root = root
        if ids is None:
            ids = sorted(
                int(os.path.splitext(os.path.basename(p))[0].split("_")[-1])
                for p in glob.glob(os.path.join(root, "mel_*.npy"))
            )
        self.ids = list(ids)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample_id = self.ids[index]
        mel_path = os.path.join(self.root, f"mel_{sample_id}.npy")
        lbl_path = os.path.join(self.root, f"label_{sample_id}.npy")

        mel = np.load(mel_path)    # [T, 128]
        label = np.load(lbl_path)  # [T, 88]

        mel = _pad_or_truncate(mel, TARGET_LEN)    # [512, 128]
        label = _pad_or_truncate(label, TARGET_LEN)  # [512, 88]

        # mel: [512, 128] -> [128, 512] -> [1, 128, 512]
        mel = torch.from_numpy(mel).float().transpose(0, 1).unsqueeze(0)
        # label: [512, 88]，不转置
        label = torch.from_numpy(label).float()

        return mel, label
