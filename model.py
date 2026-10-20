import torch
import torch.nn as nn


class CNNFeatureExtractor(nn.Module):
    """
    CNN 特征提取器：将 [B, 1, 128, 512] 的梅尔频谱转换为 [B, 512, hidden_dim] 的时序特征。

    设计思路：
    - 使用 3 个卷积块，每个块包含 Conv2d + BatchNorm2d + ReLU + MaxPool
    - 池化仅在频率维（128 方向）进行，保持时间维 512 不变
    - 频率维变化：128 -> 64 -> 32 -> 16
    - 通道数变化：1 -> 16 -> 32 -> 64
    - 最终展平：[B, 64, 16, 512] -> [B, 512, 64*16] = [B, 512, 1024]
    """

    def __init__(self, cnn_channels: list[int] | None = None):
        """
        Args:
            cnn_channels: 每个卷积块的输出通道数列表，默认 [16, 32, 64]
        """
        super().__init__()

        if cnn_channels is None:
            cnn_channels = [16, 32, 64]

        blocks = []
        in_ch = 1
        for out_ch in cnn_channels:
            blocks.append(self._make_block(in_ch, out_ch))
            in_ch = out_ch

        self.conv_blocks = nn.Sequential(*blocks)
        self.output_dim = cnn_channels[-1] * 16  # 通道数 * 频率维高度

    def _make_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """
        构建单个卷积块：Conv(3x3, padding=1) + BN + ReLU + MaxPool(2x2 on freq)

        MaxPool kernel=(1, 2) 表示仅在频率维（高度）做 2 倍下采样，时间维（宽度）不变。
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),  # 频率维（高度）减半，时间维（宽度）不变
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 1, 128, 512]

        Returns:
            [B, 512, hidden_dim] 的时序特征，供后续 Bi-LSTM 使用
        """
        x = self.conv_blocks(x)  # [B, 64, 16, 512]

        B, C, F, T = x.shape
        # 展平通道和频率维：[B, C, F, T] -> [B, T, C*F]
        x = x.permute(0, 3, 1, 2).reshape(B, T, C * F)

        return x  # [B, 512, 1024]


class BiLSTMClassifier(nn.Module):
    """
    双向 LSTM 分类器：将时序特征映射到 88 个钢琴键的激活概率。

    设计思路：
    - 使用双向 LSTM 捕获前后文上下文（钢琴演奏中，当前音符与前后音符强相关）
    - 输出层使用 Linear + Sigmoid，将每个时间步的特征映射到 88 维
    - Sigmoid 输出 [0, 1] 区间，表示每个键被按下的概率
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 2):
        """
        Args:
            input_dim:  输入特征维度（来自 CNN 的 hidden_dim）
            hidden_dim: LSTM 隐藏层维度（双向，实际输出为 2*hidden_dim）
            num_layers: LSTM 层数，层数越多表达能力越强
        """
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,  # 双向 LSTM
            dropout=0.3 if num_layers > 1 else 0.0,  # 多层时加 dropout 防过拟合
        )

        # 双向 LSTM 输出维度是 2*hidden_dim
        self.fc = nn.Linear(2 * hidden_dim, 88)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 512, hidden_dim] 时序特征

        Returns:
            [B, 512, 88] 每个时间步 88 个键的激活概率
        """
        lstm_out, _ = self.lstm(x)  # [B, 512, 2*hidden_dim]
        out = self.fc(lstm_out)     # [B, 512, 88]
        out = self.sigmoid(out)     # [B, 512, 88]，概率值

        return out


class PianoErrorModel(nn.Module):
    """
    钢琴错误检测模型：CNN + Bi-LSTM 架构

    数据流：
    [B, 1, 128, 512]  (梅尔频谱)
        ↓ CNN
    [B, 512, 1024]    (时序特征)
        ↓ Bi-LSTM
    [B, 512, 88]      (键激活概率，Sigmoid 输出)

    适用任务：多标签分类，每个时间步独立判断 88 个键是否被按下
    """

    def __init__(
        self,
        cnn_channels: list[int] | None = None,
        lstm_hidden_dim: int = 256,
        lstm_num_layers: int = 2,
    ):
        """
        Args:
            cnn_channels:    CNN 各块通道数，默认 [16, 32, 64]
            lstm_hidden_dim: LSTM 隐藏层维度，默认 256
            lstm_num_layers: LSTM 层数，默认 2
        """
        super().__init__()

        self.cnn = CNNFeatureExtractor(cnn_channels)
        self.lstm = BiLSTMClassifier(
            input_dim=self.cnn.output_dim,
            hidden_dim=lstm_hidden_dim,
            num_layers=lstm_num_layers,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 1, 128, 512] 梅尔频谱输入

        Returns:
            [B, 512, 88] 每个时间步 88 个键的激活概率（Sigmoid 输出）
        """
        features = self.cnn(x)  # [B, 512, 1024]
        output = self.lstm(features)  # [B, 512, 88]

        return output


if __name__ == "__main__":
    # 测试模型前向传播
    model = PianoErrorModel()
    dummy_input = torch.randn(4, 1, 128, 512)  # batch=4
    output = model(dummy_input)

    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输出范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
