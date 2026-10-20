# 钢琴错误检测系统

基于 **CNN + Bi-LSTM** 的钢琴演奏错误检测。系统读取演奏音频（WAV）与参考乐谱（MIDI），
逐帧检测 88 个琴键的激活状态，转换为音符序列后与参考做 **DTW 对齐**，自动判定
**错音 / 漏音 / 多弹**，并给出可视化时间轴与评估指标。

## 功能概览

| 模块 | 说明 |
| --- | --- |
| `make_dataset.py` | 数据合成：基于 `original.mid` 变速 + 注入错误，生成 mel 频谱与标签 |
| `piano_dataset.py` | `PianoDataset`：读取 `mel_*.npy` / `label_*.npy`，补齐/截断到 512 帧 |
| `model.py` | `PianoErrorModel`：CNN 提特征 + Bi-LSTM 逐帧输出 88 键概率 |
| `train.py` | 训练：8:2 划分训练/测试集，BCELoss + Adam，20 epoch，绘制 Loss 曲线 |
| `evaluate.py` | 评估：阈值 0.5 判定音符，sklearn 计算准确率/精确率/召回率/F1 + 混淆矩阵 |
| `post_process.py` | 后处理：概率矩阵 → 离散音符事件序列（`{pitch, start_time, end_time}`） |
| `dtw_align.py` | DTW 对齐找错：输出 JSON 错误列表 + DTW 路径图 |
| `app.py` | Gradio 网页：上传 WAV + MIDI → 预测 → DTW → JSON + 时间轴标红 + 展示评估图 |

## 环境安装

需要 Python 3.10+（本项目在 3.12 上验证）。

```powershell
# 1. 创建虚拟环境
python -m venv .venv

# 2. 激活虚拟环境（PowerShell）
.\.venv\Scripts\Activate.ps1

# 3. 安装依赖
pip install -r requirements.txt
```

> **注意**：`requirements.txt` 中的 `torch` 为 CPU 版本。如需 GPU 加速，请先按
> [PyTorch 官网](https://pytorch.org) 指引安装对应 CUDA 版本的 torch，再安装其余依赖。

## 运行

在项目根目录（`Piano_Project/`）下按顺序执行：

```powershell
# 1. 合成数据集（生成 data/features/ 下的 .npy，并输出 mel_spectrogram.png）
python make_dataset.py

# 2. 训练模型（20 epoch，输出 loss_curve.png、piano_model.pth、test_ids.json）
python train.py

# 3. 在测试集上评估（输出 evaluation_report.txt、confusion_matrix.png）
python evaluate.py

# 4. DTW 对齐模块自测（输出 dtw_result.json、dtw_path.png）
python dtw_align.py

# 5. 启动网页演示，浏览器访问 http://127.0.0.1:7860
python app.py
```

> **执行顺序很重要**：`train.py` 会生成 `piano_model.pth` 与 `test_ids.json`，
> `evaluate.py` 和 `app.py` 都依赖它们。因此请先完成训练。

## 数据合成流程

`make_dataset.py` 以 `original.mid` 为基准，合成 1000 段"带错误的演奏"，具体步骤：

1. **读取基准乐谱**：加载 `original.mid`，取出全部音符事件。
2. **速度扰动**：对每个样本施加 0.9~1.1 的随机速度缩放（`n.start *= spd; n.end *= spd`），
   模拟演奏中的快慢变化。
3. **注入演奏错误**（`inject_errors`，每个样本随机 0~2 处）：
   - `wrong`（错音）：将某个音符的音高偏移 ±1 或 ±2 个半音
   - `miss`（漏音）：删除某个音符
   - `extra`（多弹）：额外插入一个随机音符（MIDI 55~72，时长 0.4s）
4. **波形合成**（`midi_to_audio`）：将 MIDI 音符渲染为音频——
   每个音符用基频 + 2 次谐波 + 3 次谐波的叠加，并乘以指数衰减包络，
   按起始时间叠加到整段波形上。
5. **特征提取**：用 librosa 计算 log-mel 频谱（`SR=22050, hop_length=512, n_mels=128`），
   转置为 `[T, 128]` 保存为 `mel_*.npy`。
6. **标签生成**（`midi_to_label`）：按 `帧号 = 时间 × SR / hop_length` 将音符区间
   映射为 `[T, 88]` 的多热标签（`lab[start_frame:end_frame, pitch-21] = 1`），
   保存为 `label_*.npy`。
7. **可视化**：读取 `mel_0.npy` 绘制 mel 频谱图存为 `mel_spectrogram.png`。

### 关键参数

| 参数 | 值 | 说明 |
| --- | --- | --- |
| `SR` | 22050 | 采样率 |
| `HOP` | 512 | STFT hop 长度 |
| `N_MELS` | 128 | mel 频率维数 |
| `N_SAMPLES` | 1000 | 生成样本数 |
| 帧率 | `SR/HOP ≈ 43.07` | 帧号与秒的换算，`post_process.py` 中需保持一致 |

### 数据约定

数据集遵循 `piano_dataset.py` 中的约定，`mel` 与 `label` 的时间维一一对应：

```
mel_{id}.npy    -> [T, 128]  log-mel 频谱
label_{id}.npy  -> [T, 88]   每个时刻 88 键的激活标签（21=A0 ... 108=C8）
```

训练时 `PianoDataset` 会把两者补齐/截断到 `TARGET_LEN=512` 帧。

## 输出产物

| 文件 | 说明 |
| --- | --- |
| `data/features/*.npy` | 合成数据集 |
| `mel_spectrogram.png` | 样本 mel 频谱可视化 |
| `loss_curve.png` | 训练 Loss 下降曲线 |
| `piano_model.pth` | 训练好的模型权重 |
| `test_ids.json` | 训练/测试集划分（固定种子，保证可复现） |
| `evaluation_report.txt` | 测试集评估报告（准确率/精确率/召回率/F1 + 混淆矩阵） |
| `confusion_matrix.png` | 测试集混淆矩阵图 |
| `dtw_result.json` | DTW 对齐自测的 JSON 错误列表 |
| `dtw_path.png` | DTW 累积代价矩阵与最优路径图 |

## 常见问题

- **matplotlib 中文显示为方框**：代码中已设置 `SimHei` / `Microsoft YaHei` 字体，
  若系统缺少这些字体，请安装中文字体或修改脚本顶部的 `font.sans-serif` 配置。
- **首次运行 `make_dataset.py` 较慢**：librosa/numba 首次调用需要 JIT 编译，
  通常只影响第一批样本，之后会明显加速。
- **磁盘空间**：每个样本的 mel 约 20MB，1000 个样本约需 20GB 以上空间。
