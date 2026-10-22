# 钢琴错误检测系统

基于 **CNN + Bi-LSTM** 的钢琴演奏错误检测。系统读取演奏音频（WAV / MP3）或演奏视频（MP4，自动抽取音轨）与参考乐谱（MIDI），
逐帧检测 88 个琴键的激活状态，转换为音符序列后与参考做 **DTW 对齐**，自动判定
**错音 / 漏音 / 多弹**，并给出可视化时间轴与评估指标。

## 功能概览

| 模块 | 说明 |
| --- | --- |
| `make_dataset.py` | 数据合成：从 `midi_songs/` 多曲目语料随机选曲，移调 + 变速 + 注入错误，SoundFont 渲染后生成 mel 频谱与标签 |
| `piano_dataset.py` | `PianoDataset`：读取 `mel_*.npy` / `label_*.npy`，补齐/截断到 512 帧 |
| `model.py` | `PianoErrorModel`：CNN 提特征 + Bi-LSTM 逐帧输出 88 键概率 |
| `train.py` | 训练：8:2 划分训练/测试集，BCELoss + Adam，20 epoch，绘制 Loss 曲线 |
| `evaluate.py` | 评估：阈值 0.5 判定音符，sklearn 计算准确率/精确率/召回率/F1 + 混淆矩阵 |
| `post_process.py` | 后处理：概率矩阵 → 离散音符事件序列（`{pitch, start_time, end_time}`） |
| `dtw_align.py` | DTW 对齐找错：输出 JSON 错误列表 + DTW 路径图 |
| `app.py` | Gradio 网页：上传 WAV / MP3 / MP4 + MIDI → 预测 → DTW → JSON + 时间轴标红 + 展示评估图 |

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

## 快速运行（仅启动网页演示）

仓库已内置训练好的 `piano_model.pth` 与三张评估图（`loss_curve.png` / `dtw_path.png` / `confusion_matrix.png`），
**clone 后无需重新训练**，安装依赖即可直接启动网页：

```powershell
# 1. 创建并激活虚拟环境
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. 安装依赖
pip install -r requirements.txt

# 3. 启动网页
python app.py
```

浏览器访问 <http://127.0.0.1:7860> 。

> `app.py` 中以 `share=True` 启动，联网时会额外生成一个公网临时链接（终端里会打印），
> 不联网时不影响本地访问。

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

`make_dataset.py` 从 `midi_songs/` 目录的 9 首单音曲目中随机选曲，合成 1000 段"带错误的演奏"，具体步骤：

1. **随机选曲**：扫描 `midi_songs/*.mid`（小星星、欢乐颂、两只老虎、生日歌、铃儿响叮当、
   多年以前、奇异恩典、C 大调音阶、琶音），每个样本随机选一首；目录为空时回退到 `original.mid`。
2. **tempo 归一化**：将整曲线性缩放到 14 秒，使 512 帧窗口（约 11.9 秒）能覆盖多个音高。
3. **随机移调**：整体移调 ±5 个半音（自动限制在 MIDI 21~108 内），扩大音高覆盖范围，
   缓解"只见过一首曲子就换歌失效"的泛化问题。
4. **速度扰动**：对每个样本施加 0.9~1.1 的随机速度缩放（`n.start *= spd; n.end *= spd`），
   模拟演奏中的快慢变化。
5. **轻量增强**：力度随机 ±15、起止时间抖动 ±30ms，模拟真实演奏的不完美。
6. **注入演奏错误**（`inject_errors`，每个样本随机 0~2 处）：
   - `wrong`（错音）：将某个音符的音高偏移 ±1 或 ±2 个半音
   - `miss`（漏音）：删除某个音符
   - `extra`（多弹）：额外插入一个随机音符（MIDI 55~76）
7. **音频渲染**（`midi_to_audio`）：用 FluidSynth + `piano.sf2` 钢琴 SoundFont
   将 MIDI 渲染为接近真实钢琴的单声道音频（替换早期版本的正弦叠加）。
8. **特征提取**：用 librosa 计算 log-mel 频谱（`SR=22050, hop_length=512, n_mels=128`），
   转置为 `[T, 128]` 保存为 `mel_*.npy`。
9. **标签生成**（`midi_to_label`）：按 `帧号 = 时间 × SR / hop_length` 将音符区间
   映射为 `[T, 88]` 的多热标签（`lab[start_frame:end_frame, pitch-21] = 1`），
   保存为 `label_*.npy`。
10. **可视化**：读取 `mel_0.npy` 绘制 mel 频谱图存为 `mel_spectrogram.png`。

> 曲目语料全部为单音旋律（无和弦），与模型"逐帧 88 键激活检测"的设定一致。

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

- **仓库里没有 `piano.sf2`**：该 SoundFont 体积约 148 MB，超过 GitHub 单文件 100 MB 限制，
  因此未纳入版本控制。仅在需要重新执行 `make_dataset.py` 合成数据时才需要它：
  自行下载钢琴 SoundFont（如 FluidR3_GM、Salamander Grand Piano）放到项目根目录并命名为 `piano.sf2`。
  **只运行 `app.py` 打开网页演示不需要该文件。**
- **matplotlib 中文显示为方框**：代码中已设置 `SimHei` / `Microsoft YaHei` 字体，
  若系统缺少这些字体，请安装中文字体或修改脚本顶部的 `font.sans-serif` 配置。
- **首次运行 `make_dataset.py` 较慢**：librosa/numba 首次调用需要 JIT 编译，
  通常只影响第一批样本，之后会明显加速。
- **磁盘空间**：每个样本的 mel 约 350 KB、label 约 240 KB，1000 个样本合计约 0.5 GB。
