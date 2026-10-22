"""钢琴错误检测系统 - Gradio 演示网页。

完整流程：
    上传演奏音频（WAV / MP3）或视频（MP4，自动抽取音轨）+ 参考 MIDI
        → 提取 log-mel 频谱（与 make_dataset.py 相同参数）
        → PianoErrorModel 预测每帧 88 键激活概率
        → post_process 转换为音符事件序列
        → dtw_align 与参考 MIDI 音符 DTW 对齐找错
        → 页面展示 JSON 错误列表 + 时间轴标红图
        → 页面下方展示 loss_curve.png / dtw_path.png / confusion_matrix.png

运行方式：
    python app.py
    然后浏览器打开 http://127.0.0.1:7860
"""

import os
import subprocess
import tempfile

import gradio as gr
import librosa
import matplotlib

# 无图形界面环境下只做文件输出
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pretty_midi
import torch

from dtw_align import midi_to_name
from model import PianoErrorModel
from post_process import post_process

# ---------------- 常量（与 make_dataset.py / piano_dataset.py 保持一致）----------------
SR = 22050       # 采样率
HOP = 512        # STFT hop 长度
N_MELS = 128     # mel 频率维数
TARGET_LEN = 512 # 模型输入时间帧数
THRESHOLD = 0.6  # 概率阈值：> 0.6 视为琴键按下（收紧以减少误报）
MIN_DURATION = 5 # 最短持续帧数（去毛刺）
FPS = SR / HOP   # 帧率 ≈ 43.07 帧/秒，用于帧号 <-> 秒换算
EMA_ALPHA = 0.5  # EMA 平滑系数（0 表示不平滑），抑制帧间抖动
PITCH_MIN = 36   # 音域下限 MIDI（C2）
PITCH_MAX = 96   # 音域上限 MIDI（C7）
ANALYSIS_SEC = TARGET_LEN / FPS  # 模型单次可分析时长 ≈ 11.9 秒

# ---------------- 支持的输入格式 ----------------
# 音频容器：librosa（soundfile 后端）可直接读取
AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")
# 视频容器：需先用 ffmpeg 抽取音轨再分析
VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".avi", ".webm")
SUPPORTED_EXTS = AUDIO_EXTS + VIDEO_EXTS

# ---------------- 路径 ----------------
_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_HERE, "piano_model.pth")
LOSS_PNG = os.path.join(_HERE, "loss_curve.png")
DTW_PNG = os.path.join(_HERE, "dtw_path.png")
CM_PNG = os.path.join(_HERE, "confusion_matrix.png")

# Windows 下显示中文
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# ---------------- 启动时加载一次模型 ----------------
print(f"正在加载模型: {MODEL_PATH}")
model = PianoErrorModel()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()
print("模型加载完成")


def _pad_or_truncate(arr: np.ndarray, target_len: int) -> np.ndarray:
    """沿第 0 维截断或补零到 target_len（与 piano_dataset.py 一致）。"""
    T = arr.shape[0]
    if T >= target_len:
        return arr[:target_len]
    pad_width = [(0, target_len - T)] + [(0, 0)] * (arr.ndim - 1)
    return np.pad(arr, pad_width, mode="constant", constant_values=0)


def _find_ffmpeg() -> str | None:
    """定位 ffmpeg 可执行文件：优先 imageio-ffmpeg 自带二进制，其次系统 PATH。"""
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        from shutil import which

        return which("ffmpeg")


def ensure_wav(path: str) -> tuple[str, bool]:
    """把上传文件统一成可供 librosa 读取的 WAV 路径。

    - 音频格式（WAV/MP3/FLAC/OGG/M4A）：直接返回原路径，is_temp=False
    - 视频格式（MP4/MOV/MKV...）：调用 ffmpeg 抽取音轨到临时 WAV，
      返回 (临时路径, True)，调用方负责删除临时文件
    """
    ext = os.path.splitext(path)[1].lower()
    if ext not in VIDEO_EXTS:
        return path, False

    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        raise gr.Error("未找到 ffmpeg，无法从视频中抽取音轨。请先执行：pip install imageio-ffmpeg")

    tmp = tempfile.NamedTemporaryFile(suffix="_extracted.wav", delete=False, dir=tempfile.gettempdir())
    tmp_path = tmp.name
    tmp.close()

    # -vn 丢弃视频流，-ac 1 单声道，-ar 统一采样率，保证与训练参数一致
    cmd = [ffmpeg, "-y", "-i", path, "-vn", "-ac", "1", "-ar", str(SR), "-f", "wav", tmp_path]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")

    if proc.returncode != 0 or not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        detail = (proc.stderr or "").strip()[-300:]
        raise gr.Error(f"音轨提取失败（文件可能无音轨或已损坏）：{detail}")

    return tmp_path, True


def extract_mel(audio_path: str) -> tuple[np.ndarray, float]:
    """音频/视频文件 -> (log-mel 频谱 [512, 128], 音频时长秒)。"""
    wav_path, is_temp = ensure_wav(audio_path)
    try:
        y, _ = librosa.load(wav_path, sr=SR)
    finally:
        # 只清理自己生成的临时 WAV，不碰用户上传的原文件
        if is_temp:
            try:
                os.remove(wav_path)
            except OSError:
                pass
    duration = len(y) / SR
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, hop_length=HOP).T  # [T, 128]
    mel = librosa.power_to_db(mel)
    return _pad_or_truncate(mel, TARGET_LEN), duration


def predict_notes(mel: np.ndarray) -> list[dict]:
    """模型预测 -> post_process 转音符序列 [{pitch, start_time, end_time}]。"""
    x = torch.from_numpy(mel.T).float().unsqueeze(0).unsqueeze(0)  # [1, 1, 128, 512]
    with torch.no_grad():
        prob = model(x)[0].numpy()  # [512, 88]
    notes = post_process(
        prob,
        threshold=THRESHOLD,
        min_duration=MIN_DURATION,
        fps=FPS,
        ema_alpha=EMA_ALPHA,
        pitch_min=PITCH_MIN,
        pitch_max=PITCH_MAX,
    )
    return notes


def midi_to_notes(midi_path: str, audio_duration: float | None = None) -> list[dict]:
    """参考 MIDI -> 音符序列 [{pitch, start_time, end_time}]，仅保留钢琴 88 键范围。

    若提供 audio_duration，则将 MIDI 的时间轴线性归一化到该时长，
    用于处理 tempo 编码异常的 MIDI（如 original.mid 每音 12.8s）。
    """
    mid = pretty_midi.PrettyMIDI(midi_path)
    mid_dur = mid.get_end_time()
    scale = 1.0
    if audio_duration is not None and mid_dur > 0:
        scale = audio_duration / mid_dur
    notes = []
    for inst in mid.instruments:
        for n in inst.notes:
            if 21 <= n.pitch <= 108:
                notes.append({
                    "pitch": n.pitch,
                    "start_time": float(n.start * scale),
                    "end_time": float(n.end * scale),
                })
    notes.sort(key=lambda n: n["start_time"])
    return notes


def plot_timeline(ref_notes: list[dict], perf_notes: list[dict], errors: list[dict]) -> plt.Figure:
    """绘制时间轴：参考音符（蓝）、检测音符（绿），错误位置标红并加文字标注。"""
    fig, ax = plt.subplots(figsize=(12, 5))

    # 底层：参考（蓝实条）与检测（绿描边条）
    for n in ref_notes:
        ax.barh(n["pitch"], n["end_time"] - n["start_time"], left=n["start_time"],
                height=1.6, color="#4C88D6", alpha=0.55, edgecolor="none", zorder=2)
    for n in perf_notes:
        ax.barh(n["pitch"], n["end_time"] - n["start_time"], left=n["start_time"],
                height=1.6, color="none", edgecolor="#3AA655", linewidth=1.4, zorder=3)

    # 错误标红 + 文字标注
    t_max = 1.0
    for e in errors:
        if e["type"] == "wrong":
            # 参考期望位置（橙）与实际演奏位置（红）都标出
            ax.barh(_pitch_by_name(e["expected"]), 0.3,
                    left=e["ref_onset"], height=2.2, color="#F5A623", zorder=4)
            ax.barh(_pitch_by_name(e["played"]), 0.3, left=e["played_onset"],
                    height=2.2, color="#E23B3B", zorder=4)
            ax.annotate(f"错音 {e['expected']}→{e['played']}",
                        (e["played_onset"], _pitch_by_name(e["played"]) + 3),
                        color="#C02020", fontsize=10, fontweight="bold", zorder=5)
            t_max = max(t_max, e["played_onset"])
        elif e["type"] == "missed":
            ax.barh(_pitch_by_name(e["expected"]), 0.3, left=e["ref_onset"],
                    height=2.2, color="none", edgecolor="#E23B3B", linewidth=2.2, zorder=4)
            ax.annotate(f"漏音 {e['expected']}", (e["ref_onset"], _pitch_by_name(e["expected"]) + 3),
                        color="#C02020", fontsize=10, fontweight="bold", zorder=5)
            t_max = max(t_max, e["ref_onset"])
        elif e["type"] == "extra":
            ax.barh(_pitch_by_name(e["played"]), 0.3, left=e["played_onset"],
                    height=2.2, color="#E23B3B", zorder=4)
            ax.annotate(f"多弹 {e['played']}", (e["played_onset"], _pitch_by_name(e["played"]) + 3),
                        color="#C02020", fontsize=10, fontweight="bold", zorder=5)
            t_max = max(t_max, e["played_onset"])

    # 坐标轴：y 轴每隔一个八度（C 音）标注音名
    ax.set_ylim(20, 109)
    ticks = [p for p in range(24, 109, 12)]
    ax.set_yticks(ticks)
    ax.set_yticklabels([midi_to_name(p) for p in ticks])
    ax.set_xlim(0, max(t_max + 2.0, 5.0))
    ax.set_xlabel("时间（秒）")
    ax.set_ylabel("音高")
    ax.set_title("演奏时间轴（蓝=参考 MIDI，绿框=模型检测，红=错误）")

    # 图例（用代理句柄）
    from matplotlib.patches import Patch
    handles = [Patch(color="#4C88D6", alpha=0.55, label="参考音符"),
               Patch(facecolor="none", edgecolor="#3AA655", label="检测音符"),
               Patch(color="#E23B3B", label="错误")]
    ax.legend(handles=handles, loc="upper right")
    fig.tight_layout()
    return fig


def _pitch_by_name(name: str) -> int:
    """音名转 MIDI 编号（复用 dtw_align 的实现逻辑，避免反向依赖私有函数）。"""
    from dtw_align import name_to_midi
    return name_to_midi(name)


def analyze(audio_path, midi_path):
    """主处理流程：抽取音轨（视频）→ 频谱 → 预测 → 音符 → DTW 找错 → JSON + 时间轴图。"""
    if not audio_path:
        raise gr.Error("请先上传演奏录音或视频（WAV / MP3 / MP4）")
    if not midi_path:
        raise gr.Error("请先上传参考乐谱 MIDI 文件")

    # 1. 提取频谱（MP4 等视频格式会先抽音轨） + 2. 模型预测 + 3. 转音符序列
    mel, audio_dur = extract_mel(audio_path)
    perf_notes = predict_notes(mel)

    # 4. 参考 MIDI 音符序列（按音频时长归一化 tempo，兼容异常 MIDI）
    ref_notes = midi_to_notes(midi_path, audio_duration=audio_dur)
    if not ref_notes:
        raise gr.Error("参考 MIDI 中没有可用的钢琴音符（需 MIDI 21-108 范围）")

    # 5. DTW 对齐找错
    from dtw_align import dtw_align
    D, path, errors = dtw_align(ref_notes, perf_notes)

    summary = {
        "num_ref_notes": len(ref_notes),
        "num_detected_notes": len(perf_notes),
        "num_wrong": sum(1 for e in errors if e["type"] == "wrong"),
        "num_missed": sum(1 for e in errors if e["type"] == "missed"),
        "num_extra": sum(1 for e in errors if e["type"] == "extra"),
        "dtw_total_cost": round(float(D[-1, -1]), 3),
        "errors": errors,
    }

    # 6. 时间轴标红图
    fig = plot_timeline(ref_notes, perf_notes, errors)
    return summary, fig


with gr.Blocks(title="钢琴错误检测系统") as demo:
    gr.Markdown(
        """
        # 钢琴错误检测系统
        上传**演奏音频（WAV / MP3）**或**演奏视频（MP4）**，以及**参考乐谱（MIDI）**。
        视频会自动抽取音轨；系统提取频谱、用 CNN+Bi-LSTM 模型逐帧检测琴键激活，
        转换为音符序列后与参考做 DTW 对齐，找出**错音 / 漏音 / 多弹**。
        """
    )
    with gr.Row():
        audio_in = gr.File(
            label="演奏录音 / 视频（WAV / MP3 / MP4）",
            file_types=[".wav", ".mp3", ".mp4"],
            file_count="single",
            type="filepath",
        )
        midi_in = gr.File(label="参考乐谱 MIDI", file_types=[".mid", ".midi"])
    btn = gr.Button("开始分析", variant="primary")

    out_json = gr.JSON(label="错误列表（JSON）")
    out_timeline = gr.Plot(label="时间轴（红=错误）")

    gr.Markdown("### 训练与评估产物")
    with gr.Row():
        img_loss = gr.Image(LOSS_PNG, label="训练 Loss 曲线", interactive=False)
        img_dtw = gr.Image(DTW_PNG, label="DTW 对齐路径（自测）", interactive=False)
        img_cm = gr.Image(CM_PNG, label="测试集混淆矩阵", interactive=False)

    btn.click(analyze, inputs=[audio_in, midi_in], outputs=[out_json, out_timeline])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=False, share=True)
