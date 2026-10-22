"""训练数据集生成脚本：基于多曲目 MIDI 语料 + 钢琴 SoundFont 渲染 + 错误注入。

核心改动（相比单曲目版本）：
  1. 从 midi_songs/ 目录随机选取曲目（9 首单音旋律/音阶/琶音），
     使模型见过更宽的音高分布和更多旋律走向，缓解"换曲子就失效"的泛化问题。
  2. 每段样本随机整体移调 ±5 个半音（自动限制在 MIDI 21~108 内），
     进一步扩大音高覆盖范围。
  3. 用 FluidSynth + 钢琴 SoundFont 渲染 MIDI 音频（替换正弦叠加），
     使训练音色更接近真实钢琴，缓解"音色域不匹配"问题。
  4. 对所有曲目统一做 tempo 归一化到 ~14s，保证 512 帧窗口能覆盖多个音符。
  5. 保留变速（0.9~1.1x）、错误注入（错音/漏音/多弹）逻辑。
  6. 轻量数据增强：随机力度、微小时序抖动。

输出：
  data/features/mel_*.npy   : [T, 128] log-mel 频谱
  data/features/label_*.npy : [T, 88]  逐帧 88 键激活标签
  mel_spectrogram.png       : 首个样本的频谱可视化
"""

import argparse
import glob
import os
import random
from collections import Counter

import librosa
import matplotlib
import numpy as np
import pretty_midi

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- 常量 ----------------
SR, HOP, N_MELS = 22050, 512, 128
N_SAMPLES = 1000
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOUNDFONT_PATH = os.path.join(BASE_DIR, "piano.sf2")
MIDI_DIR = os.path.join(BASE_DIR, "midi_songs")  # 多曲目语料目录
FALLBACK_MIDI = os.path.join(BASE_DIR, "original.mid")
TARGET_DURATION = 14.0  # 归一化后整段 MIDI 的目标时长（秒），使 512 帧窗口覆盖多音符
TRANSPOSE_RANGE = 5     # 每段样本随机整体移调的最大半音数

# 统一使用绝对路径，避免从其他工作目录启动时产物写错位置
DATA_DIR = os.path.join(BASE_DIR, "data", "features")
os.makedirs(DATA_DIR, exist_ok=True)

# 全局复用一个 FluidSynth 实例，避免每个样本重复加载 SoundFont（~140MB）
_synth = None
_sfid = None


def _ensure_fluidsynth_on_path() -> None:
    """Windows 下把含 libfluidsynth-3.dll 的目录加入 PATH。

    pyfluidsynth 通过 ctypes.util.find_library 查找 DLL，只认 PATH；
    本项目的 DLL 随 venv 放在 .venv/Scripts 下，因此这里自动补齐，
    避免每次手动设置环境变量。
    """
    if os.name != "nt":
        return
    for d in (os.path.join(BASE_DIR, ".venv", "Scripts"), os.path.join(BASE_DIR, "fluidsynth", "bin")):
        if os.path.isdir(d) and d.lower() not in os.environ.get("PATH", "").lower():
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
            try:
                os.add_dll_directory(d)  # 同时注册依赖 DLL 的搜索目录
            except (AttributeError, OSError):
                pass


def get_synth():
    """获取（懒加载）全局 FluidSynth 合成器实例。"""
    global _synth, _sfid
    if _synth is None:
        _ensure_fluidsynth_on_path()
        import fluidsynth
        _synth = fluidsynth.Synth(samplerate=SR, gain=0.5)
        _sfid = _synth.sfload(SOUNDFONT_PATH)
        # bank=0, preset=0 → Acoustic Grand Piano（钢琴 SoundFont 的默认钢琴音色）
        _synth.program_select(0, _sfid, 0, 0)
    return _synth


def list_midi_files() -> list[str]:
    """列出 midi_songs/ 下的所有 MIDI 曲目；目录为空时回退到 original.mid。"""
    files = sorted(
        glob.glob(os.path.join(MIDI_DIR, "*.mid"))
        + glob.glob(os.path.join(MIDI_DIR, "*.midi"))
    )
    return files if files else [FALLBACK_MIDI]


def transpose_midi(mid: pretty_midi.PrettyMIDI, max_shift: int = TRANSPOSE_RANGE) -> int:
    """随机整体移调，返回实际移调的半音数（0 表示不移调）。

    移调范围自动收窄，保证所有音高仍落在 MIDI 21~108（对应 88 键标签）内。
    """
    pitches = [n.pitch for inst in mid.instruments for n in inst.notes]
    if not pitches:
        return 0
    low = max(-max_shift, 21 - min(pitches))
    high = min(max_shift, 108 - max(pitches))
    if low > high:
        return 0
    shift = random.randint(low, high)
    if shift:
        for inst in mid.instruments:
            for n in inst.notes:
                n.pitch += shift
    return shift


def normalize_tempo(mid: pretty_midi.PrettyMIDI, target_duration: float = TARGET_DURATION) -> None:
    """将所有曲目统一缩放到 target_duration。

    原始 original.mid 的 tempo 异常（每音 12.8s，共 512s），归一化后可让
    512 帧窗口覆盖多个音符；其他曲目同样归一化，保证各曲目时长一致。
    直接原地缩放所有音符的 start/end。
    """
    old_dur = mid.get_end_time()
    if old_dur <= 0:
        return
    scale = target_duration / old_dur
    for inst in mid.instruments:
        for n in inst.notes:
            n.start *= scale
            n.end *= scale


def midi_to_audio(mid: pretty_midi.PrettyMIDI) -> np.ndarray:
    """用 FluidSynth + 钢琴 SoundFont 将 PrettyMIDI 渲染为单声道音频。

    实现：按时间顺序处理 note_on / note_off 事件，分段渲染音频并拼接。
    返回归一化到 [-1, 1] 的 float32 单声道数组。
    """
    synth = get_synth()

    # 收集所有音符事件（on/off），按时间排序
    events = []
    for inst in mid.instruments:
        for n in inst.notes:
            events.append((n.start, 0, n.pitch, n.velocity))  # type 0 = on
            events.append((n.end, 1, n.pitch, 0))              # type 1 = off
    events.sort(key=lambda e: (e[0], e[1]))  # 同时间 off 先于 on？用 type 保证顺序

    total_duration = mid.get_end_time()
    audio = np.zeros(int((total_duration + 1.0) * SR), dtype=np.float32)  # 多留 1s 给 release

    current_time = 0.0
    for ev_time, ev_type, pitch, vel in events:
        # 渲染 current_time → ev_time 之间的音频
        duration = ev_time - current_time
        if duration > 1e-6:
            n_samples = int(duration * SR)
            samples = synth.get_samples(n_samples)  # 交错立体声 [L, R, L, R, ...]
            left = samples[::2].astype(np.float32)
            start_idx = int(current_time * SR)
            actual = min(len(left), len(audio) - start_idx)
            audio[start_idx:start_idx + actual] += left[:actual]
            current_time = ev_time

        if ev_type == 0:
            synth.noteon(0, pitch, vel)
        else:
            synth.noteoff(0, pitch)

    # 渲染尾部残余（音符 release 衰减），再静默 0.3s
    tail = 0.8
    n_samples = int(tail * SR)
    samples = synth.get_samples(n_samples)
    left = samples[::2].astype(np.float32)
    start_idx = int(current_time * SR)
    actual = min(len(left), len(audio) - start_idx)
    audio[start_idx:start_idx + actual] += left[:actual]

    # 归一化到 [-0.9, 0.9]，避免削波
    peak = float(np.abs(audio).max())
    if peak > 1e-8:
        audio = audio / peak * 0.9
    return audio


def inject_errors(mid: pretty_midi.PrettyMIDI, max_err: int = 2) -> None:
    """随机注入 0~max_err 个演奏错误：错音 / 漏音 / 多弹。"""
    for _ in range(random.randint(0, max_err)):
        notes = mid.instruments[0].notes
        if len(notes) < 2:
            break
        k = random.randrange(len(notes))
        kind = random.choice(["wrong", "miss", "extra"])
        if kind == "wrong":
            notes[k].pitch = min(108, max(21, notes[k].pitch + random.choice([-2, -1, 1, 2])))
        elif kind == "miss":
            del notes[k]
        else:  # extra：在第 k 个音符附近插入一个额外音符
            ref = notes[k]
            dur = max(0.2, (ref.end - ref.start) * 0.8)
            notes.append(pretty_midi.Note(
                velocity=random.randint(70, 110),
                pitch=random.randint(55, 76),
                start=ref.start,
                end=ref.start + dur,
            ))


def add_augmentation(mid: pretty_midi.PrettyMIDI) -> None:
    """轻量数据增强：随机力度 + 微小时序抖动，提升对真实演奏的泛化能力。"""
    for inst in mid.instruments:
        for n in inst.notes:
            # 力度随机化（模拟真实演奏力度变化）
            n.velocity = int(np.clip(n.velocity + random.randint(-15, 15), 30, 120))
            # 微小时序抖动（±30ms），模拟真实演奏的不完美 timing
            jitter = random.uniform(-0.03, 0.03)
            n.start = max(0.0, n.start + jitter)
            n.end = n.end + jitter


def midi_to_label(mid: pretty_midi.PrettyMIDI, n_frames: int) -> np.ndarray:
    """将 MIDI 音符转为 [T, 88] 的逐帧激活标签（88 键对应 MIDI 21~108）。"""
    lab = np.zeros((n_frames, 88), dtype=np.float32)
    for inst in mid.instruments:
        for n in inst.notes:
            s_f = int(n.start * SR / HOP)
            e_f = int(n.end * SR / HOP)
            if 0 <= s_f < n_frames:
                lab[s_f:min(e_f, n_frames), n.pitch - 21] = 1.0
    return lab


def main(start: int = 0, count: int = N_SAMPLES) -> None:
    """生成数据集。

    Args:
        start: 起始样本编号（用于分块生成，避免单次运行超时）。
        count: 本次生成的样本数量，默认 N_SAMPLES=1000。
    每个样本使用独立随机种子（42 + i），因此分块生成与一次性生成的样本完全一致。
    """
    print(f"SoundFont: {SOUNDFONT_PATH}")
    midi_files = list_midi_files()
    print(f"曲目语料: {len(midi_files)} 首 -> {[os.path.basename(p) for p in midi_files]}")
    print(f"目标时长: {TARGET_DURATION}s | 本次生成: {count} 条（起始 {start}） | 随机移调 ±{TRANSPOSE_RANGE} 半音")

    used = Counter()
    for i in range(start, start + count):
        # 每个样本独立种子：保证分块运行也能得到确定、可复现的数据集
        random.seed(42 + i)

        # 0. 随机选一首曲目
        src = random.choice(midi_files)
        used[os.path.basename(src)] += 1
        mid = pretty_midi.PrettyMIDI(src)

        # 1. 归一化 tempo（统一到目标时长）
        normalize_tempo(mid)

        # 2. 随机整体移调（扩大音高覆盖范围）
        transpose_midi(mid)

        # 3. 变速 0.9~1.1 倍
        spd = random.uniform(0.9, 1.1)
        for inst in mid.instruments:
            for n in inst.notes:
                n.start *= spd
                n.end *= spd

        # 4. 轻量数据增强（力度 + 时序抖动）
        add_augmentation(mid)

        # 5. 注入演奏错误
        inject_errors(mid)

        # 6. SoundFont 渲染音频
        audio = midi_to_audio(mid)

        # 7. 提取 log-mel 频谱
        mel = librosa.feature.melspectrogram(y=audio, sr=SR, n_mels=N_MELS, hop_length=HOP).T
        mel = librosa.power_to_db(mel)
        lab = midi_to_label(mid, mel.shape[0])

        np.save(os.path.join(DATA_DIR, f"mel_{i}.npy"), mel)
        np.save(os.path.join(DATA_DIR, f"label_{i}.npy"), lab)

        if (i + 1) % 100 == 0 or i == start + count - 1:
            print(f"  已生成 {i + 1}/{start + count}")

    # 可视化：优先用第 0 号样本，分块时用本块首样本
    plot_id = 0 if os.path.exists(os.path.join(DATA_DIR, "mel_0.npy")) else start
    mel0 = np.load(os.path.join(DATA_DIR, f"mel_{plot_id}.npy"))
    plt.figure(figsize=(10, 4))
    plt.imshow(mel0.T, aspect="auto", origin="lower", cmap="magma")
    plt.title("Mel Spectrogram (SoundFont Piano, multi-song corpus)")
    plt.colorbar()
    plt.ylabel("Mel bin")
    plt.xlabel("Frame")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "mel_spectrogram.png"), dpi=120)
    plt.close()
    print(f"✅ 本次生成 {count} 条，累计目标 {start + count} 条；曲目分布: "
          + ", ".join(f"{k}×{v}" for k, v in used.most_common()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多曲目钢琴训练数据生成")
    parser.add_argument("--start", type=int, default=0, help="起始样本编号")
    parser.add_argument("--count", type=int, default=N_SAMPLES, help="本次生成数量")
    args = parser.parse_args()
    main(args.start, args.count)
