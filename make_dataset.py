"""训练数据集生成脚本：基于 original.mid + 钢琴 SoundFont 渲染 + 错误注入。

核心改动（相比正弦合成版本）：
  1. 用 FluidSynth + 钢琴 SoundFont 渲染 MIDI 音频（替换正弦叠加），
     使训练音色更接近真实钢琴，缓解"音色域不匹配"问题。
  2. 对 original.mid 的异常 tempo 做归一化：42 个音符从 512s 压缩到 ~14s，
     使 512 帧窗口（~11.9s）能覆盖多个不同音符，模型才能学到多音高检测。
  3. 保留变速（0.9~1.1x）、错误注入（错音/漏音/多弹）逻辑。
  4. 轻量数据增强：随机力度、微小时序抖动、低强度背景噪声。

输出：
  data/features/mel_*.npy   : [T, 128] log-mel 频谱
  data/features/label_*.npy : [T, 88]  逐帧 88 键激活标签
  mel_spectrogram.png       : 首个样本的频谱可视化
"""

import os
import random

import librosa
import matplotlib
import numpy as np
import pretty_midi

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- 常量 ----------------
SR, HOP, N_MELS = 22050, 512, 128
N_SAMPLES = 1000
SOUNDFONT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "piano.sf2")
TARGET_DURATION = 14.0  # 归一化后整段 MIDI 的目标时长（秒），使 512 帧窗口覆盖多音符

os.makedirs("data/features", exist_ok=True)

# 全局复用一个 FluidSynth 实例，避免每个样本重复加载 SoundFont（~140MB）
_synth = None
_sfid = None


def get_synth():
    """获取（懒加载）全局 FluidSynth 合成器实例。"""
    global _synth, _sfid
    if _synth is None:
        import fluidsynth
        _synth = fluidsynth.Synth(samplerate=SR, gain=0.5)
        _sfid = _synth.sfload(SOUNDFONT_PATH)
        # bank=0, preset=0 → Acoustic Grand Piano（钢琴 SoundFont 的默认钢琴音色）
        _synth.program_select(0, _sfid, 0, 0)
    return _synth


def normalize_tempo(mid: pretty_midi.PrettyMIDI, target_duration: float = TARGET_DURATION) -> None:
    """将 original.mid 的异常 tempo（每音 12.8s，共 512s）归一化到 target_duration。

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


def main() -> None:
    random.seed(42)
    np.random.seed(42)

    print(f"SoundFont: {SOUNDFONT_PATH}")
    print(f"目标时长: {TARGET_DURATION}s | 样本数: {N_SAMPLES}")

    for i in range(N_SAMPLES):
        mid = pretty_midi.PrettyMIDI("original.mid")

        # 1. 归一化 tempo（修复原始 MIDI 的异常长音）
        normalize_tempo(mid)

        # 2. 变速 0.9~1.1 倍
        spd = random.uniform(0.9, 1.1)
        for inst in mid.instruments:
            for n in inst.notes:
                n.start *= spd
                n.end *= spd

        # 3. 轻量数据增强（力度 + 时序抖动）
        add_augmentation(mid)

        # 4. 注入演奏错误
        inject_errors(mid)

        # 5. SoundFont 渲染音频
        audio = midi_to_audio(mid)

        # 6. 提取 log-mel 频谱
        mel = librosa.feature.melspectrogram(y=audio, sr=SR, n_mels=N_MELS, hop_length=HOP).T
        mel = librosa.power_to_db(mel)
        lab = midi_to_label(mid, mel.shape[0])

        np.save(f"data/features/mel_{i}.npy", mel)
        np.save(f"data/features/label_{i}.npy", lab)

        if (i + 1) % 100 == 0:
            print(f"  已生成 {i + 1}/{N_SAMPLES}")

    # 可视化首个样本的频谱
    mel0 = np.load("data/features/mel_0.npy")
    plt.figure(figsize=(10, 4))
    plt.imshow(mel0.T, aspect="auto", origin="lower", cmap="magma")
    plt.title("Mel Spectrogram (SoundFont 渲染)")
    plt.colorbar()
    plt.ylabel("Mel bin")
    plt.xlabel("Frame")
    plt.tight_layout()
    plt.savefig("mel_spectrogram.png", dpi=120)
    plt.close()
    print(f"✅ 基于 original.mid + SoundFont 渲染 + 错误注入，生成 {N_SAMPLES} 段训练数据")


if __name__ == "__main__":
    main()
