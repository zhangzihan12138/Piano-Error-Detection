import numpy as np

# ---------------- 默认参数（与 make_dataset.py 的特征提取参数保持一致）----------------
SAMPLE_RATE = 22050.0   # 采样率
HOP_LENGTH = 512        # STFT hop 长度
DEFAULT_FPS = SAMPLE_RATE / HOP_LENGTH  # 帧率 ≈ 43.07 帧/秒

# 音域过滤：只保留 MIDI 36~96（对应 C2~C7），丢弃钢琴键盘两端的极端音
PITCH_MIN = 36
PITCH_MAX = 96

# 最短持续帧数：连续激活帧数低于此值的片段视为毛刺，直接丢弃
MIN_DURATION = 5

# 激活阈值：概率 > 此值才视为"琴键按下"
DEFAULT_THRESHOLD = 0.6

# 同音高去重窗口（秒）：同一音高在此窗口内只保留最早的一个事件，消除密集重复多弹
DEDUPE_SEC = 0.3


def _ema_smooth(probs: np.ndarray, alpha: float) -> np.ndarray:
    """对概率矩阵做指数移动平均（EMA）平滑，抑制帧间抖动。

    s[t] = alpha * p[t] + (1 - alpha) * s[t-1]

    Args:
        probs: [T, 88] 概率矩阵
        alpha: 平滑系数，取值 (0, 1]；越小越平滑。alpha <= 0 表示不平滑。
    """
    if alpha is None or alpha <= 0:
        return probs

    out = np.empty_like(probs)
    prev = probs[0].copy()
    out[0] = prev
    for t in range(1, probs.shape[0]):
        prev = alpha * probs[t] + (1.0 - alpha) * prev
        out[t] = prev
    return out


def post_process(
    logits: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
    min_duration: int = MIN_DURATION,
    fps: float = DEFAULT_FPS,
    ema_alpha: float = 0.0,
    pitch_min: int = PITCH_MIN,
    pitch_max: int = PITCH_MAX,
    dedupe_sec: float = DEDUPE_SEC,
) -> list[dict]:
    """
    将模型输出的概率矩阵转换为离散的音符事件序列，供 DTW 对齐使用。

    处理流程：
    1. （可选）对概率矩阵做 EMA 平滑，减少帧间抖动
    2. 音域过滤：只处理 MIDI pitch_min~pitch_max 范围内的琴键
    3. 逐键把连续的超阈值帧合并为一个音符段（note segment）
    4. 过滤持续时间短于 min_duration 帧的片段（去毛刺）
    5. onset 使用真实帧索引换算：start_time = start_frame * HOP_LENGTH / SAMPLE_RATE
    6. 同音高去重：同一音高在 dedupe_sec 秒内只保留最早的一个事件
    7. 按起始时间排序，返回音符事件列表

    Args:
        logits:       [T, 88] 概率矩阵，值域 [0, 1]
                      T 是时间帧数（如 512），88 对应钢琴 88 键（21=A0 ... 108=C8）
        threshold:    激活阈值，概率 > 此值视为"琴键按下"，默认 0.6
        min_duration: 最短持续帧数，低于此值的音符段被过滤（去毛刺），默认 5
        fps:          帧率（frames per second），用于把帧索引换算成秒，
                      默认 SAMPLE_RATE / HOP_LENGTH ≈ 43.07
        ema_alpha:    EMA 平滑系数，> 0 时启用平滑（建议 0.3~0.6），默认 0（不平滑）
        pitch_min:    音域下限（MIDI），默认 36（C2）
        pitch_max:    音域上限（MIDI），默认 96（C7）
        dedupe_sec:   同音高去重窗口（秒），0 表示不去重，默认 0.3

    Returns:
        notes: 按 start_time 升序排列的音符事件列表，每个元素为 dict：
               {
                   "pitch":      int,   # MIDI 编号（36=C2 ... 96=C7）
                   "start_time": float, # 起始时间（秒）= start_frame / fps
                   "end_time":   float, # 结束时间（秒）= end_frame / fps
               }

    示例:
        >>> logits = np.random.rand(512, 88)
        >>> notes = post_process(logits, threshold=0.6, ema_alpha=0.5)
        >>> print(notes[0])
        {'pitch': 60, 'start_time': 1.23, 'end_time': 1.45}
    """
    probs = np.asarray(logits, dtype=np.float32)
    if probs.ndim != 2 or probs.shape[1] != 88:
        raise ValueError(f"期望 [T, 88] 的概率矩阵，但输入形状为 {probs.shape}")

    T = probs.shape[0]

    # 1. 可选：EMA 平滑，减少帧间抖动导致的碎片化音符
    probs = _ema_smooth(probs, ema_alpha)

    notes: list[dict] = []

    # 2. 音域过滤：把 MIDI 范围换算成 0~87 的琴键索引区间
    key_lo = max(0, pitch_min - 21)
    key_hi = min(87, pitch_max - 21)

    # 3. 逐键检测连续激活段（向量化：用差分找上升/下降沿）
    for key_idx in range(key_lo, key_hi + 1):
        active = probs[:, key_idx] > threshold  # [T] 布尔数组

        # 前后各补一个 False，差分后 +1 为音符起点、-1 为终点
        padded = np.concatenate(([False], active, [False]))
        diffs = np.diff(padded.astype(np.int8))
        starts = np.where(diffs == 1)[0]   # 起点帧索引
        ends = np.where(diffs == -1)[0]    # 终点帧索引（不含）

        midi_pitch = key_idx + 21

        for s, e in zip(starts, ends):
            # 4. 过滤过短片段（单帧/双帧毛刺）
            if e - s < min_duration:
                continue

            # 5. onset / offset 用真实帧索引换算为秒，绝不为 0（除非真的在第 0 帧起音）
            notes.append({
                "pitch": int(midi_pitch),
                "start_time": float(s) / fps,
                "end_time": float(e) / fps,
            })

    # 6. 同音高去重：同一音高在 dedupe_sec 秒内只保留最早 onset 的事件，消除密集重复多弹
    if dedupe_sec > 0 and notes:
        notes.sort(key=lambda n: (n["pitch"], n["start_time"]))
        deduped: list[dict] = []
        last_start: dict[int, float] = {}  # pitch -> 上一个被保留事件的 onset
        for n in notes:
            prev = last_start.get(n["pitch"])
            if prev is None or n["start_time"] - prev >= dedupe_sec:
                deduped.append(n)
                last_start[n["pitch"]] = n["start_time"]
        notes = deduped

    # 7. 按起始时间升序排列，便于 DTW 逐帧对齐
    notes.sort(key=lambda n: (n["start_time"], n["pitch"]))

    return notes
