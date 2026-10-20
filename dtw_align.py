"""DTW 对齐与演奏错误判定模块。

职责：
1. 将参考音符序列（来自 original.mid）与演奏/检测出的音符序列
   （来自 post_process.py 的输出，格式 {pitch, start_time, end_time}）做 DTW 对齐
2. 基于对齐路径做一对一匹配，并分类四类结果：
   - correct（正确）：音高一致且起始时间偏差在容差内
   - wrong（错音）：位置对上了但音高不同，如 D4 弹成了 E4
   - missed（漏音）：参考中存在但演奏中没有对应音符
   - extra（多弹）：演奏中存在但参考中没有对应音符
3. 输出 JSON 结果（含音名，如 C4）和 DTW 累积代价矩阵 + 路径图 dtw_path.png

直接运行本脚本会用模拟数据自测：
    python dtw_align.py
"""

import json
import os

import matplotlib

# 无图形界面环境下只做文件输出
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(_HERE, "dtw_result.json")
PNG_PATH = os.path.join(_HERE, "dtw_path.png")

# Windows 下显示中文标题
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# 起始时间匹配容差（秒）：超过此值视为"位置没有对上"
ONSET_TOL = 0.15
# 代价权重：pitch 差（半音数）与 onset 差（秒）的加权
W_PITCH = 1.0
W_TIME = 1.0


def midi_to_name(midi: int) -> str:
    """MIDI 编号转音名，如 60 -> 'C4'，21 -> 'A0'，108 -> 'C8'。"""
    return f"{_NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def name_to_midi(name: str) -> int:
    """音名转 MIDI 编号，如 'C4' -> 60，'F#3' -> 54。"""
    pitch_class = _NOTE_NAMES.index(name[:-1])
    octave = int(name[-1])
    return pitch_class + (octave + 1) * 12


def _cost(ref: dict, perf: dict) -> float:
    """两个音符的配对代价：音高差（半音）+ 起始时间差（秒）加权。"""
    return W_PITCH * abs(ref["pitch"] - perf["pitch"]) + W_TIME * abs(ref["start_time"] - perf["start_time"])


def dtw_align(
    ref_notes: list[dict],
    perf_notes: list[dict],
    onset_tol: float = ONSET_TOL,
) -> tuple[np.ndarray, list[tuple[int, int]], list[dict]]:
    """对参考与演奏音符序列做 DTW 对齐并判定错误。

    Args:
        ref_notes:  参考音符列表，元素为 {pitch, start_time, end_time}
        perf_notes: 演奏音符列表，格式同上（来自 post_process 或模拟数据）
        onset_tol:  一对一匹配允许的最大起始时间偏差（秒）

    Returns:
        D:      [n_ref+1, n_perf+1] 的 DTW 累积代价矩阵（含边界行列）
        path:   最优对齐路径 [(i, j), ...]，i/j 为音符索引（边界用 -1 表示）
        errors: 错误列表，每项含 type/expected/played/onset 等字段
    """
    n, m = len(ref_notes), len(perf_notes)
    assert n > 0 and m > 0, "参考与演奏序列均不能为空"

    # ---- 1. 局部代价矩阵（仅音符索引区，边界另算）----
    C = np.array([[_cost(r, p) for p in perf_notes] for r in ref_notes])

    # ---- 2. DTW 累积代价矩阵，含边界（-1 表示序列端点）----
    INF = np.inf
    D = np.full((n + 1, m + 1), INF)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = C[i - 1, j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])

    # ---- 3. 回溯最优路径（音符索引，遇边界记 -1）----
    path: list[tuple[int, int]] = []
    i, j = n, m
    while i > 0 or j > 0:
        path.append((i - 1, j - 1))
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            step = min([(D[i - 1, j], (i - 1, j)), (D[i, j - 1], (i, j - 1)), (D[i - 1, j - 1], (i - 1, j - 1))])
            i, j = step[1]
    path.reverse()

    # ---- 4. 从路径收集候选对齐，并做一对一贪心匹配 ----
    # 候选条件：路径经过 (i, j) 且起始时间差在容差内（防止"借"远处同音名的音符）
    candidates: list[tuple[float, int, int]] = []
    for i, j in path:
        if i >= 0 and j >= 0 and abs(ref_notes[i]["start_time"] - perf_notes[j]["start_time"]) <= onset_tol:
            candidates.append((C[i, j], i, j))
    candidates.sort(key=lambda x: x[0])

    ref_used: set[int] = set()
    perf_used: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for c, i, j in candidates:
        if i not in ref_used and j not in perf_used:
            ref_used.add(i)
            perf_used.add(j)
            pairs.append((i, j))

    # ---- 5. 分类：correct / wrong / missed / extra ----
    errors: list[dict] = []
    num_correct = 0
    for i, j in pairs:
        r, p = ref_notes[i], perf_notes[j]
        if r["pitch"] == p["pitch"]:
            num_correct += 1
        else:
            errors.append({
                "type": "wrong",
                "expected": midi_to_name(r["pitch"]),
                "played": midi_to_name(p["pitch"]),
                "ref_onset": round(r["start_time"], 3),
                "played_onset": round(p["start_time"], 3),
            })

    for i in range(n):
        if i not in ref_used:
            errors.append({
                "type": "missed",
                "expected": midi_to_name(ref_notes[i]["pitch"]),
                "ref_onset": round(ref_notes[i]["start_time"], 3),
            })
    for j in range(m):
        if j not in perf_used:
            errors.append({
                "type": "extra",
                "played": midi_to_name(perf_notes[j]["pitch"]),
                "played_onset": round(perf_notes[j]["start_time"], 3),
            })

    # 错误按时间排序，便于阅读
    errors.sort(key=lambda e: e.get("ref_onset") or e.get("played_onset") or 0.0)

    return D[1:, 1:], path, errors


def plot_dtw(D: np.ndarray, path: list[tuple[int, int]], ref_notes: list[dict],
             perf_notes: list[dict], out_path: str) -> None:
    """绘制 DTW 累积代价矩阵热图并叠加最优对齐路径。"""
    plt.figure(figsize=(9, 8))
    plt.imshow(D, origin="lower", cmap="viridis", aspect="auto")
    plt.colorbar(label="累积代价")

    pi = [i for i, _ in path]
    pj = [j for _, j in path]
    plt.plot(pj, pi, color="red", linewidth=2, marker="o", markersize=4, label="最优路径")

    plt.xticks(range(len(perf_notes)), [midi_to_name(p["pitch"]) for p in perf_notes])
    plt.yticks(range(len(ref_notes)), [midi_to_name(r["pitch"]) for r in ref_notes])
    plt.xlabel("演奏序列（检测/模拟）")
    plt.ylabel("参考序列（original.mid）")
    plt.title("DTW 对齐路径")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def run_self_test() -> dict:
    """模拟数据自测：构造一段带已知错误的"演奏"，验证三类错误均能被检出。

    参考序列（C4 起，间隔 0.5s）：C4 D4 E4 F4 G4 A4 B4
    模拟演奏：
        C4  -> 正确（时间微偏 0.02s）
        E4  -> 错音（应为 D4）
        F4  -> 正确（但参考中的 E4 被跳过 -> 漏音）
        G4  -> 正确
        A4  -> 正确
        B4  -> 正确
        C5  -> 多弹（参考中不存在）
    期望检出：1 错音（D4→E4）、1 漏音（E4）、1 多弹（C5）
    """
    ref = [
        {"pitch": name_to_midi(n), "start_time": 0.5 * k, "end_time": 0.5 * k + 0.45}
        for k, n in enumerate(["C4", "D4", "E4", "F4", "G4", "A4", "B4"])
    ]
    perf = [
        {"pitch": name_to_midi("C4"), "start_time": 0.02, "end_time": 0.42},
        {"pitch": name_to_midi("E4"), "start_time": 0.52, "end_time": 0.92},  # 错音：应 D4
        {"pitch": name_to_midi("F4"), "start_time": 1.52, "end_time": 1.92},
        {"pitch": name_to_midi("G4"), "start_time": 2.02, "end_time": 2.42},
        {"pitch": name_to_midi("A4"), "start_time": 2.52, "end_time": 2.92},
        {"pitch": name_to_midi("B4"), "start_time": 3.02, "end_time": 3.42},
        {"pitch": name_to_midi("C5"), "start_time": 3.40, "end_time": 3.80},  # 多弹
    ]

    D, path, errors = dtw_align(ref, perf)

    result = {
        "summary": {
            "num_ref_notes": len(ref),
            "num_played_notes": len(perf),
            "num_correct": len(ref) - sum(1 for e in errors if e["type"] in ("wrong", "missed")),
            "num_wrong": sum(1 for e in errors if e["type"] == "wrong"),
            "num_missed": sum(1 for e in errors if e["type"] == "missed"),
            "num_extra": sum(1 for e in errors if e["type"] == "extra"),
            "dtw_total_cost": round(float(D[-1, -1]), 3),
        },
        "errors": errors,
    }

    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    plot_dtw(D, path, ref, perf, PNG_PATH)

    print("=== DTW 对齐自测 ===")
    print(f"参考音符 {len(ref)} 个 | 演奏音符 {len(perf)} 个")
    print(f"正确: {result['summary']['num_correct']} | 错音: {result['summary']['num_wrong']} "
          f"| 漏音: {result['summary']['num_missed']} | 多弹: {result['summary']['num_extra']}")
    for e in errors:
        if e["type"] == "wrong":
            print(f"  [错音] 应为 {e['expected']}，实际弹了 {e['played']}（参考 {e['ref_onset']}s）")
        elif e["type"] == "missed":
            print(f"  [漏音] {e['expected']} 未弹出（参考 {e['ref_onset']}s）")
        elif e["type"] == "extra":
            print(f"  [多弹] {e['played']}（演奏 {e['played_onset']}s）")
    print(f"JSON 已保存: {JSON_PATH}")
    print(f"DTW 路径图已保存: {PNG_PATH}")

    # 自检断言：三类错误都必须被检出，否则报错
    assert result["summary"]["num_wrong"] == 1, "应检出 1 个错音"
    assert result["summary"]["num_missed"] == 1, "应检出 1 个漏音"
    assert result["summary"]["num_extra"] == 1, "应检出 1 个多弹"
    print("✅ 自测通过：错音/漏音/多弹均正确检出")

    return result


if __name__ == "__main__":
    run_self_test()
