"""
生成钢琴错音检测系统的两张架构图：
1. 系统总体架构图 (architecture.png)
2. 开发流程图 (development_flow.png)
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 图1：系统总体架构图
# ============================================================
def draw_architecture():
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.set_xlim(0, 17)
    ax.set_ylim(-1.8, 10)
    ax.axis('off')
    fig.patch.set_facecolor('#f8f8f8')

    # --- 标题 ---
    ax.text(8.5, 9.7, '钢琴错音检测系统 · 总体开发架构', fontsize=16, fontweight='bold', ha='center')
    ax.text(8.5, 9.35, 'WAV音频 → Mel频谱 → CNN+BiLSTM模型 → 音符事件 → DTW对齐 → JSON报告+路径图',
            fontsize=9, color='#666', ha='center')

    # --- 辅助函数 ---
    def box(x, y, w, h, text, subtext='', fc='#ffffff', ec='#999999', lw=1.2, fs=10):
        rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                               boxstyle="round,pad=0.08",
                               facecolor=fc, edgecolor=ec, linewidth=lw)
        ax.add_patch(rect)
        ax.text(x, y + 0.05, text, fontsize=fs, ha='center', va='center', fontweight='bold')
        if subtext:
            lines = subtext.split('\n')
            for i, line in enumerate(lines):
                ax.text(x, y - 0.15 - i*0.17, line, fontsize=7.5, ha='center', va='center', color='#555')

    def dashed_box(x, y, w, h, label, fc='#fafafa'):
        rect = Rectangle((x - w/2, y - h/2), w, h,
                          facecolor=fc, edgecolor='#888', linewidth=1.2,
                          linestyle='--', alpha=0.6)
        ax.add_patch(rect)
        ax.text(x - w/2 + 0.3, y + h/2 - 0.25, label, fontsize=8.5,
                color='#666', ha='left', va='top')

    def arrow(x1, y1, x2, y2, color='#555', lw=1.2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=lw))

    def dashed_arrow(x1, y1, x2, y2, color='#2E7D32', lw=1.5):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=lw, linestyle='--'))

    # ===== 顶层：数据源 =====
    box(4, 8.5, 3.5, 0.9, 'WAV 音频输入', '.wav / .mp3 / .mp4 文件', fc='#fff', ec='#aaa')
    box(13, 8.5, 3.5, 0.9, '标准 MIDI 乐谱', 'original.mid 参考音符序列', fc='#fff', ec='#aaa')

    # ===== 中层：离线训练 =====
    dashed_box(8, 5.8, 14.5, 3.0, '离线训练 · 模型开发')

    box(3, 5.8, 3.0, 1.5, 'Mel 频谱特征',
        'librosa melspectrogram\nsr=22050, n_mels=128, hop=512\npower_to_db → [T, 128]',
        fc='#fff', ec='#aaa')
    box(8, 5.8, 3.0, 1.5, 'CNN+BiLSTM 模型',
        'CNN: 3×ConvBlock, MaxPool(2,1)\nBiLSTM: hidden=256, layers=2\n输出 [B, 512, 88] 概率',
        fc='#fff', ec='#aaa')
    box(13, 5.8, 3.0, 1.5, '模型权重', 'piano_model.pth\nCNN-BiLSTM 训练权重', fc='#fff', ec='#aaa')

    # 离线训练内部流程（短箭头）
    arrow(4, 8.05, 3, 6.55)   # WAV → Mel
    arrow(4.5, 5.8, 6.5, 5.8)  # Mel → CNN
    arrow(9.5, 5.8, 11.5, 5.8) # CNN → 模型权重
    # MIDI → Mel：沿顶部走线，从 MIDI 左侧出发，上行绕过数据源框，水平到 Mel 正上方，再下行
    ax.plot([11.25, 11.25, 3.0, 3.0], [8.95, 9.1, 9.1, 7.55], color='#555', lw=1.0, linestyle='--')
    ax.annotate('', xy=(3.0, 7.35), xytext=(3.0, 7.55),
                arrowprops=dict(arrowstyle='->', color='#555', lw=1.0))

    # ===== 共享核心 =====
    box(8, 3.8, 5.5, 0.85, '共享核心 · 一致设计原则',
        '训练/推理 mel 参数一致 · 色度距离 DTW · zero-padding 语义',
        fc='#e8f8e8', ec='#2E7D32', lw=2)
    dashed_arrow(8, 4.25, 8, 5.05)
    dashed_arrow(8, 3.35, 8, 2.75)

    # ===== 下层：在线推理管线 =====
    dashed_box(8, 1.4, 14.5, 2.6, '在线推理 · 错音检测（model → post_process → dtw_align → app）')

    box(3.0, 1.5, 3.0, 1.1, '频谱提取', 'librosa mel\n→ [1,128,512]', fc='#fff', ec='#aaa')
    box(7.0, 1.5, 3.0, 1.1, '模型推理', 'CNN-BiLSTM\n→ [512,88] 概率', fc='#fff', ec='#aaa')
    box(11.0, 1.5, 3.0, 1.1, '后处理', '阈值化+连续段\n→ 音符事件列表', fc='#fff', ec='#aaa')
    box(14.5, 1.5, 2.5, 1.1, 'DTW 对齐', '色度距离+路径遍历\n→ 错音/漏音/多弹', fc='#fff', ec='#aaa')

    # 推理管线内部流程（短箭头）
    arrow(4.5, 1.5, 5.5, 1.5)
    arrow(8.5, 1.5, 9.5, 1.5)
    arrow(12.5, 1.5, 13.25, 1.5)

    # 跨层连接标注（用文字说明代替长箭头，避免穿越混乱）
    ax.text(1.2, 5.0, 'WAV 输入', fontsize=7.5, color='#888', ha='center',
            rotation=90, va='center')
    ax.text(14.8, 5.0, 'MIDI 参考', fontsize=7.5, color='#888', ha='center',
            rotation=90, va='center')
    # 短指示箭头：WAV → 频谱提取（左侧边缘）
    ax.annotate('', xy=(2.2, 2.05), xytext=(2.2, 3.0),
                arrowprops=dict(arrowstyle='->', color='#999', lw=1.0, linestyle='--'))
    # 短指示箭头：MIDI → DTW（右侧边缘）
    ax.annotate('', xy=(14.5, 2.05), xytext=(14.5, 3.0),
                arrowprops=dict(arrowstyle='->', color='#999', lw=1.0, linestyle='--'))
    ax.text(14.5, 2.5, 'MIDI', fontsize=7, color='#999', ha='center')
    # 短指示箭头：模型权重 → 模型推理
    ax.annotate('', xy=(10.5, 2.05), xytext=(10.5, 3.0),
                arrowprops=dict(arrowstyle='->', color='#999', lw=1.0, linestyle='--'))
    ax.text(10.5, 2.5, '模型加载', fontsize=7, color='#999', ha='center')

    # ===== 输出层 =====
    box(11, -0.7, 3.0, 0.9, 'JSON 错误报告', 'time · type · expect · actual\n音名标注 精确到0.1秒',
        fc='#fff', ec='#aaa')
    box(14.5, -0.7, 2.5, 0.9, 'DTW 路径图', 'dtw_path.png\n红色对齐路径可视化',
        fc='#fff', ec='#aaa')
    box(7, -0.7, 3.0, 0.9, 'Web 分析服务', 'app.py Flask上传\n在线分析+报告下载',
        fc='#fff', ec='#aaa')

    # 输出箭头（短垂直，不交叉）
    arrow(11.0, 0.95, 11.0, -0.25)   # 后处理 → JSON
    arrow(14.5, 0.95, 14.5, -0.25)   # DTW → 路径图
    arrow(11.0, 0.95, 7.0, -0.25)    # 后处理 → Web

    # ===== 图例 =====
    ly = -1.5
    ax.plot([0.5, 1.5], [ly, ly], color='#555', lw=1.2)
    ax.text(1.7, ly, '数据流 / 模型加载', fontsize=8, va='center')
    ax.plot([4.5, 5.5], [ly, ly], color='#2E7D32', lw=1.5, linestyle='--')
    ax.text(5.7, ly, '训练-推理共用同一实现', fontsize=8, va='center', color='#2E7D32')
    rect = Rectangle((7.8, ly-0.12), 0.25, 0.25, facecolor='#e8f8e8', edgecolor='#2E7D32', lw=1.5)
    ax.add_patch(rect)
    ax.text(8.2, ly, '跨模块共享核心', fontsize=8, va='center')
    rect2 = Rectangle((10.5, ly-0.12), 0.25, 0.25, facecolor='none', edgecolor='#888', lw=1.2, linestyle='--')
    ax.add_patch(rect2)
    ax.text(10.9, ly, '域边界（离线 / 在线）', fontsize=8, va='center')

    plt.tight_layout()
    plt.savefig('architecture.png', dpi=150, bbox_inches='tight', facecolor='#f8f8f8')
    plt.close()
    print('OK architecture.png generated')


# ============================================================
# 图2：开发流程图
# ============================================================
def draw_development_flow():
    fig, ax = plt.subplots(figsize=(12, 20))
    ax.set_xlim(0, 12)
    ax.set_ylim(-0.5, 20)
    ax.axis('off')
    fig.patch.set_facecolor('#f8f8f8')

    # --- 标题 ---
    ax.text(6, 19.6, '钢琴错音检测系统 · 总体开发流程', fontsize=16, fontweight='bold', ha='center')
    ax.text(6, 19.2, 'MIDI数据 → 特征工程 → 模型训练 → 后处理 → DTW对齐 → Web服务 → 评估，共 8 个模块分段推进',
            fontsize=9, color='#666', ha='center')

    def box(x, y, w, h, text, subtext='', fc='#ffffff', ec='#999999', lw=1.2, fs=10):
        rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                               boxstyle="round,pad=0.08",
                               facecolor=fc, edgecolor=ec, linewidth=lw)
        ax.add_patch(rect)
        ax.text(x, y + 0.08, text, fontsize=fs, ha='center', va='center', fontweight='bold')
        if subtext:
            lines = subtext.split('\n')
            for i, line in enumerate(lines):
                ax.text(x, y - 0.12 - i*0.2, line, fontsize=7.5, ha='center', va='center', color='#555')

    def dashed_box(cx, cy, w, h, label, fc='#fafafa'):
        rect = Rectangle((cx - w/2, cy - h/2), w, h,
                          facecolor=fc, edgecolor='#888', linewidth=1.2,
                          linestyle='--', alpha=0.6)
        ax.add_patch(rect)
        ax.text(cx - w/2 + 0.25, cy + h/2 - 0.22, label, fontsize=8.5,
                color='#666', ha='left', va='top')

    def arrow(x1, y1, x2, y2, color='#555', lw=1.3):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=lw))

    BW = 6.5  # box width
    BH = 1.3  # box height
    CX = 6    # center x

    # ===== 数据源 =====
    box(CX, 18.2, BW, 0.8, 'original.mid + make_dataset.py',
        'MIDI 错误注入 · 音频合成 · mel 频谱 + 标签生成 → data/features/',
        fc='#fff', ec='#aaa')

    # ===== 阶段 ① 数据与特征工程 =====
    dashed_box(CX, 16.3, 11, 2.6, '阶段 ①  数据与特征工程')

    box(CX, 16.3, BW, BH, '数据合成 · make_dataset.py',
        '速度扰动 ±10% · 错音/漏音/多弹注入\nmel [T,128] + label [T,88] → 1000 样本',
        fc='#fff', ec='#aaa')

    arrow(CX, 17.8, CX, 16.95)

    # ===== 阶段  模型训练与封装 =====
    dashed_box(CX, 13.0, 11, 4.6, '阶段 ②  模型训练与封装')

    box(CX, 14.6, BW, BH, '数据集封装 · piano_dataset.py',
        'zero-padding T→512 · mel [1,128,512] · label [512,88]\n不转置 label · 返回 (mel, label) 张量对',
        fc='#fff', ec='#aaa')

    box(CX, 13.0, BW, BH, '模型训练 · train.py',
        'CNN-BiLSTM · BCELoss · Adam(lr=1e-3)\nbatch=8 · epochs=20 · loss_curve.png',
        fc='#fff', ec='#aaa')

    arrow(CX, 15.65, CX, 15.25)
    arrow(CX, 13.95, CX, 13.65)

    # ===== 阶段 ③ 后处理与 DTW 对齐 =====
    dashed_box(CX, 9.2, 11, 4.6, '阶段 ③  后处理与 DTW 对齐')

    box(CX, 10.8, BW, BH, '后处理 · post_process.py',
        '阈值 >0.5 · 连续段检测 · 过滤 <3 帧噪声\nframe→秒 · MIDI=key_idx+21 · 排序',
        fc='#fff', ec='#aaa')

    box(CX, 9.2, BW, BH, 'DTW 对齐 · dtw_align.py',
        '色度距离代价矩阵 · librosa.sequence.dtw\n遍历路径分类：错音/漏音/多弹\nJSON 报告 + dtw_path.png',
        fc='#fff', ec='#aaa')

    arrow(CX, 12.35, CX, 11.45)
    arrow(CX, 10.15, CX, 9.85)

    # ===== 模型入库上线（绿色高亮） =====
    box(CX, 7.8, 5.5, 0.85, '模型入库 · piano_model.pth',
        'CNN-BiLSTM 权重保存 · 可直接加载推理',
        fc='#e8f8e8', ec='#2E7D32', lw=2.5)

    arrow(CX, 8.55, CX, 8.25)

    # ===== 阶段 ④ 应用部署与评估 =====
    dashed_box(CX, 5.0, 11, 4.0, '阶段 ④  应用部署与评估')

    box(CX, 6.5, BW, BH, 'Web 服务 · app.py',
        'Flask 上传接口 · WAV/MP3/MP4 支持\n在线分析 → JSON 报告 + 路径图下载',
        fc='#fff', ec='#aaa')

    box(CX, 4.9, BW, 1.3, '评估验证 · evaluate.py',
        '多阈值指标(0.5/0.6/0.7) · 混淆矩阵\nF1=0.9876 · 准确率 0.9997 · 精确率 0.9884',
        fc='#fff', ec='#aaa')

    arrow(CX, 7.35, CX, 7.15)
    arrow(CX, 5.85, CX, 5.55)

    # ===== 图例 =====
    ly = 3.3
    ax.plot([1.0, 2.0], [ly, ly], color='#555', lw=1.3)
    ax.text(2.2, ly, '主流程', fontsize=8, va='center')
    rect = Rectangle((4.5, ly-0.12), 0.25, 0.25, facecolor='#e8f8e8', edgecolor='#2E7D32', lw=2)
    ax.add_patch(rect)
    ax.text(4.9, ly, '上线模型（训练成果）', fontsize=8, va='center')
    rect2 = Rectangle((7.5, ly-0.12), 0.25, 0.25, facecolor='none', edgecolor='#888', lw=1.2, linestyle='--')
    ax.add_patch(rect2)
    ax.text(7.9, ly, '阶段分区（虚线框）', fontsize=8, va='center')

    plt.tight_layout()
    plt.savefig('development_flow.png', dpi=150, bbox_inches='tight', facecolor='#f8f8f8')
    plt.close()
    print('OK development_flow.png generated')


if __name__ == '__main__':
    draw_architecture()
    draw_development_flow()
    print('All done!')
