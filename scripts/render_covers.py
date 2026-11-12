#!/usr/bin/env python3
"""
Generate institutional research covers for MiniMax H3 articles.
Adheres to institutional-chart and kuhung-ip-cover standards:
- Pure white background (#FFFFFF)
- Editorial Chinese typography
- Brand signature: 谷粒 · KUHUNG | kuhung.me
- Clean vector concept diagrams on the right, zero 3D metallic bloat
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'Hiragino Sans GB', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

NAVY = '#0F2C59'
SLATE_DARK = '#334155'
SLATE_MID = '#64748B'
SLATE_LIGHT = '#94A3B8'
SILVER = '#CBD5E1'
BG_CARD = '#F8FAFC'
TEAL_DARK = '#115E59'
TEAL_BG = '#F0FDFA'
GOLD_DARK = '#9A6B1F'
GOLD_BG = '#FEF9C3'


def render_cover_01(output_path):
    fig = plt.figure(figsize=(13.33, 7.5), facecolor='#FFFFFF')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')

    # Top Navy bar
    ax.add_patch(patches.Rectangle((0.06, 0.93), 0.88, 0.008, color=NAVY))

    # Left: Editorial Typography
    ax.text(0.06, 0.86, "MiniMax H3 视频生成实战 · 篇一", fontsize=11, fontweight='bold', color=TEAL_DARK)
    ax.text(0.06, 0.74, "技术选型：海螺 3.0 处于什么水平？", fontsize=24, fontweight='bold', color='#0F172A')
    ax.text(0.06, 0.65, "核心指标解析与 2026 主流视频生成模型横向评测", fontsize=13, color=SLATE_MID)

    # Bullet badges on the left
    badges = [
        "时序与上下文一致性",
        "最高 12 个多模态参考配额",
        "原生 32kHz 高保真立体声音画同出",
        "开放 33B 基础权重与 24GB 显存私有化部署"
    ]
    curr_y = 0.54
    for b in badges:
        ax.add_patch(patches.Circle((0.07, curr_y + 0.008), 0.005, color=NAVY))
        ax.text(0.085, curr_y, b, fontsize=10.5, color=SLATE_DARK)
        curr_y -= 0.055

    # Brand Footer
    ax.add_artist(plt.Line2D([0.06, 0.94], [0.12, 0.12], color=SILVER, linewidth=0.8))
    ax.text(0.06, 0.08, "谷粒 · KUHUNG", fontsize=10, fontweight='bold', color=NAVY)
    ax.text(0.18, 0.08, "|   kuhung.me", fontsize=10, color=SLATE_MID)
    ax.text(0.94, 0.08, "2026 技术选型研报", fontsize=9.5, color=SLATE_LIGHT, ha='right')

    # Right: Clean Schematic Diagram
    right_box = patches.FancyBboxPatch(
        (0.53, 0.22), 0.41, 0.66,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        edgecolor=SILVER, facecolor=BG_CARD, linewidth=1.2
    )
    ax.add_patch(right_box)
    ax.text(0.55, 0.83, "全模态统一上下文架构示意 (Ref2VA)", fontsize=11, fontweight='bold', color=NAVY)

    # Channel cards inside right schematic
    channels = [
        ("图像通道", "≤ 9 张图片约束", TEAL_DARK, TEAL_BG),
        ("视频通道", "≤ 3 段视频 (≤15s)", NAVY, '#EEF2F6'),
        ("音频通道", "≤ 3 段音频 (≤15s)", GOLD_DARK, GOLD_BG),
        ("自然语言", "时间线提示词与 @ 语法", SLATE_DARK, '#F1F5F9')
    ]
    cy = 0.74
    for title, desc, col, bg in channels:
        cb = patches.FancyBboxPatch(
            (0.55, cy), 0.17, 0.052,
            boxstyle="round,pad=0.006,rounding_size=0.01",
            edgecolor=col, facecolor=bg, linewidth=0.8
        )
        ax.add_patch(cb)
        ax.text(0.56, cy + 0.028, title, fontsize=8.5, fontweight='bold', color=col)
        ax.text(0.56, cy + 0.010, desc, fontsize=7.2, color=SLATE_DARK)
        
        # Arrow to center
        ax.annotate("", xy=(0.75, 0.53), xytext=(0.725, cy + 0.026),
                    arrowprops=dict(arrowstyle="->", color=SLATE_LIGHT, lw=1.0))
        cy -= 0.075

    # Center Hub: 33B Omni-DiT
    hub = patches.FancyBboxPatch(
        (0.75, 0.40), 0.17, 0.26,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        edgecolor=NAVY, facecolor='#FFFFFF', linewidth=1.5
    )
    ax.add_patch(hub)
    ax.text(0.765, 0.61, "H3-Omni-DiT", fontsize=10.5, fontweight='bold', color=NAVY)
    ax.text(0.765, 0.56, "33B 单流生成基座", fontsize=8, color=SLATE_MID)
    ax.text(0.765, 0.50, "• Qwen3-VL 文本编码", fontsize=7.2, color=SLATE_DARK)
    ax.text(0.765, 0.46, "• 32x 空间时序 VAE", fontsize=7.2, color=SLATE_DARK)
    ax.text(0.765, 0.42, "• 8 步 Turbo 蒸馏加速", fontsize=7.2, color=SLATE_DARK)

    # Output Arrows
    ax.annotate("", xy=(0.835, 0.28), xytext=(0.835, 0.40),
                arrowprops=dict(arrowstyle="->", color=TEAL_DARK, lw=1.5))
    out_b = patches.FancyBboxPatch(
        (0.73, 0.23), 0.19, 0.048,
        boxstyle="round,pad=0.006,rounding_size=0.01",
        edgecolor=TEAL_DARK, facecolor=TEAL_BG, linewidth=1.0
    )
    ax.add_patch(out_b)
    ax.text(0.74, 0.245, "768p 视频 + 原生 32kHz 声景", fontsize=8, fontweight='bold', color=TEAL_DARK)

    fig.savefig(output_path, dpi=300, facecolor='#FFFFFF', bbox_inches='tight')
    plt.close(fig)
    print(f"[+] Saved Cover 01 to {output_path}")


def render_cover_02(output_path):
    fig = plt.figure(figsize=(13.33, 7.5), facecolor='#FFFFFF')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')

    # Top Navy bar
    ax.add_patch(patches.Rectangle((0.06, 0.93), 0.88, 0.008, color=NAVY))

    # Left: Editorial Typography
    ax.text(0.06, 0.86, "MiniMax H3 视频生成实战 · 篇二", fontsize=11, fontweight='bold', color=NAVY)
    ax.text(0.06, 0.74, "流程落地：我们跑通的基准工作流", fontsize=24, fontweight='bold', color='#0F172A')
    ax.text(0.06, 0.65, "核心参数对算力开销、显存控制与成片质感的决定性影响", fontsize=13, color=SLATE_MID)

    # Badges on the left
    badges = [
        "单路 BasicGuider (CFG=1.0) 较 v1.0 544p 提速 47%",
        "960×544 烟测峰值 21.91 GB，适配 L4 22GB 可用显存",
        "8 步 Turbo 采样适配单路快速收敛",
        "上海街景实测：稳态 165.4s / ~0.089 CU 单镜"
    ]
    curr_y = 0.54
    for b in badges:
        ax.add_patch(patches.Circle((0.07, curr_y + 0.008), 0.005, color=TEAL_DARK))
        ax.text(0.085, curr_y, b, fontsize=10.5, color=SLATE_DARK)
        curr_y -= 0.055

    # Brand Footer
    ax.add_artist(plt.Line2D([0.06, 0.94], [0.12, 0.12], color=SILVER, linewidth=0.8))
    ax.text(0.06, 0.08, "谷粒 · KUHUNG", fontsize=10, fontweight='bold', color=NAVY)
    ax.text(0.18, 0.08, "|   kuhung.me", fontsize=10, color=SLATE_MID)
    ax.text(0.94, 0.08, "Google Colab L4 工程实录", fontsize=9.5, color=SLATE_LIGHT, ha='right')

    # Right: Pipeline Architecture Block
    right_box = patches.FancyBboxPatch(
        (0.53, 0.22), 0.41, 0.66,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        edgecolor=SILVER, facecolor=BG_CARD, linewidth=1.2
    )
    ax.add_patch(right_box)
    ax.text(0.55, 0.83, "ComfyUI 原生无扩展基准流水线拓扑", fontsize=11, fontweight='bold', color=NAVY)

    # 4 Pipeline Stage Nodes
    stages = [
        ("输入构建", "Prompt Text + MiniMaxH3ImageToVideo", NAVY, '#EEF2F6'),
        ("条件引导", "BasicGuider (强制 CFG=1.0 单路无冗余)", TEAL_DARK, TEAL_BG),
        ("潜空间采样", "SamplerCustomAdvanced (res_multistep + 8步 Turbo)", NAVY, '#EEF2F6'),
        ("音画封装", "CreateVideo + SaveVideo (原生 32kHz MP4)", TEAL_DARK, TEAL_BG)
    ]
    sy = 0.73
    for i, (title, desc, col, bg) in enumerate(stages):
        sb = patches.FancyBboxPatch(
            (0.56, sy), 0.35, 0.075,
            boxstyle="round,pad=0.008,rounding_size=0.015",
            edgecolor=col, facecolor=bg, linewidth=1.0
        )
        ax.add_patch(sb)
        ax.text(0.58, sy + 0.045, f"阶段 0{i+1} · {title}", fontsize=9, fontweight='bold', color=col)
        ax.text(0.58, sy + 0.018, desc, fontsize=7.8, color=SLATE_DARK)

        if i < len(stages) - 1:
            ax.annotate("", xy=(0.735, sy - 0.03), xytext=(0.735, sy),
                        arrowprops=dict(arrowstyle="->", color=SLATE_LIGHT, lw=1.2))
        sy -= 0.125

    fig.savefig(output_path, dpi=300, facecolor='#FFFFFF', bbox_inches='tight')
    plt.close(fig)
    print(f"[+] Saved Cover 02 to {output_path}")


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent.parent / "docs" / "articles" / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)
    render_cover_01(out_dir / "cover_01_selection.png")
    render_cover_02(out_dir / "cover_02_pipeline.png")
