#!/usr/bin/env python3
"""
Generate Artificial Analysis (AA) real benchmark chart for Article 1.
100% objective, factual, no exaggeration.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'Hiragino Sans GB', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

NAVY = '#0F2C59'
SLATE_DARK = '#334155'
SLATE_MID = '#64748B'
SLATE_LIGHT = '#94A3B8'
SILVER = '#CBD5E1'
BG_CARD = '#F8FAFC'
GRID_COLOR = '#E2E8F0'
CRIMSON = '#8B1E24'
CRIMSON_BG = '#FDF2F2'
TEAL_DARK = '#115E59'
TEAL_BG = '#F0FDFA'
ORANGE = '#E05A1F'
BLUE = '#2563EB'
GREEN = '#16A34A'
MAGENTA = '#D92662'


def add_research_chrome(fig, title, subtitle, source_text):
    fig.add_artist(plt.Line2D([0.04, 0.96], [0.965, 0.965], color=NAVY, linewidth=2.5, transform=fig.transFigure))
    fig.text(0.04, 0.925, title, fontsize=13.5, fontweight='bold', color='#0F172A', ha='left', va='top')
    fig.text(0.04, 0.885, subtitle, fontsize=9.2, color='#475569', ha='left', va='top')
    fig.add_artist(plt.Line2D([0.04, 0.96], [0.065, 0.065], color=SILVER, linewidth=0.8, transform=fig.transFigure))
    fig.text(0.04, 0.045, source_text, fontsize=8, color=SLATE_MID, ha='left', va='top')


def render_aa_chart(output_path):
    fig = plt.figure(figsize=(13.33, 7.5), facecolor='#FFFFFF')
    add_research_chrome(
        fig,
        title="AA 视频竞技场核心数据：人类偏好 Elo 评分与商业/自建成本对照",
        subtitle="基于 Artificial Analysis 官方实测盲测数据与厂商公开费率 (2026)",
        source_text="数据来源：Artificial Analysis 视频竞技场官方实测榜单及厂商公开费率 [^5] | 视觉呈现：谷粒 · KUHUNG"
    )

    # Top Factual Summary Box
    c_top = patches.FancyBboxPatch(
        (0.04, 0.77), 0.92, 0.085,
        boxstyle="round,pad=0.010,rounding_size=0.015",
        edgecolor=NAVY, facecolor=BG_CARD, linewidth=1.0, transform=fig.transFigure
    )
    fig.add_artist(c_top)
    fig.text(0.055, 0.835, "【实测数据摘要】AA Video Arena 人类盲测与官方公布定价 (2026)：",
             fontsize=9.0, fontweight='bold', color=NAVY, transform=fig.transFigure)
    fig.text(0.055, 0.805, "• 盲测评分：Wan 3.0 (1192)、Hyperflow H3 (1149)、Seedance 2.5 (1147)、Gemini Omni Flash (1146)、MiniMax H3 768p (1119)；\n"
                           "• 推理费率：MiniMax H3 768p 官方为 $9.60/min ($0.16/s)，Seedance 2.5 为 $40.82/min ($0.68/s)，本地单卡 L4 硬件折算为 $0.14/min ($0.0023/s)。",
             fontsize=8.0, color=SLATE_DARK, transform=fig.transFigure)

    # Left: Elo Rating Bar Chart
    ax1 = fig.add_axes([0.22, 0.16, 0.32, 0.56])
    # Right: Pricing per minute Bar Chart
    ax2 = fig.add_axes([0.64, 0.16, 0.30, 0.56])

    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(SLATE_LIGHT)
        ax.spines['bottom'].set_color(SLATE_LIGHT)
        ax.xaxis.grid(True, linestyle='--', linewidth=0.7, color=GRID_COLOR)
        ax.set_axisbelow(True)

    # 7 Focused Models from AA Leaderboard
    models_rev = [
        "Kling 3.0 Omni (快手基准)",
        "Wan 2.7 (Alibaba)",
        "MiniMax H3 768p (原生开源)",
        "Gemini Omni Flash 1.1",
        "Dreamina Seedance 2.5",
        "Hyperflow H3 (H3衍生加速)",
        "Wan 3.0 (Alibaba 旗舰)"
    ]
    elos_rev = [1000, 1071, 1119, 1146, 1147, 1149, 1192]
    ci_rev = [0, 8, 10, 10, 11, 11, 10]
    samples_rev = ["20,213", "22,001", "8,941", "8,271", "7,099", "5,734", "9,205"]
    prices_min_rev = [10.91, 16.90, 9.60, 9.64, 40.82, 18.00, 12.00]

    colors_elo = [
        CRIMSON,      # Kling
        SLATE_LIGHT,  # Wan 2.7
        MAGENTA,      # MiniMax H3 (768p)
        GREEN,        # Gemini Omni Flash
        BLUE,         # Seedance 2.5
        NAVY,         # Hyperflow H3
        ORANGE        # Wan 3.0
    ]

    y_pos = list(range(len(models_rev)))
    ax1.barh(y_pos, elos_rev, color=colors_elo, height=0.52, zorder=3)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(models_rev, fontsize=8.6, color=SLATE_DARK)
    ax1.set_xlabel("AA Video Arena 盲测 Elo 等级分", fontsize=8.6, color=SLATE_DARK)
    ax1.set_xlim(950, 1260)
    ax1.set_title("视频综合质量盲测评分 (Elo)", fontsize=10.0, fontweight='bold', color=NAVY, pad=8, loc='left')

    for i, (v, ci, s) in enumerate(zip(elos_rev, ci_rev, samples_rev)):
        bold_flag = 'bold' if i in [2, 5] else 'normal'
        col = MAGENTA if i == 2 else (NAVY if i == 5 else SLATE_DARK)
        ci_str = f"±{ci}" if ci > 0 else ""
        tag = f" {v}{ci_str} (样本 {s})"
        ax1.text(v + 5, i, tag, va='center', fontsize=7.8, fontweight=bold_flag, color=col)

    # Right: Cost per minute
    ax2.barh(y_pos, prices_min_rev, color=colors_elo, height=0.52, zorder=3)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(["" for _ in y_pos])
    ax2.set_xlabel("官方公布商业费率 (美元 / 分钟)", fontsize=8.6, color=SLATE_DARK)
    ax2.set_xlim(0, 48)
    ax2.set_title("商业推理定价对比 (美元/分钟 与 美元/秒)", fontsize=10.0, fontweight='bold', color=NAVY, pad=8, loc='left')

    for i, p in enumerate(prices_min_rev):
        bold_flag = 'bold' if i in [2, 5] else 'normal'
        col = MAGENTA if i == 2 else (NAVY if i == 5 else SLATE_DARK)
        per_sec = p / 60.0
        tag = f" ${p:.2f}/min (${per_sec:.2f}/s)"
        ax2.text(p + 0.8, i, tag, va='center', fontsize=7.8, fontweight=bold_flag, color=col)

    # Note at bottom
    fig.text(0.50, 0.082, "注：本地自建单卡 L4 工作流 (544p 规格) 实测耗时 165.4 秒/镜，单镜 0.089 CU ($0.012)，换算分钟费率约 $0.14/min ($0.0023/s) [^6]",
             fontsize=7.8, color=SLATE_MID, ha='center', transform=fig.transFigure)

    fig.savefig(output_path, dpi=300, facecolor='#FFFFFF', bbox_inches='tight')
    plt.close(fig)
    print(f"[+] Saved AA Benchmark Chart to {output_path}")


if __name__ == "__main__":
    out_file = Path("/Users/kuhung/roy/3_Active_Labs/minimax-h3-colab-workflow/docs/articles/assets/chart_01_aa_benchmark.png")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    render_aa_chart(out_file)
