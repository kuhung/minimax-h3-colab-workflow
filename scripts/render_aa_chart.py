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
        title="AA 视频竞技场：人类偏好 Elo 评分与商业推理定价对照",
        subtitle="基于 Artificial Analysis AA-Video-T2V v2.1 榜单与公开 API 费率（核对日期：2026-11-12，以 AA 实时榜单为准）",
        source_text="数据来源：Artificial Analysis AA-Video-T2V v2.1 (2026-11-12) 及公开 API 定价 | 视觉呈现：谷粒 · KUHUNG"
    )

    # Top Factual Summary Box
    c_top = patches.FancyBboxPatch(
        (0.04, 0.745), 0.92, 0.105,
        boxstyle="round,pad=0.010,rounding_size=0.015",
        edgecolor=NAVY, facecolor=BG_CARD, linewidth=1.0, transform=fig.transFigure
    )
    fig.add_artist(c_top)
    fig.text(0.055, 0.838, "【核验数据摘要】AA-Video-T2V v2.1 实测榜单与公开定价（核对日期：2026-11-12）：",
             fontsize=9.0, fontweight='bold', color=NAVY, va='top', transform=fig.transFigure, parse_math=False)
    fig.text(0.055, 0.808, "• 盲测评分：Wan 3.0 (1175±13)、MiniMax H3 768p (1161±13)、Dreamina Seedance 2.5 (1155±13)、MiniMax H3 Max (1149±15)；\n"
                           "• 公开费率：MiniMax H3 768p 为 $4.80/min ($0.08/s)，Wan 3.0 为 $12.00/min ($0.20/s)，Seedance 2.5 为 $34.12/min ($0.57/s)（以实时榜单为准）。",
             fontsize=8.0, color=SLATE_DARK, va='top', linespacing=1.45, transform=fig.transFigure, parse_math=False)

    # Left: Elo Rating Bar Chart
    ax1 = fig.add_axes([0.23, 0.15, 0.31, 0.51])
    # Right: Pricing per minute Bar Chart
    ax2 = fig.add_axes([0.64, 0.15, 0.30, 0.51])

    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(SLATE_LIGHT)
        ax.spines['bottom'].set_color(SLATE_LIGHT)
        ax.xaxis.grid(True, linestyle='--', linewidth=0.7, color=GRID_COLOR)
        ax.set_axisbelow(True)

    # Verified Models from AA-Video-T2V v2.1 Leaderboard (Checked 2026-11-12)
    models_rev = [
        "MiniMax H3 Max (闭源高规)",
        "Dreamina Seedance 2.5",
        "MiniMax H3 768p (开放同源权重)",
        "Wan 3.0 (Alibaba)"
    ]
    elos_rev = [1149, 1155, 1161, 1175]
    ci_rev = [15, 13, 13, 13]
    prices_min_rev = [9.60, 34.12, 4.80, 12.00]

    colors_elo = [
        NAVY,         # MiniMax H3 Max
        BLUE,         # Seedance 2.5
        MAGENTA,      # MiniMax H3 768p
        ORANGE        # Wan 3.0
    ]

    y_pos = list(range(len(models_rev)))
    ax1.barh(y_pos, elos_rev, color=colors_elo, height=0.45, zorder=3)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(models_rev, fontsize=8.8, color=SLATE_DARK)
    ax1.set_xlabel("AA-Video-T2V v2.1 盲测 Elo 等级分", fontsize=8.6, color=SLATE_DARK)
    ax1.set_xlim(1100, 1210)
    ax1.set_title("视频综合质量盲测评分 (Elo ±95% CI)", fontsize=10.0, fontweight='bold', color=NAVY, pad=8, loc='left')

    for i, (v, ci) in enumerate(zip(elos_rev, ci_rev)):
        bold_flag = 'bold' if i in [0, 2] else 'normal'
        col = MAGENTA if i == 2 else (NAVY if i == 0 else SLATE_DARK)
        tag = f" {v} (±{ci})"
        ax1.text(v + 2, i, tag, va='center', fontsize=8.0, fontweight=bold_flag, color=col)

    # Right: Cost per minute
    ax2.barh(y_pos, prices_min_rev, color=colors_elo, height=0.45, zorder=3)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(["" for _ in y_pos])
    ax2.set_xlabel("公开 API 推理费率 (美元 / 分钟)", fontsize=8.6, color=SLATE_DARK)
    ax2.set_xlim(0, 42)
    ax2.set_title("商业 API 推理定价对比 (美元/分钟 与 美元/秒)", fontsize=10.0, fontweight='bold', color=NAVY, pad=8, loc='left')

    for i, p in enumerate(prices_min_rev):
        bold_flag = 'bold' if i in [0, 2] else 'normal'
        col = MAGENTA if i == 2 else (NAVY if i == 0 else SLATE_DARK)
        per_sec = p / 60.0
        tag = f" ${p:.2f}/min (${per_sec:.2f}/s)"
        ax2.text(p + 0.6, i, tag, va='center', fontsize=8.0, fontweight=bold_flag, color=col, parse_math=False)

    # Note at bottom
    fig.text(
        0.50,
        0.082,
        "注：本仓库 Colab L4 (544p 124帧 ≈ 5.17s) 稳态实测 165.4s/镜 (~0.089 CU)；按 Pay-As-You-Go ($9.99/100 CU ≈ $0.10/CU) 折算约 $0.0089/镜 (~$0.10/min 输出时长，不含冷启)。",
        fontsize=7.6,
        color=SLATE_MID,
        ha='center',
        transform=fig.transFigure,
        parse_math=False
    )

    fig.savefig(output_path, dpi=300, facecolor='#FFFFFF', bbox_inches='tight')
    plt.close(fig)
    print(f"[+] Saved AA Benchmark Chart to {output_path}")


if __name__ == "__main__":
    out_file = Path(__file__).resolve().parent.parent / "docs" / "articles" / "assets" / "chart_01_aa_benchmark.png"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    render_aa_chart(out_file)
