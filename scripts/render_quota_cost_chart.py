#!/usr/bin/env python3
"""
Generate institutional comparison chart for Colab bundled quota video capacity
and equivalent mainstream commercial video API costs.
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
TEAL_DARK = '#115E59'
TEAL_MID = '#0D9488'
TEAL_BG = '#F0FDFA'
BLUE = '#2563EB'
ORANGE = '#E05A1F'


def add_research_chrome(fig, title, subtitle, source_text):
    fig.add_artist(plt.Line2D([0.04, 0.96], [0.965, 0.965], color=NAVY, linewidth=2.5, transform=fig.transFigure))
    fig.text(0.04, 0.930, title, fontsize=13.2, fontweight='bold', color='#0F172A', ha='left', va='top', parse_math=False)
    fig.text(0.04, 0.890, subtitle, fontsize=9.0, color='#475569', ha='left', va='top', parse_math=False)
    fig.add_artist(plt.Line2D([0.04, 0.96], [0.068, 0.068], color=SILVER, linewidth=0.8, transform=fig.transFigure))
    fig.text(0.04, 0.048, source_text, fontsize=7.6, color=SLATE_MID, ha='left', va='top', parse_math=False)


def render_quota_cost_chart(output_path: Path):
    fig = plt.figure(figsize=(13.33, 7.5), facecolor='#FFFFFF')
    add_research_chrome(
        fig,
        title="每月 $19.99 附带额度可出 167–193 分钟视频，折合主流 API 需 $801–$6,593",
        subtitle="Colab L4 实测单分钟约 $0.10（544p 立体声 MP4，1.94 CU/h），比主流商业视频 API 低 46–330 倍（2026-11-28 核验）",
        source_text="口径：v2_0 L4 实测 1.94 CU/h（含冷启均摊 0.103 至稳态 0.089 CU/镜，5.17s/镜）；按高耗 3.5 CU/h 计，200 CU 仍可出 ~107 分钟 ($514–$3,654) | 谷粒 · KUHUNG"
    )

    # Left panel: Per-minute output cost (USD / min)
    ax1 = fig.add_axes([0.21, 0.285, 0.25, 0.49])
    # Right panel: Google AI Pro (200 CU/mo, $19.99) equivalent commercial API spend
    ax2 = fig.add_axes([0.62, 0.285, 0.33, 0.49])

    for ax in (ax1, ax2):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(SLATE_LIGHT)
        ax.spines['bottom'].set_color(SLATE_LIGHT)
        ax.xaxis.grid(True, linestyle='--', linewidth=0.7, color=GRID_COLOR)
        ax.set_axisbelow(True)

    # --- Left Panel Data (bottom to top) ---
    labels_left = [
        "Seedance 2.5 (API)",
        "Veo 3.1 标准版 (有声 API)",
        "Kling 4.0 (AA 基准 API)",
        "Wan 3.0 (API)",
        "MiniMax H3 Max (API)",
        "MiniMax H3 768p (API)",
        "Colab L4 高耗参考 (3.5 CU/h)",
        "Colab L4 三镜均摊 (含冷启)",
        "Colab L4 稳态 (1.94 CU/h)",
    ]
    costs_per_min = [34.12, 24.00, 12.60, 12.00, 9.60, 4.80, 0.19, 0.12, 0.10]
    colors_left = [
        CRIMSON,
        ORANGE,
        BLUE,
        SLATE_DARK,
        SLATE_MID,
        NAVY,
        SLATE_LIGHT,
        TEAL_MID,
        TEAL_DARK,
    ]
    tags_left = [
        " $34.12/min (~330x)",
        " $24.00/min (~232x)",
        " $12.60/min (~122x)",
        " $12.00/min (~116x)",
        " $9.60/min (~93x)",
        " $4.80/min (~46x)",
        " $0.19/min (高耗灵敏度)",
        " $0.12/min (含冷启)",
        " $0.10/min (基准 1.0x)",
    ]

    y1 = list(range(len(labels_left)))
    ax1.barh(y1, costs_per_min, color=colors_left, height=0.56, zorder=3)
    ax1.set_yticks(y1)
    ax1.set_yticklabels(labels_left, fontsize=8.3, color=SLATE_DARK)
    ax1.set_xlim(0, 58)
    ax1.tick_params(axis='x', labelsize=8.3, colors=SLATE_DARK)
    ax1.set_xlabel("每分钟成片费率 (美元 / 分钟)", fontsize=8.6, color=SLATE_DARK, labelpad=6)
    ax1.set_title("每分钟视频产出成本对照 (USD / min)", fontsize=10.2, fontweight='bold', color=NAVY, pad=10, loc='left')

    for i, (val, tag) in enumerate(zip(costs_per_min, tags_left)):
        col = TEAL_DARK if i == 8 else (CRIMSON if i == 0 else SLATE_DARK)
        weight = 'bold' if i in (0, 5, 8) else 'normal'
        ax1.text(val + 0.8, i, tag, va='center', fontsize=8.0, fontweight=weight, color=col, parse_math=False)

    # --- Right Panel Data (Google AI Pro 200 CU/mo -> ~167-193 min) ---
    labels_right = [
        "Seedance 2.5 等效账单",
        "Veo 3.1 标准版等效账单",
        "Kling 4.0 等效账单",
        "Wan 3.0 等效账单",
        "MiniMax H3 768p 等效账单",
        "Google AI Pro 月订阅费",
    ]
    vals_right = [6593, 4637, 2435, 2319, 927, 19.99]
    colors_right = [CRIMSON, ORANGE, BLUE, SLATE_DARK, NAVY, TEAL_DARK]
    tags_right = [
        " $5,693–$6,593",
        " $4,005–$4,637",
        " $2,102–$2,435",
        " $2,002–$2,319",
        " $801–$927",
        " $19.99/月 (含 200 CU)",
    ]

    y2 = list(range(len(labels_right)))
    ax2.barh(y2, vals_right, color=colors_right, height=0.52, zorder=3)
    ax2.set_yticks(y2)
    ax2.set_yticklabels(labels_right, fontsize=8.5, color=SLATE_DARK)
    ax2.set_xlim(0, 9000)
    ax2.tick_params(axis='x', labelsize=8.3, colors=SLATE_DARK)
    ax2.set_xlabel("同产能对应费用 (美元 USD)", fontsize=8.6, color=SLATE_DARK, labelpad=6)
    ax2.set_title("以 Google AI Pro (200 CU/月) 产出 167–193 分钟为例", fontsize=10.2, fontweight='bold', color=NAVY, pad=10, loc='left')

    for i, (val, tag) in enumerate(zip(vals_right, tags_right)):
        col = TEAL_DARK if i == 5 else (CRIMSON if i == 0 else SLATE_DARK)
        weight = 'bold' if i in (0, 4, 5) else 'normal'
        ax2.text(val + 145, i, tag, va='center', fontsize=8.3, fontweight=weight, color=col, parse_math=False)

    # --- Bottom 4-Card KPI Strip ---
    cards = [
        {
            "x": 0.040,
            "plan": "Colab Pro / 按量 · 100 CU ($9.99)",
            "duration": "83–97 分钟",
            "clips": "~969–1,122 镜 (5.17s/镜)",
            "api": "等效商业 API $400–$3,296",
            "focal": False,
        },
        {
            "x": 0.273,
            "plan": "Google AI Pro · 200 CU ($19.99/月)",
            "duration": "167–193 分钟",
            "clips": "~1,938–2,244 镜 (5.17s/镜)",
            "api": "等效商业 API $801–$6,593",
            "focal": True,
        },
        {
            "x": 0.506,
            "plan": "Colab Pro+ · 600 CU ($49.99/月)",
            "duration": "501–580 分钟",
            "clips": "~5,813–6,732 镜 (5.17s/镜)",
            "api": "等效商业 API $2,403–$19,778",
            "focal": False,
        },
        {
            "x": 0.740,
            "plan": "AI Ultra 20x · 2,000 CU ($199.99/月)",
            "duration": "1,669–1,932 分钟",
            "clips": "~19,377–22,439 镜 (5.17s/镜)",
            "api": "等效商业 API $8,009–$65,927",
            "focal": False,
        },
    ]

    cy = 0.088
    cw = 0.220
    ch = 0.115
    for c in cards:
        edge_c = TEAL_DARK if c["focal"] else SILVER
        bg_c = TEAL_BG if c["focal"] else BG_CARD
        lw = 1.6 if c["focal"] else 0.9
        box = patches.FancyBboxPatch(
            (c["x"], cy), cw, ch,
            boxstyle="round,pad=0.0,rounding_size=0.010",
            edgecolor=edge_c, facecolor=bg_c, linewidth=lw, transform=fig.transFigure
        )
        fig.add_artist(box)

        pad_x = c["x"] + 0.010
        plan_col = TEAL_DARK if c["focal"] else SLATE_DARK
        dur_col = TEAL_DARK if c["focal"] else NAVY
        fig.text(pad_x, cy + ch - 0.013, c["plan"], fontsize=7.9, fontweight='bold', color=plan_col, va='top', transform=fig.transFigure, parse_math=False)
        fig.text(pad_x, cy + ch - 0.041, c["duration"], fontsize=12.0, fontweight='bold', color=dur_col, va='top', transform=fig.transFigure, parse_math=False)
        fig.text(pad_x, cy + ch - 0.073, c["clips"], fontsize=7.5, color=SLATE_MID, va='top', transform=fig.transFigure, parse_math=False)
        fig.text(pad_x, cy + 0.013, c["api"], fontsize=7.8, fontweight='bold', color=CRIMSON if c["focal"] else SLATE_DARK, va='bottom', transform=fig.transFigure, parse_math=False)

    fig.savefig(output_path, dpi=300, facecolor='#FFFFFF', bbox_inches='tight')
    plt.close(fig)
    print(f"[+] Saved Quota Cost Chart to {output_path}")


if __name__ == "__main__":
    out_file = Path(__file__).resolve().parent.parent / "docs" / "articles" / "assets" / "chart_03_quota_cost_comparison.png"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    render_quota_cost_chart(out_file)
