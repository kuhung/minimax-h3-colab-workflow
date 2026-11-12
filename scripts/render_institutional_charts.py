#!/usr/bin/env python3
"""
Generate institutional research charts for MiniMax H3 articles.
Adheres to institutional-chart skill standards.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# Setup typography and colors
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
GOLD_DARK = '#9A6B1F'
GOLD_BG = '#FEF9C3'


def add_research_chrome(fig, title, subtitle, source_text):
    # Top Navy line (2.5pt)
    fig.add_artist(plt.Line2D([0.04, 0.96], [0.965, 0.965], color=NAVY, linewidth=2.5, transform=fig.transFigure))
    # Title
    fig.text(0.04, 0.925, title, fontsize=14, fontweight='bold', color='#0F172A', ha='left', va='top')
    # Subtitle
    fig.text(0.04, 0.880, subtitle, fontsize=9.5, color='#475569', ha='left', va='top')
    # Bottom silver divider (0.8pt)
    fig.add_artist(plt.Line2D([0.04, 0.96], [0.065, 0.065], color=SILVER, linewidth=0.8, transform=fig.transFigure))
    # Footer source text
    fig.text(0.04, 0.045, source_text, fontsize=8, color=SLATE_MID, ha='left', va='top')


def render_chart_01(output_path):
    fig = plt.figure(figsize=(13.33, 7.5), facecolor='#FFFFFF')
    add_research_chrome(
        fig,
        title="2026 主流 AI 视频生成模型多维选型矩阵与核心特性对标",
        subtitle="围绕时序一致性、多模态参考上限、原生音画同频、物理模拟及开源推理成本横向对标（核对日期：2026-11-12）",
        source_text="数据来源：各模型官方文档与公开规格（核对日期：2026-11-12，以各官网最新说明为准） | 视觉设计：谷粒 · KUHUNG"
    )

    ax = fig.add_axes([0.04, 0.08, 0.92, 0.77])
    ax.axis('off')

    models = [
        {
            "name": "MiniMax H3 (海螺 3.0)",
            "badge": "开源基座 / 原生声画",
            "badge_color": TEAL_DARK,
            "badge_bg": TEAL_BG,
            "arch": "33B 单流 DiT，开放 768p 基础权重",
            "ref": "最高 12 个文件 (≤9图、≤3视频、≤3音频)",
            "audio": "原生 32kHz 双声道立体声 (音画同频扩散)",
            "motion": "平稳镜头推移、微观物理动态、环境拟音",
            "cost": "支持 NVFP4/INT8 量化，适配 24GB 单卡",
            "best_for": "商品外观锚定、市井自然声景、本地私有化批量部署",
            "is_focal": True
        },
        {
            "name": "Seedance 2.5 (即梦高端线)",
            "badge": "闭源旗舰 / 叙事规划",
            "badge_color": NAVY,
            "badge_bg": BG_CARD,
            "arch": "双分支 DiT，内置多镜头叙事规划器",
            "ref": "最高 12 个文件，支持 @ 标签角色绑定",
            "audio": "双分支毫秒级同步，支持 3 人音色克隆",
            "motion": "复杂动作连贯性优秀，支持跨镜头叙事",
            "cost": "托管云端 API，商业级分层按秒计费",
            "best_for": "商业广告全案、多镜头故事短片、定制配音",
            "is_focal": False
        },
        {
            "name": "可灵 AI (Kling 3.0)",
            "badge": "闭源旗舰 / 动力学标杆",
            "badge_color": NAVY,
            "badge_bg": BG_CARD,
            "arch": "自研 DiT，强动力学与身体结构建模",
            "ref": "支持多图主体参考与首尾帧控制",
            "audio": "内置拟真音色对口型 (无环境声原生同出)",
            "motion": "大范围肢体对抗、大幅重心转移标杆",
            "cost": "算力成本相对较高，侧重单镜头高表现力",
            "best_for": "人物高动态表演、打斗跳跃、长达 2 分钟延长",
            "is_focal": False
        },
        {
            "name": "Google Veo 3.1",
            "badge": "海外旗舰 / 光学级质感",
            "badge_color": SLATE_MID,
            "badge_bg": BG_CARD,
            "arch": "深度扩展 DiT，专注光学级真实反射",
            "ref": "支持首尾帧控制与场景扩展功能",
            "audio": "分层独立音频合成 (无声/带声分级)",
            "motion": "高保真人群、微观流体与水面光影",
            "cost": "Google Cloud 托管，按秒分层阶梯计费",
            "best_for": "4 到 8 秒极高光影写实空镜、影视级概念片",
            "is_focal": False
        },
        {
            "name": "通义万相 Wan 2.1",
            "badge": "开源标杆 / 首尾帧控制",
            "badge_color": SLATE_MID,
            "badge_bg": BG_CARD,
            "arch": "Apache 2.0 开源 14B DiT 架构",
            "ref": "720p 首尾帧生成 (FLF2V / VACE)",
            "audio": "纯画面生成，无内置音频扩散轨道",
            "motion": "平稳镜头推移、开源 720p FLF2V / VACE 形变过渡",
            "cost": "ComfyUI 开源部署，消费级显卡可用",
            "best_for": "抽象空间转场、形态演变、开源二开实验",
            "is_focal": False
        }
    ]

    card_width = 0.185
    card_gap = 0.015
    start_x = 0.005

    for i, m in enumerate(models):
        cx = start_x + i * (card_width + card_gap)
        cy = 0.02
        ch = 0.95

        edge_c = NAVY if m["is_focal"] else SILVER
        bg_c = '#FFFFFF' if m["is_focal"] else BG_CARD
        lw = 1.8 if m["is_focal"] else 0.8
        card = patches.FancyBboxPatch(
            (cx, cy), card_width, ch,
            boxstyle="round,pad=0.015,rounding_size=0.02",
            edgecolor=edge_c, facecolor=bg_c, linewidth=lw
        )
        ax.add_patch(card)

        title_c = NAVY if m["is_focal"] else SLATE_DARK
        ax.text(cx + 0.012, cy + ch - 0.045, m["name"], fontsize=9.8, fontweight='bold', color=title_c)

        b_box = patches.FancyBboxPatch(
            (cx + 0.012, cy + ch - 0.085), 0.16, 0.032,
            boxstyle="round,pad=0.005,rounding_size=0.008",
            edgecolor=m["badge_color"], facecolor=m["badge_bg"], linewidth=0.8
        )
        ax.add_patch(b_box)
        ax.text(cx + 0.018, cy + ch - 0.078, m["badge"], fontsize=7.5, fontweight='bold', color=m["badge_color"])

        ax.add_artist(plt.Line2D([cx + 0.012, cx + card_width - 0.012], [cy + ch - 0.105, cy + ch - 0.105], color=SILVER, linewidth=0.6))

        fields = [
            ("架构与开源", m["arch"]),
            ("多模态参考配额", m["ref"]),
            ("音频生成模式", m["audio"]),
            ("物理与时序动态", m["motion"]),
            ("算力与显存部署", m["cost"]),
            ("最佳生产适用场景", m["best_for"])
        ]

        curr_y = cy + ch - 0.135
        for label, val in fields:
            ax.text(cx + 0.012, curr_y, label, fontsize=7.8, fontweight='bold', color=NAVY if m["is_focal"] else SLATE_MID)
            curr_y -= 0.025
            # Wrap text cleanly by display width (~24 half-width units per line)
            lines, buf, w = [], "", 0
            for ch_char in val:
                cw = 2 if ord(ch_char) > 127 else 1
                if w + cw > 26:
                    lines.append(buf)
                    buf, w = ch_char, cw
                else:
                    buf += ch_char
                    w += cw
            if buf:
                lines.append(buf)
            ax.text(cx + 0.012, curr_y, "\n".join(lines), fontsize=7.0, color=SLATE_DARK, va='top', linespacing=1.35)
            curr_y -= 0.105

    fig.savefig(output_path, dpi=300, facecolor='#FFFFFF', bbox_inches='tight')
    plt.close(fig)
    print(f"[+] Saved Chart 01 to {output_path}")


def render_chart_02(output_path):
    fig = plt.figure(figsize=(13.33, 7.5), facecolor='#FFFFFF')
    add_research_chrome(
        fig,
        title="MiniMax H3 基准工作流核心参数实测基准与物理影响机理",
        subtitle="基于 Google Colab L4 (24GB Ada Lovelace, 1.94 CU/h) 的真实物理实测日志",
        source_text="数据来源：minimax-h3-colab-workflow 阶段一真实基准测试日志 | 视觉设计：谷粒 · KUHUNG"
    )

    # Top charts
    ax1 = fig.add_axes([0.08, 0.45, 0.38, 0.34])
    ax2 = fig.add_axes([0.55, 0.45, 0.38, 0.34])

    # Bottom explanation cards
    ax_card1 = fig.add_axes([0.08, 0.10, 0.38, 0.27])
    ax_card2 = fig.add_axes([0.55, 0.10, 0.38, 0.27])

    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(SLATE_LIGHT)
        ax.spines['bottom'].set_color(SLATE_LIGHT)
        ax.yaxis.grid(True, linestyle='--', linewidth=0.7, color=GRID_COLOR)
        ax.set_axisbelow(True)

    for ax in [ax_card1, ax_card2]:
        ax.axis('off')

    # --- Left Chart: CFG Mode (Verified v1.0 vs v2.0 @ 960x544) ---
    ax1.set_title("引导方式（CFG）对同规格实测耗时与 CU 的影响", fontsize=10.2, fontweight='bold', color=NAVY, pad=10, loc='left')
    x1 = [0, 1]
    times = [315.8, 165.4]
    ax1.bar(x1, times, color=[SLATE_LIGHT, NAVY], width=0.38, zorder=3)
    ax1.set_xticks(x1)
    ax1.set_xticklabels(["v1.0 双路 CFG (3.0)\n16 次 DiT 前向 (544p 实测)", "v2.0 单路 BasicGuider (1.0)\n8 次 DiT 前向 (544p 稳态)"], fontsize=8.3, color=SLATE_DARK)
    ax1.set_ylabel("生成耗时 (秒)", fontsize=8.5, color=SLATE_DARK)
    ax1.set_ylim(0, 410)

    ax1.text(0, 323, "315.8 秒 (~5.3 分钟)\n折算 ~0.170 CU (无音频)", ha='center', va='bottom', fontsize=8, fontweight='bold', color=CRIMSON)
    ax1.text(1, 173, "165.4 秒 (~2 分 45 秒)\n稳态 ~0.089 CU (含 32kHz 音轨)", ha='center', va='bottom', fontsize=8, fontweight='bold', color=TEAL_DARK)

    # --- Right Chart: Verified Resolution Scaling & Smoke VRAM Headroom ---
    ax2.set_title("分辨率切换对实测单镜耗时与 L4 显存边界的影响", fontsize=10.2, fontweight='bold', color=NAVY, pad=10, loc='left')
    x2 = [0, 1, 2]
    res_times = [874.7, 315.8, 165.4]
    ax2.bar(x2, res_times, color=[CRIMSON, SLATE_LIGHT, NAVY], width=0.42, zorder=3)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(["v1.0 1344×768\n103.2 万像素 (CFG=3.0)", "v1.0 960×544\n52.2 万像素 (CFG=3.0)", "v2.0 960×544\n52.2 万像素 (CFG=1.0)"], fontsize=8.0, color=SLATE_DARK)
    ax2.set_ylabel("实测耗时 (秒)", fontsize=8.5, color=SLATE_DARK)
    ax2.set_ylim(0, 1120)

    ax2.text(0, 890, "874.7 秒 (~14.6 分钟)\n像素 +97.6%，单镜 ~0.471 CU", ha='center', va='bottom', fontsize=7.8, fontweight='bold', color=CRIMSON)
    ax2.text(1, 332, "315.8 秒\n冒烟峰值 21.91/22.03 GB", ha='center', va='bottom', fontsize=7.8, fontweight='bold', color=SLATE_DARK)
    ax2.text(2, 182, "165.4 秒 (稳态)\n~0.089 CU / 首镜冷启 243.5s", ha='center', va='bottom', fontsize=7.8, fontweight='bold', color=TEAL_DARK)

    # --- Bottom Left Card ---
    c1 = patches.FancyBboxPatch(
        (0.01, 0.02), 0.98, 0.94,
        boxstyle="round,pad=0.015,rounding_size=0.03",
        edgecolor=TEAL_DARK, facecolor=TEAL_BG, linewidth=1.0
    )
    ax_card1.add_patch(c1)
    ax_card1.text(0.04, 0.82, "【CFG=1.0 单路无引导的工程机理】", fontsize=9, fontweight='bold', color=TEAL_DARK)
    ax_card1.text(0.04, 0.58, "• 架构机理：Turbo 蒸馏模型已将提示词条件内化至单次扩散漂移场；", fontsize=8, color=SLATE_DARK)
    ax_card1.text(0.04, 0.38, "• 实测收益：8 步采样精确执行 8 次 DiT 前向，较 v1.0 (315.8s) 提速 47.6%；", fontsize=8, color=SLATE_DARK)
    ax_card1.text(0.04, 0.18, "• 音画成片：省下冗余负向引导分支后，同卡封装 32kHz 立体声 MP4。", fontsize=8, color=SLATE_DARK)

    # --- Bottom Right Card ---
    c2 = patches.FancyBboxPatch(
        (0.01, 0.02), 0.98, 0.94,
        boxstyle="round,pad=0.015,rounding_size=0.03",
        edgecolor=NAVY, facecolor=BG_CARD, linewidth=1.0
    )
    ax_card2.add_patch(c2)
    ax_card2.text(0.04, 0.82, "【960×544 规格的实测显存与成本边界】", fontsize=9, fontweight='bold', color=NAVY)
    ax_card2.text(0.04, 0.58, "• 像素复杂度：768p 像素增加 97.6%，实测耗时从 315.8s 升至 874.1~875.4s；", fontsize=8, color=SLATE_DARK)
    ax_card2.text(0.04, 0.38, "• 显存实测：v1.0 544p 冒烟去噪 21.77 GB、峰值 21.91 / 22.03 GB (余量 0.12 GB)；", fontsize=8, color=SLATE_DARK)
    ax_card2.text(0.04, 0.18, "• 生产基准：v2.0 544p 稳态 165.4s (~0.089 CU/镜，含冷启 3 镜约 0.309 CU)。", fontsize=8, color=SLATE_DARK)

    fig.savefig(output_path, dpi=300, facecolor='#FFFFFF', bbox_inches='tight')
    plt.close(fig)
    print(f"[+] Saved Chart 02 to {output_path}")


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent.parent / "docs" / "articles" / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)
    render_chart_01(out_dir / "chart_01_model_matrix.png")
    render_chart_02(out_dir / "chart_02_params_benchmark.png")
