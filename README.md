# MiniMax H3 Colab 纯本地视频生产工作流

> 基于 Google AI Pro 每月赠送的 200 Colab 计算单元，全流程**纯本地白嫖**运行 MiniMax H3（Hailuo 3.0）全模态视频模型。
> **硬约束**：不调用任何 MiniMax 外部 API、不产生二次计费、聚焦无脸视频（Faceless Video）高品质批量出片。

---

## 🏗️ 架构与工作流全景

```mermaid
flowchart TD
    A["Google AI Pro 账号 (200 计算单元)"] --> B["Colab 硬件门禁检测 (check_gpu.py)"]
    B -->|T4 (算力 7.5)| C["❌ 立即中断退出，防白白扣费"]
    B -->|L4 (24GB) / A100 (40GB)| D["✅ 硬件通过 (算力 >= 8.0)"]
    D --> E["Google Drive 挂载与软链接缓存 (download_models.py)"]
    E --> F["ComfyUI 后台服务启动 (动态显存卸载)"]
    F --> G["冒烟快筛 (544p / 5秒 / Turbo 4步)"]
    G --> H["无头批量生产队列 (768p / PDD 8步 / batch_runner.py)"]
    H --> I["产物落盘: Google Drive (MP4 + 原生 32kHz 音频 + 结构化日志)"]
    I --> J["🛑 自动提醒断开运行时 (防闲置偷跑算力单元)"]
```

---

## 📁 仓库目录结构

- 📓 [notebooks/minimax_h3_l4_production.ipynb](file:///Users/kuhung/.gemini/antigravity/scratch/minimax-h3-colab-workflow/notebooks/minimax_h3_l4_production.ipynb) — **Colab 核心交互笔记本**（集成门禁、模型挂载、WebUI 隧道与批量渲染）。
- ⚙️ [scripts/check_gpu.py](file:///Users/kuhung/.gemini/antigravity/scratch/minimax-h3-colab-workflow/scripts/check_gpu.py) — **硬件环境门禁脚本**（严格校验计算能力 ≥ 8.0、显存与磁盘）。
- 📦 [scripts/download_models.py](file:///Users/kuhung/.gemini/antigravity/scratch/minimax-h3-colab-workflow/scripts/download_models.py) — **权重断点与 Drive 缓存管理**（只拉取 40GB 剪枝最小集，支持二次启动 0 秒软链接）。
- 🎬 [scripts/batch_runner.py](file:///Users/kuhung/.gemini/antigravity/scratch/minimax-h3-colab-workflow/scripts/batch_runner.py) — **无头批量队列渲染器**（直连本地 8188 接口，批量出片并自动写 Drive 与日志）。
- 🧩 [workflows/h3_t2v_l4_baseline.json](file:///Users/kuhung/.gemini/antigravity/scratch/minimax-h3-colab-workflow/workflows/h3_t2v_l4_baseline.json) — ComfyUI 544p 快速冒烟基线工作流。
- 🧩 [workflows/h3_t2v_pdd8_production.json](file:///Users/kuhung/.gemini/antigravity/scratch/minimax-h3-colab-workflow/workflows/h3_t2v_pdd8_production.json) — ComfyUI 768p PDD 8步高画质出片工作流。
- 📋 [configs/prompts.example.json](file:///Users/kuhung/.gemini/antigravity/scratch/minimax-h3-colab-workflow/configs/prompts.example.json) — 无脸视频排期任务模板清单。
- 📖 [docs/PROMPT_RECIPES.md](file:///Users/kuhung/.gemini/antigravity/scratch/minimax-h3-colab-workflow/docs/PROMPT_RECIPES.md) — H3 音画同出提示词结构公式与无脸镜头配方库。
- 📚 [docs/RESEARCH_SUMMARY.md](file:///Users/kuhung/.gemini/antigravity/scratch/minimax-h3-colab-workflow/docs/RESEARCH_SUMMARY.md) — 前期全网调研核查报告归档。

---

## ⚡ 核心避坑法则（保卫 200 计算单元）

| 规则 | 说明与处置方式 |
|---|---|
| **1. 认准主力显卡** | **首选 L4 (24GB)**：算力 8.9，消耗约 1.5 单元/小时，性价比最高；**次选 A100 (40GB)**（消耗约 5.5 单元/小时）。 |
| **2. 严禁在 T4 上跑** | T4 算力 7.5，不支持 Int8 ConvRot 算子，强行跑会立即 OOM 崩溃。门禁脚本会自动熔断退出。 |
| **3. 模型不要重复下载** | 首次下载后将权重同步至 Google Drive；下次开机直接执行软链接，节省 30 分钟下载等待。 |
| **4. 分级画幅策略** | 快筛验证用 960×544，决赛成片上 1344×768。960×544 比 768p 每步提速约 2.3 倍。 |
| **5. 拒绝本地硬升 2K** | H3 官方开源权重上限为 768p（2K 模块闭源）。本地最高产出 768p，后期按需用 Topaz/Real-ESRGAN 超分。 |
| **6. 跑完立即断开** | 渲染任务完成后，**务必在 Colab 点击“断开连接并删除运行时”**，防止后台空转持续扣费。 |

---

## 🚀 极速起步 SOP

### 第一步：打开笔记本与挂载
1. 将 [minimax_h3_l4_production.ipynb](file:///Users/kuhung/.gemini/antigravity/scratch/minimax-h3-colab-workflow/notebooks/minimax_h3_l4_production.ipynb) 上传或导入至 Google Colab。
2. 运行时配置：**GPU -> L4 (高内存)**。
3. 顺序执行单元格 1（挂载 Google Drive）与单元格 2（硬件门禁检测）。

### 第二步：一键环境与模型挂载
1. 执行单元格 3 与 4，脚本会自动从 HuggingFace 拉取剪枝量化最小集（约 40GB）或软链接 Drive 缓存：
   - 变压器：`minimax_h3_fl2va_pruned_int8_convrot.safetensors` (~20.97GB)
   - 文本编码器：`qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` (~15.69GB)
   - VAE & PDD 8步蒸馏 LoRA
2. 执行单元格 5 启动 ComfyUI，并获取 Cloudflare 网页管理端（可选，用于可视化微调节点）。

### 第三步：无头批处理生产（批量出片）
1. 在 [configs/prompts.example.json](file:///Users/kuhung/.gemini/antigravity/scratch/minimax-h3-colab-workflow/configs/prompts.example.json) 中填入无脸视频镜头脚本。
2. 执行批处理单元格或在终端运行：
   ```bash
   python scripts/batch_runner.py \
       --workflow workflows/h3_t2v_pdd8_production.json \
       --prompts configs/prompts.example.json \
       --output /content/drive/MyDrive/MiniMax_Outputs
   ```
3. 渲染生成的 MP4 文件与 `render_history.jsonl` 日志将实时自动存入您的 Google Drive。
4. 执行完毕，断开 Colab 运行时。
