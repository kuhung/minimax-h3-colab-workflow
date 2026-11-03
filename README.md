# MiniMax H3 Colab 纯本地视频生产工作流 (2.0 Architecture)

> 基于 Google AI Pro/Ultra 赠送的 Colab 计算单元（Pro 100 CU/月，Ultra 500 CU/月，90天可囤积），全流程**纯本地白嫖**运行 MiniMax H3（Hailuo 3.0）全模态视频模型。
> **硬约束**：不调用任何 MiniMax 外部商业 API、不产生二次计费、聚焦无脸视频（Faceless Video）高品质批量出片。

---

## 🚀 2.0 架构核心技术飞跃

1. **单路无 CFG 架构 (`BasicGuider`, CFG=1.0)**：
   - 彻底废除正负双路 `CLIPTextEncode` + `KSampler (cfg=3.0)` 架构。
   - 采用 `MiniMaxH3ImageToVideo` -> `BasicGuider` -> `SamplerCustomAdvanced` 标准拓扑。
   - **单镜推理算力直接减半（耗时从 306s 降低至 ~150s）**，彻底消除蒸馏模型在外推 CFG 下的过曝与相位破音。
2. **CUDA 与硬件自适应感知路由**：
   - `int8_convrot` 算子依赖 PyTorch `cu130` 与 `comfy-kitchen`；
   - 脚本自动检测当前环境：若处于 Colab 默认的 CUDA 12.x + L4 (Ada CC 8.9)，自动无缝切换为 `fp8_scaled` (DiT) + `video_vae_fp16` (VAE)，完全释放 Ada 第四代 Tensor Core 硬件吞吐，杜绝非 cu130 慢路径回退！
3. **4-Step / 8-Step Turbo LoRA 双模式**：
   - **8-Step**（默认，~150s）：电影级工业品质交付；
   - **4-Step**（极速，~75s）：快速分镜验证与批量快筛。
4. **动态节点契约与网格对齐**：
   - 提交前根据 `/object_info` 校验输入契约，自动安全裁剪未注册参数；
   - 帧数向上严格自动对齐到 `17k + 5`（如 121 -> 124 帧）。

---

## 🏗️ 2.0 工作流全景架构图

```mermaid
flowchart TD
    A["Google AI Pro/Ultra 账号 (100~500 CU/月，90天有效期)"] --> B["Colab 硬件门禁检测 (check_gpu.py)"]
    B -->|T4 (算力 7.5)| C["❌ 立即中断并销毁实例，防扣费"]
    B -->|L4 (24GB) 算力 8.9| D["✅ 硬件通过 (实测 1.9 CU/h)"]
    D --> E["CUDA 自适应挂载 (cu130: int8_convrot / cu12x: fp8_scaled)"]
    E --> F["ComfyUI 后台守护启动 (低显存卸载模式)"]
    F --> G["/object_info 节点契约预检 (14 类核心节点)"]
    G --> H["2.0 单路无 CFG 批处理 (BasicGuider, 8步/4步 Turbo, ~150s/镜)"]
    H --> I["产物落盘: Google Drive (.mp4 视频 + 32kHz 原生立体声 + 日志)"]
    I --> J["🛑 三重关机保护: finally 销毁 + Watchdog 看门狗 (杜绝闲置扣费)"]
```

---

## 🧩 2.0 节点链路拓扑 (15 节点闭环)

```
[1] UNETLoader (minimax_h3_fl2va_pruned_fp8_scaled / int8_convrot)
 └──> [10] LoraLoaderModelOnly (minimax_h3_fl2v_turbo_8step / 4step)
       ├──> [7] BasicGuider (CFG=1.0) <── conditioning ── [6] MiniMaxH3ImageToVideo <── [2] CLIPLoader (Qwen3-VL 32B NVFP4)
       │         │                                               │                      └── [4] VAELoader (Video VAE)
       │         │                                         latent_image
       │         ▼                                               │
       │    [3] SamplerCustomAdvanced <──────────────────────────┘
       │         ├── noise <── [14] RandomNoise (seed)
       │         ├── sampler <── [15] KSamplerSelect (res_multistep)
       │         └── sigmas <── [16] BasicScheduler (steps: 8, scheduler: simple)
       │
       ├──> [8] VAEDecode (Video) <── vae ── [4] VAELoader
       └──> [12] VAEDecodeAudio (Audio) <── vae ── [11] VAELoader (32kHz Stereo FP32)
                 │                                        │
                 ▼                                        ▼
             [13] CreateVideo (24.0 fps) ──────────> [9] SaveVideo (.mp4)
```

---

## 📁 仓库目录结构

- 📓 [notebooks/minimax_h3_l4_production.ipynb](file:///Users/kuhung/roy/3_Active_Labs/minimax-h3-colab-workflow/notebooks/minimax_h3_l4_production.ipynb) — **Colab 核心交互笔记本 (2.0)**（自适应权重、看门狗、门禁、完整性缓存、2.0 批处理与 HTML5 试听）。
- 📓 [notebooks/test_shanghai_mp4_audio.ipynb](file:///Users/kuhung/roy/3_Active_Labs/minimax-h3-colab-workflow/notebooks/test_shanghai_mp4_audio.ipynb) — **上海街景 10~20s 快速测试笔记本 (2.0)**。
- 🤖 [scripts/agent_colab_runner.py](file:///Users/kuhung/roy/3_Active_Labs/minimax-h3-colab-workflow/scripts/agent_colab_runner.py) — **Agent 自动化托管执行器**（支持三重防扣费兜底、自适应权重、看门狗熔断）。
- ⚙️ [scripts/check_gpu.py](file:///Users/kuhung/roy/3_Active_Labs/minimax-h3-colab-workflow/scripts/check_gpu.py) — **硬件环境门禁脚本**（严格校验计算能力 ≥ 8.0、显存与磁盘；支持不达标立即销毁）。
- 📦 [scripts/download_models.py](file:///Users/kuhung/roy/3_Active_Labs/minimax-h3-colab-workflow/scripts/download_models.py) — **权重自适应校验与 Drive 缓存管理**（`--variant` / `--turbo-steps`、精确字节校验、safetensors 头部元数据边界校验）。
- 🎬 [scripts/batch_runner.py](file:///Users/kuhung/roy/3_Active_Labs/minimax-h3-colab-workflow/scripts/batch_runner.py) — **无头批量队列渲染器 (2.0)**（`BasicGuider` 拓扑、`format_h3_prompt` 多镜头组装、`align_frame_count` 17k+5 帧数对齐、`/object_info` 契约校验）。
- 🧩 [workflows/h3_t2v_pdd8_production.json](file:///Users/kuhung/roy/3_Active_Labs/minimax-h3-colab-workflow/workflows/h3_t2v_pdd8_production.json) — 2.0 标准 768p 8步无 CFG MP4 工作流。
- 🧩 [workflows/h3_t2v_pdd8_mp4.json](file:///Users/kuhung/roy/3_Active_Labs/minimax-h3-colab-workflow/workflows/h3_t2v_pdd8_mp4.json) — 2.0 标准 MP4 音画同出工作流。
- 🧩 [workflows/h3_t2v_pdd8_webp.json](file:///Users/kuhung/roy/3_Active_Labs/minimax-h3-colab-workflow/workflows/h3_t2v_pdd8_webp.json) — 2.0 WebP 快速工作流。
- 🧩 [workflows/h3_t2v_l4_baseline.json](file:///Users/kuhung/roy/3_Active_Labs/minimax-h3-colab-workflow/workflows/h3_t2v_l4_baseline.json) — 2.0 544p 冒烟基线工作流。
- 📋 [configs/prompts.example.json](file:///Users/kuhung/roy/3_Active_Labs/minimax-h3-colab-workflow/configs/prompts.example.json) — 任务排期模板（已对齐 124 帧）。
- 📋 [configs/story_cyber_dawn.json](file:///Users/kuhung/roy/3_Active_Labs/minimax-h3-colab-workflow/configs/story_cyber_dawn.json) — 短视频分镜剧本案例（《极客破晓：硅基脉动》，8 镜头成片流水线）。
- 📋 [configs/shanghai_street.json](file:///Users/kuhung/roy/3_Active_Labs/minimax-h3-colab-workflow/configs/shanghai_street.json) — 上海街景 3 镜案例。
- 📖 [docs/COLAB_MCP_INTEGRATION.md](file:///Users/kuhung/roy/3_Active_Labs/minimax-h3-colab-workflow/docs/COLAB_MCP_INTEGRATION.md) — Colab MCP 深度集成与 Agent 全自动化托管操作手册。
- 📖 [docs/PROMPT_RECIPES.md](file:///Users/kuhung/roy/3_Active_Labs/minimax-h3-colab-workflow/docs/PROMPT_RECIPES.md) — 官方原生结构化多镜头语法与 10 大风格 Embedding 指南。
- 📚 [docs/RESEARCH_SUMMARY.md](file:///Users/kuhung/roy/3_Active_Labs/minimax-h3-colab-workflow/docs/RESEARCH_SUMMARY.md) — 全网五大方案横评与 2.0 核心性能避坑指南。

---

## ⚡ 核心避坑法则与 2.0 实测基准

| 规则 | 2.0 实测数据与规范说明 |
|---|---|
| **1. 算力费率与单镜成本减半** | **L4 实测费率 1.90 CU/h。2.0 架构消除 CFG 双倍计算后，单镜 124 帧稳态耗时降低至 ~150 秒，单镜算力成本低至 0.0818 CU（1 CU 可产出 12 镜）！Pro 方案月产 1,200 镜，Ultra 月产 6,000 镜！** |
| **2. 严禁在 T4 上跑** | T4 算力 7.5，不支持量化算子加速，强行跑会立即 OOM 崩溃。门禁脚本自动阻断并调用 `runtime.unassign()` 关机。 |
| **3. CUDA 12.x 自动用 fp8_scaled** | 在 Colab 默认 CUDA 12.x 环境下，`download_models.py` 与 Notebooks 自动选用 `fp8_scaled` + `video_vae_fp16`，消除 `int8_convrot` 缺少 cu130 算子导致的慢路径回退。 |
| **4. BasicGuider 单路无 CFG** | 严禁使用 `KSampler (cfg=3.0)`，必须用 `BasicGuider`（CFG=1.0）。防止蒸馏模型伪影与双倍时间浪费。 |
| **5. 采样器首选 res_multistep** | `KSamplerSelect (res_multistep)` + `BasicScheduler (simple)` 在极少步数下提供高阶微分修正，微观动态平滑连贯。 |
| **6. 缓存完整性与 0 秒软链接** | 模型首次拉取自动原子备份到 Drive 缓存，后续启动 0 秒软链接秒开，双重校验 safetensors 数据边界。 |

---

## 🤖 方案 A：Colab MCP + Agent 自动化托管模式（强烈推荐）

详见 [docs/COLAB_MCP_INTEGRATION.md](file:///Users/kuhung/roy/3_Active_Labs/minimax-h3-colab-workflow/docs/COLAB_MCP_INTEGRATION.md)。

```bash
python scripts/agent_colab_runner.py \
    --workflow workflows/h3_t2v_pdd8_production.json \
    --prompts configs/story_cyber_dawn.json \
    --output /content/drive/MyDrive/MiniMax_Outputs \
    --auto-shutdown
```

---

## 🛠️ 方案 B：传统交互 Notebook 运行 SOP

1. 上传或在 Google Colab 中打开 [minimax_h3_l4_production.ipynb](file:///Users/kuhung/roy/3_Active_Labs/minimax-h3-colab-workflow/notebooks/minimax_h3_l4_production.ipynb)。
2. 更改运行时类型：**GPU -> L4 (高内存)**。
3. 点击「代码执行程序」->「全部运行」。
4. 步骤 2 授权 Google Drive 挂载即可，运行完毕自动导出 MP4 并安全关机。
