# MiniMax H3 & H3-Image Verified Facts Ledger (2026-11-12)

> **Purpose**: Single dated source of truth for external benchmarks, ecosystem status, pricing, and repository execution baselines. Core model/L4 execution records reflect sources verified on **2026-11-12**; Section 4 (Google AI bundled Colab CU allotments) and Section 6 (quota output capacity & Kling/Veo API cost equivalence) include updates verified on **2026-11-28**. External leaderboards and cloud GPU rates change over time; refer to official documentation and live leaderboards for the latest figures.

---

## 1. Artificial Analysis Video Arena (Verified 2026-11-12)

- **Track**: `AA-Video-T2V v2.1` (Text-to-Video; real-time Elo leaderboard & published API price per minute)
  - **Wan 3.0**: `1175 (±13)` Elo, `$12.00 / min`
  - **MiniMax H3 (768p)**: `1161 (±13)` Elo, `$4.80 / min`
  - **Dreamina Seedance 2.5**: `1155 (±13)` Elo, `$34.12 / min`
  - **MiniMax H3 Max**: `1149 (±15)` Elo, `$9.60 / min`
- **Other / Earlier Tracks (Reference Ranges)**:
  - **AA-Video-I2V**: Dreamina Seedance 2.5 ~`1161` Elo; MiniMax H3 ~`1115–1133` Elo; Hyperflow H3 ~`1120–1149` Elo (exact rank depends on track filter and snapshot time; always cite snapshot date and live leaderboard).

## 2. OpenAI Sora Availability (Verified 2026-11-12)

- **Consumer Web & Mobile App**: Shut down on **April 26, 2026**.
- **Sora API (`sora-2`, `sora-2-pro`)**: Shut down on **September 24, 2026**.

## 3. Official MiniMax H3 & ComfyUI Documentation (`docs.comfy.org`, Verified 2026-11-12)

- **MiniMax H3 (Video)**:
  - Supported workflows in ComfyUI: Text-to-Video (T2V), Image-to-Video (I2V), Reference-to-Video (R2V) with native stereo audio generation (`AudioVAE`).
  - Open weights target ~`768px` short-edge resolution at `24 fps`, up to ~`15s` duration locally.
  - **Licensing** (`MiniMax H3 Community License Agreement`, Aug 2, 2026, <https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE>): Worldwide excluding Excluded Territories (EU, UK, South Korea, USA). Royalty-free commercial use is permitted within the Applicable Territory if yearly revenue from commercial products/services does not exceed **$20M USD** and `MiniMax H3` is prominently displayed on the commercial UI; separate prior written authorization (`api@minimax.io`) is required above **$20M USD** or for Excluded Territories. Prohibits using weights/outputs to improve other AI models (except MiniMax H3 derivatives) and forbids resale/unrestricted redistribution.
- **MiniMax-H3-Image**:
  - Single unified model for native high-resolution Text-to-Image (up to ~`16 MP`, e.g., `4096×4096`, `5024×3360`, `5472×3072`) and multi-reference instruction editing (`In-Context Edit`).

## 4. Google Colab Compute Units (CU) & Pricing (Verified 2026-11-12; Google AI tiers updated 2026-11-28)

- **Pay-As-You-Go**:
  - `$9.99` for `100 CU` (~`$0.10 / CU`)
  - `$49.99` for `500 CU` (~`$0.10 / CU`)
- **Subscriptions**:
  - **Colab Pro**: `$9.99 / mo` (`100 CU / mo`)
  - **Colab Pro+**: `$49.99 / mo` (`600 CU` = `100 + 500 CU`, includes background execution)
  - **CU Expiration**: Compute Units expire after **90 days** (unused monthly units roll over within that 90-day window).
  - **Colab Free**: Does not include guaranteed free L4 GPU allocation.
  - **Google AI Plans (bundled Colab CU, verified 2026-11-28)**:
    - **Google AI Plus** (`$4.99 / mo`): `50 CU / mo`
    - **Google AI Pro** (`$19.99 / mo`): `200 CU / mo`
    - **Google AI Ultra 5x** (`$99.99 / mo`): `1,000 CU / mo` (priority GPUs + 24h background execution)
    - **Google AI Ultra 20x** (`$199.99 / mo`): `2,000 CU / mo` (priority GPUs + 24h background execution)
    - Benefits stack with standalone Colab subscriptions and are subject to official Google AI / Colab pricing pages and regional availability.
- **L4 GPU Burn Rate Reference**:
  - External community / Colab documentation reports typical L4 burn rates around `3.0–4.82 CU/h` depending on region, runtime class, and dynamic allocation.
  - This repository's actual Colab L4 runtime panel logs recorded **`1.90 CU/h`** during `v2_0` Cell 14 execution and a calibrated session baseline of **`1.94 CU/h`**. Dynamic Colab CU metering means future sessions are governed by live Colab runtime billing.

## 5. Repository Verified Execution Baselines (Direct Colab Notebook Logs)

The 3 original executed notebooks below have recorded Colab output logs; V3 execution records are listed in the added V3 section (`v1.0`, `v1.5`, `v2.0`, `v2.0_test`, `v4.0` template files remain unexecuted blueprints with 0 cell outputs):

- **`notebooks/v1_0_minimax_h3_l4_benchmark_webp.ipynb`** (NVIDIA L4 `22.03 GB` VRAM):
  - **Cell 12 Smoke Test** (`960×544`, `CFG=3.0`): `393.78s`; denoising steady VRAM `21.77 GB`, peak VRAM `21.91 GB` (`0.12 GB` headroom).
  - **Cell 14 Benchmark** (`CFG=3.0`, VRAM not logged in Cell 14): Shot 1 (`960×544`) `315.83s`; Shot 2 (`1344×768`) `875.35s`; Shot 3 (`1344×768`) `874.08s`.
- **`notebooks/v2_0_minimax_h3_l4_production_mp4.ipynb`** (NVIDIA L4 `22.49 GB` VRAM, PyTorch CUDA `13.0` / `cu130`):
  - **HF Weight Sync**: `136.32s` total.
  - **Cell 14 Benchmark** (`960×544`, `124` frames, `8-step Turbo`, `BasicGuider CFG=1.0`, explicit `gc`/VRAM cleanup):
    - Shot 1 (cold start incl. compile/warmup): `243.5s` (`2.42 MB` MP4)
    - Shot 2 (steady): `165.3s` (`0.63 MB` MP4)
    - Shot 3 (steady): `165.5s` (`1.23 MB` MP4)
    - Steady average: **`165.4s / shot`**
    - Total 3-shot elapsed: `574.6s` (`9.58 min`)
    - Logged raw CU estimate: `~0.303 CU` (`1.90 CU/h`); calibrated at `1.94 CU/h`: **`~0.0891 CU / steady shot`**, 3-shot historical record `~0.309 CU` (`0.310 CU` by exact rate, `~0.103 CU / shot` avg).
- **`notebooks/v4_0_minimax_h3_image_colab_demo.ipynb`** (NVIDIA L4 `22.5 GB` `sm_89`, `stable-diffusion.cpp` `sd-cli` restored from Drive in `<1s`):
  - **Weights**: `minimax_h3_image_pruned_int8_convrot.safetensors` (`19.54 GB`), `vae_fp16` (`1.09 GB`), `qwen3vl_32b_minimax_h3-Q4_K_M.gguf` (`16.97 GB`).
  - **Demo 1 (Text-to-Image, `2048×2048`, `6 steps`, `CFG=1.0`)**: **`157.8s`** total (`generate_image 149.09s`, `vae decode 8.27s`, `~0.085 CU` @ `1.94 CU/h`).
  - **Demo 2 (Instruction Image Edit, `2048×2048`, `<Picture 1>`)**: **`226.1s`** total (`generate_image 217.53s`, `vae decode 9.93s`, `~0.122 CU` @ `1.94 CU/h`).


## V3 多模态 L4 实测记录（2026-11-12）

三种模式均已保留执行输出与内嵌 MP4。配置为 NVIDIA L4（设备总显存 22.49 GB）、INT8 ConvRot、960×544、124 帧、24fps、BasicGuider。每种模式仅记录一次首镜运行；以下耗时从提交任务后开始计时，包含首镜加载及推理，不代表稳态性能。

| 模式 | Turbo 步数 | 首镜任务耗时 | 推算 CU | 五个权重加载阶段耗时 |
|---|---:|---:|---:|---:|
| I2V | 8 | 258.1 秒 | 0.1391 | 122 秒 |
| FLF2V | 8 | 273.2 秒 | 0.1472 | 122 秒 |
| Ref2VA | 4 | 369.2 秒 | 0.1990 | 118 秒 |

CU 按历史费率 1.94 CU/h × 任务秒数 / 3600 推算，并非本次面板实扣；不包含安装、下载、启动服务和空闲。权重阶段耗时来自日志时间戳，包含校验，为单次网络环境观察。22.49 GB 是设备容量，未采集本次峰值显存。Ref2VA 使用橙色小车横向行驶素材，生成结果有可见位移；尚未通过对照实验隔离提示词和参考视频各自作用。I2V / FLF2V 的完成日志与 MP4 存证证明链路执行完成，效果质量另行验收。

执行存证：

- [I2V](../notebooks/v3_1_minimax_h3_i2v.ipynb)
- [FLF2V](../notebooks/v3_2_minimax_h3_flf2v.ipynb)
- [Ref2VA](../notebooks/v3_3_minimax_h3_ref2va.ipynb)


## V5 三镜头接续实测（2026-11-12）

执行存证：[v5_0 腕表分镜](../notebooks/v5_0_minimax_h3_watch_storyboard.ipynb)。本次使用上一镜实际尾帧接续下一镜，第三镜约束全貌终点；原生音轨做切点交叉淡化，未提供统一 BGM。

- 三镜耗时：265.4 / 183.5 / 195.2 秒，合计约 644.1 秒；包含各镜输出检查与 Drive 保存。按历史 1.94 CU/h 推算约 0.3471 CU，不包含环境安装、下载和空闲，也非本次面板实扣。
- 从服务就绪 14:17:22 到成片保存 14:28:10，约 648 秒。日志确认成片保存、运行时释放调用及 Drive 刷盘完成。
- 提取 Notebook 内嵌成片检查：H.264、960×544、373 帧、15.541667 秒；AAC、32kHz 双声道。实际成片比三镜理论合计 372 帧多 1 帧，尚未归因。
- 已检查每秒抽帧，构图随推进与拉远变化；该检查不等于逐帧连续性验收。音轨淡化分支执行成功，尚不能据此宣称音乐旋律连续；统一 BGM 分支未在本次运行中验证。

---

## 6. Bundled Quota Video Output & Commercial API Cost Equivalence (Verified 2026-11-28)

All video volume calculations below use the verified `v2_0` Colab L4 output specification: **`960×544`, `124` frames @ `24 fps` (`5.1667s` / `5.17s` = `0.086111 min` per clip)**, 8-step Turbo LoRA, `BasicGuider (CFG=1.0)`, 32kHz stereo MP4.

### 6.1 Per-Minute & Per-Clip Cost vs. Commercial Video APIs

*(L4 effective USD cost is derived at `$9.99 / 100 CU` = `$0.0999 / CU`.)*

| Route / Model | Resolution & Audio | Cost per Minute | Cost per `5.17s` Clip | Multiple vs. L4 Steady |
|---|---|---:|---:|---:|
| **Colab L4 Steady (`1.94 CU/h`, `165.4s/clip`)** | `960×544`, 32kHz stereo | **`~$0.103 / min`** (`1.035 CU/min`) | **`~$0.0089`** (`0.0891 CU`) | `1.0×` (baseline) |
| **Colab L4 3-Shot Batch Avg (`1.94 CU/h`, `191.5s/clip`)** | `960×544`, 32kHz stereo (incl. cold start) | **`~$0.120 / min`** (`1.199 CU/min`) | **`~$0.0103`** (`0.1032 CU`) | `~1.2×` |
| **Colab L4 High-Burn Sensitivity (`3.5 CU/h`, `165.4s/clip`)** | `960×544`, 32kHz stereo | **`~$0.187 / min`** (`1.867 CU/min`) | **`~$0.0161`** (`0.1608 CU`) | `~1.8×` |
| **MiniMax H3 768p API** | `768p` cloud w/ stereo | **`$4.80 / min`** (`$0.080 / s`) | **`$0.413`** | **`~46×`** |
| **MiniMax H3 Max API** | Cloud high-tier w/ stereo | **`$9.60 / min`** (`$0.160 / s`) | **`$0.827`** | **`~93×`** |
| **Wan 3.0 API (Alibaba)** | Cloud `720p/1080p` w/ audio | **`$12.00 / min`** (`$0.200 / s`) | **`$1.033`** | **`~116×`** |
| **Kling 4.0 API (Kuaishou, AA benchmark)** | Cloud `720p` w/ audio | **`$12.60 / min`** (`$0.210 / s`) | **`$1.085`** | **`~122×`** |
| **Google Veo 3.1 Standard API** | `720p/1080p` w/ audio | **`$24.00 / min`** (`$0.400 / s`) | **`$2.067`** | **`~232×`** |
| **Dreamina Seedance 2.5 API (ByteDance)** | Cloud flagship w/ audio | **`$34.12 / min`** (`$0.569 / s`) | **`$2.938`** | **`~330×`** |

*(Additional commercial tier references verified `2026-11-28`: Kling 3.0 API `720p/1080p` w/ native audio is `$0.1135–$0.168 / s` = **`$6.81–$10.08 / min`**; Google Veo 3.1 Fast `720p` w/ audio is `$0.10 / s` = **`$6.00 / min`**.)*

### 6.2 Bundled Quota Capacity & Equivalent Commercial API Spend (`1.94 CU/h` L4 Baseline)

Ranges show **`3-shot batch avg incl. cold start (0.1032 CU/clip)`** to **`steady-state inference (0.0891 CU/clip)`**:

| Plan & Monthly Price | Included CU | Output Clips (`5.17s`) | Total Video Duration | MiniMax H3 API (`$4.80/m`) | Wan 3.0 API (`$12.00/m`) | Kling 4.0 API (`$12.60/m`) | Veo 3.1 Std (`$24.00/m`) | Seedance 2.5 (`$34.12/m`) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Colab Pro / PAYG (`$9.99`)** | `100 CU` | `~969 – 1,122` | `~83 – 97 min` | `$400 – $464` | `$1,001 – $1,159` | `$1,051 – $1,217` | `$2,002 – $2,319` | `$2,847 – $3,296` |
| **Google AI Pro (`$19.99/mo`)** | `200 CU` | `~1,938 – 2,244` | `~167 – 193 min` | `$801 – $927` | `$2,002 – $2,319` | `$2,102 – $2,435` | `$4,005 – $4,637` | `$5,693 – $6,593` |
| **Colab Pro+ (`$49.99/mo`)** | `600 CU` | `~5,813 – 6,732` | `~501 – 580 min` | `$2,403 – $2,782` | `$6,007 – $6,956` | `$6,307 – $7,304` | `$12,014 – $13,912` | `$17,080 – $19,778` |
| **Google AI Ultra 5x (`$99.99/mo`)** | `1,000 CU` | `~9,688 – 11,219` | `~834 – 966 min` | `$4,005 – $4,637` | `$10,011 – $11,593` | `$10,512 – $12,173` | `$20,023 – $23,187` | `$28,466 – $32,964` |
| **Google AI Ultra 20x (`$199.99/mo`)** | `2,000 CU` | `~19,377 – 22,439` | `~1,669 – 1,932 min` | `$8,009 – $9,275` | `$20,023 – $23,187` | `$21,024 – $24,346` | `$40,046 – $46,373` | `$56,932 – $65,927` |

### 6.3 Sensitivity Reference (`3.5 CU/h` Dynamic L4 Burn Rate) & Image Quota Note

- **Higher Colab Burn Rate (`3.5 CU/h` mid-range of community `3.0–4.82 CU/h`, steady `165.4s/clip` = `0.1608 CU/clip`)**:
  - `100 CU` (Colab Pro / PAYG): `~622` clips (`~53.5 min`) → `$257` (H3 API) / `$643` (Wan 3.0) / `$675` (Kling 4.0) / `$1,285` (Veo 3.1 Std) / `$1,827` (Seedance 2.5).
  - `200 CU` (Google AI Pro): `~1,244` clips (`~107.1 min`) → `$514` (H3 API) / `$1,285` (Wan 3.0) / `$1,350` (Kling 4.0) / `$2,571` (Veo 3.1 Std) / `$3,654` (Seedance 2.5).
  - `600 CU` (Colab Pro+): `~3,731` clips (`~321.3 min`) → `$1,542` (H3 API) / `$3,856` (Wan 3.0) / `$4,049` (Kling 4.0) / `$7,712` (Veo 3.1 Std) / `$10,963` (Seedance 2.5).
- **`MiniMax-H3-Image` (`v4_0` `2048×2048` @ `1.94 CU/h`)**: T2I (`157.8s`, `0.0850 CU`) yields `~1,176` images per `100 CU` (`~2,352` per `200 CU`); `<Picture 1>` instruction edit (`226.1s`, `0.1218 CU`) yields `~821` edits per `100 CU` (`~1,641` per `200 CU`).
- **Comparison Boundaries**: Local Colab runs output quantized `960×544` open weights (`H3-Base`), whereas commercial APIs benchmark `720p/1080p` cloud inference with `H3-Context-IR` / `H3-Regenerate-2K` or flagship closed architectures. Local CU figures exclude model sync (`~136.3s` = `~0.073 CU` once per fresh VM), pip/Colab setup, idle time, and failed retakes; commercial APIs likewise bill per generated second including discarded drafts.
