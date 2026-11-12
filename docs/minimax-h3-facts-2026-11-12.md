# MiniMax H3 & H3-Image Verified Facts Ledger (2026-11-12)

> **Purpose**: Single dated source of truth for external benchmarks, ecosystem status, pricing, and repository execution baselines. All figures below reflect sources verified on **2026-11-12** or direct Google Colab execution logs in this repository. External leaderboards and cloud GPU rates change over time; refer to official documentation and live leaderboards for the latest figures.

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
  - **Licensing**: Open weights permit non-commercial / local research workflows; a **commercial license** is required for commercial use of local open weights (refer to official MiniMax license terms).
- **MiniMax-H3-Image**:
  - Single unified model for native high-resolution Text-to-Image (up to ~`16 MP`, e.g., `4096×4096`, `5024×3360`, `5472×3072`) and multi-reference instruction editing (`In-Context Edit`).

## 4. Google Colab Compute Units (CU) & Pricing (Verified 2026-11-12)

- **Pay-As-You-Go**:
  - `$9.99` for `100 CU` (~`$0.10 / CU`)
  - `$49.99` for `500 CU` (~`$0.10 / CU`)
- **Subscriptions**:
  - **Colab Pro**: `$9.99 / mo` (`100 CU / mo`)
  - **Colab Pro+**: `$49.99 / mo` (`600 CU` = `100 + 500 CU`, includes background execution)
  - **CU Expiration**: Compute Units expire after **90 days**.
  - **Colab Free**: Does not include guaranteed free L4 GPU allocation.
  - **Google AI Pro / Ultra**: Plans bundle Colab Compute Units (exact CU allotment and priority/background perks vary by plan tier and Google account terms; subject to official Google AI / Colab pricing pages).
- **L4 GPU Burn Rate Reference**:
  - External community / Colab documentation reports typical L4 burn rates around `3.0–4.82 CU/h` depending on region, runtime class, and dynamic allocation.
  - This repository's actual Colab L4 runtime panel logs recorded **`1.90 CU/h`** during `v2_0` Cell 14 execution and a calibrated session baseline of **`1.94 CU/h`**. Dynamic Colab CU metering means future sessions are governed by live Colab runtime billing.

## 5. Repository Verified Execution Baselines (Direct Colab Notebook Logs)

Only the 3 executed notebooks below have recorded Colab output logs in this repository (`v1.0`, `v1.5`, `v2.0`, `v2.0_test`, `v3.1`, `v3.2`, `v3.3`, `v4.0` template files remain unexecuted blueprints with 0 cell outputs):

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
