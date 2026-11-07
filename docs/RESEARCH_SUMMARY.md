# 用 Google AI Pro/Ultra 额度跑 MiniMax 视频模型：实测基准与架构归档

首轮 Benchmark 核准日期：2026-11-02。
架构 2.0 升级核准日期：2026-11-03。

> **核心硬约束：全程本地 Colab 推理，不引入 MiniMax API 请求；Colab 额度仍会扣 CU。** 
> 任何“改走 MiniMax API / API 对照 / API 出 2K / API fallback”均作废，不执行；2K、H3-Context-IR、H3-Max 等托管能力不在本仓库路线内。

---

## 一、 首轮实测（Benchmark）核准结论

1. **模型指认**：MiniMax H3（海螺 AI 3.0 开源权重），采用 Comfy-Org 官方免鉴权公开源 `Comfy-Org/MiniMax-H3`。
2. **算力额度与实测费率（v2.0 实测物理数据已校准）**：
   - **Google AI Pro 方案**：每月赠送 **100 计算单元（CU）**，有效期 **90 天**（支持滚动囤积至最高 **~300 CU**）。
   - **Google AI Ultra 方案**：每月赠送 **500 计算单元（CU）**，有效期 **90 天**（支持滚动囤积至最高 **~1,500 CU**）。
   - **L4 官方实测扣费费率**：**1.94 CU/小时**（实测自 Google Colab 资源监控面板）。
   - **单镜头实测端到端耗时（960x544, 124 帧 @ 24fps 原生 32kHz 音画同出）**：
     - 首镜冷启（含模型显存分步加载开销）：**243.5 秒**；
     - 稳态热跑（第 2、3 镜稳态渲染）：**165.3 ~ 165.5 秒（约 2 分 45 秒）**；
     - 3 镜头连续批处理总耗时：**574.6 秒（9.58 分钟）**。
     - 对比 1.0 架构（315.8 秒无声 WebP 动画），稳态渲染速度提速 **47%**，且音画双轨成片；对比 1.5 架构（双路 CFG 631s+ 超时），算力消耗直接降低 **74%**！
   - **实测算力消耗（现行口径）**：$$1.94 \times \frac{165.4}{3600} \approx \mathbf{0.089\ \text{CU / 稳态镜头}}$$（适用条件：960×544、124 帧、24fps、Turbo 8 步、BasicGuider）。三镜历史记录 **0.309 CU** / 574.6s；按费率折算约 0.310 CU，显示精度与计时边界待复核，不能描述为精确相等。三镜批次均摊约 **0.103 CU/镜**（含首镜开销），不能替代稳态口径。
   - **产能说明**：按稳态 0.089 CU/镜做理论折算时，100 CU 约对应 ~1,120 镜、500 CU 约对应 ~5,600 镜；**未计入**冷启、下载、空闲、失败与返工，**不构成月产量保证**。费率以当时 Colab 面板为准；旧文 0.087 CU 为历史口径，已停用。
   - **关于耗时差异的深度探究（分辨率与冷热机效应）**：
     - 在 v1.0 实测中，同一批次分镜 1 耗时 315.8 秒，而分镜 2 与 3 耗时 875 秒与 874 秒。这是因为分镜 1 规格为 960x544（52.2 万像素），而分镜 2、3 为 1344x768（103.2 万像素，**像素暴增 97.6%**）。DiT 模型的自注意力计算呈非线性激增，加上 v1.0 的双路 CFG(3.0) 强制每步计算 2 次前向传播（共 16 次），导致耗时逼近 15 分钟！
     - 相同分辨率下性能极度稳定（分镜 2 的 875.35s vs 分镜 3 的 874.08s，波动 <0.15%）。这一实测事实坚定了本仓库 2.0 架构**将 960x544 锁定为 L4 黄金规格**的工程决策。
3. **硬件可行性关卡（实测核准）**：
   - **T4（16GB，算力 7.5）**：**不可用**。量化算子要求 CUDA Compute Capability ≥ 8.0。硬跑必崩溃，门禁脚本自动熔断销毁。
   - **L4（24GB，算力 8.9 Ada）**：**主力终极推荐**。实测整卡峰值显存精确停在 **21.77 GB / 22.03 GB**（留有约 260MB 安全余量），是物理与性价比的最佳契合点。
   - **A100（40GB/80GB）**：速度略快，但费率高达 7.5~11.8 CU/h，单镜成本反升至 0.38~0.45 CU，性价比较低。
4. **模型体量瘦身与缓存**：
   - 原始全精度约 131GB+，单卡绝对无法装下。
   - 方案采用 Comfy-Org 剪枝量化方案：Diffusion (~19.5GB) + Qwen3-VL 32B NVFP4 (~14.61GB) + VAEs + Turbo LoRA，总计 ~42.5GB。
   - 首次下载自动原子备份至 Google Drive，后续启动 0 秒软链接秒开。
5. **成片封装演进**：
   - 冒烟阶段：`SaveAnimatedWEBP`（零依赖，快速跑通闭环）。
   - 生产阶段：`VAEDecodeAudio` + `CreateVideo` + `SaveVideo`（封装 32kHz 原生立体声环境音的标准 `.mp4` 容器）。

---

## 二、 全网开源方案横评与 2.0 架构最佳实践升级（2026-11-03）

### 1. 全网五大开源方案深度横评对比

| 对比维度 | 本仓库 2.0 (`minimax-h3-colab-workflow`) | `killkli/minimax-h3-colab-skill` | 官方模板 (`Comfy-Org/workflow_templates`) | `modelscope/DiffSynth-Studio` | `MiniMax_H3_on_Colab_T4` (Gist) |
|---|---|---|---|---|---|
| **输入模式** | **纯文本 T2VA 原生批量出片（无脸视频工业流）** | ⚠️ **强制要求上传 1~9 张参考图/首帧图** (无法纯文本批量出片) | 基础图生视频 / 文生视频演示 | 纯 Python 代码脚本调用 | 单图/单提示词交互式运行 |
| **采样拓扑** | **`BasicGuider` (单路无 CFG, CFG=1.0)** | `BasicGuider` (单路无 CFG) | `BasicGuider` (单路无 CFG) | 框架内置 Pipeline | `KSampler` (双路 CFG=3.0) |
| **DiT 前向次数 (8步)** | **精确 8 次（零多余算力浪费）** | 精确 8 次 | 精确 8 次 | 8 次 | 16 次（正负双路，耗时翻倍） |
| **CUDA 与量化自适应** | **自动检测：cu130 走 `int8_convrot`；cu12x + L4 走 `fp8_scaled`** | 支持环境变量切换 `int8`/`fp8`/`bf16` | 仅固定模板权重 | 默认拉取全精度 Text Encoder (60GB+)，依赖磁盘卸载 | 依赖 3 阶段内存拆分强跑 T4 |
| **音画同出 (MP4)** | **原生 32kHz 双声道立体声 MP4 (`CreateVideo`+`SaveVideo`)** | 支持 MP4 (依赖第三方 `ComfyUI-VideoHelperSuite` 插件) | 基础视频保存 | 支持音视频分离输出 | 仅 WebP/GIF (无原生音频合流) |
| **插件依赖性** | **100% ComfyUI 官方原生内置节点 (零第三方插件)** | 依赖第三方扩展 `ComfyUI-VideoHelperSuite` | 官方原生内置 | 需安装独立 DiffSynth 库 | 需安装自定义插件 |
| **Google Drive 缓存** | **官方原生挂载，原子缓存与 0 秒软链接秒开** | ❌ **无 Drive 缓存机制**（每次新建会话需重下 40GB 权重） | 无云端持久化设计 | 无云盘缓存集成 | 无云端持久化设计 |
| **节点契约校验** | **动态 `/object_info` 预检与参数自适应裁剪** | 支持 `/object_info` 契约校验 | 无校验 | 框架内置 | 无校验 |
| **帧数网格对齐** | **强制向上对齐 `17k + 5` (精确公式)** | 依赖用户自行输入 | 手动指定 | 手动指定 | 随意填写（引发时域断裂伪影） |
| **硬件与防扣费守门** | **锁定 L4 (1.94 CU/h) + 三重守门 (finally+Watchdog+Agent)** | 默认绑死高价 A100，支持 `colab stop` 但无线程级看门狗 | 无关机与防扣费机制 | 无关机机制 | 跑在 T4 (单片耗时 35 分钟+，无生产力) |

---

### 2. 开源竞品方案深度源码级剖析

1. **`killkli/minimax-h3-colab-skill` 源码级事实分析**：
   - **优势**：同样采用了单路无 CFG 的 `BasicGuider` 拓扑，具备 CUDA 量化感知与 `/object_info` 契约校验能力，并实现了基于 `VHS_VideoCombine` 的 MP4 导出。
   - **核心缺陷**：
     - ① **强制要求上传参考图**：工作流强依赖外部图片输入（1~9 张首尾帧或风格图），无法直接基于分镜文本批量生成纯原创无脸视频；
     - ② **缺乏持久化缓存**：每次启动必须耗时数十分钟从 HuggingFace 重新拉取 40GB+ 权重，无法在 Colab 频繁断连下复用 Drive 缓存；
     - ③ **依赖第三方自定义节点**：依赖 `ComfyUI-VideoHelperSuite`，增加了扩展编译与安装失败风险（本仓库 2.0 采用 ComfyUI 原生 `CreateVideo` + `SaveVideo` 实现零第三方插件依赖）；
     - ④ **算力浪费严重**：默认绑死单价昂贵的 A100 运行时，且无后台线程级算力看门狗熔断保护。

2. **`modelscope/DiffSynth-Studio` 源码级事实分析**：
   - 其 `model_inference_low_vram` 默认拉取 `MiniMax/MiniMax-H3` 官方全精度 Qwen3-VL 文本编码器（体量高达 60GB+），并依赖 `"offload_device": "disk"` 进行磁盘分页卸载。在 Google Colab 的虚拟磁盘 IO 瓶颈下，每次前向传播频繁触发磁盘吞吐颠簸，整体出片耗时远慢于 ComfyUI 结合 AWQ `NVFP4` (14.6GB) 的常驻显存流水线。

3. **`MiniMax_H3_on_Colab_T4.ipynb` (Gist) 源码级事实分析**：
   - 针对免费版 Colab T4（16GB VRAM, 12.7GB Host RAM）设计，采用极端的**三阶段独立子进程架构**（阶段 1：子进程执行文本编码并落盘退出以清空内存；阶段 2：子进程加载 DiT 执行去噪并落盘退出；阶段 3：子进程加载 VAE 执行解码）。
   - 由于 T4 算力仅 7.5（不支持硬件量化指令加速），且进程反复初始化与全量张量落盘带来了巨大的系统开销，**单片生成耗时超过 35 分钟**，完全不具备工业级批量生产可行性。

---

### 3. 三大核心性能与版本避坑指南

#### 💡 避坑 1：为什么 Pruned + Turbo 严禁使用 `KSampler (cfg=3.0)`，必须用 `BasicGuider (cfg=1.0)`？
- **数学与架构机理**：MiniMax 官方开源的 `minimax_h3_fl2va_pruned_*` 模型与配套的 Turbo LoRA 属于 **CFG-Distilled（无分类器引导蒸馏）模型**。模型在蒸馏训练阶段已经将 Prompt 文本向量强条件内化到单次扩散漂移场中。
- **双倍算力浪费的陷阱**：旧版工作流使用 `CLIPTextEncode (Positive)` + `CLIPTextEncode (Negative)` + `KSampler (cfg=3.0)`，ComfyUI 内部会执行：
  $$\epsilon_{\text{pred}} = \epsilon_{\text{uncond}} + 3.0 \times (\epsilon_{\text{cond}} - \epsilon_{\text{uncond}})$$
  这导致 32B 的 Qwen3-VL 文本编码器必须前向计算两次，且 32B 的 DiT 在去噪的每一步都必须做两次前向评估（8 步实际跑了 16 次 DiT 前向传播）！生成耗时从 ~150 秒直接飙升到 306 秒以上，白白烧掉双倍 Colab 算力。
- **画质失真与过曝伪影**：由于模型本身已完成 CFG 蒸馏，强行在外部施加 3.0 的外推引导差值，会导致潜空间数值剧烈偏移动态范围，引发画面颜色严重过饱和、边缘发焦、运动抽搐以及 32kHz 音频相位破音。
- **2.0 升级解法**：采用 `MiniMaxH3ImageToVideo` -> `BasicGuider (model, conditioning)` -> `SamplerCustomAdvanced` 标准拓扑，强制等效 CFG=1.0 单路推理，**推理算力直接降低 50%，画面呈现纯正电影级柔和光影，音频纯净饱满**。

---

#### 💡 避坑 2：CUDA 12.x vs CUDA 13.0 的量化算子选择（`fp8_scaled` vs `int8_convrot`）
- **底层依赖差异**：
  - `int8_convrot` 算子深度依赖 PyTorch `cu130` 运行库与 `comfy-kitchen` 专有编译内核。
  - Google Colab 实测：最新实例已默认搭载 PyTorch CUDA 13.0 (`cu130`) 与 NVIDIA L4 GPU（Ada Lovelace 架构，Compute Capability 8.9），可直接运行原生 `int8_convrot` 算子！
- **非原生回退的性能陷阱**：若运行在缺少 cu130 支持的 CUDA 12.x 旧环境中强行加载 `int8_convrot` 权重，PyTorch 会回退到极其缓慢的通用 CPU/GPU 内存搬运与模拟反量化逻辑，单步耗时增加 30%~50%。
- **2.0 升级解法（自适应感知路由）**：
  - 动态检测 `torch.version.cuda`：
    - 若运行环境为 CUDA >= 13.0（PyTorch `cu130`，Colab 最新实测），自动选用：
      `minimax_h3_fl2va_pruned_int8_convrot.safetensors` (DiT) + `minimax_h3_video_vae_int8_convrot.safetensors` (VAE)；
    - 若运行环境处于 CUDA 12.x 旧环境，**自动无缝切换**为：
      `minimax_h3_fl2va_pruned_fp8_scaled.safetensors` (DiT) + `minimax_h3_video_vae_fp16.safetensors` (VAE)；
  - L4 (Ada CC 8.9) 具备第四代 Tensor Core 原生支持 FP8（E4M3 / E5M2）张量计算的能力，选用 `fp8_scaled` 可以完全释放 Ada 架构的高硬件吞吐与低显存优势。

---

#### 💡 避坑 3：采样器（`res_multistep` + `simple`）与 4-Step / 8-Step LoRA 选型对比
- **采样器对比 (`res_multistep` vs `euler`)**：
  - 在常规 20~30 步推理中，`euler` 表现良好；
  - 但在 MiniMax H3 经 Turbo 蒸馏后的 4 步或 8 步极少步数场景下，`KSamplerSelect (res_multistep)` 结合 `BasicScheduler (simple)` 能利用多步历史梯度做精确的高阶修正，使微观动态（如咖啡液滴滴落、冷雾翻滚、法租界梧桐叶缓缓滑落）的流体运动更加平滑，杜绝步数不足时的画面抖动与噪点闪烁。
- **4-Step vs 8-Step Turbo LoRA 选型指南**：
  1. **8-Step Turbo LoRA** (`minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors`, 1,956,193,000 字节)：
     - **推荐场景**：成片终稿、高精度 32kHz 拟声音画合流、复杂光影长镜头；
     - **本仓库实测口径**：稳态约 165.4 秒 / 约 **0.089 CU/稳态镜**（1.94 CU/h；见上文适用条件）。
  2. **4-Step Turbo LoRA** (`minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors`, 1,956,192,992 字节)：
     - **推荐场景**：前期分镜快筛、提示词概念筛选（候选路径）；
     - **耗时与成本**：未经本仓库同条件实测，**不作承诺**；阶段四再与 8 步成对对照。
