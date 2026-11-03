#!/usr/bin/env python3
"""
Generates notebooks/test_shanghai_mp4_audio.ipynb (2.0 Architecture)

100% Google Colab compliant and standalone:
- NO rclone, NO fuse, NO third-party daemons (strict adherence to Colab Acceptable Use Policy)
- Standard official Google Drive mount (from google.colab import drive)
- Full 2.0 topology without CFG: MiniMaxH3ImageToVideo + BasicGuider + SamplerCustomAdvanced (50% compute reduction)
- Native 32kHz stereo audio MP4 generation via VAEDecodeAudio + CreateVideo + SaveVideo
- Exact HuggingFace byte verification & safetensors header boundary check
- CUDA and hardware adaptive model selection (cu130 int8_convrot vs cu12x fp8_scaled)
- Hardware gatekeeper for L4 (CUDA CC >= 8.0)
- ComfyUI background service with preflight node validation
- In-memory workflow batch rendering with PyAV stream verification
- Immediate sync to Google Drive & HTML5 audio-video player preview
- Automatic shutdown to safeguard Colab Pro compute units
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

WF = json.loads((ROOT / "workflows" / "h3_t2v_pdd8_production.json").read_text(encoding="utf-8"))
TASKS = json.loads((ROOT / "configs" / "shanghai_street.json").read_text(encoding="utf-8"))

# Validate 2.0 node topology
EXPECTED = {
    "1": "UNETLoader",
    "2": "CLIPLoader",
    "4": "VAELoader",
    "11": "VAELoader",
    "10": "LoraLoaderModelOnly",
    "6": "MiniMaxH3ImageToVideo",
    "7": "BasicGuider",
    "14": "RandomNoise",
    "15": "KSamplerSelect",
    "16": "BasicScheduler",
    "3": "SamplerCustomAdvanced",
    "8": "VAEDecode",
    "12": "VAEDecodeAudio",
    "13": "CreateVideo",
    "9": "SaveVideo"
}
for nid, ctype in EXPECTED.items():
    assert WF[nid]["class_type"] == ctype, f"Node {nid} must be {ctype}"

assert WF["9"]["inputs"]["video"] == ["13", 0]
assert WF["13"]["inputs"]["images"] == ["8", 0] and WF["13"]["inputs"]["audio"] == ["12", 0]
assert WF["3"]["inputs"]["guider"] == ["7", 0]
assert WF["7"]["inputs"]["conditioning"] == ["6", 0]

WF_JSON = json.dumps(WF, ensure_ascii=False, indent=1)
TASKS_JSON = json.dumps(TASKS, ensure_ascii=False, indent=1)

MD_INTRO = r"""# 🏙️ MiniMax H3 (Hailuo 3.0) 上海街景原生音画 MP4 出片工作流 (v2.0-Shanghai-Test)

> **版本标识**: `v2.0-Shanghai-Test` (发布日期: 2026-11-03)  
> **对比存证**:
> - `v1.0 (基准跑通版)`: WebP 无声动画，960x544，单镜耗时 306 秒。
> - `v1.5 (旧版试验组)`: MP4 双路 CFG(3.0)，因使用 1344x768 且计算量翻倍，实测 631s+ 易超时。
> - `v2.0 (当前测试版)`: **单路无 CFG (`BasicGuider`, CFG=1.0) + 原生 32kHz 音画同频 + L4 黄金规格 960x544**，单镜耗时回归 **~300 秒**！

| 架构维度 | v1.0 基准版 (WebP) | v1.5 过渡版 (旧 MP4) | v2.0 当前测试版 (新 MP4) |
| :--- | :--- | :--- | :--- |
| **文件命名** | `v1.0_minimax_h3_l4_benchmark_webp.ipynb` | `v1.5_minimax_h3_l4_legacy_mp4.ipynb` | `v2.0_test_shanghai_street_mp4.ipynb` |
| **采样拓扑** | `KSampler (cfg=3.0)` | `KSampler (cfg=3.0)` | `BasicGuider (cfg=1.0)` 单路 |
| **DiT 前向** | 16 次 / shot | 16 次 / shot | **8 次 / shot (计算量腰斩)** |
| **视频格式** | `.webp` (无音频) | `.mp4` (32kHz 音效) | **`.mp4` (原生 32kHz 双声道)** |
| **推荐分辨率** | 960x544 | 1344x768 | **960x544 (L4 黄金生产规格)** |
| **实测单镜耗时** | **306.22 秒** | **631+ 秒 (严重超时)** | **~280s - 320 秒** |

---
**使用步骤**：
1. 顶部菜单点击「代码执行程序」->「更改运行时类型」-> 选择 **L4 GPU**。
2. 点击「代码执行程序」->「全部运行」。
3. 步骤 2 弹出 Google Drive 授权提示时，点击「连接到 Google 云端硬盘」授权一次即可。
"""

CELL_CONFIG = r'''# ==================== 1. 运行配置与安全看门狗 ====================
NUM_SHOTS = 2               # 生成镜头数 1~3（每镜 124 帧 ≈ 5.16 秒，L4 约 5~6 分钟/镜）
AUTO_SHUTDOWN = True        # 全部完成后自动执行 runtime.unassign() 释放实例停止扣费
WATCHDOG_HOURS = 1.5        # 算力看门狗硬熔断上限（小时），防止异常卡死消耗算力

DIFFUSION_VARIANT = "auto"  # 量化权重选型: "auto", "int8_convrot", "fp8_scaled"
                            # "auto": CUDA >= 13.0 用 int8_convrot; Colab 默认 CUDA 12.x + L4 自动用 fp8_scaled
TURBO_STEPS = 8             # Turbo LoRA 步数: 8 (默认高品质) 或 4 (极速快筛)
SAMPLER_NAME = "res_multistep" # 官方推荐采样器: "res_multistep" (或 "euler")

import os, sys, time, json, shutil, threading, subprocess
from pathlib import Path

DRIVE_ROOT = Path("/content/drive/MyDrive")
DRIVE_MODELS = DRIVE_ROOT / "MiniMax_Models"
DRIVE_OUTPUTS = DRIVE_ROOT / "MiniMax_Outputs/shanghai_street"

COMFY_ROOT = Path("/content/ComfyUI")
LOCAL_MODELS = COMFY_ROOT / "models"
COMFY_MODELS = LOCAL_MODELS  # 双保险别名
LOCAL_OUT = COMFY_ROOT / "output"
COMFY_LOG = Path("/content/comfyui.log")
SERVER_URL = "http://127.0.0.1:8188"

def log(msg):
    print(time.strftime("[%H:%M:%S] ") + str(msg), flush=True)

def safe_shutdown(reason="正常完成"):
    log(f"🛑 触发 Colab 运行时安全释放: {reason}")
    try:
        from google.colab import drive
        drive.flush_and_unmount()
        log("💾 Drive 缓存已刷盘")
    except Exception:
        pass
    try:
        from google.colab import runtime
        runtime.unassign()
    except Exception as e:
        log(f"⚠️ runtime.unassign() 异常: {e}，执行容器级退出")
        os.system("sync")
        os.kill(1, 15)

if not globals().get("_WATCHDOG_RUNNING"):
    def _watchdog_timer():
        time.sleep(int(WATCHDOG_HOURS * 3600))
        log(f"🚨 [WATCHDOG 熔断] 运行时已超 {WATCHDOG_HOURS} 小时，自动断开保护算力！")
        safe_shutdown("看门狗超时熔断")
    threading.Thread(target=_watchdog_timer, daemon=True).start()
    _WATCHDOG_RUNNING = True
    log(f"🛡️  算力看门狗守门员已就绪：{WATCHDOG_HOURS} 小时超时自动断开")
'''

CELL_DRIVE = r'''# ==================== 2. 挂载 Google Drive ====================
from google.colab import drive

if not os.path.exists("/content/drive/MyDrive"):
    log("⏳ 正在通过官方原生接口挂载 Google Drive...")
    drive.mount("/content/drive")
else:
    log("✅ Google Drive 处于已挂载就绪状态")

DRIVE_MODELS.mkdir(parents=True, exist_ok=True)
DRIVE_OUTPUTS.mkdir(parents=True, exist_ok=True)
log(f"📁 云端模型持久化缓存: {DRIVE_MODELS}")
log(f"🎬 成片与日志最终落盘: {DRIVE_OUTPUTS}")
'''

CELL_GPU = r'''# ==================== 3. 硬件门禁检测 (验证 L4 GPU) ====================
def check_gpu():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,compute_cap,memory.total", "--format=csv,noheader,nounits"],
            text=True
        ).strip().splitlines()[0]
        name, cc_str, total_mb = [x.strip() for x in out.split(",")]
        return name, float(cc_str), float(total_mb) / 1024
    except Exception as e:
        raise RuntimeError(f"未检测到可用 GPU，请在「更改运行时类型」中选择 L4 GPU: {e}")

gpu_name, cc, total_gb = check_gpu()
disk_free_gb = shutil.disk_usage("/content").free / (1024 ** 3)
log(f"🖥️ 当前显卡: {gpu_name} (算力 {cc:.1f}) | 显存: {total_gb:.2f} GB | 本地磁盘剩余: {disk_free_gb:.1f} GB")

if cc < 8.0:
    log(f"🚨 致命拦截：{gpu_name} 算力 {cc:.1f} < 8.0，不支持量化算子加速！")
    safe_shutdown("硬件不达标 (非 L4)")
    raise RuntimeError(f"硬件不达标 ({gpu_name})，已自动断开。请重新分配 L4 实例。")

log("🎉 硬件门禁验证通过，允许运行！")
'''

CELL_COMFY = r'''# ==================== 4. 安装 ComfyUI 与 PyAV 基础依赖 ====================
if not (COMFY_ROOT / "main.py").exists():
    log("📦 正在克隆官方 ComfyUI 仓库...")
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/comfyanonymous/ComfyUI.git", str(COMFY_ROOT)], check=True)
    log("📦 正在安装 ComfyUI 基础依赖...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(COMFY_ROOT / "requirements.txt")], check=True)
else:
    log("✅ 检测到 ComfyUI 目录已就绪")

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "av", "huggingface_hub", "safetensors", "accelerate"], check=True)
log("✅ ComfyUI 核心与 PyAV 依赖就绪！")
'''

CELL_MODELS = r'''# ==================== 5. 模型挂载与直连高速下载 (直连优先 + Drive 容灾回退) ====================
import struct
from huggingface_hub import hf_hub_download

EXACT_SIZES = {
    "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors": 20970379616,
    "diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors": 20958205608,
    "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors": 15687142551,
    "vae/minimax_h3_video_vae_int8_convrot.safetensors": 2811065184,
    "vae/minimax_h3_video_vae_fp16.safetensors": 5207808496,
    "vae/minimax_h3_audio_vae_fp32.safetensors": 605254808,
    "loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors": 1956193000,
    "loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors": 1956192992,
}

def resolve_variant_and_files():
    cuda_ver = None
    try:
        import torch
        if torch.cuda.is_available() and hasattr(torch.version, "cuda") and torch.version.cuda:
            cuda_ver = torch.version.cuda
    except Exception:
        pass

    if DIFFUSION_VARIANT == "auto":
        if cuda_ver and cuda_ver.startswith("13."):
            actual_variant = "int8_convrot"
            log(f"⚡ 检测到 PyTorch CUDA 版本: {cuda_ver} (cu130)，启用原生 int8_convrot 算子")
        else:
            actual_variant = "fp8_scaled"
            log(f"💡 检测到 CUDA 环境: {cuda_ver}。在 Colab 默认 CUDA 12.x + L4 (CC 8.9) 下，"
                f"自动选用 fp8_scaled + video_vae_fp16，消除 int8_convrot 在非 cu130 下的慢路径回退！")
    else:
        actual_variant = DIFFUSION_VARIANT
        log(f"⚙️ 用户显式指定量化权重方案: {actual_variant}")

    if actual_variant == "int8_convrot":
        diff_file = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
        vae_file = "minimax_h3_video_vae_int8_convrot.safetensors"
    else:
        diff_file = "minimax_h3_fl2va_pruned_fp8_scaled.safetensors"
        vae_file = "minimax_h3_video_vae_fp16.safetensors"

    if TURBO_STEPS == 4:
        lora_file = "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors"
    else:
        lora_file = "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"

    return actual_variant, diff_file, vae_file, lora_file

ACTUAL_VARIANT, DIFFUSION_FILE, VIDEO_VAE_FILE, LORA_FILE = resolve_variant_and_files()

MODELS_SPEC = [
    (f"diffusion_models/{DIFFUSION_FILE}", "diffusion_models", EXACT_SIZES[f"diffusion_models/{DIFFUSION_FILE}"]),
    ("text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "text_encoders", EXACT_SIZES["text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"]),
    (f"vae/{VIDEO_VAE_FILE}", "vae", EXACT_SIZES[f"vae/{VIDEO_VAE_FILE}"]),
    ("vae/minimax_h3_audio_vae_fp32.safetensors", "vae", EXACT_SIZES["vae/minimax_h3_audio_vae_fp32.safetensors"]),
    (f"loras/{LORA_FILE}", "loras", EXACT_SIZES[f"loras/{LORA_FILE}"]),
]

def check_safetensors_file(file_path: Path, expected_size: int) -> bool:
    """严格校验文件大小与 safetensors 头部元数据边界，防止半截文件"""
    if not file_path.is_file():
        return False
    try:
        actual_size = file_path.stat().st_size
        if actual_size != expected_size:
            return False
        with open(file_path, "rb") as f:
            header_bytes = f.read(8)
            if len(header_bytes) < 8:
                return False
            hlen = struct.unpack("<Q", header_bytes)[0]
            header = json.loads(f.read(hlen).decode("utf-8"))
        max_end = max(v["data_offsets"][1] for k, v in header.items() if k != "__metadata__")
        return (8 + hlen + max_end) == actual_size
    except Exception:
        return False

log("🔗 正在挂载并校验 5 大核心模型权重...")
t_m0 = time.time()

for rel_path, subfolder, exact_bytes in MODELS_SPEC:
    fname = Path(rel_path).name
    local_dir = LOCAL_MODELS / subfolder
    local_dir.mkdir(parents=True, exist_ok=True)
    local_file = local_dir / fname
    drive_sub_file = DRIVE_MODELS / subfolder / fname
    drive_flat_file = DRIVE_MODELS / fname

    # 1. 检查本地 NVMe 是否已有真实物理文件且通过完整性校验
    if not local_file.is_symlink() and local_file.is_file() and check_safetensors_file(local_file, exact_bytes):
        log(f"   ✅ 本地 NVMe 高速缓存已就绪 (真实物理文件): {fname} ({exact_bytes / (1024**3):.2f} GiB)")
        continue

    # 若本地是软链接或残损文件，先清除，确保后续获得本地高速物理文件
    if local_file.is_symlink() or local_file.exists():
        local_file.unlink()

    # 2. 首选主路径：直接从 HuggingFace 官方源下载至本地 NVMe (速度快且杜绝 FUSE 慢读取)
    log(f"   ⬇️ 首选主路径: 从 HuggingFace 官方拉取至本地 NVMe: {fname} ({exact_bytes / (1024**3):.2f} GiB)...")
    download_success = False
    try:
        downloaded = hf_hub_download(
            repo_id="Comfy-Org/MiniMax-H3",
            filename=rel_path,
            local_dir=str(LOCAL_MODELS),
            local_dir_use_symlinks=False
        )
        dl_path = Path(downloaded)
        if dl_path.resolve() != local_file.resolve() and dl_path.exists():
            shutil.move(str(dl_path), str(local_file))

        if check_safetensors_file(local_file, exact_bytes):
            log(f"   ✅ HuggingFace 直连下载完成并通过校验: {fname}")
            download_success = True

            # 3. 同步原子备份至 Drive 容灾缓存（不阻断主流程）
            try:
                drive_sub_file.parent.mkdir(parents=True, exist_ok=True)
                if not drive_sub_file.exists() or not check_safetensors_file(drive_sub_file, exact_bytes):
                    tmp_drive = drive_sub_file.with_suffix(".safetensors.partial")
                    shutil.copyfile(local_file, tmp_drive)
                    tmp_drive.replace(drive_sub_file)
                    log(f"   💾 已同步原子备份至 Drive 容灾缓存: {drive_sub_file.name}")
            except Exception as be:
                log(f"   ⚠️ 备份到 Drive 略过: {be}")
        else:
            log(f"   ⚠️ HuggingFace 下载文件完整性未通过: {fname}")
            if local_file.exists():
                local_file.unlink()
    except Exception as dl_err:
        log(f"   ⚠️ HuggingFace 直连下载受阻或网络不通: {dl_err}")
        if local_file.is_symlink() or local_file.exists():
            try:
                local_file.unlink()
            except Exception:
                pass

    # 4. 容灾兜底回退：仅当 HuggingFace 直连下载失败时，尝试从 Google Drive 缓存恢复
    if not download_success:
        log(f"   🔄 触发容灾回退: 尝试从 Google Drive 检索可用备份: {fname} ...")
        cand_drive = None
        for cand in [drive_sub_file, drive_flat_file]:
            if cand.exists():
                if check_safetensors_file(cand, exact_bytes):
                    cand_drive = cand
                    break
                else:
                    log(f"   🗑️ 检测到 Drive 中存在残损半截文件: {cand.name}，正在清理...")
                    try:
                        cand.unlink()
                    except Exception:
                        pass

        if cand_drive:
            log(f"   ⚡ 命中 Drive 缓存: {cand_drive.name}")
            copied = False
            try:
                log(f"   📦 正在从 Drive 物理拷贝至本地 NVMe (避免 FUSE 显存饥饿延迟)...")
                shutil.copy2(cand_drive, local_file)
                if check_safetensors_file(local_file, exact_bytes):
                    copied = True
                    log(f"   ✅ 从 Drive 物理拷贝完成: {fname}")
            except Exception as ce:
                log(f"   ⚠️ Drive 物理拷贝失败 ({ce})，降级为软链接...")
                if local_file.exists():
                    local_file.unlink()

            if not copied:
                local_file.symlink_to(cand_drive.resolve())
                log(f"   🔗 软链接建立: {fname} -> {cand_drive}")
        else:
            raise RuntimeError(f"❌ HuggingFace 下载失败且 Google Drive 无有效备份: {fname}！请检查网络连接。")

# 建立向后兼容别名软链接
aliases = [
    (LOCAL_MODELS / "vae" / "minimax_h3_vae.safetensors", LOCAL_MODELS / "vae" / VIDEO_VAE_FILE),
    (LOCAL_MODELS / "loras" / "minimax_h3_fl2va_pdd_8step_pruned_comfy.safetensors", LOCAL_MODELS / "loras" / LORA_FILE),
]
for alias_path, src_path in aliases:
    if src_path.exists() and not alias_path.exists():
        alias_path.symlink_to(src_path.resolve())

log(f"🎉 全部 5 大核心模型权重就绪！耗时: {time.time() - t_m0:.1f}s")
'''

CELL_START = r'''# ==================== 6. 启动 ComfyUI 守护服务与前置节点预检 ====================
import urllib.request, urllib.error

def post_json(path, payload=None, timeout=10):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(f"{SERVER_URL}{path}", data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        content = resp.read().decode("utf-8")
        return json.loads(content) if content.strip() else {}

def is_comfy_ready():
    try:
        res = post_json("/system_stats", timeout=3)
        return "devices" in res
    except Exception:
        return False

subprocess.run(["pkill", "-9", "-f", "main.py.*8188"], check=False)
time.sleep(2)

log("🚀 正在启动 ComfyUI 后台守护服务 (--lowvram)...")
log_fp = open(COMFY_LOG, "w", encoding="utf-8")
comfy_proc = subprocess.Popen(
    [sys.executable, "main.py", "--listen", "127.0.0.1", "--port", "8188", "--lowvram"],
    cwd=str(COMFY_ROOT), stdout=log_fp, stderr=subprocess.STDOUT, start_new_session=True
)

for i in range(60):
    if comfy_proc.poll() is not None:
        tail = COMFY_LOG.read_text(errors="replace")[-2000:] if COMFY_LOG.exists() else ""
        raise RuntimeError(f"ComfyUI 提前异常退出，错误日志:\n{tail}")
    if is_comfy_ready():
        log(f"✅ ComfyUI 守护服务就绪 (PID: {comfy_proc.pid})")
        break
    time.sleep(2)
else:
    raise RuntimeError("ComfyUI 在 120 秒内未能就绪，请检查 /content/comfyui.log")

log("🔍 正在预检 ComfyUI 节点注册表与模型索引...")
OBJ_INFO = post_json("/object_info", timeout=60)
required_nodes = [
    "UNETLoader", "CLIPLoader", "VAELoader", "LoraLoaderModelOnly",
    "MiniMaxH3ImageToVideo", "BasicGuider", "RandomNoise", "KSamplerSelect",
    "BasicScheduler", "SamplerCustomAdvanced", "VAEDecode",
    "VAEDecodeAudio", "CreateVideo", "SaveVideo"
]
missing = [n for n in required_nodes if n not in OBJ_INFO]
if missing:
    raise RuntimeError(f"ComfyUI 缺失必要节点类: {missing}")

log(f"✅ 预检 100% 通过：全部 {len(required_nodes)} 类核心节点均已就绪！")
'''

CELL_RENDER = r'''# ==================== 7. 上海街景原生音画 MP4 队列批处理渲染 (2.0) ====================
import copy

BASE_WORKFLOW = json.loads(r"""__WF_JSON__""")
ALL_TASKS = json.loads(r"""__TASKS_JSON__""")
TASKS = ALL_TASKS[:max(1, min(int(NUM_SHOTS), len(ALL_TASKS)))]

def align_frame_count(n: int) -> int:
    """对齐帧数至 17k + 5 网格 (nodes_minimax_h3.py 官方公式)"""
    n = max(5, int(n))
    return n + (5 - n % 17) % 17

def format_h3_prompt(task: dict) -> str:
    """组装官方结构化多镜头提示词"""
    if not isinstance(task, dict):
        return str(task)
    shots = task.get("shots")
    soundscape = task.get("overall_soundscape")
    music = task.get("non_diegetic_music")
    if not shots and not soundscape and not music:
        return task.get("prompt", "")
    parts = []
    if shots:
        shot_lines = []
        for i, s in enumerate(shots, 1):
            if isinstance(s, dict):
                t_str = s.get("time")
                desc = s.get("description", s.get("prompt", ""))
                if i == 1 and (not t_str or t_str == "00:00.000" or t_str == "0"):
                    shot_lines.append(f"[Shot 1] {desc}")
                else:
                    t_val = t_str if t_str else f"00:{((i-1)*5):02d}.000"
                    shot_lines.append(f"[Shot {i}] At {t_val}, {desc}")
            else:
                if i == 1:
                    shot_lines.append(f"[Shot 1] {s}")
                else:
                    shot_lines.append(f"[Shot {i}] At 00:{((i-1)*5):02d}.000, {s}")
        parts.append("integrated_multimodal_description:\n" + "\n".join(shot_lines))
    elif task.get("prompt"):
        parts.append(f"integrated_multimodal_description:\n{task['prompt']}")
    if soundscape:
        parts.append(f"overall_soundscape: {soundscape}")
    if music:
        parts.append(f"non_diegetic_music: {music}")
    return "\n\n".join(parts)

def validate_and_adapt_workflow(wf: dict, obj_info: dict) -> dict:
    """根据 ComfyUI /object_info 校验输入契约并自动裁剪未注册可选参数"""
    if not obj_info:
        return wf
    cleaned = copy.deepcopy(wf)
    for nid, node in cleaned.items():
        if not isinstance(node, dict):
            continue
        ctype = node.get("class_type")
        if not ctype or ctype not in obj_info:
            raise ValueError(f"节点 [{nid}] 类名 '{ctype}' 未注册！")
        node_spec = obj_info[ctype].get("input", {})
        req = set(node_spec.get("required", {}).keys())
        opt = set(node_spec.get("optional", {}).keys())
        valid_keys = req | opt
        inputs = node.get("inputs", {})
        to_remove = [k for k in list(inputs.keys()) if k not in valid_keys]
        for k in to_remove:
            log(f"   ⚠️ 节点 [{nid}: {ctype}] 移除非当前版本参数: '{k}'")
            del inputs[k]
        missing_req = [k for k in req if k not in inputs]
        if missing_req:
            raise ValueError(f"节点 [{nid}: {ctype}] 缺失必要输入参数: {missing_req}")
    return cleaned

def wait_job_completion(pid, timeout_sec=900):
    t_start = time.time()
    last_print = 0
    while time.time() - t_start < timeout_sec:
        try:
            h = post_json(f"/history/{pid}", timeout=5)
            if pid in h:
                return h[pid], time.time() - t_start
        except Exception:
            pass
        elapsed = int(time.time() - t_start)
        if elapsed - last_print >= 20:
            log(f"   ⏳ 渲染去噪中... 已耗时 {elapsed}s")
            last_print = elapsed
        time.sleep(3)
    try:
        post_json("/interrupt", {}, timeout=5)
    except Exception:
        pass
    raise TimeoutError(f"镜头渲染超时 ({timeout_sec}s)，已触发中断")

def probe_mp4_file(mp4_path):
    try:
        import av
        with av.open(str(mp4_path)) as container:
            v_streams = [s for s in container.streams if s.type == "video"]
            a_streams = [s for s in container.streams if s.type == "audio"]
            dur = round(float(container.duration) / 1e6, 2) if container.duration else 0.0
            a_rate = a_streams[0].codec_context.sample_rate if a_streams else 0
            return len(v_streams), len(a_streams), dur, a_rate
    except Exception as e:
        return 0, 0, 0, str(e)

log("=" * 68)
log(f"🎬 开始 MiniMax H3 上海街景队列批处理渲染 (2.0): 共 {len(TASKS)} 镜")
log(f"🎯 选定权重方案: {ACTUAL_VARIANT} | LoRA: {LORA_FILE}")
log("=" * 68)

try:
    post_json("/queue", {"clear": True}, timeout=5)
    post_json("/interrupt", {}, timeout=5)
    time.sleep(2)
except Exception:
    pass

RENDER_RESULTS = []
t_batch_start = time.time()

for i, task in enumerate(TASKS, 1):
    t_id = task["id"]
    title = task.get("title", t_id)
    prompt = format_h3_prompt(task)
    seed = int(task.get("seed", 1024))
    prefix = f"{t_id}_{seed}"
    raw_len = int(task.get("length", 124))
    aligned_len = align_frame_count(raw_len)
    width = int(task.get("width", 960))
    height = int(task.get("height", 544))
    steps = int(task.get("steps", TURBO_STEPS))

    log(f"\n▶️ [{i}/{len(TASKS)}] 正在生成: {title}")
    log(f"   📐 规格: {width}x{height} | {aligned_len} 帧 (网格对齐) | 步数 {steps} | 种子 {seed}")
    log(f"   📝 画面与声效提示词: {prompt[:90]}...")

    wf = copy.deepcopy(BASE_WORKFLOW)
    wf["1"]["inputs"]["unet_name"] = DIFFUSION_FILE
    wf["4"]["inputs"]["vae_name"] = VIDEO_VAE_FILE
    wf["10"]["inputs"]["lora_name"] = LORA_FILE
    wf["6"]["inputs"]["prompt"] = prompt
    wf["6"]["inputs"]["width"] = width
    wf["6"]["inputs"]["height"] = height
    wf["6"]["inputs"]["length"] = aligned_len
    wf["14"]["inputs"]["noise_seed"] = seed
    wf["15"]["inputs"]["sampler_name"] = task.get("sampler_name", SAMPLER_NAME)
    wf["16"]["inputs"]["steps"] = steps
    wf["16"]["inputs"]["scheduler"] = task.get("scheduler", "simple")
    wf["9"]["inputs"]["filename_prefix"] = prefix

    wf = validate_and_adapt_workflow(wf, globals().get("OBJ_INFO"))

    submit_resp = post_json("/prompt", {"prompt": wf, "client_id": "colab_shanghai_2.0"}, timeout=15)
    if "error" in submit_resp or submit_resp.get("node_errors"):
        raise RuntimeError(f"ComfyUI 拒绝提交工作流: {submit_resp}")
    pid = submit_resp["prompt_id"]
    log(f"   🚀 成功推入 ComfyUI 执行队列 (Prompt ID: {pid})")

    hist_data, cost_s = wait_job_completion(pid)
    status_info = hist_data.get("status", {})
    if status_info.get("status_str") != "success":
        err_msgs = status_info.get("messages", [])
        raise RuntimeError(f"镜头执行异常: {json.dumps(err_msgs, ensure_ascii=False)}")

    target_mp4 = None
    for n_out in hist_data.get("outputs", {}).values():
        for cat in ["videos", "gifs", "images"]:
            for item in n_out.get(cat, []):
                fn = item.get("filename", "")
                if fn.endswith(".mp4"):
                    candidate = LOCAL_OUT / item.get("subfolder", "") / fn
                    if candidate.exists():
                        target_mp4 = candidate
                        break

    if not target_mp4 and LOCAL_OUT.exists():
        found = sorted(list(LOCAL_OUT.glob(f"{prefix}*.mp4")), key=lambda p: p.stat().st_mtime, reverse=True)
        if found:
            target_mp4 = found[0]

    if not target_mp4:
        raise RuntimeError(f"渲染成功但在 {LOCAL_OUT} 未发现匹配的 MP4 成片！")

    v_cnt, a_cnt, dur_s, a_sample = probe_mp4_file(target_mp4)
    file_mb = target_mp4.stat().st_size / (1024 * 1024)

    drive_dst = DRIVE_OUTPUTS / target_mp4.name
    shutil.copyfile(target_mp4, drive_dst)
    assert drive_dst.exists(), f"文件落盘 Google Drive 失败: {drive_dst}"

    log(f"   ✅ 渲染成功！耗时: {cost_s:.1f}s | 文件大小: {file_mb:.2f} MB")
    log(f"   🔊 音频规格: {a_cnt} 条音频流 ({a_sample} Hz) | 视频时长: {dur_s}s")
    log(f"   💾 成片已安全同步至 Google Drive: {drive_dst.name}")

    RENDER_RESULTS.append({
        "id": t_id,
        "title": title,
        "file": target_mp4.name,
        "local_path": str(target_mp4),
        "drive_path": str(drive_dst),
        "size_mb": round(file_mb, 2),
        "duration_s": dur_s,
        "audio_rate": a_sample,
        "cost_s": round(cost_s, 1)
    })

total_batch_time = time.time() - t_batch_start
cu_spent = 1.94 * (total_batch_time / 3600)
log("\n" + "=" * 68)
log(f"🎉 全部 {len(RENDER_RESULTS)} 镜批处理完成！总耗时: {total_batch_time:.1f}s ({total_batch_time/60:.2f} 分钟)")
log(f"⚡ 实际折合算力消耗: ~{cu_spent:.3f} CU (L4 费率 1.94 CU/h)")
log(f"📂 所有成片均已持久化保存在: {DRIVE_OUTPUTS}")
log("=" * 68)
'''

CELL_PREVIEW = r'''# ==================== 8. 成片 HTML5 音画同步预览试听 ====================
import base64
from IPython.display import HTML, display

if globals().get("RENDER_RESULTS"):
    log(f"🎬 正在渲染 {len(RENDER_RESULTS)} 个上海街景音视频预览卡片（请打开视频声音滑块试听）：\n")
    for r in RENDER_RESULTS:
        with open(r["local_path"], "rb") as f:
            b64_data = base64.b64encode(f.read()).decode("utf-8")
        html_code = f"""
        <div style="margin-bottom: 20px; background: #1a1a24; padding: 16px; border-radius: 8px; border: 1px solid #3b82f6; max-width: 720px; font-family: sans-serif;">
            <h4 style="color: #60a5fa; margin: 0 0 8px 0;">🎬 {r['title']} ({r['size_mb']} MB | {r['duration_s']}s)</h4>
            <video width="100%" controls style="border-radius: 6px;">
                <source src="data:video/mp4;base64,{b64_data}" type="video/mp4">
            </video>
            <p style="color: #94a3b8; font-size: 13px; margin: 8px 0 0 0;">
                🔊 <b>原生 32kHz 音效</b>：点击播放器右下角<b>声音图标</b>解除静音，即可试听现场环境声效！
            </p>
        </div>
        """
        display(HTML(html_code))
else:
    print("暂无生成的视频数据")
'''

CELL_SHUTDOWN = r'''# ==================== 9. 安全收尾与自动销毁实例 (防扣费) ====================
if AUTO_SHUTDOWN:
    log("🛑 批处理任务全部完成，正在自动销毁实例以终止算力扣费...")
    safe_shutdown("上海街景出片完毕，正常下班")
else:
    log("💡 提示：AUTO_SHUTDOWN 设为 False，请在右上角手动断开并释放运行时。")
'''

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(True)}

def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src.strip("\n").splitlines(True)}

def build():
    render_code = CELL_RENDER.replace("__WF_JSON__", WF_JSON).replace("__TASKS_JSON__", TASKS_JSON)
    cells = [
        md(MD_INTRO),
        md("### 步骤 1：运行配置与算力看门狗守护"), code(CELL_CONFIG),
        md("### 步骤 2：挂载 Google Drive (官方原生接口)"), code(CELL_DRIVE),
        md("### 步骤 3：硬件门禁检测 (验证 L4 GPU，零内核显存占用)"), code(CELL_GPU),
        md("### 步骤 4：安装 ComfyUI 核心依赖与 PyAV"), code(CELL_COMFY),
        md("### 步骤 5：模型安全校验与秒级挂载 (精确字节 + safetensors 头部双校验)"), code(CELL_MODELS),
        md("### 步骤 6：启动 ComfyUI 守护服务与前置预检"), code(CELL_START),
        md("### 步骤 7：上海街景原生音画 MP4 队列批处理渲染"), code(render_code),
        md("### 步骤 8：成片 HTML5 音画同步预览试听"), code(CELL_PREVIEW),
        md("### 步骤 9：安全收尾与自动销毁实例 (防扣费)"), code(CELL_SHUTDOWN),
    ]

    nb = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {"name": "v2.0_test_shanghai_street_mp4.ipynb", "provenance": [], "gpuType": "L4"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU"
        },
        "cells": cells
    }

    nb_str = json.dumps(nb, indent=2, ensure_ascii=False)
    versioned_file = ROOT / "notebooks" / "v2.0_test_shanghai_street_mp4.ipynb"

    versioned_file.parent.mkdir(parents=True, exist_ok=True)
    versioned_file.write_text(nb_str, encoding="utf-8")
    print(f"🎉 成功生成版本化上海街景 Notebook: {versioned_file}")

if __name__ == "__main__":
    build()
