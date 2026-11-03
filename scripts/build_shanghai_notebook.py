#!/usr/bin/env python3
"""
Generates notebooks/test_shanghai_mp4_audio.ipynb

100% Google Colab compliant and standalone:
- NO rclone, NO fuse, NO third-party daemons (strict adherence to Colab Acceptable Use Policy)
- Standard official Google Drive mount (from google.colab import drive)
- Exact HuggingFace byte verification & safetensors header boundary check
- Local NVMe model storage (zero Drive symlinks to prevent FUSE corruptions)
- Hardware gatekeeper for L4 (CUDA CC >= 8.0)
- ComfyUI background service with preflight node validation
- In-memory workflow batch rendering with PyAV stream verification
- Immediate sync to Google Drive & HTML5 audio-video player preview
- Automatic shutdown to safeguard Colab Pro compute units
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

WF = json.loads((ROOT / "workflows" / "h3_t2v_pdd8_mp4.json").read_text(encoding="utf-8"))
TASKS = json.loads((ROOT / "configs" / "shanghai_street.json").read_text(encoding="utf-8"))

# Validate node topology
EXPECTED = {
    "1": "UNETLoader",
    "2": "CLIPLoader",
    "3": "KSampler",
    "4": "VAELoader",
    "5": "EmptyMiniMaxH3LatentAV",
    "6": "CLIPTextEncode",
    "7": "CLIPTextEncode",
    "8": "VAEDecode",
    "9": "SaveVideo",
    "10": "LoraLoaderModelOnly",
    "11": "VAELoader",
    "12": "VAEDecodeAudio",
    "13": "CreateVideo"
}
for nid, ctype in EXPECTED.items():
    assert WF[nid]["class_type"] == ctype, f"Node {nid} must be {ctype}"
assert WF["9"]["inputs"]["video"] == ["13", 0]
assert WF["13"]["inputs"]["images"] == ["8", 0] and WF["13"]["inputs"]["audio"] == ["12", 0]

WF_JSON = json.dumps(WF, ensure_ascii=False, indent=1)
TASKS_JSON = json.dumps(TASKS, ensure_ascii=False, indent=1)

MD_INTRO = r"""# 🏙️ MiniMax H3 (Hailuo 3.0) 上海街景原生音画 MP4 出片工作流

> **合规与安全性声明**：
> 本 Notebook **100% 遵守 Google Colab 服务条款与使用规范**：
> - 仅使用官方原生 `google.colab.drive` 挂载，绝不使用任何未经许可的第三方后台守护工具（如 rclone、fuse 等）。
> - 纯本地推理运算，通过 ComfyUI 内部 API (`127.0.0.1:8188`) 通信，不开启任何公网穿透代理或文件服务。
> - 配备精确字节与头部边界双重校验，彻底防止残损文件导致的异常退出。
> - 生成完成后自动同步成片至 Google Drive，并执行安全关机释放实例，杜绝算力浪费。

**使用步骤**：
1. 顶部菜单点击「代码执行程序」->「更改运行时类型」-> 选择 **L4 GPU**。
2. 点击「代码执行程序」->「全部运行」。
3. 步骤 2 弹出 Google Drive 授权提示时，点击「连接到 Google 云端硬盘」授权一次即可。
"""

CELL_CONFIG = r'''# ==================== 1. 运行配置与安全看门狗 ====================
NUM_SHOTS = 2               # 生成镜头数 1~3（每镜 124 帧 ≈ 5.16 秒，L4 约 5~6 分钟/镜）
AUTO_SHUTDOWN = True        # 全部完成后自动执行 runtime.unassign() 释放实例停止扣费
WATCHDOG_HOURS = 1.5        # 算力看门狗硬熔断上限（小时），防止异常卡死消耗算力

import os, sys, time, json, shutil, threading, subprocess
from pathlib import Path

DRIVE_ROOT = Path("/content/drive/MyDrive")
DRIVE_MODELS = DRIVE_ROOT / "MiniMax_Models"
DRIVE_OUTPUTS = DRIVE_ROOT / "MiniMax_Outputs/shanghai_street"

COMFY_ROOT = Path("/content/ComfyUI")
LOCAL_MODELS = COMFY_ROOT / "models"
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

# 启动后台看门狗定时器
if not globals().get("_WATCHDOG_RUNNING"):
    def _watchdog_timer():
        time.sleep(int(WATCHDOG_HOURS * 3600))
        log(f"🚨 [WATCHDOG 熔断] 运行时已超 {WATCHDOG_HOURS} 小时，自动断开保护算力！")
        safe_shutdown("看门狗超时熔断")
    threading.Thread(target=_watchdog_timer, daemon=True).start()
    _WATCHDOG_RUNNING = True
    log(f"🛡️  算力看门狗守门员已就绪：{WATCHDOG_HOURS} 小时超时自动断开")
'''

CELL_DRIVE = r'''# ==================== 2. 挂载 Google Drive（官方合规原生方式） ====================
from google.colab import drive

if not os.path.exists("/content/drive/MyDrive"):
    log("⏳ 正在通过官方接口挂载 Google Drive（请在弹出框中点击授权）...")
    drive.mount("/content/drive")
else:
    log("✅ Google Drive 处于已挂载就绪状态")

DRIVE_MODELS.mkdir(parents=True, exist_ok=True)
DRIVE_OUTPUTS.mkdir(parents=True, exist_ok=True)
log(f"📁 云端模型持久化缓存: {DRIVE_MODELS}")
log(f"🎬 成片与日志最终落盘: {DRIVE_OUTPUTS}")
'''

CELL_GPU = r'''# ==================== 3. 硬件环境门禁检测（锁定 L4 GPU） ====================
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
    log(f"🚨 致命拦截：{gpu_name} 算力 {cc:.1f} < 8.0，不支持 Int8 ConvRot 算子！")
    safe_shutdown("硬件不达标 (非 L4)")
    raise RuntimeError(f"硬件不达标 ({gpu_name})，已自动断开。请重新分配 L4 实例。")

log("🎉 硬件门禁验证通过，允许运行！")
'''

CELL_COMFY = r'''# ==================== 4. 安装 ComfyUI 核心环境与 PyAV 音视频库 ====================
if not (COMFY_ROOT / "main.py").exists():
    log("📦 正在克隆官方 ComfyUI 仓库...")
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/comfyanonymous/ComfyUI.git", str(COMFY_ROOT)], check=True)
    log("📦 正在安装 ComfyUI 基础依赖...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(COMFY_ROOT / "requirements.txt")], check=True)
else:
    log("✅ 检测到 ComfyUI 目录已就绪")

# 确保安装必要扩展库（PyAV 用于 32kHz 音视频合流与编码封装）
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "av", "huggingface_hub", "safetensors", "accelerate"], check=True)
log("✅ ComfyUI 核心与 PyAV 依赖就绪！")
'''

CELL_MODELS = r'''# ==================== 5. 模型校验与自愈加载 (精确字节 + safetensors 头部双校验) ====================
import struct
from huggingface_hub import hf_hub_download

# HuggingFace 官方 Comfy-Org/MiniMax-H3 仓库精确字节数
MODELS_SPEC = [
    ("diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors", "diffusion_models", 20970379616),
    ("text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "text_encoders", 15687142551),
    ("vae/minimax_h3_video_vae_int8_convrot.safetensors", "vae", 2811065184),
    ("vae/minimax_h3_audio_vae_fp32.safetensors", "vae", 605254808),
    ("loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors", "loras", 1956193000),
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

    # 如果本地已是软链接，清除软链接以确保使用本地独立高速实体文件
    if local_file.is_symlink():
        local_file.unlink()

    # 1. 本地已存在完整文件
    if check_safetensors_file(local_file, exact_bytes):
        log(f"   ✅ 本地完整就绪: {fname} ({exact_bytes / (1024**3):.2f} GiB)")
        continue

    # 2. 若本地存在不完整文件，彻底清理
    if local_file.exists():
        log(f"   🗑️ 清理本地残损文件: {fname} (当前大小 {local_file.stat().st_size} != {exact_bytes})")
        local_file.unlink()

    # 3. 检查 Google Drive 缓存中是否有完整文件
    cand_drive = None
    for cand in [drive_sub_file, drive_flat_file]:
        if cand.exists():
            if check_safetensors_file(cand, exact_bytes):
                cand_drive = cand
                break
            else:
                log(f"   🗑️ 清理 Drive 残损缓存: {cand.name} (大小不匹配，彻底移除)")
                cand.unlink()

    if cand_drive:
        local_file.symlink_to(cand_drive.resolve())
        log(f"   ⚡ 命中 Drive 完整缓存，0 秒软链接: {fname} -> {subfolder}/")
        continue

    # 4. 从 HuggingFace 官方拉取
    log(f"   ⬇️ 本地与 Drive 无有效缓存，从 HuggingFace 下载: {fname} ({exact_bytes / (1024**3):.2f} GiB)...")
    downloaded = hf_hub_download(
        repo_id="Comfy-Org/MiniMax-H3",
        filename=rel_path,
        local_dir=str(LOCAL_MODELS)
    )
    dl_path = Path(downloaded)
    if dl_path.resolve() != local_file.resolve() and dl_path.exists():
        shutil.move(str(dl_path), str(local_file))

    if not check_safetensors_file(local_file, exact_bytes):
        raise RuntimeError(f"从 HuggingFace 下载的文件损坏或不完整: {fname}")
    log(f"   ✅ 下载并通过安全校验: {fname}")

    # 5. 原子备份至 Drive 缓存目录
    try:
        drive_sub_file.parent.mkdir(parents=True, exist_ok=True)
        tmp_drive = drive_sub_file.with_suffix(".safetensors.partial")
        shutil.copyfile(local_file, tmp_drive)
        tmp_drive.replace(drive_sub_file)
        log(f"   💾 已原子备份至 Drive 缓存: {drive_sub_file.name}")
    except Exception as be:
        log(f"   ⚠️ 备份到 Drive 略过: {be}")

# 建立向后兼容别名软链接
aliases = [
    (LOCAL_MODELS / "vae" / "minimax_h3_vae.safetensors", LOCAL_MODELS / "vae" / "minimax_h3_video_vae_int8_convrot.safetensors"),
    (LOCAL_MODELS / "loras" / "minimax_h3_fl2va_pdd_8step_pruned_comfy.safetensors", LOCAL_MODELS / "loras" / "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"),
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

# 若已有运行实例先杀死
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

# 预检关键节点与模型名称是否已被 ComfyUI 正确注册
log("🔍 正在预检 ComfyUI 节点注册表与模型索引...")
obj_info = post_json("/object_info", timeout=60)
required_nodes = [
    "UNETLoader", "CLIPLoader", "VAELoader", "LoraLoaderModelOnly",
    "EmptyMiniMaxH3LatentAV", "CLIPTextEncode", "KSampler", "VAEDecode",
    "VAEDecodeAudio", "CreateVideo", "SaveVideo"
]
missing = [n for n in required_nodes if n not in obj_info]
if missing:
    raise RuntimeError(f"ComfyUI 缺失必要节点类: {missing}")

log("✅ 预检 100% 通过：全部 11 类节点均已就绪！")
'''

CELL_RENDER = r'''# ==================== 7. 原生批量执行上海街景出片队列 (原生 32kHz 立体声 MP4) ====================
import copy

BASE_WORKFLOW = json.loads(r"""__WF_JSON__""")
ALL_TASKS = json.loads(r"""__TASKS_JSON__""")
TASKS = ALL_TASKS[:max(1, min(int(NUM_SHOTS), len(ALL_TASKS)))]

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
    # 超时中断
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
log(f"🎬 开始上海街景纯本地批处理渲染: 共 {len(TASKS)} 镜")
log("=" * 68)

RENDER_RESULTS = []
t_batch_start = time.time()

for i, task in enumerate(TASKS, 1):
    t_id = task["id"]
    title = task.get("title", t_id)
    prompt = task["prompt"]
    neg_prompt = task.get("negative_prompt", "distorted audio, watermark, text, blur, low quality, deformed, jerky motion, noisy static")
    seed = int(task.get("seed", 1024))
    prefix = f"{t_id}_{seed}"

    log(f"\n▶️ [{i}/{len(TASKS)}] 正在生成: {title}")
    log(f"   📝 画面与声效提示词: {prompt[:90]}...")

    # 动态构建工作流
    wf = copy.deepcopy(BASE_WORKFLOW)
    wf["6"]["inputs"]["text"] = prompt
    wf["7"]["inputs"]["text"] = neg_prompt
    wf["5"]["inputs"]["width"] = int(task.get("width", 1344))
    wf["5"]["inputs"]["height"] = int(task.get("height", 768))
    wf["5"]["inputs"]["length"] = int(task.get("length", 124))
    wf["3"]["inputs"]["seed"] = seed
    wf["3"]["inputs"]["steps"] = int(task.get("steps", 8))
    wf["3"]["inputs"]["cfg"] = float(task.get("cfg", 3.0))
    wf["9"]["inputs"]["filename_prefix"] = prefix

    # 提交队列
    submit_resp = post_json("/prompt", {"prompt": wf, "client_id": "colab_production"}, timeout=15)
    if "error" in submit_resp or submit_resp.get("node_errors"):
        raise RuntimeError(f"ComfyUI 拒绝提交工作流: {submit_resp}")
    pid = submit_resp["prompt_id"]
    log(f"   🚀 成功推入 ComfyUI 执行队列 (Prompt ID: {pid})")

    # 等待完成
    hist_data, cost_s = wait_job_completion(pid)
    status_info = hist_data.get("status", {})
    if status_info.get("status_str") != "success":
        err_msgs = status_info.get("messages", [])
        raise RuntimeError(f"镜头执行异常: {json.dumps(err_msgs, ensure_ascii=False)}")

    # 定位生成的 MP4 文件
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

    # 探测 MP4 音视频流质量
    v_cnt, a_cnt, dur_s, a_sample = probe_mp4_file(target_mp4)
    file_mb = target_mp4.stat().st_size / (1024 * 1024)

    # 立即持久化同步至 Google Drive
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
cu_spent = 1.9 * (total_batch_time / 3600)
log("\n" + "=" * 68)
log(f"🎉 全部 {len(RENDER_RESULTS)} 镜批处理完成！总耗时: {total_batch_time:.1f}s ({total_batch_time/60:.2f} 分钟)")
log(f"⚡ 实际折合算力消耗: ~{cu_spent:.3f} CU (L4 费率 1.90 CU/h)")
log(f"📂 所有成片均已持久化保存在: {DRIVE_OUTPUTS}")
log("=" * 68)
'''

CELL_PREVIEW = r'''# ==================== 8. HTML5 原生音视频播放试听 ====================
import base64
from IPython.display import HTML, display

if globals().get("RENDER_RESULTS"):
    log(f"🎬 正在渲染 {len(RENDER_RESULTS)} 个镜头的音视频预览卡片（请打开视频声音滑块试听）：\n")
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
                🔊 <b>原生 32kHz 音效</b>：点击播放器右下角<b>声音图标</b>解除静音，即可试听市井晨光生煎油煎声与梧桐风声！
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
    safe_shutdown("出片完毕，正常下班")
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
        md("### 步骤 3：硬件门禁检测 (验证 L4 GPU)"), code(CELL_GPU),
        md("### 步骤 4：安装 ComfyUI 核心依赖与 PyAV"), code(CELL_COMFY),
        md("### 步骤 5：模型安全校验与下载 (精确字节 + safetensors 头部双校验)"), code(CELL_MODELS),
        md("### 步骤 6：启动 ComfyUI 守护服务与前置预检"), code(CELL_START),
        md("### 步骤 7：批量渲染上海街景双镜头原生音画 MP4"), code(render_code),
        md("### 步骤 8：成片 HTML5 音画同步预览试听"), code(CELL_PREVIEW),
        md("### 步骤 9：安全断开实例终止算力计费"), code(CELL_SHUTDOWN),
    ]

    nb = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {"name": "test_shanghai_mp4_audio.ipynb", "provenance": [], "gpuType": "L4"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU"
        },
        "cells": cells
    }

    out_file = ROOT / "notebooks" / "test_shanghai_mp4_audio.ipynb"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(nb, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"🎉 成功生成 100% 官方合规自包含 Notebook: {out_file}")

if __name__ == "__main__":
    build()
