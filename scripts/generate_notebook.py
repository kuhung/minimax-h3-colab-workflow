#!/usr/bin/env python3
"""
Notebook Generator for MiniMax H3 Colab Workflow.
Generates notebooks/minimax_h3_l4_production.ipynb.
Fully self-contained, deadlock-free, and real benchmark smoke tested.
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Read existing templates and scripts for embedded fallback
wf_pdd8_content = (PROJECT_ROOT / "workflows" / "h3_t2v_pdd8_production.json").read_text(encoding="utf-8")
wf_baseline_content = (PROJECT_ROOT / "workflows" / "h3_t2v_l4_baseline.json").read_text(encoding="utf-8")
prompts_content = (PROJECT_ROOT / "configs" / "prompts.example.json").read_text(encoding="utf-8")
story_content = (PROJECT_ROOT / "configs" / "story_cyber_dawn.json").read_text(encoding="utf-8")
batch_runner_content = (PROJECT_ROOT / "scripts" / "batch_runner.py").read_text(encoding="utf-8")

def make_cell(cell_type: str, content: str):
    return {
        "cell_type": cell_type,
        "metadata": {},
        "execution_count": None if cell_type == "code" else None,
        "outputs": [] if cell_type == "code" else None,
        "source": [line + "\n" for line in content.splitlines()]
    }

# Remove None outputs if markdown
def clean_cell(cell):
    if cell["cell_type"] == "markdown":
        cell.pop("execution_count", None)
        cell.pop("outputs", None)
    return cell

cells = []

# Title & Principles
cells.append(clean_cell(make_cell("markdown", """# 🚀 MiniMax H3 (Hailuo 3.0) 纯本地 Colab 视频生产工作流

> **核心原则**：
> 1. **纯本地运行**：运行开源剪枝与量化权重，绝不调用外部商业 API。
> 2. **硬件门禁与三重防扣费**：强制要求算力 ≥ 8.0（首选 **L4 24GB**）。遇 T4 自动阻断并关机；配备后台 Watchdog 超时熔断守门员。
> 3. **无脸视频批产流**：提示词融合环境声效与视觉构图，产物直存 Google Drive。""")))

# Step 1: Drive mount & Watchdog
cells.append(clean_cell(make_cell("markdown", """### 步骤 1：挂载 Google Drive 与启动算力单元看门狗（Watchdog）
- 自动挂载 Drive 以便持久化保存成片和模型缓存。
- **看门狗安全线程**：若执行时间超过设定时限（默认 2.5 小时），强制触发关机，彻底防止遗忘网页导致 200 算力单元耗尽。""")))

cells.append(clean_cell(make_cell("code", """from google.colab import drive, runtime
import os, time, threading

print("⏳ 正在挂载 Google Drive...")
drive.mount('/content/drive')

os.makedirs('/content/drive/MyDrive/MiniMax_Outputs', exist_ok=True)
os.makedirs('/content/drive/MyDrive/MiniMax_Models', exist_ok=True)
print("✅ Google Drive 挂载就绪！")

# 算力看门狗守护线程 (防闲置偷跑算力)
WATCHDOG_HOURS = 2.5 # 设置单次最大运行小时上限
def watchdog():
    time.sleep(int(WATCHDOG_HOURS * 3600))
    print("\\n🚨 [WATCHDOG 熔断] 运行时已超最大限制，正在自动销毁实例以保卫算力！")
    try:
        os.sync()
        runtime.unassign()
    except Exception:
        os.kill(1, 15)

threading.Thread(target=watchdog, daemon=True).start()
print(f"🛡️  算力看门狗已激活：{WATCHDOG_HOURS} 小时未退出将强制终止实例。")""")))

# Step 2: Hardware Gatekeeper Check
cells.append(clean_cell(make_cell("markdown", """### 步骤 2：硬件环境门禁检测（阻断 T4，验证 L4/A100）
- 若检测到 T4（算力 7.5），会自动触发关机拦截，保护额度。""")))

cells.append(clean_cell(make_cell("code", """import torch, shutil
from google.colab import runtime

if not torch.cuda.is_available():
    print("❌ 未检测到 GPU！")
    runtime.unassign()
    raise RuntimeError("请在菜单：代码执行程序 -> 更改运行时类型 中分配 GPU (首选 L4)。")

gpu_name = torch.cuda.get_device_name(0)
major, minor = torch.cuda.get_device_capability(0)
cc = major + minor / 10.0
vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
disk_free_gb = shutil.disk_usage('/content').free / (1024 ** 3)

print(f"🖥️ 当前显卡: {gpu_name} (算力 {cc:.1f})")
print(f"💾 显存容量: {vram_gb:.2f} GB | 💽 可用磁盘: {disk_free_gb:.1f} GB")

if cc < 8.0:
    print(f"\\n🚨 致命阻断：{gpu_name} 算力 {cc:.1f} < 8.0，不支持 Int8 ConvRot 算子！")
    print("🛑 正在自动销毁实例，防止白白消耗算力单元...")
    runtime.unassign()
    raise RuntimeError(f"硬件不达标 ({gpu_name})，已自动断开。请重新分配 L4 实例。")

print("\\n🎉 硬件门禁通过（≥ 8.0），允许执行！")""")))

# Step 3: Install ComfyUI & Setup Workspace
step3_code = f'''%cd /content

# 1. 安装最新版 ComfyUI
if not os.path.exists('/content/ComfyUI'):
    print("📦 正在安装 ComfyUI 核心环境...")
    !git clone https://github.com/comfyanonymous/ComfyUI.git /content/ComfyUI
    %cd /content/ComfyUI
    !pip install -q -r requirements.txt
    !pip install -q huggingface_hub accelerate safetensors av
    %cd /content

# 2. 准备项目工作空间 (/content/minimax-h3-colab-workflow)
import os, sys, json
from pathlib import Path

REPO_DIR = Path("/content/minimax-h3-colab-workflow")
if not (REPO_DIR / "scripts" / "batch_runner.py").exists():
    print("📦 尝试克隆项目仓库...")
    !git clone https://github.com/kuhung/minimax-h3-colab-workflow.git /content/minimax-h3-colab-workflow 2>/dev/null || true

# 3. 自包含兜底初始化：即使仅单文件上传，也能自动写入内置的标准工作流与脚本
(REPO_DIR / "workflows").mkdir(parents=True, exist_ok=True)
(REPO_DIR / "configs").mkdir(parents=True, exist_ok=True)
(REPO_DIR / "scripts").mkdir(parents=True, exist_ok=True)

embedded_wf_pdd8 = {repr(wf_pdd8_content)}
embedded_wf_baseline = {repr(wf_baseline_content)}
embedded_prompts = {repr(prompts_content)}
embedded_story = {repr(story_content)}
embedded_batch_runner = {repr(batch_runner_content)}

if not (REPO_DIR / "workflows" / "h3_t2v_pdd8_production.json").exists():
    (REPO_DIR / "workflows" / "h3_t2v_pdd8_production.json").write_text(embedded_wf_pdd8, encoding="utf-8")
    print("✅ 已自包含写入: workflows/h3_t2v_pdd8_production.json")

if not (REPO_DIR / "workflows" / "h3_t2v_l4_baseline.json").exists():
    (REPO_DIR / "workflows" / "h3_t2v_l4_baseline.json").write_text(embedded_wf_baseline, encoding="utf-8")
    print("✅ 已自包含写入: workflows/h3_t2v_l4_baseline.json")

if not (REPO_DIR / "configs" / "prompts.example.json").exists():
    (REPO_DIR / "configs" / "prompts.example.json").write_text(embedded_prompts, encoding="utf-8")
    print("✅ 已自包含写入: configs/prompts.example.json")

if not (REPO_DIR / "configs" / "story_cyber_dawn.json").exists():
    (REPO_DIR / "configs" / "story_cyber_dawn.json").write_text(embedded_story, encoding="utf-8")
    print("✅ 已自包含写入: configs/story_cyber_dawn.json")

if not (REPO_DIR / "scripts" / "batch_runner.py").exists():
    (REPO_DIR / "scripts" / "batch_runner.py").write_text(embedded_batch_runner, encoding="utf-8")
    print("✅ 已自包含写入: scripts/batch_runner.py")

print("✅ 项目运行环境与工作流模板准备就绪: /content/minimax-h3-colab-workflow")'''

cells.append(clean_cell(make_cell("markdown", """### 步骤 3：获取项目脚本与安装 ComfyUI 核心运行环境
- 自动准备 ComfyUI 运行环境。
- 具备完全自包含兜底：若未联网克隆仓库，自动嵌入标准生产工作流与批处理模块，确保单文件上传也能 100% 运行。""")))

cells.append(clean_cell(make_cell("code", step3_code)))

# Step 4: Download or link model weights
cells.append(clean_cell(make_cell("markdown", """### 步骤 4：下载或软链接 MiniMax H3 剪枝权重（校验完整性与 Drive 缓存）
- 优先检索 Drive 缓存；若文件受损自动重新拉取；仅同步完整文件。""")))

cells.append(clean_cell(make_cell("code", """from huggingface_hub import hf_hub_download
from pathlib import Path
import os, shutil

# 官方公开免鉴权仓库: Comfy-Org/MiniMax-H3
REPO_ID = "Comfy-Org/MiniMax-H3"

# 极简剪枝量化最小集 (支持 L4 24GB 低显存运行)
models_to_fetch = [
    ("diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors", "diffusion_models", 18.0),
    ("text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "text_encoders", 13.5),
    ("vae/minimax_h3_video_vae_int8_convrot.safetensors", "vae", 2.4),
    ("vae/minimax_h3_audio_vae_fp32.safetensors", "vae", 0.5),
    ("loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors", "loras", 1.6),
]

drive_base = Path("/content/drive/MyDrive/MiniMax_Models")
comfy_base = Path("/content/ComfyUI/models")

for rel_path, folder, min_size_gb in models_to_fetch:
    filename = Path(rel_path).name
    target_dir = comfy_base / folder
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / filename

    drive_dir = drive_base / folder
    drive_dir.mkdir(parents=True, exist_ok=True)
    drive_file = drive_dir / filename
    min_bytes = int(min_size_gb * (1024 ** 3))

    # 1. 优先命中 Google Drive 完整缓存 (秒级软链接)
    if drive_file.exists() and drive_file.stat().st_size >= min_bytes:
        print(f"⚡ 命中 Google Drive 完整缓存，软链接: {filename}")
        if target_file.exists() or target_file.is_symlink():
            target_file.unlink()
        target_file.symlink_to(drive_file.resolve())
    elif target_file.exists() and target_file.stat().st_size >= min_bytes:
        print(f"✅ 本地已存在完整文件: {filename}")
    else:
        print(f"⬇️ 正在从 HuggingFace ({REPO_ID}) 下载: {rel_path} ...")
        downloaded = hf_hub_download(
            repo_id=REPO_ID,
            filename=rel_path,
            local_dir="/content/ComfyUI/models",
            local_dir_use_symlinks=False
        )
        dl_path = Path(downloaded)
        if dl_path.resolve() != target_file.resolve() and dl_path.exists():
            shutil.move(str(dl_path), str(target_file))

        if drive_base.exists() and target_file.exists() and target_file.stat().st_size >= min_bytes:
            print(f"💾 原子备份至 Drive 缓存目录: {filename} ...")
            tmp_drive = drive_file.with_suffix(drive_file.suffix + '.tmp')
            shutil.copy2(target_file, tmp_drive)
            tmp_drive.replace(drive_file)
            print(f"✅ Drive 备份完成: {filename}")

# 建立向后兼容别名软链接
aliases = [
    (comfy_base / "vae" / "minimax_h3_vae.safetensors", comfy_base / "vae" / "minimax_h3_video_vae_int8_convrot.safetensors"),
    (comfy_base / "loras" / "minimax_h3_fl2va_pdd_8step_pruned_comfy.safetensors", comfy_base / "loras" / "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"),
]
for alias_path, src_path in aliases:
    if src_path.exists() and not alias_path.exists():
        alias_path.symlink_to(src_path)
        print(f"🔗 建立兼容别名: {alias_path.name} -> {src_path.name}")

print("\\n🎉 所有模型权重配置与缓存校验完毕！")""")))

# Step 5: ComfyUI daemon startup (deadlock-free)
cells.append(clean_cell(make_cell("markdown", """### 步骤 5：启动 ComfyUI 守护服务（防死锁日志重定向与就绪轮询）
- 严禁使用 `subprocess.PIPE` 避免缓冲区满死锁；将日志直接输出至 `/content/comfyui.log`。
- 轮询 120 秒检测服务状态，若提前退出自动打印末尾日志进行诊断。""")))

cells.append(clean_cell(make_cell("code", """import subprocess, time, urllib.request, os

%cd /content/ComfyUI
# 启动 ComfyUI 守护服务 (--lowvram 保证 L4 24GB 稳定卸载)
# 必须将 stdout/stderr 重定向到文件，严禁使用 subprocess.PIPE 避免缓冲区死锁
log_file = open('/content/comfyui.log', 'w', encoding='utf-8')
comfy_proc = subprocess.Popen(
    ["python", "main.py", "--listen", "127.0.0.1", "--port", "8188", "--lowvram"],
    stdout=log_file,
    stderr=subprocess.STDOUT,
    start_new_session=True
)
print(f"🚀 ComfyUI 后台服务已拉起 (PID: {comfy_proc.pid})，日志输出至 /content/comfyui.log")

# 轮询本地接口就绪状态 (最多等待 120s)
ready = False
for i in range(60):
    if comfy_proc.poll() is not None:
        print(f"\\n❌ ComfyUI 进程提前异常退出，退出码: {comfy_proc.returncode}")
        break
    try:
        req = urllib.request.Request("http://127.0.0.1:8188/system_stats")
        with urllib.request.urlopen(req, timeout=2) as r:
            if r.status == 200:
                print("✅ ComfyUI 本地 API 就绪 (127.0.0.1:8188)！")
                ready = True
                break
    except Exception:
        pass
    time.sleep(2)

if not ready:
    log_file.flush()
    tail = ""
    if os.path.exists('/content/comfyui.log'):
        with open('/content/comfyui.log', 'r', encoding='utf-8', errors='replace') as f:
            tail = "".join(f.readlines()[-50:])
    print("\\n" + "=" * 65)
    print(f"❌ ComfyUI 启动失败或超时，诊断日志如下:\\n{tail}")
    print("=" * 65)
    raise RuntimeError("ComfyUI 未能在 120 秒内就绪，请核对上方日志。")""")))

# Step 6: Real Benchmark Smoke Test
cells.append(clean_cell(make_cell("markdown", """### 步骤 6：实测 Benchmark 验证（真实冒烟测试：测算单帧耗时与 CU 消耗率）
- 真正向 ComfyUI `/prompt` 提交 1 条短镜头测试任务（960x544, 61帧, 8步），轮询直到出片。
- **跨进程全局显存采样**：通过 CUDA 驱动级接口 `torch.cuda.mem_get_info()` 与 ComfyUI `/system_stats` 实时跟踪整卡显存峰值；若遇超时自动下发 `/interrupt` 释放 GPU 显存。""")))

cells.append(clean_cell(make_cell("code", """import json, time, shutil, urllib.request, urllib.error, os
from pathlib import Path
import torch

def interrupt_comfyui(server_url="http://127.0.0.1:8188"):
    try:
        ireq = urllib.request.Request(f"{server_url}/interrupt", data=b"{}", headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(ireq, timeout=5) as r:
            pass
        print("🛑 已下发 /interrupt 指令中断任务并释放 GPU 显存。")
    except Exception as e:
        print(f"⚠️ 下发 /interrupt 异常: {e}")

print("=" * 65)
print("🔍 执行 MiniMax H3 真实冒烟基准评测 (Smoke Benchmark)")
print("=" * 65)

# 加载工作流模板
wf_path = Path("/content/minimax-h3-colab-workflow/workflows/h3_t2v_pdd8_production.json")
if not wf_path.exists():
    wf_path = Path("/content/minimax-h3-colab-workflow/workflows/h3_t2v_l4_baseline.json")

with open(wf_path, "r", encoding="utf-8") as f:
    wf = json.load(f)

# 配置冒烟规格: 960x544, 124 帧, 8 步, seed 42
smoke_task_id = "benchmark_smoke_01"
smoke_seed = 42
smoke_steps = 8
smoke_width = 960
smoke_height = 544
smoke_length = 124
smoke_prompt = "A cinematic macro shot of an espresso machine brewing rich coffee into a glass cup. Slow motion droplets forming. Ambient steam hissing, smooth mechanical pump humming, gentle liquid dripping acoustic sound."

for node_id, node in wf.items():
    ctype = node.get("class_type")
    inputs = node.get("inputs", {})
    tmeta = node.get("_meta", {}).get("title", "")
    if ctype == "CLIPTextEncode" and (tmeta.startswith("Positive") or "Positive" in tmeta):
        inputs["text"] = smoke_prompt
    elif ctype in ("EmptyMiniMaxH3LatentAV", "EmptyH3LatentVideo", "EmptyLatentVideo"):
        inputs["width"] = smoke_width
        inputs["height"] = smoke_height
        inputs["length"] = smoke_length
    elif ctype == "KSampler":
        inputs["steps"] = smoke_steps
        inputs["seed"] = smoke_seed
        inputs["cfg"] = 3.0
    elif ctype in ("SaveAnimatedWEBP", "VHS_VideoCombine"):
        inputs["filename_prefix"] = f"{smoke_task_id}_{smoke_seed}"
        if ctype == "SaveAnimatedWEBP":
            inputs.setdefault("fps", 24.0)
            inputs.setdefault("lossless", False)
            inputs.setdefault("quality", 85)
            inputs.setdefault("method", "default")

print(f"🚀 提交冒烟测试任务至 ComfyUI API (960x544, {smoke_length} 帧, {smoke_steps} 步)...")
t_start = time.time()
prompt_data = json.dumps({"prompt": wf}).encode("utf-8")
req = urllib.request.Request("http://127.0.0.1:8188/prompt", data=prompt_data, headers={"Content-Type": "application/json"})

try:
    with urllib.request.urlopen(req, timeout=15) as resp:
        res_json = json.loads(resp.read().decode("utf-8"))
        prompt_id = res_json.get("prompt_id")
        print(f"✅ 冒烟任务提交成功，Prompt ID: {prompt_id}")
except urllib.error.HTTPError as e:
    err_body = e.read().decode("utf-8", errors="replace")
    print(f"\\n❌ ComfyUI 返回 HTTP {e.code} 校验失败，详细 JSON 报错:")
    print(err_body)
    raise RuntimeError(f"ComfyUI 节点参数或权重校验未通过: HTTP {e.code}")

# 轮询 /history/{prompt_id} 直到完成，并在等待期间采样整卡真实显存峰值
history_data = None
last_log = 0
peak_used_gb = 0.0

while time.time() - t_start < 600:
    # 采样 CUDA 驱动级整卡显存 (跨进程统计 ComfyUI 真实显存)
    curr_used_gb = 0.0
    if torch.cuda.is_available():
        try:
            free_b, total_b = torch.cuda.mem_get_info(0)
            curr_used_gb = (total_b - free_b) / (1024**3)
            if curr_used_gb > peak_used_gb:
                peak_used_gb = curr_used_gb
        except Exception:
            pass

    try:
        with urllib.request.urlopen(f"http://127.0.0.1:8188/history/{prompt_id}", timeout=5) as resp:
            hist = json.loads(resp.read().decode("utf-8"))
            if prompt_id in hist:
                history_data = hist[prompt_id]
                break
    except Exception:
        pass

    elapsed = int(time.time() - t_start)
    if elapsed - last_log >= 15:
        print(f"   ⏳ 冒烟去噪中... 已耗时 {elapsed}s | 当前整卡已用显存: {curr_used_gb:.2f} GB (峰值: {peak_used_gb:.2f} GB)")
        last_log = elapsed
    time.sleep(3)

cost_sec = round(time.time() - t_start, 2)
if not history_data:
    print("❌ 冒烟任务超时 (600s)，正在下发中断指令以释放显存...")
    interrupt_comfyui("http://127.0.0.1:8188")
    raise TimeoutError("冒烟任务超时 (600s)，未在规定时间内完成！已下发中断指令释放 GPU。")

# 状态检查
status_info = history_data.get("status", {})
if status_info.get("status_str") != "success":
    raise RuntimeError(f"冒烟测试执行失败: {status_info.get('messages')}")

avg_step_sec = round(cost_sec / smoke_steps, 2)
print("\\n" + "=" * 65)
print("📊 [MiniMax H3 冒烟评测报告]")
print(f"⏱️ 端到端总耗时: {cost_sec} 秒 ({cost_sec/60:.2f} 分钟)")
print(f"⚡ 单步平均耗时: {avg_step_sec} 秒/步 ({smoke_steps} 步去噪)")

# 显存状态查询与系统资源报告
if torch.cuda.is_available():
    try:
        free_b, total_b = torch.cuda.mem_get_info(0)
        curr_used_gb = (total_b - free_b) / (1024**3)
        total_gb = total_b / (1024**3)
        print(f"💾 GPU 显存总量: {total_gb:.2f} GB")
        print(f"💾 整卡显存监控: 去噪峰值 {peak_used_gb:.2f} GB | 当前占用 {curr_used_gb:.2f} GB | 剩余可用 {free_b / (1024**3):.2f} GB")
    except Exception as e:
        print(f"⚠️ 读取 CUDA mem_get_info 失败: {e}")

try:
    with urllib.request.urlopen("http://127.0.0.1:8188/system_stats", timeout=3) as stats_resp:
        stats_data = json.loads(stats_resp.read().decode("utf-8"))
        devices = stats_data.get("devices", [])
        if devices:
            dev = devices[0]
            dev_name = dev.get("name", "GPU 0")
            c_vram_free = dev.get("vram_free", 0) / (1024**3)
            c_vram_total = dev.get("vram_total", 0) / (1024**3)
            c_torch_total = dev.get("torch_vram_total", 0) / (1024**3)
            print(f"🖥️ ComfyUI 设备视图 ({dev_name}): 显存总量 {c_vram_total:.2f} GB, 空闲 {c_vram_free:.2f} GB (PyTorch 预留: {c_torch_total:.2f} GB)")
except Exception:
    pass

# 提取并验证生成文件
comfy_out = Path("/content/ComfyUI/output")
drive_out = Path("/content/drive/MyDrive/MiniMax_Outputs")
drive_out.mkdir(parents=True, exist_ok=True)

found_files = []
# 优先从 outputs 解析
for node_out in history_data.get("outputs", {}).values():
    for cat in ["images", "videos", "gifs", "audio"]:
        for item in node_out.get(cat, []):
            fn = item.get("filename")
            subfolder = item.get("subfolder", "")
            target_f = comfy_out / subfolder / fn if fn else None
            if target_f and target_f.exists():
                found_files.append(target_f)

if not found_files and comfy_out.exists():
    found_files = list(comfy_out.glob(f"{smoke_task_id}_{smoke_seed}*"))

if not found_files:
    raise RuntimeError("❌ 冒烟测试执行完毕，但在输出目录未找到任何产物文件！")

saved_smoke_files = []
for f in found_files:
    dst = drive_out / f.name
    shutil.copy2(f, dst)
    saved_smoke_files.append(dst)
    assert dst.exists(), f"文件落盘校验失败: {dst}"
    print(f"💾 产物已同步至 Google Drive 并校验存在: {dst.name} ({dst.stat().st_size / (1024**2):.2f} MB)")

print(f"🎉 冒烟测试 100% 成功闭环！")
print("=" * 65)""")))

# Step 7: Headless batch queue
cells.append(clean_cell(make_cell("markdown", """### 步骤 7：无头批量生产队列（自动出片 + 自动同步 Drive）
- 支持音画同频描述，成片与结构化日志自动保存至 `/content/drive/MyDrive/MiniMax_Outputs`。""")))

cells.append(clean_cell(make_cell("code", """import os, sys
from pathlib import Path

repo_path = Path("/content/minimax-h3-colab-workflow")
if str(repo_path) not in sys.path:
    sys.path.insert(0, str(repo_path))

from scripts.batch_runner import run_batch

# 默认排产: configs/story_cyber_dawn.json (完整 8 分镜短视频案例) 或 configs/prompts.example.json (测试集)
prompts_file = repo_path / "configs" / "story_cyber_dawn.json"
if not prompts_file.exists():
    prompts_file = repo_path / "configs" / "prompts.example.json"

print(f"🎬 准备启动批处理队列，使用排期文件: {prompts_file.name}")

run_batch(
    workflow_path=str(repo_path / "workflows" / "h3_t2v_pdd8_production.json"),
    prompts_path=str(prompts_file),
    output_dir="/content/drive/MyDrive/MiniMax_Outputs",
    server_url="http://127.0.0.1:8188",
    comfy_output_dir="/content/ComfyUI/output",
    timeout_per_shot=900
)""")))

# Step 8: Safe shutdown
cells.append(clean_cell(make_cell("markdown", """### 步骤 8：防扣费与安全收尾（跑完务必执行）
> ⚠️ **重要**：运行下方单元格立即刷盘并销毁实例，彻底切断算力扣费！""")))

cells.append(clean_cell(make_cell("code", """import os
from google.colab import runtime

print("🛑 正在执行安全收尾：刷盘并销毁 Colab 运行时，彻底终止算力计费...")
try:
    os.sync()
    runtime.unassign()
    print("✅ runtime.unassign() 已成功下发！")
except Exception as e:
    print(f"⚠️ runtime.unassign() 异常: {e}，触发容器级兜底强制断开...")
    try:
        os.sync()
        os.kill(1, 15)
    except Exception:
        os.system("kill -9 1")""")))

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "gpuType": "L4",
            "provenance": []
        },
        "kernelspec": {
            "display_name": "Python 3",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 0
}

output_path = PROJECT_ROOT / "notebooks" / "minimax_h3_l4_production.ipynb"
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print("Notebook generated successfully at", output_path)
