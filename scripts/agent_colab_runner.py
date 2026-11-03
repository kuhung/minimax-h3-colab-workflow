#!/usr/bin/env python3
"""
Agent-Driven Headless Colab Runner for MiniMax H3.
Designed for autonomous execution via Colab MCP or terminal.
Orchestrates:
1. Hardware Gatekeeper Check (Blocks T4, approves L4/A100)
2. Auto-Install ComfyUI & Dependencies (if fresh instance)
3. Model Cache & Drive Symlinking (with integrity verification)
4. Background ComfyUI Daemon Lifecycle
5. Prompt Queue Batch Rendering & Drive Sync
6. Triple-layer Auto-Shutdown & Watchdog (safeguards monthly 200 compute units)
"""

import os
import sys
import time
import argparse
import subprocess
import urllib.request
import json
import threading
from pathlib import Path
from typing import Optional

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.check_gpu import check_system
from scripts.download_models import download_or_link
from scripts.batch_runner import run_batch

def safe_colab_shutdown():
    """Attempts to unassign and terminate the Colab instance to prevent CU drain."""
    print("\n" + "!" * 65)
    print("🛑 正在触发 Colab 实例销毁 (runtime.unassign)，坚决杜绝闲置扣费！")
    print("!" * 65)
    in_colab = Path("/content").exists()
    try:
        from google.colab import runtime
        runtime.unassign()
        print("✅ runtime.unassign() 已成功下发！实例将在数秒内断开。")
    except ImportError:
        print("ℹ️ 当前非 Colab 环境（未找到 google.colab），跳过 runtime.unassign()。")
        if in_colab:
            print("⚠️ 检测到处于 Colab 容器环境 (/content)，执行容器级关机兜底...")
            try:
                os.sync()
                print("🛑 刷盘完成，向容器初始进程 (PID 1) 发送 SIGTERM 终止信号...")
                os.kill(1, 15)
            except Exception as kill_err:
                print(f"⚠️ os.kill(1, 15) 失败: {kill_err}，尝试 kill -9 1")
                os.system("kill -9 1")
    except Exception as e:
        print(f"⚠️ runtime.unassign() 调用异常: {e}。")
        if in_colab:
            print("⚠️ 当前处于 Colab 容器环境，执行容器级关机兜底 (os.sync() + 强制终止 PID 1)...")
            try:
                os.sync()
                print("🛑 刷盘完成，向容器初始进程 (PID 1) 发送 SIGTERM 终止信号...")
                os.kill(1, 15)
            except Exception as kill_err:
                print(f"⚠️ os.kill(1, 15) 失败: {kill_err}，尝试 kill -9 1")
                os.system("kill -9 1")
        else:
            print("💡 请在 Colab 网页端手动断开运行时！")

def start_watchdog_timer(timeout_hours: float, auto_shutdown: bool = True):
    """Starts background watchdog thread. Force-quits if runtime exceeds hard threshold."""
    if timeout_hours <= 0:
        return
    timeout_seconds = int(timeout_hours * 3600)
    def watchdog_loop():
        time.sleep(timeout_seconds)
        print("\n" + "!" * 70)
        print(f"⏰ [WATCHDOG 熔断] 任务总执行时间已超硬性上限 {timeout_hours} 小时 ({timeout_seconds}s)！")
        print("⏰ [WATCHDOG 熔断] 触发紧急销毁以保卫算力单元！")
        print("!" * 70)
        if auto_shutdown:
            safe_colab_shutdown()
        os._exit(124)

    t = threading.Thread(target=watchdog_loop, daemon=True, name="CU_Watchdog")
    t.start()
    print(f"🛡️  [WATCHDOG 守门员已启动] 超时上限: {timeout_hours} 小时，超时将强制自动关机熔断。")

def wait_for_comfyui(server_url: str = "http://127.0.0.1:8188", timeout: int = 90) -> bool:
    """Polls ComfyUI system stats until server is ready."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            req = urllib.request.Request(f"{server_url}/system_stats")
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False

def ensure_comfyui_installed(comfy_root: Path):
    """Installs ComfyUI and base dependencies if not present."""
    if not (comfy_root / "main.py").exists():
        print(f"📦 未在 {comfy_root} 检测到 ComfyUI，正在克隆官方仓库...")
        comfy_root.parent.mkdir(parents=True, exist_ok=True)
        res = subprocess.run(
            ["git", "clone", "https://github.com/comfyanonymous/ComfyUI.git", str(comfy_root)],
            capture_output=True,
            text=True
        )
        if res.returncode != 0:
            raise RuntimeError(f"Git clone ComfyUI 失败: {res.stderr}")

        print("📦 正在安装 ComfyUI 基础依赖包...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-r", str(comfy_root / "requirements.txt")],
            check=True
        )
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "huggingface_hub", "accelerate", "safetensors"],
            check=True
        )
        print("✅ ComfyUI 仓库与核心依赖安装完成！")
    else:
        print(f"✅ 检测到 ComfyUI 目录就绪: {comfy_root}")

def ensure_comfyui_running(comfy_root: Path, server_url: str = "http://127.0.0.1:8188") -> Optional[subprocess.Popen]:
    """Ensures ComfyUI daemon is active. Launches in background if needed."""
    if wait_for_comfyui(server_url, timeout=3):
        print("✅ ComfyUI 守护进程已处于运行就绪状态。")
        return None

    print("🚀 正在后台启动 ComfyUI 守护服务 (--lowvram)...")
    log_path = Path("/content/comfyui.log") if Path("/content").exists() else Path("comfyui.log")
    log_file = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        ["python", "main.py", "--listen", "127.0.0.1", "--port", "8188", "--lowvram"],
        cwd=str(comfy_root),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True
    )

    # 轮询检查 ComfyUI 是否在 120s 内就绪，或子进程提前异常退出
    start_time = time.time()
    ready = False
    while time.time() - start_time < 120:
        if proc.poll() is not None:
            print(f"❌ ComfyUI 子进程提前异常退出，退出码: {proc.returncode}")
            break
        if wait_for_comfyui(server_url, timeout=2):
            ready = True
            break

    if ready:
        print("✅ ComfyUI 后台服务启动成功并已就绪 (127.0.0.1:8188)！")
        return proc
    else:
        log_file.flush()
        log_tail = ""
        if log_path.exists():
            try:
                with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()
                    log_tail = "".join(lines[-50:])
            except Exception as read_err:
                log_tail = f"(读取 comfyui.log 失败: {read_err})"

        print("\n" + "=" * 65)
        print("❌ ComfyUI 服务启动失败/超时诊断 (comfyui.log 最近 50 行):")
        print(log_tail if log_tail else "(日志内容为空)")
        print("=" * 65 + "\n")
        raise RuntimeError("ComfyUI 服务在 120 秒内未就绪或子进程提前退出，请检查上方日志。")

def run_agent_workflow(
    workflow_path: str,
    prompts_path: str,
    output_dir: str,
    comfy_root: str = "/content/ComfyUI",
    drive_cache: str = "/content/drive/MyDrive/MiniMax_Models",
    auto_shutdown: bool = False,
    watchdog_hours: float = 2.0,
    timeout_per_shot: int = 900,
    variant: str = "auto",
    turbo_steps: int = 8,
):
    print("=" * 65)
    print("🤖 MiniMax H3 Agent 自动化托管作业启动 (Colab MCP / Headless 2.0)")
    print("=" * 65)

    if watchdog_hours > 0:
        start_watchdog_timer(watchdog_hours, auto_shutdown=auto_shutdown)

    success = False
    try:
        # 1. 硬件门禁检测
        print("\n[阶段 1/5] 校验硬件环境门禁...")
        passed, details = check_system(auto_shutdown_on_fail=False, verbose=True)
        if not passed:
            raise RuntimeError(f"硬件门禁拦截: {details.get('failure_reason')}")

        comfy_path = Path(comfy_root)
        drive_cache_path = Path(drive_cache) if drive_cache else None

        # 2. 安装与挂载
        print("\n[阶段 2/5] 准备 ComfyUI 环境与模型权重...")
        ensure_comfyui_installed(comfy_path)
        download_or_link(
            comfy_path,
            drive_cache_path,
            variant=variant,
            turbo_steps=turbo_steps
        )

        # 3. 启动守护进程
        print("\n[阶段 3/5] 检查与启动 ComfyUI 后台服务...")
        ensure_comfyui_running(comfy_path)

        # 4. 执行批量渲染
        print("\n[阶段 4/5] 执行批量出片生产队列...")
        run_batch(
            workflow_path=workflow_path,
            prompts_path=prompts_path,
            output_dir=output_dir,
            server_url="http://127.0.0.1:8188",
            comfy_output_dir=str(comfy_path / "output"),
            timeout_per_shot=timeout_per_shot
        )

        # 5. 校验成片
        print("\n[阶段 5/5] 校验成片与归档日志...")
        history_log = Path(output_dir) / "render_history.jsonl"
        if history_log.exists():
            with open(history_log, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            print(f"📊 累计生成日志记录: {len(lines)} 条。成片已保存在: {output_dir}")
        else:
            print("⚠️ 未发现 render_history.jsonl 日志，请核对生成状态。")

        success = True

    except Exception as e:
        print(f"\n❌ 工作流执行发生异常中断: {e}")
        raise
    finally:
        # 无论成功或失败，只要开启了 auto_shutdown，都保证执行销毁，防止闲置扣费
        if auto_shutdown:
            safe_colab_shutdown()
        else:
            print("\n💡 提示：未开启 --auto-shutdown。若在 Colab 环境中，请及时手动断开运行时以节省算力！")

    return success

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agent Colab Headless Runner for MiniMax H3")
    parser.add_argument("--workflow", default="workflows/h3_t2v_pdd8_production.json", help="Path to workflow JSON")
    parser.add_argument("--prompts", default="configs/prompts.example.json", help="Path to prompts JSON")
    parser.add_argument("--output", default="/content/drive/MyDrive/MiniMax_Outputs", help="Output directory in Drive")
    parser.add_argument("--comfy-root", default="/content/ComfyUI", help="ComfyUI root path")
    parser.add_argument("--drive-cache", default="/content/drive/MyDrive/MiniMax_Models", help="Drive models cache")
    parser.add_argument("--auto-shutdown", action="store_true", help="Automatically unassign Colab runtime after batch finishes")
    parser.add_argument("--watchdog-hours", type=float, default=2.0, help="Hard watchdog timeout limit in hours (default: 2.0h)")
    parser.add_argument("--timeout-per-shot", type=int, default=900, help="Per-shot generation timeout in seconds")
    parser.add_argument("--variant", type=str, default="auto", choices=["auto", "int8_convrot", "fp8_scaled"], help="Quantization variant")
    parser.add_argument("--turbo-steps", type=int, default=8, choices=[4, 8], help="Turbo LoRA step variant (4 or 8)")
    args = parser.parse_args()

    run_agent_workflow(
        workflow_path=args.workflow,
        prompts_path=args.prompts,
        output_dir=args.output,
        comfy_root=args.comfy_root,
        drive_cache=args.drive_cache,
        auto_shutdown=args.auto_shutdown,
        watchdog_hours=args.watchdog_hours,
        timeout_per_shot=args.timeout_per_shot,
        variant=args.variant,
        turbo_steps=args.turbo_steps,
    )
