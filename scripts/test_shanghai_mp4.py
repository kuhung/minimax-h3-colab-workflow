#!/usr/bin/env python3
"""
Dedicated Small-Scale Test Runner: Shanghai Street Scene (10-20s Audio MP4)
Demonstrates Agent/Colab autonomous scheduling with audio-visual generation:
- Hardware Gatekeeper (L4 GPU validation)
- Drive Cache Fast Symlinking & Adaptive Weight Selection (cu130 vs cu12x)
- Background ComfyUI daemon lifecycle management
- Native 32kHz Stereo Audio MP4 generation via BasicGuider (CFG=1.0) 2.0 topology
- Watchdog & Auto-shutdown to prevent idle compute unit drain
"""

import os
import sys
import time
import json
import shutil
import argparse
import subprocess
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.check_gpu import check_system
from scripts.download_models import download_or_link
from scripts.batch_runner import run_batch
from scripts.agent_colab_runner import (
    ensure_comfyui_installed,
    ensure_comfyui_running,
    safe_colab_shutdown,
    start_watchdog_timer,
)

def run_shanghai_test(
    shots: int = 2,
    auto_shutdown: bool = False,
    workflow_path: str = "workflows/h3_t2v_pdd8_production.json",
    prompts_path: str = "configs/shanghai_street.json",
    output_dir: str = "/content/drive/MyDrive/MiniMax_Outputs/shanghai_street",
    comfy_root: str = "/content/ComfyUI",
    drive_cache: str = "/content/drive/MyDrive/MiniMax_Models",
    watchdog_hours: float = 1.0,
    variant: str = "auto",
    turbo_steps: int = 8,
):
    print("=" * 68)
    print("🏙️ MiniMax H3 自动化能力测试: 上海街景原生音画 MP4 出片 (2.0)")
    print(f"🎬 规划镜头数: {shots} 镜 (每镜 124 帧 @ 24fps ≈ 5.16s | 总片长 ≈ {shots * 5.16:.1f}s)")
    print(f"⏱️ 预计耗时: ~{shots * 5.1:.1f} 分钟 | 预计消耗算力: ~{shots * 0.16:.2f} CU")
    print(f"🔊 音频规格: 32kHz 原生双声道立体声 (Audio VAE FP32)")
    print(f"📦 封装格式: MP4 (H.264 / AAC)")
    print(f"🎯 量化模式: {variant} | Turbo LoRA: {turbo_steps}-Step")
    print(f"🛑 自动关机防扣费: {'已开启 (生成完毕立即断开 runtime)' if auto_shutdown else '未开启 (手动断开)'}")
    print("=" * 68)

    # 1. 守门员定时器（最长 1 小时硬熔断）
    if watchdog_hours > 0:
        start_watchdog_timer(watchdog_hours, auto_shutdown=auto_shutdown)

    # 2. 硬件环境检测
    print("\n[步骤 1/5] 校验硬件环境 (L4 GPU 门禁)...")
    passed, details = check_system(auto_shutdown_on_fail=False, verbose=True)
    if not passed:
        raise RuntimeError(f"❌ 硬件门禁拦截: {details.get('failure_reason')}")

    # 3. 依赖包检查 (确保 PyAV 已安装，用于 MP4 封装)
    print("\n[步骤 2/5] 检查音视频合成依赖 (PyAV)...")
    try:
        import av
        print(f"✅ PyAV 已就绪 (版本: {av.__version__})")
    except ImportError:
        print("📦 正在安装 PyAV 依赖包 (用于 CreateVideo 音视频合流)...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "av"], check=True)
        print("✅ PyAV 安装完成！")

    comfy_path = Path(comfy_root)
    drive_cache_path = Path(drive_cache) if drive_cache else None

    # 4. 挂载模型软链接
    print("\n[步骤 3/5] 挂载 Drive 缓存模型软链接...")
    ensure_comfyui_installed(comfy_path)
    active_files = download_or_link(
        comfy_path,
        drive_cache_path,
        variant=variant,
        turbo_steps=turbo_steps
    )

    # 检查 Audio VAE 软链接就绪情况
    audio_vae_file = comfy_path / "models" / "vae" / "minimax_h3_audio_vae_fp32.safetensors"
    if audio_vae_file.exists():
        print(f"✅ 32kHz Audio VAE 就绪: {audio_vae_file.name}")
    else:
        print(f"⚠️ 警告: 未在 {audio_vae_file} 发现音频 VAE 权重，请核实下载列表！")

    # 5. 启动 ComfyUI 守护进程
    print("\n[步骤 4/5] 启动 ComfyUI 后台服务...")
    ensure_comfyui_running(comfy_path)

    # 6. 读取 prompts 并截取指定镜头数
    with open(prompts_path, "r", encoding="utf-8") as f:
        all_tasks = json.load(f)

    target_tasks = all_tasks[:shots]
    print(f"\n[步骤 5/5] 开始执行上海街景出片队列 (共 {len(target_tasks)} 镜)...")
    for i, t in enumerate(target_tasks, 1):
        print(f"   [{i}/{len(target_tasks)}] {t.get('title')} (ID: {t.get('id')})")

    t_batch_start = time.time()
    try:
        run_batch(
            workflow_path=workflow_path,
            prompts_path=None,
            output_dir=output_dir,
            server_url="http://127.0.0.1:8188",
            comfy_output_dir=str(comfy_path / "output"),
            timeout_per_shot=600,
            tasks_list=target_tasks,
            active_models=active_files,
        )
    finally:
        total_batch_time = round(time.time() - t_batch_start, 2)
        print("\n" + "=" * 68)
        print("📊 批处理运行总结报告:")
        print(f"⏱️ 实际总耗时: {total_batch_time}s ({total_batch_time / 60:.2f} 分钟)")
        print(f"⚡ 折合算力消耗: ~{1.9 * (total_batch_time / 3600):.3f} CU")

        out_path = Path(output_dir)
        if out_path.exists():
            mp4_files = list(out_path.glob("*.mp4"))
            print(f"🎬 生成的 MP4 成片清单 ({len(mp4_files)} 个文件):")
            for mp4 in mp4_files:
                size_mb = mp4.stat().st_size / (1024 * 1024)
                print(f"   🎥 {mp4.name} ({size_mb:.2f} MB)")
        print("=" * 68)

        if auto_shutdown:
            print("\n🛑 自动化调度收尾：触发 Colab 运行时自毁销毁，停止扣费...")
            safe_colab_shutdown()
        else:
            print("\n💡 提示：未开启 --auto-shutdown，请及时在 Colab 侧边栏断开运行时！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Shanghai Street Scene with Audio MP4")
    parser.add_argument("--shots", type=int, default=2, choices=[1, 2, 3], help="Number of shots to generate (1=5s, 2=10s, 3=15s)")
    parser.add_argument("--auto-shutdown", action="store_true", help="Automatically unassign Colab runtime after batch finishes")
    parser.add_argument("--workflow", default="workflows/h3_t2v_pdd8_production.json", help="Path to workflow JSON")
    parser.add_argument("--prompts", default="configs/shanghai_street.json", help="Path to prompts JSON")
    parser.add_argument("--output", default="/content/drive/MyDrive/MiniMax_Outputs/shanghai_street", help="Output directory in Drive")
    parser.add_argument("--comfy-root", default="/content/ComfyUI", help="ComfyUI root path")
    parser.add_argument("--drive-cache", default="/content/drive/MyDrive/MiniMax_Models", help="Drive models cache")
    parser.add_argument("--watchdog-hours", type=float, default=1.0, help="Watchdog timeout limit in hours")
    parser.add_argument("--variant", type=str, default="auto", choices=["auto", "int8_convrot", "fp8_scaled"], help="Quantization variant")
    parser.add_argument("--turbo-steps", type=int, default=8, choices=[4, 8], help="Turbo LoRA step variant (4 or 8)")
    args = parser.parse_args()

    run_shanghai_test(
        shots=args.shots,
        auto_shutdown=args.auto_shutdown,
        workflow_path=args.workflow,
        prompts_path=args.prompts,
        output_dir=args.output,
        comfy_root=args.comfy_root,
        drive_cache=args.drive_cache,
        watchdog_hours=args.watchdog_hours,
        variant=args.variant,
        turbo_steps=args.turbo_steps,
    )
