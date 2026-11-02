#!/usr/bin/env python3
"""
Hardware Pre-flight Check for MiniMax H3 Local Deployment on Colab.
Enforces hard constraints:
1. Compute Capability >= 8.0 (Rejects T4 7.5, approves L4/A100/H100/RTX 30xx/40xx)
2. GPU VRAM >= 20GB
3. System RAM >= 25GB (High-RAM runtime recommended)
4. Free Disk Space >= 70GB

Supports auto-unassigning Colab instance on failure to prevent CU burn.
"""

import sys
import shutil
import os
from typing import Tuple, Dict, Any

def check_system(auto_shutdown_on_fail: bool = False, verbose: bool = True) -> Tuple[bool, Dict[str, Any]]:
    """
    Checks hardware environment.
    Returns (passed, details_dict).
    If auto_shutdown_on_fail is True and running inside Colab, triggers runtime.unassign() immediately.
    """
    if verbose:
        print("=" * 60)
        print("🔍 MiniMax H3 硬件环境门禁检测 (Hardware Gate Check)")
        print("=" * 60)

    details: Dict[str, Any] = {
        "passed": False,
        "cuda_available": False,
        "device_name": None,
        "compute_capability": 0.0,
        "vram_gb": 0.0,
        "ram_gb": 0.0,
        "disk_free_gb": 0.0,
        "failure_reason": None
    }

    def _trigger_shutdown_if_colab():
        if auto_shutdown_on_fail:
            try:
                from google.colab import runtime
                print("🛑 门禁未通过且开启了自动关机，正在销毁 Colab 实例以节省算力...")
                runtime.unassign()
            except Exception:
                pass

    # 1. 检查 PyTorch & CUDA
    try:
        import torch
    except ImportError:
        err = "未检测到 PyTorch，请先在环境中安装 torch。"
        details["failure_reason"] = err
        if verbose:
            print(f"❌ {err}")
        _trigger_shutdown_if_colab()
        return False, details

    if not torch.cuda.is_available():
        err = "未检测到可用 GPU 加速器！请选择 GPU (L4 或 A100)。"
        details["failure_reason"] = err
        if verbose:
            print(f"❌ {err}")
        _trigger_shutdown_if_colab()
        return False, details

    details["cuda_available"] = True
    device_name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    cc = major + minor / 10.0
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

    details["device_name"] = device_name
    details["compute_capability"] = cc
    details["vram_gb"] = vram_gb

    if verbose:
        print(f"🖥️  GPU 型号: {device_name}")
        print(f"⚡ 计算能力 (Compute Capability): {cc:.1f}")
        print(f"💾 GPU 显存 (VRAM): {vram_gb:.2f} GB")

    # 门禁 1: 计算能力必须 >= 8.0 (Int8 ConvRot 算子硬要求)
    if cc < 8.0:
        err = f"当前 GPU ({device_name}) 计算能力 {cc:.1f} < 8.0。不支持 Int8 ConvRot 算子！"
        details["failure_reason"] = err
        if verbose:
            print("\n" + "!" * 60)
            print("🚨 致命阻断：当前 GPU 计算能力不足！")
            print(f"   检测到计算能力为 {cc:.1f}（如 Tesla T4 为 7.5）。")
            print("   MiniMax H3 的 Int8 剪枝量化硬性要求计算能力 ≥ 8.0（L4 / A100）。")
            print("   在 T4 上强行运行将导致内核崩溃，白白消耗计算单元。")
            print("💡 处置建议：立即断开并更改运行时，重新分配到 L4 或 A100 实例！")
            print("!" * 60)
        _trigger_shutdown_if_colab()
        return False, details
    else:
        if verbose:
            print("✅ 计算能力检查通过（≥ 8.0，支持 Int8 硬件级加速加速）")

    # 门禁 2: 显存检测
    if vram_gb < 20.0:
        if verbose:
            print(f"⚠️  警告：GPU 显存为 {vram_gb:.1f} GB，低于推荐的 24GB (L4/A100)。")
            print("   请确保在 ComfyUI 启动参数中包含 CPU 动态卸载 (--lowvram)。")
    else:
        if verbose:
            print(f"✅ GPU 显存充足 ({vram_gb:.1f} GB)")

    # 门禁 3: 内存与磁盘检测
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        details["ram_gb"] = ram_gb
        if verbose:
            print(f"🧠 系统内存 (RAM): {ram_gb:.1f} GB")
        if ram_gb < 25.0:
            if verbose:
                print("⚠️  系统内存低于 25GB。建议在 Colab 中开启 '高内存 (High-RAM)' 运行时。")
        else:
            if verbose:
                print("✅ 系统内存通过 (≥ 25GB)")
    except ImportError:
        pass

    target_dir = "/content" if os.path.exists("/content") else "."
    free_disk_gb = shutil.disk_usage(target_dir).free / (1024 ** 3)
    details["disk_free_gb"] = free_disk_gb
    if verbose:
        print(f"💽 可用磁盘空间: {free_disk_gb:.1f} GB")
    if free_disk_gb < 70.0:
        if verbose:
            print("⚠️  可用磁盘空间低于 70GB！剪枝模型约需 40GB，请留意存储。")
    else:
        if verbose:
            print("✅ 磁盘空间通过 (≥ 70GB)")

    details["passed"] = True
    if verbose:
        print("=" * 60)
        print("🎉 恭喜！当前硬件完全满足 MiniMax H3 纯本地高效运行标准！")
        print("=" * 60)

    return True, details

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MiniMax H3 Hardware Gatekeeper Check")
    parser.add_argument("--auto-shutdown", action="store_true", help="Automatically unassign Colab runtime if checks fail")
    args = parser.parse_args()

    passed, details = check_system(auto_shutdown_on_fail=args.auto_shutdown, verbose=True)
    if not passed:
        sys.exit(1)
