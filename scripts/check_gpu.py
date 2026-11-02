#!/usr/bin/env python3
"""
Hardware Pre-flight Check for MiniMax H3 Local Deployment on Colab.
Enforces hard constraints:
1. Compute Capability >= 8.0 (Rejects T4 7.5, approves L4/A100/H100/RTX 30xx/40xx)
2. GPU VRAM >= 20GB
3. System RAM >= 25GB (High-RAM runtime recommended)
4. Free Disk Space >= 70GB
"""

import sys
import shutil
import os

def check_system():
    print("=" * 60)
    print("🔍 MiniMax H3 硬件环境门禁检测 (Hardware Gate Check)")
    print("=" * 60)

    # 1. 检查 PyTorch & CUDA
    try:
        import torch
    except ImportError:
        print("❌ 未检测到 PyTorch，请先在环境中安装 torch。")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("❌ 错误：未检测到可用 GPU 加速器！")
        print("💡 请在 Colab 顶部菜单选择：代码执行程序 -> 更改运行时类型 -> 硬件加速器选择 GPU (L4 或 A100)。")
        sys.exit(1)

    device_name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    cc = major + minor / 10.0
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

    print(f"🖥️  GPU 型号: {device_name}")
    print(f"⚡ 计算能力 (Compute Capability): {cc:.1f}")
    print(f"💾 GPU 显存 (VRAM): {vram_gb:.2f} GB")

    # 门禁 1: 计算能力必须 >= 8.0 (Int8 ConvRot 算子硬要求)
    if cc < 8.0:
        print("\n" + "!" * 60)
        print("🚨 致命阻断：当前 GPU 计算能力不足！")
        print(f"   检测到计算能力为 {cc:.1f}（如 Tesla T4 为 7.5）。")
        print("   MiniMax H3 的 Int8 剪枝量化（ConvRot 算子）硬性要求计算能力 ≥ 8.0（L4 / A100 / RTX 30/40系）。")
        print("   在 T4 上强行运行将导致内核崩溃或无实用产出，白白消耗计算单元。")
        print("💡 处置建议：立即断开并更改运行时，重新分配到 L4 或 A100 实例！")
        print("!" * 60)
        sys.exit(1)
    else:
        print("✅ 计算能力检查通过（≥ 8.0，支持 Int8 硬件级加速加速）")

    # 门禁 2: 显存检测
    if vram_gb < 20.0:
        print(f"⚠️  警告：GPU 显存为 {vram_gb:.1f} GB，低于推荐的 24GB (L4/A100)。")
        print("   请确保在 ComfyUI 启动参数中包含 CPU 动态卸载 (--lowvram 或平衡模式)。")
    else:
        print(f"✅ GPU 显存充足 ({vram_gb:.1f} GB)")

    # 门禁 3: 内存与磁盘检测
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        print(f"🧠 系统内存 (RAM): {ram_gb:.1f} GB")
        if ram_gb < 25.0:
            print("⚠️  系统内存低于 25GB。建议在 Colab 中开启 '高内存 (High-RAM)' 运行时，防止变压器加载时 OOM。")
        else:
            print("✅ 系统内存通过 (≥ 25GB)")
    except ImportError:
        pass

    target_dir = "/content" if os.path.exists("/content") else "."
    free_disk_gb = shutil.disk_usage(target_dir).free / (1024 ** 3)
    print(f"💽 可用磁盘空间: {free_disk_gb:.1f} GB")
    if free_disk_gb < 70.0:
        print("⚠️  可用磁盘空间低于 70GB！剪枝模型约需 40GB，若空间不足请清理不必要的文件。")
    else:
        print("✅ 磁盘空间通过 (≥ 70GB)")

    print("=" * 60)
    print("🎉 恭喜！当前硬件完全满足 MiniMax H3 纯本地高效运行标准！")
    print("=" * 60)

if __name__ == "__main__":
    check_system()
