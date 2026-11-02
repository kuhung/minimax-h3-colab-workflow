#!/usr/bin/env python3
"""
Model Downloader & Linker for MiniMax H3 on ComfyUI.
Features:
1. Drive Cache Checking: Checks if model already exists on mounted Google Drive to save Colab compute units.
2. Minimal Set Download: Only pulls Pruned Int8 Diffusion + NVFP4 Text Encoder + VAE (Total ~40GB).
3. Resumable & Fast: Uses huggingface_hub or aria2c.
"""

import os
import sys
import argparse
from pathlib import Path

# 模型文件定义清单
MODELS_SPEC = [
    {
        "name": "Diffusion Model (FL2VA Pruned Int8)",
        "subfolder": "diffusion_models",
        "filename": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        "repo_id": "Comfy-Org/MiniMax-H3_repackaged",
        "size_gb": 20.97,
    },
    {
        "name": "Text Encoder (Qwen3-VL 32B NVFP4 AWQ)",
        "subfolder": "text_encoders",
        "filename": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        "repo_id": "Comfy-Org/MiniMax-H3_repackaged",
        "size_gb": 15.69,
    },
    {
        "name": "VAE (H3 Audio-Visual Autoencoder)",
        "subfolder": "vae",
        "filename": "minimax_h3_vae.safetensors",
        "repo_id": "Comfy-Org/MiniMax-H3_repackaged",
        "size_gb": 1.2,
    },
    {
        "name": "PDD Acc LoRA (8-Step Fast Video)",
        "subfolder": "loras",
        "filename": "minimax_h3_fl2va_pdd_8step_pruned_comfy.safetensors",
        "repo_id": "Comfy-Org/MiniMax-H3_repackaged",
        "size_gb": 0.8,
    }
]

def ensure_symlink(src_path: Path, dst_path: Path):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.is_symlink() or dst_path.exists():
        dst_path.unlink()
    dst_path.symlink_to(src_path.resolve())
    print(f"🔗 软链接建立: {dst_path.name} -> {src_path}")

def download_or_link(comfy_root: Path, drive_cache_dir: Path = None):
    models_root = comfy_root / "models"
    models_root.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("📦 MiniMax H3 权重挂载与下载检测")
    print(f"📁 ComfyUI 根目录: {comfy_root.resolve()}")
    if drive_cache_dir and drive_cache_dir.exists():
        print(f"💾 检测到 Google Drive 缓存目录: {drive_cache_dir.resolve()}")
    print("=" * 60)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("⚙️  正在安装 huggingface_hub ...")
        os.system("pip install -q huggingface_hub")
        from huggingface_hub import hf_hub_download

    for item in MODELS_SPEC:
        subfolder = models_root / item["subfolder"]
        subfolder.mkdir(parents=True, exist_ok=True)
        target_file = subfolder / item["filename"]

        print(f"\n🔍 检查 [{item['name']}] (~{item['size_gb']} GB)...")

        # 1. 优先检查本地已经存在
        if target_file.exists() and target_file.stat().st_size > 1024 * 1024 * 100:
            print(f"   ✅ 本地已有完整文件: {target_file.name}")
            continue

        # 2. 检查 Google Drive 缓存中是否有预存好的文件
        if drive_cache_dir:
            drive_file = drive_cache_dir / item["subfolder"] / item["filename"]
            if drive_file.exists() and drive_file.stat().st_size > 1024 * 1024 * 100:
                print(f"   ⚡ 在 Google Drive 中命中缓存，直接链接至本地...")
                ensure_symlink(drive_file, target_file)
                continue

        # 3. 需通过网络下载
        print(f"   ⬇️  本地与云盘无缓存，正在从 HuggingFace 拉取: {item['filename']} ...")
        try:
            downloaded_path = hf_hub_download(
                repo_id=item["repo_id"],
                filename=item["filename"],
                subfolder=None,
                local_dir=str(subfolder),
                local_dir_use_symlinks=False
            )
            print(f"   ✅ 下载完成: {downloaded_path}")

            # 如果配置了 Drive 缓存路径，拷贝或同步一份备份（可选）
            if drive_cache_dir and drive_cache_dir.exists():
                drive_subfolder = drive_cache_dir / item["subfolder"]
                drive_subfolder.mkdir(parents=True, exist_ok=True)
                drive_target = drive_subfolder / item["filename"]
                if not drive_target.exists():
                    print(f"   💾 正在备份到 Google Drive 以便下次零等待加载...")
                    os.system(f"cp '{target_file}' '{drive_target}'")
        except Exception as e:
            print(f"   ❌ 下载失败: {e}")
            print(f"   💡 若连接 HuggingFace 较慢，可配置环境变量: export HF_ENDPOINT=https://hf-mirror.com")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("🎉 所有必要模型组件就绪！无需二次下载即可启动 ComfyUI。")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMax H3 Model Downloader")
    parser.add_argument("--comfy-root", type=str, default="/content/ComfyUI", help="ComfyUI root path")
    parser.add_argument("--drive-cache", type=str, default="/content/drive/MyDrive/MiniMax_Models", help="Google Drive cache path")
    args = parser.parse_args()

    comfy_path = Path(args.comfy_root)
    drive_path = Path(args.drive_cache) if args.drive_cache else None

    download_or_link(comfy_path, drive_path)
