#!/usr/bin/env python3
"""
Model Downloader & Linker for MiniMax H3 on ComfyUI.
Features:
1. Drive Cache Checking: Checks if model already exists on mounted Google Drive with integrity verification.
2. Minimal Set Download: Pulls Pruned Int8 Diffusion + NVFP4 Text Encoder + VAE (Total ~40GB).
3. Atomic Drive Backup: Copies to Drive via temporary files to avoid corrupted partial caches.
"""

import os
import sys
import shutil
import argparse
from pathlib import Path

# 模型文件定义清单与最小完整体积（对应官方 Comfy-Org/MiniMax-H3 开源免鉴权仓库）
MODELS_SPEC = [
    {
        "name": "Diffusion Model (FL2VA Pruned Int8)",
        "subfolder": "diffusion_models",
        "filename": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        "repo_id": "Comfy-Org/MiniMax-H3",
        "repo_file": "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        "size_gb": 19.53,
        "min_bytes": int(18.0 * (1024 ** 3)),
    },
    {
        "name": "Text Encoder (Qwen3-VL 32B NVFP4 AWQ)",
        "subfolder": "text_encoders",
        "filename": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        "repo_id": "Comfy-Org/MiniMax-H3",
        "repo_file": "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        "size_gb": 14.61,
        "min_bytes": int(13.5 * (1024 ** 3)),
    },
    {
        "name": "Video VAE (H3 Video VAE Int8 ConvRot)",
        "subfolder": "vae",
        "filename": "minimax_h3_video_vae_int8_convrot.safetensors",
        "repo_id": "Comfy-Org/MiniMax-H3",
        "repo_file": "vae/minimax_h3_video_vae_int8_convrot.safetensors",
        "size_gb": 2.62,
        "min_bytes": int(2.4 * (1024 ** 3)),
    },
    {
        "name": "Audio VAE (H3 Audio VAE FP32 Stereo)",
        "subfolder": "vae",
        "filename": "minimax_h3_audio_vae_fp32.safetensors",
        "repo_id": "Comfy-Org/MiniMax-H3",
        "repo_file": "vae/minimax_h3_audio_vae_fp32.safetensors",
        "size_gb": 0.56,
        "min_bytes": int(0.5 * (1024 ** 3)),
    },
    {
        "name": "Turbo LoRA (8-Step Fast Video)",
        "subfolder": "loras",
        "filename": "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
        "repo_id": "Comfy-Org/MiniMax-H3",
        "repo_file": "loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
        "size_gb": 1.82,
        "min_bytes": int(1.6 * (1024 ** 3)),
    }
]

def ensure_symlink(src_path: Path, dst_path: Path):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.is_symlink() or dst_path.exists():
        dst_path.unlink()
    dst_path.symlink_to(src_path.resolve())
    print(f"🔗 软链接建立: {dst_path.name} -> {src_path}")

def is_file_valid(file_path: Path, min_bytes: int) -> bool:
    if not file_path.exists():
        return False
    try:
        actual_size = file_path.stat().st_size
        return actual_size >= min_bytes
    except Exception:
        return False

def atomic_backup_to_drive(src_file: Path, drive_target: Path, min_bytes: int):
    drive_target.parent.mkdir(parents=True, exist_ok=True)
    tmp_target = drive_target.with_suffix(drive_target.suffix + ".tmp")
    print(f"   💾 正在备份到 Google Drive 以便后续秒级加载: {drive_target.name} ...")
    try:
        if tmp_target.exists():
            tmp_target.unlink()
        shutil.copy2(src_file, tmp_target)
        if is_file_valid(tmp_target, min_bytes):
            tmp_target.replace(drive_target)
            print(f"   ✅ Drive 缓存备份成功完成: {drive_target.name}")
        else:
            print(f"   ⚠️ 备份大小异常，放弃缓存该文件: {tmp_target}")
            if tmp_target.exists():
                tmp_target.unlink()
    except Exception as e:
        print(f"   ⚠️ 备份到 Drive 失败: {e}（不影响当前任务运行）")
        if tmp_target.exists():
            try:
                tmp_target.unlink()
            except Exception:
                pass

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
        min_bytes = item["min_bytes"]

        print(f"\n🔍 检查 [{item['name']}] (~{item['size_gb']} GB)...")

        # 1. 优先检查本地已经存在且完整
        if is_file_valid(target_file, min_bytes):
            print(f"   ✅ 本地已有完整文件: {target_file.name}")
            continue

        # 2. 检查 Google Drive 缓存中是否有预存好的完整文件
        if drive_cache_dir and drive_cache_dir.exists():
            drive_file = drive_cache_dir / item["subfolder"] / item["filename"]
            if drive_file.exists():
                if is_file_valid(drive_file, min_bytes):
                    print(f"   ⚡ 在 Google Drive 中命中完整缓存，建立软链接...")
                    ensure_symlink(drive_file, target_file)
                    continue
                else:
                    print(f"   ⚠️ 检测到 Drive 缓存文件已损坏或不完整 ({drive_file.stat().st_size} bytes)，准备重新下载...")
                    try:
                        drive_file.unlink()
                    except Exception:
                        pass

        # 3. 需通过网络下载
        repo_file = item.get("repo_file", item["filename"])
        print(f"   ⬇️  本地与云盘无有效缓存，从 HuggingFace ({item['repo_id']}) 拉取: {repo_file} ...")
        try:
            downloaded_path = hf_hub_download(
                repo_id=item["repo_id"],
                filename=repo_file,
                local_dir=str(models_root),
                local_dir_use_symlinks=False
            )
            dl_file = Path(downloaded_path)
            if dl_file.resolve() != target_file.resolve() and dl_file.exists():
                target_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(dl_file), str(target_file))

            if not is_file_valid(target_file, min_bytes):
                raise RuntimeError(f"下载文件大小不足 ({target_file.stat().st_size} < {min_bytes})")
            print(f"   ✅ 下载并通过大小校验: {target_file.name}")

            # 4. 同步备份到 Drive 缓存
            if drive_cache_dir and drive_cache_dir.exists():
                drive_target = drive_cache_dir / item["subfolder"] / item["filename"]
                atomic_backup_to_drive(target_file, drive_target, min_bytes)
        except Exception as e:
            print(f"   ❌ 下载失败: {e}")
            print(f"   💡 若连接 HuggingFace 慢，可尝试设置镜像: export HF_ENDPOINT=https://hf-mirror.com")
            raise

    # 建立向后兼容别名软链接
    aliases = [
        (models_root / "vae" / "minimax_h3_vae.safetensors", models_root / "vae" / "minimax_h3_video_vae_int8_convrot.safetensors"),
        (models_root / "loras" / "minimax_h3_fl2va_pdd_8step_pruned_comfy.safetensors", models_root / "loras" / "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"),
    ]
    for alias_path, src_path in aliases:
        if src_path.exists() and not alias_path.exists():
            alias_path.symlink_to(src_path)
            print(f"🔗 建立兼容别名: {alias_path.name} -> {src_path.name}")

    print("\n" + "=" * 60)
    print("🎉 所有必要模型组件就绪！无需二次下载即可启动 ComfyUI。")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMax H3 Model Downloader")
    parser.add_argument("--comfy-root", type=str, default="/content/ComfyUI", help="ComfyUI root path")
    parser.add_argument("--drive-cache", type=str, default="/content/drive/MyDrive/MiniMax_Models", help="Google Drive cache path")
    args = parser.parse_args()

    download_or_link(Path(args.comfy_root), Path(args.drive_cache) if args.drive_cache else None)
