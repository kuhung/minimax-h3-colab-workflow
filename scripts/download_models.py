#!/usr/bin/env python3
"""
Model Downloader & Linker for MiniMax H3 on ComfyUI (2.0 Architecture).
Features:
1. CUDA & Hardware Adaptive Weight Matching:
   - CUDA >= 13.0 (cu130 with comfy-kitchen): int8_convrot DiT + int8_convrot Video VAE
   - CUDA 12.x (Colab default cu121/cu124/cu126, Ada CC 8.9): fp8_scaled DiT + fp16 Video VAE
2. 4-Step and 8-Step Turbo LoRA Selection.
3. Exact HuggingFace byte verification & safetensors header boundary checking.
4. Drive Cache Fast Symlinking & Atomic Backup.
"""

import os
import sys
import json
import struct
import shutil
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

# 精确字节数常量表（来自 HuggingFace 官方 API 实测，严禁编造）
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

def detect_cuda_variant() -> str:
    """
    Detects PyTorch CUDA version and returns 'int8_convrot' if CUDA >= 13.0,
    otherwise 'fp8_scaled' for CUDA 12.x / Ada CC 8.9.
    """
    cuda_ver = None
    try:
        import torch
        if torch.cuda.is_available() and hasattr(torch.version, "cuda") and torch.version.cuda:
            cuda_ver = torch.version.cuda
    except Exception:
        pass

    if cuda_ver and cuda_ver.startswith("13."):
        print(f"⚡ 检测到 PyTorch CUDA 版本: {cuda_ver} (cu130)，启用原生 int8_convrot 算子")
        return "int8_convrot"
    else:
        ver_str = cuda_ver if cuda_ver else "未知/未初始化"
        print(f"💡 检测到 CUDA 环境: {ver_str}。在 Colab 默认 CUDA 12.x (cu121/cu124/cu126) + L4 (CC 8.9) 下，"
              f"自动选用 fp8_scaled + video_vae_fp16，消除 int8_convrot 在非 cu130 下的慢路径回退！")
        return "fp8_scaled"

def get_models_spec(variant: str = "auto", turbo_steps: int = 8) -> List[Dict[str, Any]]:
    """Builds required model download specification list."""
    if variant == "auto":
        actual_variant = detect_cuda_variant()
    else:
        actual_variant = variant

    if actual_variant == "int8_convrot":
        diff_file = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
        vae_file = "minimax_h3_video_vae_int8_convrot.safetensors"
    else:
        diff_file = "minimax_h3_fl2va_pruned_fp8_scaled.safetensors"
        vae_file = "minimax_h3_video_vae_fp16.safetensors"

    if turbo_steps == 4:
        lora_file = "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors"
    else:
        lora_file = "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"

    diff_rel = f"diffusion_models/{diff_file}"
    vae_rel = f"vae/{vae_file}"
    lora_rel = f"loras/{lora_file}"
    text_rel = "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
    audio_rel = "vae/minimax_h3_audio_vae_fp32.safetensors"

    return [
        {
            "name": f"Diffusion Model ({actual_variant})",
            "role": "diffusion_model",
            "subfolder": "diffusion_models",
            "filename": diff_file,
            "repo_id": "Comfy-Org/MiniMax-H3",
            "repo_file": diff_rel,
            "exact_bytes": EXACT_SIZES[diff_rel],
        },
        {
            "name": "Text Encoder (Qwen3-VL 32B NVFP4 AWQ)",
            "role": "text_encoder",
            "subfolder": "text_encoders",
            "filename": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
            "repo_id": "Comfy-Org/MiniMax-H3",
            "repo_file": text_rel,
            "exact_bytes": EXACT_SIZES[text_rel],
        },
        {
            "name": f"Video VAE ({'int8_convrot' if actual_variant == 'int8_convrot' else 'fp16'})",
            "role": "video_vae",
            "subfolder": "vae",
            "filename": vae_file,
            "repo_id": "Comfy-Org/MiniMax-H3",
            "repo_file": vae_rel,
            "exact_bytes": EXACT_SIZES[vae_rel],
        },
        {
            "name": "Audio VAE (32kHz Stereo FP32)",
            "role": "audio_vae",
            "subfolder": "vae",
            "filename": "minimax_h3_audio_vae_fp32.safetensors",
            "repo_id": "Comfy-Org/MiniMax-H3",
            "repo_file": audio_rel,
            "exact_bytes": EXACT_SIZES[audio_rel],
        },
        {
            "name": f"Turbo LoRA ({turbo_steps}-Step)",
            "role": "lora",
            "subfolder": "loras",
            "filename": lora_file,
            "repo_id": "Comfy-Org/MiniMax-H3",
            "repo_file": lora_rel,
            "exact_bytes": EXACT_SIZES[lora_rel],
        },
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

def ensure_symlink(src_path: Path, dst_path: Path):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.is_symlink() or dst_path.exists():
        dst_path.unlink()
    dst_path.symlink_to(src_path.resolve())
    print(f"🔗 软链接建立: {dst_path.name} -> {src_path}")

def atomic_backup_to_drive(src_file: Path, drive_target: Path, exact_bytes: int):
    drive_target.parent.mkdir(parents=True, exist_ok=True)
    tmp_target = drive_target.with_suffix(drive_target.suffix + ".partial")
    print(f"   💾 正在原子备份到 Google Drive 缓存: {drive_target.name} ...")
    try:
        if tmp_target.exists():
            tmp_target.unlink()
        shutil.copy2(src_file, tmp_target)
        if check_safetensors_file(tmp_target, exact_bytes):
            tmp_target.replace(drive_target)
            print(f"   ✅ Drive 缓存备份成功完成并通过双校验: {drive_target.name}")
        else:
            print(f"   ⚠️ 备份文件校验失败，清理临时文件: {tmp_target}")
            if tmp_target.exists():
                tmp_target.unlink()
    except Exception as e:
        print(f"   ⚠️ 备份到 Drive 略过: {e}（不影响当前任务运行）")
        if tmp_target.exists():
            try:
                tmp_target.unlink()
            except Exception:
                pass

def download_or_link(
    comfy_root: Path,
    drive_cache_dir: Path = None,
    variant: str = "auto",
    turbo_steps: int = 8
) -> Dict[str, str]:
    models_root = comfy_root / "models"
    models_root.mkdir(parents=True, exist_ok=True)

    specs = get_models_spec(variant=variant, turbo_steps=turbo_steps)

    print("=" * 68)
    print("📦 MiniMax H3 权重挂载与下载检测 (2.0 最佳实践)")
    print(f"📁 ComfyUI 模型目录: {models_root.resolve()}")
    if drive_cache_dir and drive_cache_dir.exists():
        print(f"💾 检测到 Google Drive 缓存目录: {drive_cache_dir.resolve()}")
    print(f"🎯 选定量化方案: {specs[0]['name']} | 步数 LoRA: {turbo_steps}-Step")
    print("=" * 68)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("⚙️  正在安装 huggingface_hub ...")
        os.system("pip install -q huggingface_hub safetensors")
        from huggingface_hub import hf_hub_download

    active_files = {}

    for item in specs:
        subfolder = models_root / item["subfolder"]
        subfolder.mkdir(parents=True, exist_ok=True)
        target_file = subfolder / item["filename"]
        exact_bytes = item["exact_bytes"]
        active_files[item["role"]] = item["filename"]

        print(f"\n🔍 检查 [{item['name']}] ({exact_bytes / (1024**3):.2f} GiB)...")

        # 1. 优先检查本地已经存在且通过 safetensors 双校验
        if check_safetensors_file(target_file, exact_bytes):
            print(f"   ✅ 本地已有完整文件: {target_file.name}")
            continue

        # 若本地存在不完整文件或断链，清除
        if target_file.is_symlink() or target_file.exists():
            target_file.unlink()

        # 2. 检查 Google Drive 缓存（兼容子目录与平铺目录）
        cand_drive = None
        if drive_cache_dir and drive_cache_dir.exists():
            for cand in [drive_cache_dir / item["subfolder"] / item["filename"], drive_cache_dir / item["filename"]]:
                if cand.exists():
                    if check_safetensors_file(cand, exact_bytes):
                        cand_drive = cand
                        break
                    else:
                        print(f"   🗑️ 检测到 Drive 缓存文件残损 ({cand.name})，正在移除...")
                        try:
                            cand.unlink()
                        except Exception:
                            pass

        if cand_drive:
            print(f"   ⚡ 命中 Google Drive 完整缓存，0 秒软链接秒开...")
            ensure_symlink(cand_drive, target_file)
            continue

        # 3. 需通过网络从 HuggingFace 官方源拉取
        repo_file = item.get("repo_file", item["filename"])
        print(f"   ⬇️  本地与云盘无完整缓存，从 HuggingFace ({item['repo_id']}) 拉取: {repo_file} ...")
        try:
            downloaded_path = hf_hub_download(
                repo_id=item["repo_id"],
                filename=repo_file,
                local_dir=str(models_root),
                local_dir_use_symlinks=False
            )
            dl_file = Path(downloaded_path)
            if dl_file.resolve() != target_file.resolve() and dl_file.exists():
                shutil.move(str(dl_file), str(target_file))

            if not check_safetensors_file(target_file, exact_bytes):
                raise RuntimeError(f"下载文件完整性校验失败: {target_file.name}")
            print(f"   ✅ 下载完成并通过完整性校验: {target_file.name}")

            # 4. 同步原子备份到 Drive 缓存
            if drive_cache_dir and drive_cache_dir.exists():
                drive_target = drive_cache_dir / item["subfolder"] / item["filename"]
                atomic_backup_to_drive(target_file, drive_target, exact_bytes)
        except Exception as e:
            print(f"   ❌ 下载失败: {e}")
            print(f"   💡 若连接 HuggingFace 慢，可尝试设置镜像: export HF_ENDPOINT=https://hf-mirror.com")
            raise

    # 建立向后兼容别名软链接
    aliases = [
        (models_root / "vae" / "minimax_h3_vae.safetensors", models_root / "vae" / active_files.get("video_vae", "")),
        (models_root / "loras" / "minimax_h3_fl2va_pdd_8step_pruned_comfy.safetensors", models_root / "loras" / active_files.get("lora", "")),
    ]
    for alias_path, src_path in aliases:
        if src_path and src_path.exists() and not alias_path.exists():
            alias_path.symlink_to(src_path.resolve())
            print(f"🔗 建立兼容别名: {alias_path.name} -> {src_path.name}")

    print("\n" + "=" * 68)
    print("🎉 所有必要模型组件就绪！无需二次下载即可启动 ComfyUI。")
    print("=" * 68)
    return active_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMax H3 Model Downloader (2.0)")
    parser.add_argument("--comfy-root", type=str, default="/content/ComfyUI", help="ComfyUI root path")
    parser.add_argument("--drive-cache", type=str, default="/content/drive/MyDrive/MiniMax_Models", help="Google Drive cache path")
    parser.add_argument("--variant", type=str, default="auto", choices=["auto", "int8_convrot", "fp8_scaled"], help="Quantization variant")
    parser.add_argument("--turbo-steps", type=int, default=8, choices=[4, 8], help="Turbo LoRA step variant (4 or 8)")
    args = parser.parse_args()

    download_or_link(
        Path(args.comfy_root),
        Path(args.drive_cache) if args.drive_cache else None,
        variant=args.variant,
        turbo_steps=args.turbo_steps
    )
