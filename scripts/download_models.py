#!/usr/bin/env python3
"""
Model Downloader & Linker for MiniMax H3 on ComfyUI (2.0 Architecture).
Features:
1. CUDA & Hardware Adaptive Weight Matching:
   - CUDA >= 13.0 (cu130 with comfy-kitchen): int8_convrot DiT + int8_convrot Video VAE
   - CUDA 12.x (Colab default cu121/cu124/cu126, Ada CC 8.9): fp8_scaled DiT + fp16 Video VAE
2. 4-Step and 8-Step Turbo LoRA Selection (including Ref2V Turbo LoRA).
3. Exact HuggingFace byte verification & safetensors header boundary checking.
4. Support for FL2VA (T2V/I2V/FLF2V) and Ref2VA (Omni-Reference) backbones.
5. Drive Cache Fast Symlinking & Atomic Backup.
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
    "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors": 20970379616,
    "diffusion_models/minimax_h3_ref2va_pruned_fp8_scaled.safetensors": 20958205608,
    "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors": 15687142551,
    "vae/minimax_h3_video_vae_int8_convrot.safetensors": 2811065184,
    "vae/minimax_h3_video_vae_fp16.safetensors": 5207808496,
    "vae/minimax_h3_audio_vae_fp32.safetensors": 605254808,
    "loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors": 1956193000,
    "loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors": 1956192992,
    "loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors": 1956193000,
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

def get_models_spec(variant: str = "auto", turbo_steps: int = 8, mode: str = "fl2va") -> List[Dict[str, Any]]:
    """Builds required model download specification list based on mode and hardware."""
    if variant == "auto":
        actual_variant = detect_cuda_variant()
    else:
        actual_variant = variant

    # Video VAE selection
    if actual_variant == "int8_convrot":
        vae_file = "minimax_h3_video_vae_int8_convrot.safetensors"
    else:
        vae_file = "minimax_h3_video_vae_fp16.safetensors"

    text_rel = "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
    audio_rel = "vae/minimax_h3_audio_vae_fp32.safetensors"
    vae_rel = f"vae/{vae_file}"

    base_specs = [
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
    ]

    # Mode-specific backbones and LoRAs
    if mode in ("fl2va", "t2v", "i2v", "flf2v"):
        diff_file = f"minimax_h3_fl2va_pruned_{actual_variant}.safetensors"
        diff_rel = f"diffusion_models/{diff_file}"
        if turbo_steps == 4:
            lora_file = "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors"
        else:
            lora_file = "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"
        lora_rel = f"loras/{lora_file}"

        specs = [
            {
                "name": f"Diffusion Model FL2VA ({actual_variant})",
                "role": "diffusion_model",
                "subfolder": "diffusion_models",
                "filename": diff_file,
                "repo_id": "Comfy-Org/MiniMax-H3",
                "repo_file": diff_rel,
                "exact_bytes": EXACT_SIZES[diff_rel],
            }
        ] + base_specs + [
            {
                "name": f"Turbo LoRA ({turbo_steps}-Step)",
                "role": "lora",
                "subfolder": "loras",
                "filename": lora_file,
                "repo_id": "Comfy-Org/MiniMax-H3",
                "repo_file": lora_rel,
                "exact_bytes": EXACT_SIZES[lora_rel],
            }
        ]
        return specs

    elif mode in ("ref2va", "r2v"):
        diff_file = f"minimax_h3_ref2va_pruned_{actual_variant}.safetensors"
        diff_rel = f"diffusion_models/{diff_file}"
        lora_file = "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"
        lora_rel = f"loras/{lora_file}"

        specs = [
            {
                "name": f"Diffusion Model Ref2VA ({actual_variant})",
                "role": "diffusion_model_ref2va",
                "subfolder": "diffusion_models",
                "filename": diff_file,
                "repo_id": "Comfy-Org/MiniMax-H3",
                "repo_file": diff_rel,
                "exact_bytes": EXACT_SIZES[diff_rel],
            }
        ] + base_specs + [
            {
                "name": "Turbo Ref2V 4-Step LoRA",
                "role": "lora_ref2v",
                "subfolder": "loras",
                "filename": lora_file,
                "repo_id": "Comfy-Org/MiniMax-H3",
                "repo_file": lora_rel,
                "exact_bytes": EXACT_SIZES[lora_rel],
            }
        ]
        return specs

    elif mode == "all":
        fl2va_specs = get_models_spec(variant=variant, turbo_steps=turbo_steps, mode="fl2va")
        ref_diff = f"minimax_h3_ref2va_pruned_{actual_variant}.safetensors"
        ref_lora = "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"
        extra = [
            {
                "name": f"Diffusion Model Ref2VA ({actual_variant})",
                "role": "diffusion_model_ref2va",
                "subfolder": "diffusion_models",
                "filename": ref_diff,
                "repo_id": "Comfy-Org/MiniMax-H3",
                "repo_file": f"diffusion_models/{ref_diff}",
                "exact_bytes": EXACT_SIZES[f"diffusion_models/{ref_diff}"],
            },
            {
                "name": "Turbo Ref2V 4-Step LoRA",
                "role": "lora_ref2v",
                "subfolder": "loras",
                "filename": ref_lora,
                "repo_id": "Comfy-Org/MiniMax-H3",
                "repo_file": f"loras/{ref_lora}",
                "exact_bytes": EXACT_SIZES[f"loras/{ref_lora}"],
            }
        ]
        return fl2va_specs + extra

    else:
        raise ValueError(f"未知 mode: {mode}。支持: fl2va, ref2va, all")

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
    turbo_steps: int = 8,
    mode: str = "fl2va"
) -> Dict[str, str]:
    models_root = comfy_root / "models"
    models_root.mkdir(parents=True, exist_ok=True)

    specs = get_models_spec(variant=variant, turbo_steps=turbo_steps, mode=mode)

    print("=" * 68)
    print("📦 MiniMax H3 权重挂载与下载检测 (2.0 最佳实践)")
    print(f"📁 ComfyUI 模型目录: {models_root.resolve()}")
    if drive_cache_dir and drive_cache_dir.exists():
        print(f"💾 检测到 Google Drive 缓存目录: {drive_cache_dir.resolve()}")
    print(f"🎯 选定量化方案: {specs[0]['name']} | 模式: {mode} | 步数 LoRA: {turbo_steps}-Step")
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

        # 1. 优先检查本地 NVMe 是否已有真实物理文件且通过 safetensors 双校验
        if not target_file.is_symlink() and target_file.is_file() and check_safetensors_file(target_file, exact_bytes):
            print(f"   ✅ 本地 NVMe 已就绪 (真实物理文件): {target_file.name}")
            continue

        # 若本地存在软链接（如旧版指向 Drive 的软链接）或残损文件，先行清除
        if target_file.is_symlink() or target_file.exists():
            print(f"   🧹 移除旧链接或损坏文件: {target_file.name}")
            target_file.unlink()

        # 2. 检查 Google Drive 缓存目录中是否存在完好备份
        drive_cached_file = None
        if drive_cache_dir:
            cand = drive_cache_dir / item["subfolder"] / item["filename"]
            cand_root = drive_cache_dir / item["filename"]
            if cand.exists() and check_safetensors_file(cand, exact_bytes):
                drive_cached_file = cand
            elif cand_root.exists() and check_safetensors_file(cand_root, exact_bytes):
                drive_cached_file = cand_root

        if drive_cached_file:
            print(f"   ⚡ 发现 Google Drive 完整缓存: {drive_cached_file}")
            print(f"   📥 正在将缓存高速拷贝至本地 NVMe (消除 Drive FUSE I/O 锁并防止断连)...")
            shutil.copy2(drive_cached_file, target_file)
            if check_safetensors_file(target_file, exact_bytes):
                print(f"   ✅ 本地 NVMe 文件已从 Drive 缓存安全恢复: {target_file.name}")
                continue
            else:
                print(f"   ⚠️ 拷贝后校验失败，将重新从官方 HuggingFace 下载")
                if target_file.exists():
                    target_file.unlink()

        # 3. 本地与 Drive 均无可用完好权重，从 HuggingFace 官方源高速下载至本地 NVMe
        print(f"   🌐 开始从官方 HuggingFace 下载: {item['repo_id']} -> {item['repo_file']}")
        downloaded_path = hf_hub_download(
            repo_id=item["repo_id"],
            filename=item["repo_file"],
            local_dir=str(models_root),
            local_dir_use_symlinks=False,
            resume_download=True
        )

        dl_path = Path(downloaded_path)
        if dl_path.resolve() != target_file.resolve():
            shutil.move(str(dl_path), str(target_file))

        print(f"   🧪 执行下载后 safetensors 头部元数据与字节数双重核验...")
        if not check_safetensors_file(target_file, exact_bytes):
            raise RuntimeError(f"❌ 严重错误: 文件 {target_file.name} 下载损坏或头部校验不通过！")

        print(f"   ✅ 下载并核验通过: {target_file.name}")

        # 4. 原子备份至 Google Drive，方便下次 0 秒秒级恢复
        if drive_cache_dir:
            drive_dest = drive_cache_dir / item["subfolder"] / item["filename"]
            atomic_backup_to_drive(target_file, drive_dest, exact_bytes)

    print("\n" + "=" * 68)
    print("🎉 选定 MiniMax H3 权重体系全部挂载就绪！")
    print("=" * 68)
    return active_files

def main():
    parser = argparse.ArgumentParser(description="MiniMax H3 Model Downloader & Linker (2.0)")
    parser.add_argument("--comfy-root", default="/content/ComfyUI", help="ComfyUI installation root")
    parser.add_argument("--drive-cache", default="/content/drive/MyDrive/MiniMax_Weights_Cache", help="Google Drive cache root")
    parser.add_argument("--variant", default="auto", choices=["auto", "int8_convrot", "fp8_scaled"], help="Weight variant")
    parser.add_argument("--turbo-steps", type=int, default=8, choices=[4, 8], help="Turbo LoRA step selection")
    parser.add_argument("--mode", default="fl2va", choices=["fl2va", "ref2va", "all"], help="Model backbone mode")
    args = parser.parse_args()

    comfy_p = Path(args.comfy_root)
    drive_p = Path(args.drive_cache) if args.drive_cache else None

    download_or_link(
        comfy_root=comfy_p,
        drive_cache_dir=drive_p,
        variant=args.variant,
        turbo_steps=args.turbo_steps,
        mode=args.mode
    )

if __name__ == "__main__":
    main()
