#!/usr/bin/env python3
"""
Input Gatekeeper & Preprocessing Module for MiniMax H3 (V3 Multimodal Control Flows).
Supports:
1. Dimension alignment to multiples of 16 and 32 (e.g. 960x544: 16x60 / 32x30, 16x34 / 32x17) and RGB/BT.709 color preservation.
2. Temporal frame alignment to 17k + 5 grid (MiniMax H3 latent temporal stride @ 24fps).
3. Ref2VA hard quota & binding enforcement:
   - Images <= 9
   - Videos <= 3 (each 2-15s, total <= 15.0s)
   - Audios <= 3 (each 2-15s, total <= 15.0s)
   - Combined total <= 12 files
   - Audio Red Line: Pure audio cannot be submitted standalone!
   - Dual Reference Tag Normalization: '@图片N' / '@视频N' / '@音频N' <-> '<Picture N>' / '<Video N>' / '<Audio N>'
     and video-attached audio priority indexing before standalone audio.
4. Video 24fps constant frame rate transcoding (ComfyUI headless loader requirement).
5. FLF2V start/end frame alignment & anti-gradient melting bug luminance / color tone check.
6. Standalone CLI & Python importable API with rigorous self-testing.
"""

import os
import re
import sys
import json
import math
import shutil
import argparse
import subprocess
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional

try:
    from PIL import Image, ImageOps, ImageStat
except ImportError:
    Image = None
    ImageOps = None
    ImageStat = None


# ==============================================================================
# 1. 尺寸与帧数网格对齐 (16/32倍数与 17k + 5)
# ==============================================================================

def align_dimensions_16(width: int, height: int) -> Tuple[int, int]:
    """
    Checks and aligns width and height to integer multiples of 16.
    Example: 960x544 -> (960, 544) since 960 = 16 * 60, 544 = 16 * 34.
    """
    w = max(16, int(round(width / 16.0)) * 16)
    h = max(16, int(round(height / 16.0)) * 16)
    return w, h


def align_dimensions_32(width: int, height: int) -> Tuple[int, int]:
    """
    Checks and aligns width and height to integer multiples of 32.
    Note: 960x544 satisfies both 16 and 32 multiples (960 = 32 * 30, 544 = 32 * 17).
    """
    w = max(32, int(round(width / 32.0)) * 32)
    h = max(32, int(round(height / 32.0)) * 32)
    return w, h


def align_frame_count(n: int) -> int:
    """
    Align frame count upwards to 17k + 5 (MiniMax H3 latent temporal grid at 24fps).
    Example: 124 frames (17 * 7 + 5 = 124).
    """
    n = max(5, int(n))
    return n + (5 - n % 17) % 17


def normalize_h3_reference_tags(prompt: str) -> str:
    """
    Normalizes multimodal reference tags between Chinese/common notation and
    ComfyUI qwen3vl expected English tags:
      @图片N / @imageN -> <Picture N>
      @视频N / @videoN -> <Video N>
      @音频N / @audioN -> <Audio N>
    """
    if not prompt:
        return ""
    p = re.sub(r'[@<]图片(\d+)[>]?', r'<Picture \1>', prompt)
    p = re.sub(r'[@<](?:image|picture)\s*(\d+)[>]?', r'<Picture \1>', p, flags=re.IGNORECASE)
    p = re.sub(r'[@<]视频(\d+)[>]?', r'<Video \1>', p)
    p = re.sub(r'[@<]video\s*(\d+)[>]?', r'<Video \1>', p, flags=re.IGNORECASE)
    p = re.sub(r'[@<]音频(\d+)[>]?', r'<Audio \1>', p)
    p = re.sub(r'[@<]audio\s*(\d+)[>]?', r'<Audio \1>', p, flags=re.IGNORECASE)
    return p


def ensure_image_aligned_16(
    image_path: str,
    target_width: int = 960,
    target_height: int = 544,
    out_path: Optional[str] = None
) -> str:
    """
    Loads an image, resizes and center-crops it to target dimensions (aligned to 16 multiples),
    and saves as RGB to maintain BT.709 standard color space without drift.
    """
    if Image is None:
        raise RuntimeError("Pillow (PIL) is required for ensure_image_aligned_16.")

    in_p = Path(image_path)
    if not in_p.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    target_w, target_h = align_dimensions_16(target_width, target_height)

    if out_path:
        out_p = Path(out_path)
    else:
        out_p = in_p.parent / f"{in_p.stem}_aligned_16{in_p.suffix}"

    with Image.open(in_p) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")

        processed = ImageOps.fit(img, (target_w, target_h), Image.Resampling.LANCZOS)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        processed.save(str(out_p), format="PNG", quality=95)

    return str(out_p)


# ==============================================================================
# 2. 媒体元数据探测 (ffprobe / av / PIL)
# ==============================================================================

def probe_media_file(file_path: str) -> Dict[str, Any]:
    """
    Probes video/audio/image file using ffprobe (if available) or PIL.
    """
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Media file not found: {file_path}")

    info: Dict[str, Any] = {
        "path": str(p),
        "duration": 0.0,
        "fps": 0.0,
        "width": 0,
        "height": 0,
        "has_audio": False,
        "has_video": False,
        "channels": 0,
        "sample_rate": 0,
    }

    ffprobe_bin = shutil.which("ffprobe")
    if ffprobe_bin:
        cmd = [
            ffprobe_bin,
            "-v", "error",
            "-show_entries", "format=duration:stream=codec_type,width,height,r_frame_rate,channels,sample_rate",
            "-of", "json",
            str(p)
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(res.stdout)
            fmt = data.get("format", {})
            if "duration" in fmt:
                try:
                    info["duration"] = float(fmt["duration"])
                except (ValueError, TypeError):
                    pass

            for s in data.get("streams", []):
                ctype = s.get("codec_type")
                if ctype == "video":
                    info["has_video"] = True
                    info["width"] = int(s.get("width", 0))
                    info["height"] = int(s.get("height", 0))
                    r_fps = s.get("r_frame_rate", "0/1")
                    if "/" in r_fps:
                        num, den = r_fps.split("/")
                        if float(den) > 0:
                            info["fps"] = round(float(num) / float(den), 3)
                    else:
                        try:
                            info["fps"] = round(float(r_fps), 3)
                        except ValueError:
                            pass
                elif ctype == "audio":
                    info["has_audio"] = True
                    info["channels"] = int(s.get("channels", 0))
                    info["sample_rate"] = int(s.get("sample_rate", 0))
            return info
        except Exception:
            pass

    if Image is not None and p.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp", ".bmp"]:
        try:
            with Image.open(p) as img:
                info["width"], info["height"] = img.size
                info["has_video"] = False
                return info
        except Exception:
            pass

    return info


# ==============================================================================
# 3. Ref2VA 配额门禁与显式 @ 绑定校验
# ==============================================================================

def validate_ref2va_quotas(
    images: List[Any],
    videos: List[Any],
    audios: List[Any],
    prompt: str = ""
) -> Dict[str, Any]:
    """
    Strict Ref2VA Gatekeeper:
    1. Images <= 9
    2. Videos <= 3, each duration 2.0s - 15.0s, total video duration <= 15.0s
    3. Audios <= 3, each duration 2.0s - 15.0s, total audio duration <= 15.0s
    4. Combined total <= 12 files
    5. Pure audio red line: len(audios) > 0 and len(images) + len(videos) == 0 -> ValueError!
    6. Explicit @ binding validation & indexing:
       Video-attached audio takes precedence in index numbering before standalone audio.
       Dual tag detection (@图片N and <Picture N>).
    """
    num_img = len(images)
    num_vid = len(videos)
    num_aud = len(audios)
    total_files = num_img + num_vid + num_aud

    # Rule 5: Audio Red Line
    if num_aud > 0 and (num_img + num_vid) == 0:
        raise ValueError(
            "Ref2VA 硬性红线拦截：音频绝不可作为唯一输入独立提交！必须搭配参考图像或参考视频。"
        )

    # Rule 1: Images <= 9
    if num_img > 9:
        raise ValueError(f"Ref2VA 图片配额超标：当前输入 {num_img} 张图片，硬性上限为 ≤ 9 张！")

    # Rule 2: Videos <= 3
    if num_vid > 3:
        raise ValueError(f"Ref2VA 视频配额超标：当前输入 {num_vid} 段视频，硬性上限为 ≤ 3 段！")

    # Rule 3: Audios <= 3
    if num_aud > 3:
        raise ValueError(f"Ref2VA 音频配额超标：当前输入 {num_aud} 段音频，硬性上限为 ≤ 3 段！")

    # Rule 4: Total files <= 12
    if total_files > 12:
        raise ValueError(f"Ref2VA 总文件配额超标：当前输入 {total_files} 个文件，硬性混合上限为 ≤ 12 个！")

    # Probe video durations
    total_vid_sec = 0.0
    vid_meta = []
    for idx, v in enumerate(videos, 1):
        v_path = v if isinstance(v, str) else v.get("path", "")
        dur = 0.0
        if isinstance(v, dict) and "duration" in v:
            dur = float(v["duration"])
        elif v_path and Path(v_path).exists():
            probe = probe_media_file(v_path)
            dur = probe.get("duration", 0.0)
        else:
            dur = 5.0

        if dur > 0:
            if dur < 1.99:
                raise ValueError(f"参考视频 #{idx} 时长不足 2 秒 ({dur:.2f}s)，Ref2VA 要求单段 2.0s ~ 15.0s！")
            if dur > 15.05:
                raise ValueError(f"参考视频 #{idx} 时长超过 15 秒 ({dur:.2f}s)，Ref2VA 要求单段 ≤ 15.0s！")
        total_vid_sec += dur
        vid_meta.append({"index": idx, "path": v_path, "duration": dur})

    if total_vid_sec > 15.05:
        raise ValueError(f"参考视频累计总时长超标：当前总计 {total_vid_sec:.2f}s，硬性上限为 ≤ 15.0s！")

    # Probe audio durations
    total_aud_sec = 0.0
    aud_meta = []
    for idx, a in enumerate(audios, 1):
        a_path = a if isinstance(a, str) else a.get("path", "")
        dur = 0.0
        if isinstance(a, dict) and "duration" in a:
            dur = float(a["duration"])
        elif a_path and Path(a_path).exists():
            probe = probe_media_file(a_path)
            dur = probe.get("duration", 0.0)
        else:
            dur = 5.0

        if dur > 0:
            if dur < 1.99:
                raise ValueError(f"参考音频 #{idx} 时长不足 2 秒 ({dur:.2f}s)，Ref2VA 要求单段 2.0s ~ 15.0s！")
            if dur > 15.05:
                raise ValueError(f"参考音频 #{idx} 时长超过 15 秒 ({dur:.2f}s)，Ref2VA 要求单段 ≤ 15.0s！")
        total_aud_sec += dur
        aud_meta.append({"index": idx, "path": a_path, "duration": dur})

    if total_aud_sec > 15.05:
        raise ValueError(f"参考音频累计总时长超标：当前总计 {total_aud_sec:.2f}s，硬性上限为 ≤ 15.0s！")

    # Index mapping and @ binding check
    attached_audios = [{"ref_type": "video_audio", "index": i, "source_video": i} for i in range(1, num_vid + 1)]
    standalone_audios = [{"ref_type": "standalone_audio", "index": num_vid + i, "source_audio": i} for i in range(1, num_aud + 1)]

    bindings = {
        "images": [{"ref_tag": f"@图片{i}", "alt_tag": f"<Picture {i}>", "index": i} for i in range(1, num_img + 1)],
        "videos": [{"ref_tag": f"@视频{i}", "alt_tag": f"<Video {i}>", "index": i} for i in range(1, num_vid + 1)],
        "audio_tracks": attached_audios + standalone_audios
    }

    report = {
        "status": "passed",
        "counts": {
            "images": num_img,
            "videos": num_vid,
            "audios": num_aud,
            "total_files": total_files
        },
        "durations": {
            "videos_total_sec": round(total_vid_sec, 2),
            "audios_total_sec": round(total_aud_sec, 2)
        },
        "bindings_map": bindings,
        "video_attached_audio_priority": True
    }

    if prompt:
        normalized_prompt = normalize_h3_reference_tags(prompt)
        found_tags = []
        for i in range(1, num_img + 1):
            if f"@图片{i}" in prompt or f"<Picture {i}>" in normalized_prompt:
                found_tags.append(f"<Picture {i}>")
        for i in range(1, num_vid + 1):
            if f"@视频{i}" in prompt or f"<Video {i}>" in normalized_prompt:
                found_tags.append(f"<Video {i}>")
        for a in bindings["audio_tracks"]:
            idx = a["index"]
            if f"@音频{idx}" in prompt or f"<Audio {idx}>" in normalized_prompt:
                found_tags.append(f"<Audio {idx}>")
        report["detected_prompt_tags"] = found_tags
        report["normalized_prompt"] = normalized_prompt

    return report


# ==============================================================================
# 4. 参考视频 24fps 恒定帧率转码与时长限制
# ==============================================================================

def transcode_video_to_24fps(
    input_video: str,
    output_video: str,
    max_duration_sec: float = 15.0
) -> Dict[str, Any]:
    """
    ComfyUI reads reference video frames assuming constant 24fps without metadata parsing.
    This function transcodes input video to:
    - Constant Frame Rate 24.0 fps
    - Dimensions aligned to multiples of 16
    - BT.709 color space flags
    - Truncated duration <= max_duration_sec (<= 15.0s)
    - 32kHz stereo audio
    """
    in_p = Path(input_video)
    if not in_p.exists():
        raise FileNotFoundError(f"Input video not found: {input_video}")

    out_p = Path(output_video)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise RuntimeError("ffmpeg is required for video transcoding.")

    probe = probe_media_file(str(in_p))
    orig_w = probe.get("width", 960)
    orig_h = probe.get("height", 544)
    target_w, target_h = align_dimensions_16(orig_w, orig_h)

    vf_filters = [
        f"scale={target_w}:{target_h}:flags=lanczos",
        "fps=fps=24:round=near",
        "format=yuv420p"
    ]
    vf_str = ",".join(vf_filters)

    cmd = [
        ffmpeg_bin,
        "-y",
        "-i", str(in_p),
        "-t", f"{max_duration_sec:.2f}",
        "-vf", vf_str,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-colorspace", "bt709",
        "-color_primaries", "bt709",
        "-color_trc", "bt709",
        "-r", "24",
        "-c:a", "aac",
        "-b:a", "192k",
        "-ar", "32000",
        "-ac", "2",
        str(out_p)
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"FFmpeg transcode failed: {res.stderr}")

    out_probe = probe_media_file(str(out_p))
    return {
        "status": "success",
        "input_path": str(in_p),
        "output_path": str(out_p),
        "width": out_probe.get("width", target_w),
        "height": out_probe.get("height", target_h),
        "fps": out_probe.get("fps", 24.0),
        "duration": out_probe.get("duration", 0.0),
        "bt709_color": True,
        "sample_rate": 32000
    }


# ==============================================================================
# 5. FLF2V 首尾帧配对与防渐变融化 (Anti-Gradient Melting) 校验
# ==============================================================================

def validate_flf2v_pair(start_image: str, end_image: str) -> Dict[str, Any]:
    """
    Validates start and end frame pairing:
    - Both images exist
    - Dimensions are identical and multiple of 16
    - Computes luminance delta and RGB histogram difference.
    - If luminance shift > 40.0, warns about potential latent gradient melting / fading artifacts.
    """
    if Image is None or ImageStat is None:
        raise RuntimeError("Pillow is required for validate_flf2v_pair.")

    s_p = Path(start_image)
    e_p = Path(end_image)
    if not s_p.exists():
        raise FileNotFoundError(f"Start image not found: {start_image}")
    if not e_p.exists():
        raise FileNotFoundError(f"End image not found: {end_image}")

    with Image.open(s_p) as s_img, Image.open(e_p) as e_img:
        s_rgb = s_img.convert("RGB")
        e_rgb = e_img.convert("RGB")

        s_w, s_h = s_rgb.size
        e_w, e_h = e_rgb.size

        if (s_w, s_h) != (e_w, e_h):
            raise ValueError(
                f"FLF2V 尺寸不一致：首帧尺寸为 {s_w}x{s_h}，尾帧尺寸为 {e_w}x{e_h}！首尾帧必须保持严格相同的像素规格。"
            )

        if s_w % 16 != 0 or s_h % 16 != 0:
            raise ValueError(
                f"FLF2V 尺寸非 16 整数倍：当前规格 {s_w}x{s_h}，请先执行 align_dimensions_16 对齐！"
            )

        s_stats = ImageStat.Stat(s_rgb)
        e_stats = ImageStat.Stat(e_rgb)

        s_r_mean, s_g_mean, s_b_mean = s_stats.mean[0], s_stats.mean[1], s_stats.mean[2]
        e_r_mean, e_g_mean, e_b_mean = e_stats.mean[0], e_stats.mean[1], e_stats.mean[2]

        s_luma = 0.2126 * s_r_mean + 0.7152 * s_g_mean + 0.0722 * s_b_mean
        e_luma = 0.2126 * e_r_mean + 0.7152 * e_g_mean + 0.0722 * e_b_mean
        luma_diff = abs(e_luma - s_luma)

        color_diff = math.sqrt(
            (e_r_mean - s_r_mean) ** 2 +
            (e_g_mean - s_g_mean) ** 2 +
            (e_b_mean - s_b_mean) ** 2
        )

        warning = None
        recommend_tone_compensate = False
        if luma_diff > 40.0:
            recommend_tone_compensate = True
            warning = (
                f"⚠️ 检测到首尾帧亮度均值差较大 (ΔLuma = {luma_diff:.1f} > 40.0)。"
                f"为防止潜空间生成中出现生硬淡入淡出的'渐变融化Bug' (Gradient Melting Bug)，"
                f"建议启用 ToneCompensate (frame_shift 亮度补偿) 或调整首尾帧光影构图对齐！"
            )

        return {
            "status": "passed",
            "width": s_w,
            "height": s_h,
            "is_multiple_of_16": True,
            "is_multiple_of_32": (s_w % 32 == 0 and s_h % 32 == 0),
            "start_luma": round(s_luma, 2),
            "end_luma": round(e_luma, 2),
            "luma_diff": round(luma_diff, 2),
            "color_diff": round(color_diff, 2),
            "recommend_tone_compensate": recommend_tone_compensate,
            "warning": warning
        }


# ==============================================================================
# 6. 配置综合检验器与 CLI 入口
# ==============================================================================

def validate_config(config_path: str, dry_run: bool = False) -> Dict[str, Any]:
    """
    Validates a structured JSON configuration for MiniMax H3 generation.
    Supports modes: t2v, i2v, flf2v, ref2va.
    Supports both start_image/first_frame and end_image/last_frame naming.
    """
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(p, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    tasks = cfg if isinstance(cfg, list) else [cfg]
    results = []

    for idx, t in enumerate(tasks, 1):
        mode = t.get("mode", "t2v").lower()
        task_id = t.get("id", f"task_{idx}")
        w = t.get("width", 960)
        h = t.get("height", 544)
        raw_len = t.get("length", 124)

        aligned_w, aligned_h = align_dimensions_16(w, h)
        aligned_len = align_frame_count(raw_len)

        task_res: Dict[str, Any] = {
            "task_id": task_id,
            "mode": mode,
            "original_resolution": f"{w}x{h}",
            "aligned_resolution": f"{aligned_w}x{aligned_h}",
            "original_frames": raw_len,
            "aligned_frames": aligned_len,
            "is_multiple_of_32": (aligned_w % 32 == 0 and aligned_h % 32 == 0),
            "status": "valid"
        }

        if mode == "i2v":
            first_frame = t.get("first_frame") or t.get("start_image")
            if not first_frame:
                raise ValueError(f"Task '{task_id}' (mode=i2v) 缺少必需的 'first_frame' (或 'start_image') 字段！")
            if not dry_run and not Path(first_frame).exists():
                raise FileNotFoundError(f"Task '{task_id}' first_frame 不存在: {first_frame}")
            task_res["first_frame"] = first_frame

        elif mode == "flf2v":
            first_frame = t.get("first_frame") or t.get("start_image")
            last_frame = t.get("last_frame") or t.get("end_image")
            if not first_frame or not last_frame:
                raise ValueError(f"Task '{task_id}' (mode=flf2v) 必须同时提供 'first_frame' 与 'last_frame'！")
            if not dry_run:
                pair_report = validate_flf2v_pair(first_frame, last_frame)
                task_res["pair_report"] = pair_report
            task_res["first_frame"] = first_frame
            task_res["last_frame"] = last_frame

        elif mode == "ref2va":
            imgs = t.get("ref_images", [])
            vids = t.get("ref_videos", [])
            auds = t.get("ref_audios", [])
            prompt = t.get("prompt", "")
            quota_report = validate_ref2va_quotas(imgs, vids, auds, prompt)
            task_res["quota_report"] = quota_report

        elif mode == "t2v":
            pass
        else:
            raise ValueError(f"不支持的生成模式: '{mode}'。仅支持 t2v, i2v, flf2v, ref2va。")

        results.append(task_res)

    return {"status": "all_passed", "tasks": results}


def run_unit_tests():
    """Runs rigorous self-tests verifying positive and negative validation boundaries."""
    print("🧪 正在运行 scripts/validate_inputs.py 单元测试套件...\n")

    # Test 1: Dimension alignment 16 and 32
    w16, h16 = align_dimensions_16(955, 540)
    assert (w16, h16) == (960, 544), f"align_dimensions_16 failed: {(w16, h16)}"
    assert w16 % 16 == 0 and h16 % 16 == 0
    w32, h32 = align_dimensions_32(955, 540)
    assert (w32, h32) == (960, 544), f"align_dimensions_32 failed: {(w32, h32)}"
    assert w32 % 32 == 0 and h32 % 32 == 0
    print("  ✅ [Test 1] 尺寸 16/32 倍数对齐验证通过 (955x540 -> 960x544, 满足 16x60 / 32x30 与 16x34 / 32x17)")

    # Test 2: Frame grid alignment (17k + 5)
    f = align_frame_count(120)
    assert f == 124, f"align_frame_count failed: {f}"
    assert (f - 5) % 17 == 0
    print("  ✅ [Test 2] 帧数 17k+5 网格对齐验证通过 (120 -> 124 帧)")

    # Test 3: Tag normalization
    p_cn = "镜头参考 @视频1，主体外观锁定 @图片1，背景声 @音频1 与 @音频2。"
    p_norm = normalize_h3_reference_tags(p_cn)
    assert "<Picture 1>" in p_norm and "<Video 1>" in p_norm and "<Audio 1>" in p_norm and "<Audio 2>" in p_norm
    print(f"  ✅ [Test 3] 提示词双向兼容转换通过: {p_norm}")

    # Test 4: Ref2VA Pure Audio Red Line
    try:
        validate_ref2va_quotas(images=[], videos=[], audios=["test.wav"])
        assert False, "Should have raised ValueError on pure audio"
    except ValueError as e:
        assert "音频绝不可作为唯一输入独立提交" in str(e)
        print("  ✅ [Test 4] Ref2VA 纯音频红线拦截验证通过 (已阻止提交)")

    # Test 5: Ref2VA Quota Limits
    try:
        validate_ref2va_quotas(images=[f"img_{i}.png" for i in range(10)], videos=[], audios=[])
        assert False, "Should have rejected > 9 images"
    except ValueError as e:
        assert "图片配额超标" in str(e)
        print("  ✅ [Test 5a] Ref2VA 图片上限 >9 张拦截验证通过")

    try:
        validate_ref2va_quotas(images=["img.png"], videos=[{"path": f"v_{i}.mp4", "duration": 3.0} for i in range(4)], audios=[])
        assert False, "Should have rejected > 3 videos"
    except ValueError as e:
        assert "视频配额超标" in str(e)
        print("  ✅ [Test 5b] Ref2VA 视频上限 >3 段拦截验证通过")

    try:
        validate_ref2va_quotas(
            images=[f"i_{i}.png" for i in range(8)],
            videos=[{"path": f"v_{i}.mp4", "duration": 3.0} for i in range(3)],
            audios=[{"path": f"a_{i}.wav", "duration": 3.0} for i in range(3)]
        )
        assert False, "Should have rejected > 12 total files"
    except ValueError as e:
        assert "总文件配额超标" in str(e)
        print("  ✅ [Test 5c] Ref2VA 混合总文件 >12 个拦截验证通过")

    try:
        validate_ref2va_quotas(
            images=["i.png"],
            videos=[{"path": "v1.mp4", "duration": 10.0}, {"path": "v2.mp4", "duration": 6.0}],
            audios=[]
        )
        assert False, "Should have rejected total video duration > 15s"
    except ValueError as e:
        assert "累计总时长超标" in str(e)
        print("  ✅ [Test 5d] Ref2VA 视频累计时长 >15s 拦截验证通过")

    # Test 6: Valid Ref2VA & @ binding check with priority order
    prompt_sample = "镜头以 @视频1 的 24fps 弧形推镜为准，主体角色外观锁定 @图片1，背景声景复刻 @音频1，独立低频脉冲参考 @音频2。"
    valid_res = validate_ref2va_quotas(
        images=["char_01.png"],
        videos=[{"path": "motion.mp4", "duration": 5.16}],
        audios=[{"path": "pulse.wav", "duration": 5.16}],
        prompt=prompt_sample
    )
    assert valid_res["status"] == "passed"
    assert valid_res["counts"]["total_files"] == 3
    tracks = valid_res["bindings_map"]["audio_tracks"]
    assert len(tracks) == 2
    assert tracks[0]["index"] == 1 and tracks[0]["ref_type"] == "video_audio"
    assert tracks[1]["index"] == 2 and tracks[1]["ref_type"] == "standalone_audio"
    assert "<Picture 1>" in valid_res["detected_prompt_tags"]
    assert "<Video 1>" in valid_res["detected_prompt_tags"]
    assert "<Audio 1>" in valid_res["detected_prompt_tags"]
    assert "<Audio 2>" in valid_res["detected_prompt_tags"]
    print("  ✅ [Test 6] Ref2VA 合法组合与视频附带音轨优先编号验证通过")

    print("\n🎉 全套单元测试执行通过！所有硬性物理门禁均准确就绪。")


def main():
    parser = argparse.ArgumentParser(description="MiniMax H3 Input Validation & Preprocessing Gatekeeper")
    parser.add_argument("--mode", choices=["i2v", "flf2v", "ref2va", "t2v"], help="Validation mode")
    parser.add_argument("--config", help="Path to JSON config template to validate")
    parser.add_argument("--dry-run", action="store_true", help="Skip file existence checks")
    parser.add_argument("--test", action="store_true", help="Run internal unit tests")
    args = parser.parse_args()

    if args.test:
        run_unit_tests()
        sys.exit(0)

    if args.config:
        try:
            report = validate_config(args.config, dry_run=args.dry_run)
            print("=" * 68)
            print("🛡️ 输入门禁校验结果: 全部通过 (PASS)")
            print(json.dumps(report, ensure_ascii=False, indent=2))
            print("=" * 68)
            sys.exit(0)
        except Exception as e:
            print(f"❌ 输入门禁拦截失败: {e}", file=sys.stderr)
            sys.exit(1)

    parser.print_help()


if __name__ == "__main__":
    main()
