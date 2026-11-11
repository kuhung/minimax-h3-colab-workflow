#!/usr/bin/env python3
"""
MiniMax-H3-Image Pure Open-Source Demo Script (T2I & Multi-Reference Editing).

Supports two 100% open-source local backends (NO MiniMax commercial API calls):
1. `comfyui` (recommended for 24GB GPUs like Colab L4 / RTX 4090 / 5090):
   Pure Python stdlib client driving ComfyUI >= 0.45.0 with Comfy-Org quantized
   weights (`int8_convrot` or `w4a8`) via `MiniMaxH3ImageConditioning` +
   6-step distilled `KSampler` (cfg=1.0, sampler=euler, scheduler=simple).
2. `diffusers` (official HF Modular Diffusers integration, merged in PR #15230):
   Uses `ModularPipeline.from_pretrained("MiniMaxAI/MiniMax-H3-Image")` with
   `ComponentsManager` auto CPU offload (~65 GB peak VRAM at 4K BF16; A100-80G /
   H100 / H200+ recommended). Optional open-weights prompt expansion via
   `diffusers/minimax-h3-image-pe-blocks` (`MiniMaxAI/MiniMax-H3-Image-PE`).

Examples:
  # 1. Inspect ComfyUI API payload without a running server
  python3 scripts/demo_h3_image.py --backend comfyui --dry-run

  # 2. ComfyUI Text-to-Image (24GB L4 / 4090 with INT8 ConvRot or W4A8)
  python3 scripts/demo_h3_image.py --backend comfyui \
      --comfy-url http://127.0.0.1:8188 \
      --size 2048x2048 \
      --output outputs/h3_image_t2i.png

  # 3. ComfyUI Image Editing (<Picture 1> reference)
  python3 scripts/demo_h3_image.py --backend comfyui \
      --image reference1.png \
      --prompt "Keep the street and buildings in <Picture 1> unchanged, replace the yellow taxi with a pastel pink vintage ice-cream truck parked at the same curb in soft golden-hour sunlight." \
      --output outputs/h3_image_edit.png

  # 4. Official Diffusers Text-to-Image (80GB+ GPU, 16MP native 4:3)
  python3 scripts/demo_h3_image.py --backend diffusers \
      --size 4736x3552 \
      --output outputs/h3_image_diffusers_t2i.png

  # 5. Official Diffusers with open-source Prompt Expander (MiniMax-H3-Image-PE)
  python3 scripts/demo_h3_image.py --backend diffusers \
      --expand-prompt \
      --prompt "A brass astrolabe resting on an oak map table in a sunlit library" \
      --output outputs/h3_image_pe.png
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import mimetypes
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any

# Official 16MP preset dimensions (all multiples of 32)
PRESET_SIZES_16MP: dict[str, tuple[int, int]] = {
    "1:1": (4096, 4096),
    "4:3": (4736, 3552),
    "3:4": (3552, 4736),
    "3:2": (5024, 3360),
    "2:3": (3360, 5024),
    "16:9": (5472, 3072),
    "9:16": (3072, 5472),
    "21:9": (6272, 2688),
}

DEFAULT_T2I_PROMPT = (
    "A photorealistic architectural photograph of a tranquil glass-and-cedar tea pavilion "
    "built beside a misty mountain koi pond at early dawn. Warm amber lantern light glows "
    "softly inside the wooden lattice screens, casting crisp reflections across the dark "
    "rippling water and wet slate stepping stones. Surrounding moss-covered boulders, twisted "
    "red Japanese maples, and fine cedar grain are rendered in sharp natural detail under "
    "cool morning mist, shot on medium-format film with realistic depth and natural contrast."
)

# Default Comfy-Org quantized filenames (ComfyUI >= 0.45.0)
DEFAULT_COMFY_UNET = "minimax_h3_image_pruned_int8_convrot.safetensors"
DEFAULT_COMFY_CLIP = "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
DEFAULT_COMFY_VAE = "minimax_h3_image_vae_fp16.safetensors"


def round_to_32(value: int) -> int:
    """Align a pixel dimension to the model's required 32px grid (16x VAE * 2x2 patch)."""
    return max(32, round(value / 32) * 32)


def parse_size(size_str: str) -> tuple[int, int]:
    """Parse WIDTHxHEIGHT or an aspect ratio preset (e.g. '16:9') into (width, height)."""
    s = size_str.strip()
    if s in PRESET_SIZES_16MP:
        return PRESET_SIZES_16MP[s]
    if "x" in s.lower():
        w_str, h_str = s.lower().split("x", 1)
        w, h = round_to_32(int(w_str)), round_to_32(int(h_str))
        return w, h
    raise ValueError(f"Invalid --size '{size_str}'. Use WIDTHxHEIGHT (e.g. 2048x2048) or preset ({', '.join(PRESET_SIZES_16MP)}).")


def size_for_aspect_ratio(aspect_ratio: str, target_area: int = 4096 * 4096) -> tuple[int, int]:
    """Compute ~16MP (or target_area) dimensions aligned to 32px for an aspect ratio like '4:3'."""
    if aspect_ratio in PRESET_SIZES_16MP and target_area == 4096 * 4096:
        return PRESET_SIZES_16MP[aspect_ratio]
    rx, ry = (int(v) for v in aspect_ratio.split(":"))
    width = round_to_32(round(math.sqrt(target_area * rx / ry)))
    height = round_to_32(round(math.sqrt(target_area * ry / rx)))
    return width, height


# ---------------------------------------------------------------------------
# Backend 1: ComfyUI >= 0.45.0 Headless API (Quantized INT8 / W4A8 for 24GB+)
# ---------------------------------------------------------------------------


def _http_json(url: str, payload: dict[str, Any] | None = None, timeout: int = 30) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc


def _upload_comfy_image(server_url: str, image_path: Path) -> str:
    """Upload a local reference image to ComfyUI input/ via POST /upload/image."""
    if not image_path.is_file():
        raise FileNotFoundError(f"Reference image not found: {image_path}")

    boundary = f"minimax-h3-image-{uuid.uuid4().hex}"
    ctype = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
    parts = [
        f'--{boundary}\r\nContent-Disposition: form-data; name="type"\r\n\r\ninput\r\n'.encode(),
        f'--{boundary}\r\nContent-Disposition: form-data; name="overwrite"\r\n\r\ntrue\r\n'.encode(),
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="image"; '
            f'filename="{image_path.name}"\r\nContent-Type: {ctype}\r\n\r\n'
        ).encode(),
        image_path.read_bytes(),
        f"\r\n--{boundary}--\r\n".encode(),
    ]
    body = b"".join(parts)
    req = urllib.request.Request(
        f"{server_url.rstrip('/')}/upload/image",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        info = json.loads(resp.read().decode("utf-8"))
    subfolder = info.get("subfolder", "")
    name = info["name"]
    return f"{subfolder}/{name}" if subfolder else name


def build_comfyui_workflow(
    prompt: str,
    width: int,
    height: int,
    seed: int,
    unet_name: str,
    clip_name: str,
    vae_name: str,
    uploaded_images: list[str],
    filename_prefix: str = "minimax_h3_image_demo",
) -> dict[str, Any]:
    """Build the ComfyUI v0.45.0+ API prompt graph for MiniMax-H3-Image (T2I or Edit)."""
    cond_inputs: dict[str, Any] = {
        "clip": ["2", 0],
        "vae": ["3", 0],
        "prompt": prompt,
        "width": width,
        "height": height,
    }
    workflow: dict[str, Any] = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": unet_name, "weight_dtype": "default"},
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": clip_name, "type": "minimax", "device": "default"},
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": vae_name},
        },
    }

    next_id = 10
    for idx, img_ref in enumerate(uploaded_images, start=1):
        load_id = str(next_id)
        next_id += 1
        workflow[load_id] = {
            "class_type": "LoadImage",
            "inputs": {"image": img_ref},
        }
        cond_inputs[f"images.image_{idx}"] = [load_id, 0]

    workflow.update(
        {
            "4": {"class_type": "MiniMaxH3ImageConditioning", "inputs": cond_inputs},
            "5": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}},
            "6": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["1", 0],
                    "positive": ["4", 0],
                    "negative": ["5", 0],
                    "latent_image": ["4", 1],
                    "seed": seed,
                    "steps": 6,
                    "cfg": 1.0,
                    "sampler_name": "euler",
                    "scheduler": "simple",
                    "denoise": 1.0,
                },
            },
            "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["3", 0]}},
            "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": filename_prefix}},
        }
    )
    return workflow


def run_comfyui(args: argparse.Namespace, width: int, height: int) -> None:
    server_url = args.comfy_url.rstrip("/")
    if args.dry_run:
        fake_refs = [p.name for p in args.image]
        wf = build_comfyui_workflow(
            prompt=args.prompt,
            width=width,
            height=height,
            seed=args.seed,
            unet_name=args.comfy_unet,
            clip_name=args.comfy_clip,
            vae_name=args.comfy_vae,
            uploaded_images=fake_refs,
            filename_prefix=args.output.stem,
        )
        print(json.dumps(wf, indent=2, ensure_ascii=False))
        return

    # 1. Verify ComfyUI >= 0.45.0 registers MiniMaxH3ImageConditioning
    try:
        info = _http_json(f"{server_url}/object_info/MiniMaxH3ImageConditioning", timeout=10)
    except Exception as exc:
        raise RuntimeError(
            f"Cannot reach ComfyUI at {server_url} or MiniMaxH3ImageConditioning is missing. "
            "Ensure ComfyUI >= v0.45.0 is running."
        ) from exc
    if "MiniMaxH3ImageConditioning" not in info:
        raise RuntimeError("MiniMaxH3ImageConditioning node not found. Update ComfyUI to v0.45.0+.")

    # 2. Upload reference images (if any) in exact <Picture 1>, <Picture 2> order
    if len(args.image) > 16:
        raise ValueError(f"MiniMax-H3-Image supports up to 16 reference images, got {len(args.image)}.")
    uploaded_refs = [_upload_comfy_image(server_url, p) for p in args.image]

    wf = build_comfyui_workflow(
        prompt=args.prompt,
        width=width,
        height=height,
        seed=args.seed,
        unet_name=args.comfy_unet,
        clip_name=args.comfy_clip,
        vae_name=args.comfy_vae,
        uploaded_images=uploaded_refs,
        filename_prefix=args.output.stem,
    )

    # 3. Queue prompt
    client_id = f"h3-image-demo-{uuid.uuid4().hex[:8]}"
    resp = _http_json(f"{server_url}/prompt", payload={"prompt": wf, "client_id": client_id}, timeout=30)
    if resp.get("node_errors") or resp.get("error"):
        raise RuntimeError(f"ComfyUI rejected workflow: {json.dumps(resp, ensure_ascii=False)}")
    prompt_id = resp["prompt_id"]
    print(f"Queued ComfyUI prompt_id={prompt_id} ({width}x{height}, 6 steps, CFG=1.0)...")

    # 4. Poll /history/{prompt_id}
    start = time.time()
    record = None
    while time.time() - start < args.timeout:
        history = _http_json(f"{server_url}/history/{prompt_id}", timeout=10)
        if prompt_id in history:
            record = history[prompt_id]
            break
        time.sleep(2)

    if record is None:
        try:
            _http_json(f"{server_url}/interrupt", payload={}, timeout=5)
        except Exception:
            pass
        raise TimeoutError(f"ComfyUI generation timed out after {args.timeout}s.")

    status = record.get("status", {})
    if status.get("status_str") == "error":
        raise RuntimeError(f"ComfyUI execution failed: {json.dumps(status, ensure_ascii=False)}")

    # 5. Download generated image via /view
    images_out = record.get("outputs", {}).get("8", {}).get("images", [])
    if not images_out:
        raise RuntimeError(f"No output images found in ComfyUI history: {record.get('outputs')}")

    first = images_out[0]
    query = urllib.parse.urlencode(
        {
            "filename": first["filename"],
            "subfolder": first.get("subfolder", ""),
            "type": first.get("type", "output"),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(f"{server_url}/view?{query}", timeout=60) as img_resp:
        args.output.write_bytes(img_resp.read())

    elapsed = time.time() - start
    print(f"Saved {args.output} in {elapsed:.1f}s.")


# ---------------------------------------------------------------------------
# Backend 2: HuggingFace Diffusers ModularPipeline (Native BF16 Open Weights)
# ---------------------------------------------------------------------------


def run_diffusers(args: argparse.Namespace, width: int, height: int) -> None:
    if args.dry_run:
        summary = {
            "backend": "diffusers",
            "pipeline": "ModularPipeline (MiniMaxH3ImageModularPipeline)",
            "model_id": args.model_id,
            "expand_prompt": args.expand_prompt,
            "prompt": args.prompt,
            "images": [str(p) for p in args.image],
            "width": width,
            "height": height,
            "min_generation_area": args.min_generation_area,
            "seed": args.seed,
            "offload": not args.no_offload,
            "output": str(args.output),
        }
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    import torch
    from PIL import Image
    from diffusers import ComponentsManager, ModularPipeline

    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if vram_gb < 60:
            print(
                f"[Warning] Detected {vram_gb:.1f} GB VRAM. Native BF16 Diffusers peaks around ~65 GB "
                "even with auto CPU offload (recommended: A100 80GB / H100 / H200). "
                "For 24GB GPUs (L4 / 4090 / 5090), use --backend comfyui with quantized weights.",
                file=sys.stderr,
            )

    ref_images = [Image.open(p).convert("RGB") for p in args.image]
    if len(ref_images) > 16:
        raise ValueError(f"MiniMax-H3-Image supports up to 16 reference images, got {len(ref_images)}.")

    prompt = args.prompt
    if args.expand_prompt:
        print("Expanding prompt with open-weights MiniMax-H3-Image-PE...")
        enhancer = ModularPipeline.from_pretrained("diffusers/minimax-h3-image-pe-blocks", trust_remote_code=True)
        enhancer.load_components(dtype=torch.bfloat16, device_map=args.device)
        pe_kwargs: dict[str, Any] = {"prompt": prompt}
        if ref_images:
            pe_kwargs["image"] = ref_images if len(ref_images) > 1 else ref_images[0]
        pe_state = enhancer(**pe_kwargs)
        prompt = pe_state.get("prompt") or prompt
        aspect_ratio = pe_state.get("aspect_ratio")
        if aspect_ratio and not args.size_explicit:
            width, height = size_for_aspect_ratio(aspect_ratio)
        print(f"Expanded prompt ({width}x{height}): {prompt[:160]}...")
        del enhancer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.no_offload:
        pipe = ModularPipeline.from_pretrained(args.model_id)
        pipe.load_components(dtype=torch.bfloat16, device_map=args.device)
    else:
        manager = ComponentsManager()
        manager.enable_auto_cpu_offload(device=args.device)
        pipe = ModularPipeline.from_pretrained(args.model_id, components_manager=manager)
        pipe.load_components(dtype=torch.bfloat16)

    call_kwargs: dict[str, Any] = {
        "prompt": prompt,
        "height": height,
        "width": width,
        "min_generation_area": args.min_generation_area,
        "generator": torch.Generator(device="cpu").manual_seed(args.seed),
        "output": "images",
    }
    if ref_images:
        call_kwargs["image"] = ref_images if len(ref_images) > 1 else ref_images[0]

    start = time.time()
    image = pipe(**call_kwargs)[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    print(f"Saved {args.output} ({image.size[0]}x{image.size[1]}) in {time.time() - start:.1f}s.")


# ---------------------------------------------------------------------------
# Backend 3: stable-diffusion.cpp (`sd-cli`, No ComfyUI, GGUF / INT8 for 24GB)
# ---------------------------------------------------------------------------


def build_sdcpp_command(args: argparse.Namespace, width: int, height: int) -> list[str]:
    """Build the verified `sd-cli` command line for MiniMax-H3-Image (T2I or Edit)."""
    cmd = [
        str(args.sd_cli),
        "--diffusion-model",
        str(args.sd_diffusion_model),
        "--vae",
        str(args.sd_vae),
        "--llm",
        str(args.sd_llm),
    ]
    if args.sd_llm_vision:
        cmd.extend(["--llm_vision", str(args.sd_llm_vision)])
    for ref in args.image:
        cmd.extend(["-r", str(ref)])
    cmd.extend(
        [
            "-p",
            args.prompt,
            "--cfg-scale",
            "1.0",
            "--steps",
            "6",
            "--sampling-method",
            "euler",
            "--sigmas",
            "minimax-h3-image",
            "--seed",
            str(args.seed),
            "-W",
            str(width),
            "-H",
            str(height),
            "--offload-to-cpu",
            "--rng",
            "cpu",
            "--flash-attn",
            "off,diffusion=on",
            "--vae-tiling",
            "-v",
            "-o",
            str(args.output),
        ]
    )
    return cmd


def run_sdcpp(args: argparse.Namespace, width: int, height: int) -> None:
    if len(args.image) > 16:
        raise ValueError(f"MiniMax-H3-Image supports up to 16 reference images, got {len(args.image)}.")
    cmd = build_sdcpp_command(args, width, height)
    if args.dry_run:
        print(json.dumps({"backend": "sdcpp", "command": cmd}, indent=2, ensure_ascii=False))
        return

    for label, path in [
        ("sd-cli binary", args.sd_cli),
        ("diffusion model", args.sd_diffusion_model),
        ("VAE", args.sd_vae),
        ("LLM text encoder", args.sd_llm),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")
    for ref in args.image:
        if not ref.is_file():
            raise FileNotFoundError(f"Reference image not found: {ref}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    subprocess.run(cmd, check=True)
    print(f"Saved {args.output} ({width}x{height}) in {time.time() - start:.1f}s.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MiniMax-H3-Image Open-Source Demo (sd.cpp / ComfyUI Quantized / HF Diffusers)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--backend",
        choices=["sdcpp", "comfyui", "diffusers"],
        default="sdcpp",
        help="Open-source execution backend (default: sdcpp — no ComfyUI needed, fits 24GB L4/4090)",
    )
    parser.add_argument("--prompt", default=DEFAULT_T2I_PROMPT, help="Detailed English prompt (use <Picture 1>.. for edits)")
    parser.add_argument("--image", action="append", type=Path, default=[], help="Reference image path; repeat for <Picture 2>, ...")
    parser.add_argument("--size", default="2048x2048", help="WIDTHxHEIGHT or 16MP preset (1:1, 4:3, 3:2, 16:9, 21:9, etc.)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (CPU generator)")
    parser.add_argument("--output", type=Path, default=Path("outputs/minimax_h3_image_demo.png"), help="Output PNG path")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved workflow/config without calling GPU/server")

    # stable-diffusion.cpp options
    parser.add_argument("--sd-cli", type=Path, default=Path("/content/stable-diffusion.cpp/build/bin/sd-cli"), help="Path to compiled sd-cli binary")
    parser.add_argument(
        "--sd-diffusion-model",
        type=Path,
        default=Path(f"/content/models/diffusion_models/{DEFAULT_COMFY_UNET}"),
        help="Path to MiniMax-H3-Image diffusion safetensors/gguf",
    )
    parser.add_argument(
        "--sd-vae",
        type=Path,
        default=Path(f"/content/models/vae/{DEFAULT_COMFY_VAE}"),
        help="Path to MiniMax-H3-Image 2D VAE safetensors",
    )
    parser.add_argument(
        "--sd-llm",
        type=Path,
        default=Path("/content/models/text_encoders/qwen3vl_32b_minimax_h3-Q4_K_M.gguf"),
        help="Path to MiniMax-H3 Qwen3-VL-32B text encoder (GGUF or safetensors)",
    )
    parser.add_argument("--sd-llm-vision", type=Path, default=None, help="Optional separate Qwen3-VL vision tower path (--llm_vision)")

    # ComfyUI options
    parser.add_argument("--comfy-url", default="http://127.0.0.1:8188", help="ComfyUI server base URL")
    parser.add_argument("--comfy-unet", default=DEFAULT_COMFY_UNET, help="Diffusion model filename in ComfyUI/models/diffusion_models")
    parser.add_argument("--comfy-clip", default=DEFAULT_COMFY_CLIP, help="Qwen3-VL filename in ComfyUI/models/text_encoders")
    parser.add_argument("--comfy-vae", default=DEFAULT_COMFY_VAE, help="Image VAE filename in ComfyUI/models/vae")
    parser.add_argument("--timeout", type=int, default=900, help="ComfyUI polling timeout in seconds")

    # Diffusers options
    parser.add_argument("--model-id", default="MiniMaxAI/MiniMax-H3-Image", help="HuggingFace repo ID or local path")
    parser.add_argument("--device", default="cuda", help="Torch accelerator device (cuda / mps)")
    parser.add_argument("--min-generation-area", type=int, default=2048 * 2048, help="Min internal canvas area (1024 to disable upscaling)")
    parser.add_argument("--expand-prompt", action="store_true", help="Diffusers only: expand short prompt via MiniMax-H3-Image-PE")
    parser.add_argument("--no-offload", action="store_true", help="Diffusers only: keep all components resident without CPU offload")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.size_explicit = any(a == "--size" or a.startswith("--size=") for a in sys.argv[1:])
    width, height = parse_size(args.size)

    if args.backend == "sdcpp":
        run_sdcpp(args, width, height)
    elif args.backend == "comfyui":
        run_comfyui(args, width, height)
    else:
        run_diffusers(args, width, height)


if __name__ == "__main__":
    main()
