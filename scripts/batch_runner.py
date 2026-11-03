#!/usr/bin/env python3
"""
Headless Batch Runner for MiniMax H3 on ComfyUI (2.0 Architecture).
Features:
1. Native 2.0 Topology Support: MiniMaxH3ImageToVideo + BasicGuider + SamplerCustomAdvanced (CFG=1.0).
2. Backward compatibility with legacy KSampler graphs.
3. 17k + 5 Frame Alignment (align_frame_count).
4. Structured Multi-shot / Multimodal Prompt Assembler (format_h3_prompt).
5. Dynamic /object_info Contract Validation & Unsupported Parameter Pruning.
6. Timeout Interrupt & Direct Google Drive Export.
"""

import os
import sys
import copy
import json
import time
import shutil
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

def align_frame_count(n: int) -> int:
    """Align frame count upwards to 17k + 5 (MiniMax H3 latent temporal grid at 24fps)."""
    n = max(5, int(n))
    return n + (5 - n % 17) % 17

def format_h3_prompt(task: dict) -> str:
    """
    Format MiniMax H3 structured prompt according to official multi-shot / multimodal spec.
    If task contains 'shots' or 'overall_soundscape', formats:
      integrated_multimodal_description:
      [Shot 1] ...
      [Shot 2] At MM:SS.mmm, ...

      overall_soundscape: ...

      non_diegetic_music: ...
    If task already provides plain 'prompt', returns it directly.
    """
    if not isinstance(task, dict):
        return str(task)

    shots = task.get("shots")
    soundscape = task.get("overall_soundscape")
    music = task.get("non_diegetic_music")

    if not shots and not soundscape and not music:
        return task.get("prompt", "")

    parts = []
    if shots:
        shot_lines = []
        for i, s in enumerate(shots, 1):
            if isinstance(s, dict):
                t_str = s.get("time")
                desc = s.get("description", s.get("prompt", ""))
                if i == 1 and (not t_str or t_str == "00:00.000" or t_str == "0"):
                    shot_lines.append(f"[Shot 1] {desc}")
                else:
                    t_val = t_str if t_str else f"00:{((i-1)*5):02d}.000"
                    shot_lines.append(f"[Shot {i}] At {t_val}, {desc}")
            else:
                if i == 1:
                    shot_lines.append(f"[Shot 1] {s}")
                else:
                    shot_lines.append(f"[Shot {i}] At 00:{((i-1)*5):02d}.000, {s}")
        parts.append("integrated_multimodal_description:\n" + "\n".join(shot_lines))
    elif task.get("prompt"):
        parts.append(f"integrated_multimodal_description:\n{task['prompt']}")

    if soundscape:
        parts.append(f"overall_soundscape: {soundscape}")
    if music:
        parts.append(f"non_diegetic_music: {music}")

    return "\n\n".join(parts)

def fetch_object_info(server_url: str) -> Optional[dict]:
    """Queries ComfyUI /object_info endpoint."""
    try:
        req = urllib.request.Request(f"{server_url}/object_info")
        with urllib.request.urlopen(req, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as e:
        print(f"⚠️ 获取 ComfyUI /object_info 提示: {e}")
        return None

def validate_and_adapt_workflow(wf: dict, obj_info: Optional[dict]) -> dict:
    """
    Validates node classes and input fields against ComfyUI /object_info.
    Automatically prunes unsupported optional inputs to prevent ComfyUI 400 rejection.
    """
    if not obj_info:
        return wf

    cleaned_wf = copy.deepcopy(wf)
    for nid, node in cleaned_wf.items():
        if not isinstance(node, dict):
            continue
        ctype = node.get("class_type")
        if not ctype:
            continue
        if ctype not in obj_info:
            raise ValueError(f"节点 [{nid}] 类名 '{ctype}' 未在 ComfyUI 中注册！请检查扩展是否安装完整。")

        node_spec = obj_info[ctype].get("input", {})
        req = set(node_spec.get("required", {}).keys())
        opt = set(node_spec.get("optional", {}).keys())
        valid_keys = req | opt
        inputs = node.get("inputs", {})

        to_remove = [k for k in list(inputs.keys()) if k not in valid_keys]
        for k in to_remove:
            print(f"⚠️ 节点 [{nid}: {ctype}] 移除非当前版本支持的输入参数: '{k}'")
            del inputs[k]

        missing_req = [k for k in req if k not in inputs]
        if missing_req:
            raise ValueError(f"节点 [{nid}: {ctype}] 缺失必要输入参数: {missing_req}")

    return cleaned_wf

def post_prompt(server_url: str, prompt_workflow: dict) -> Optional[str]:
    """Submits workflow JSON to ComfyUI API /prompt."""
    data = json.dumps({"prompt": prompt_workflow, "client_id": "batch_runner_2.0"}).encode("utf-8")
    req = urllib.request.Request(f"{server_url}/prompt", data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            res = json.loads(response.read().decode("utf-8"))
            if res.get("error") or res.get("node_errors"):
                print(f"❌ ComfyUI 拒绝执行工作流: {res}")
                return None
            return res.get("prompt_id")
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        print(f"❌ 请求 ComfyUI API 失败 [HTTP {e.code}]: {err_body}")
        return None
    except Exception as e:
        print(f"❌ 请求 ComfyUI API 失败: {e}")
        return None

def interrupt_comfyui(server_url: str):
    """Sends /interrupt POST to halt runaway or timed out jobs and free VRAM."""
    try:
        req = urllib.request.Request(f"{server_url}/interrupt", data=b"{}", headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as response:
            if response.status in (200, 204):
                print("🛑 已成功下发 ComfyUI 中断指令 (/interrupt)，停止去噪并释放显存。")
    except Exception as e:
        print(f"⚠️ 下发 ComfyUI 中断请求失败: {e}")

def wait_for_completion(server_url: str, prompt_id: str, timeout_sec: int = 900) -> Optional[dict]:
    """Polls ComfyUI /history/{prompt_id} until generation finishes or times out."""
    start_time = time.time()
    last_print = 0
    while time.time() - start_time < timeout_sec:
        try:
            req = urllib.request.Request(f"{server_url}/history/{prompt_id}")
            with urllib.request.urlopen(req, timeout=5) as response:
                history = json.loads(response.read().decode("utf-8"))
                if prompt_id in history:
                    return history[prompt_id]
        except Exception:
            pass

        elapsed = int(time.time() - start_time)
        if elapsed - last_print >= 20:
            print(f"   ⏳ 渲染去噪中... 已耗时 {elapsed}s")
            last_print = elapsed

        time.sleep(3)

    print(f"❌ 任务超时 ({timeout_sec}s)，正在中断当前生成...")
    interrupt_comfyui(server_url)
    return None

def extract_output_files(history_data: dict, comfy_output_dir: Path) -> List[Path]:
    """Extracts actual output files from ComfyUI history outputs dict."""
    files = []
    outputs = history_data.get("outputs", {})
    for node_id, node_data in outputs.items():
        for category in ["videos", "gifs", "images", "audio"]:
            for item in node_data.get(category, []):
                filename = item.get("filename")
                if filename:
                    subfolder = item.get("subfolder", "")
                    target_file = comfy_output_dir / subfolder / filename
                    if target_file.exists():
                        files.append(target_file)
    return files

def run_batch(
    workflow_path: str,
    prompts_path: Optional[str] = None,
    output_dir: str = "/content/drive/MyDrive/MiniMax_Outputs",
    server_url: str = "http://127.0.0.1:8188",
    comfy_output_dir: str = "/content/ComfyUI/output",
    timeout_per_shot: int = 900,
    tasks_list: Optional[List[dict]] = None,
    active_models: Optional[Dict[str, str]] = None,
):
    print("=" * 68)
    print("🎬 MiniMax H3 批处理队列执行器 (Batch Runner 2.0 - BasicGuider / No-CFG)")
    print(f"📄 工作流模板: {workflow_path}")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    log_file = out_path / "render_history.jsonl"
    comfy_out_path = Path(comfy_output_dir)

    with open(workflow_path, "r", encoding="utf-8") as f:
        base_workflow = json.load(f)

    if tasks_list is not None:
        tasks = tasks_list
        print(f"📋 任务清单: 内存任务列表 ({len(tasks)} 条)")
    elif prompts_path:
        print(f"📋 任务清单: {prompts_path}")
        with open(prompts_path, "r", encoding="utf-8") as f:
            tasks = json.load(f)
    else:
        raise ValueError("必须提供 prompts_path 或 tasks_list 其中之一！")

    # 动态解析当前环境实际权重（优先使用传入的 active_models，未传时自动探测 models 目录）
    resolved_models = dict(active_models) if active_models else {}
    if not resolved_models and comfy_out_path.parent.exists():
        models_dir = comfy_out_path.parent / "models"
        diff_dir = models_dir / "diffusion_models"
        vae_dir = models_dir / "vae"
        lora_dir = models_dir / "loras"
        if diff_dir.exists():
            for cand in ["minimax_h3_fl2va_pruned_fp8_scaled.safetensors", "minimax_h3_fl2va_pruned_int8_convrot.safetensors"]:
                if (diff_dir / cand).exists():
                    resolved_models["diffusion_model"] = cand
                    break
        if vae_dir.exists():
            for cand in ["minimax_h3_video_vae_fp16.safetensors", "minimax_h3_video_vae_int8_convrot.safetensors"]:
                if (vae_dir / cand).exists():
                    resolved_models["video_vae"] = cand
                    break
        if lora_dir.exists():
            for cand in ["minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors", "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors"]:
                if (lora_dir / cand).exists():
                    resolved_models["lora"] = cand
                    break

    if resolved_models:
        print(f"🎯 动态绑定的活动权重: {resolved_models}")

    print(f"💾 最终输出落盘路径: {output_dir}")
    print("=" * 68)
    print(f"📊 共检测到 {len(tasks)} 个排产镜头任务\n")

    # 预检 /object_info 获取节点契约
    obj_info = fetch_object_info(server_url)

    for idx, task in enumerate(tasks, 1):
        task_id = task.get("id", f"shot_{idx:02d}")
        title = task.get("title", task_id)
        prompt_text = format_h3_prompt(task)
        neg_text = task.get("negative_prompt", "")
        steps = int(task.get("steps", 8))
        seed = int(task.get("seed", int(time.time())))
        width = int(task.get("width", 1344))
        height = int(task.get("height", 768))
        raw_length = int(task.get("length", 124))
        length = align_frame_count(raw_length)
        if length != raw_length:
            print(f"   ℹ️ 帧数自动网格对齐 (17k + 5): {raw_length} -> {length}")

        print(f"[{idx}/{len(tasks)}] 正在启动: {task_id} - {title}")
        print(f"   📐 规格: {width}x{height} | {length} 帧 | 步数 {steps} | 种子 {seed}")
        print(f"   📝 提示词: {prompt_text[:90]}...")

        # 克隆工作流并动态注入参数
        wf = copy.deepcopy(base_workflow)

        # 优先解析采样器图拓扑连接，准确定位正/负提示词与 Latent 节点 ID（不单纯依赖 _meta 标题）
        pos_node_ids = set()
        neg_node_ids = set()
        latent_node_ids = set()

        for nid, n in wf.items():
            if not isinstance(n, dict):
                continue
            ctype = n.get("class_type")
            inps = n.get("inputs", {})
            if ctype in ("KSampler", "KSamplerAdvanced"):
                if isinstance(inps.get("positive"), list) and len(inps["positive"]) > 0:
                    pos_node_ids.add(str(inps["positive"][0]))
                if isinstance(inps.get("negative"), list) and len(inps["negative"]) > 0:
                    neg_node_ids.add(str(inps["negative"][0]))
                if isinstance(inps.get("latent_image"), list) and len(inps["latent_image"]) > 0:
                    latent_node_ids.add(str(inps["latent_image"][0]))
            elif ctype == "BasicGuider":
                if isinstance(inps.get("conditioning"), list) and len(inps["conditioning"]) > 0:
                    pos_node_ids.add(str(inps["conditioning"][0]))

        # 遍历节点并匹配注入
        for node_id, node in wf.items():
            if not isinstance(node, dict):
                continue
            class_type = node.get("class_type")
            inputs = node.get("inputs", {})
            title_meta = node.get("_meta", {}).get("title", "")

            # 动态权重注入
            if class_type == "UNETLoader" and "diffusion_model" in resolved_models:
                inputs["unet_name"] = resolved_models["diffusion_model"]
            elif class_type == "VAELoader" and "video_vae" in resolved_models:
                if inputs.get("vae_name") != "minimax_h3_audio_vae_fp32.safetensors":
                    inputs["vae_name"] = resolved_models["video_vae"]
            elif class_type in ("LoraLoaderModelOnly", "LoraLoader") and "lora" in resolved_models:
                inputs["lora_name"] = resolved_models["lora"]

            # 2.0 新节点注入
            elif class_type == "MiniMaxH3ImageToVideo":
                inputs["prompt"] = prompt_text
                inputs["width"] = width
                inputs["height"] = height
                inputs["length"] = length
            elif class_type == "RandomNoise":
                inputs["noise_seed"] = seed
            elif class_type == "KSamplerSelect":
                if "sampler_name" in task:
                    inputs["sampler_name"] = task["sampler_name"]
            elif class_type == "BasicScheduler":
                inputs["steps"] = steps
                if "scheduler" in task:
                    inputs["scheduler"] = task["scheduler"]
                if "denoise" in task:
                    inputs["denoise"] = float(task["denoise"])

            # 输出节点注入
            elif class_type in ("SaveVideo", "SaveAnimatedWEBP", "VHS_VideoCombine"):
                inputs["filename_prefix"] = f"{task_id}_{seed}"
                if class_type == "SaveVideo":
                    inputs.setdefault("format", "mp4")
                elif class_type == "SaveAnimatedWEBP":
                    inputs.setdefault("fps", 24.0)
                    inputs.setdefault("lossless", False)
                    inputs.setdefault("quality", 85)
                    inputs.setdefault("method", "default")
            elif class_type == "CreateVideo":
                inputs.setdefault("fps", 24.0)

            # 向后兼容经典 1.0 节点
            elif class_type == "CLIPTextEncode":
                t_lower = title_meta.lower()
                is_pos = (node_id in pos_node_ids) or ("positive" in t_lower) or ("正向" in title_meta)
                is_neg = (node_id in neg_node_ids) or ("negative" in t_lower) or ("负向" in title_meta)
                if is_pos and not is_neg:
                    inputs["text"] = prompt_text
                    print(f"   📝 [节点 {node_id}] 正向提示词注入成功: {prompt_text[:50]}...")
                elif is_neg and not is_pos:
                    if neg_text:
                        inputs["text"] = neg_text
                        print(f"   🛡️ [节点 {node_id}] 负向提示词注入成功: {neg_text[:40]}...")
                elif not pos_node_ids and not neg_node_ids:
                    inputs["text"] = prompt_text
                    print(f"   📝 [节点 {node_id}] 独立提示词注入成功: {prompt_text[:50]}...")

            elif class_type in ("EmptyMiniMaxH3LatentAV", "EmptyH3LatentVideo", "EmptyLatentVideo") or (node_id in latent_node_ids):
                inputs["width"] = width
                inputs["height"] = height
                inputs["length"] = length
                print(f"   📐 [节点 {node_id}: {class_type}] 规格尺寸注入: {width}x{height}, {length} 帧")

            elif class_type in ("KSampler", "KSamplerAdvanced"):
                inputs["steps"] = steps
                inputs["seed"] = seed
                if "cfg" in task:
                    inputs["cfg"] = float(task["cfg"])

        # 校验并适配 ComfyUI 节点输入契约
        try:
            wf = validate_and_adapt_workflow(wf, obj_info)
        except Exception as ve:
            print(f"❌ 契约校验失败: {ve}")
            continue

        # 提交至 ComfyUI
        t_start = time.time()
        prompt_id = post_prompt(server_url, wf)
        if not prompt_id:
            print(f"❌ 提交任务 {task_id} 失败，跳过并记录。")
            log_record = {
                "timestamp": datetime.now().isoformat(),
                "task_id": task_id,
                "status": "submission_failed",
                "title": title,
                "cost_seconds": 0,
                "prompt": prompt_text,
                "saved_files": []
            }
            with open(log_file, "a", encoding="utf-8") as lf:
                lf.write(json.dumps(log_record, ensure_ascii=False) + "\n")
            continue

        print(f"   🚀 已推入执行队列 (Prompt ID: {prompt_id})")
        res = wait_for_completion(server_url, prompt_id, timeout_sec=timeout_per_shot)
        t_cost = round(time.time() - t_start, 2)

        # 状态检查
        status_info = res.get("status", {}) if res else {}
        status_str = status_info.get("status_str", "unknown")
        is_completed = status_info.get("completed", False)

        copied_files = []
        if res and is_completed and status_str == "success":
            # 优先从 outputs 解析生成文件
            found_files = extract_output_files(res, comfy_out_path)
            # 若为空，兜底 glob 扫描
            if not found_files and comfy_out_path.exists():
                found_files = list(comfy_out_path.glob(f"{task_id}_{seed}*"))

            if len(found_files) == 0:
                log_status = "failed_no_output"
                print(f"   ❌ 渲染完成但未产生任何输出文件！")
            else:
                print(f"   ✅ 渲染成功！耗时: {t_cost}s")
                for f in found_files:
                    target_dst = out_path / f.name
                    shutil.copy2(f, target_dst)
                    copied_files.append(str(target_dst))
                    print(f"   💾 成片已同步至云盘: {target_dst.name}")
                log_status = "success"
        else:
            log_status = "failed" if res else "timeout"
            err_msg = status_info.get("messages", [])
            print(f"   ❌ 任务状态: {log_status}，耗时: {t_cost}s，原因: {err_msg}")

        # 写入历史日志
        log_record = {
            "timestamp": datetime.now().isoformat(),
            "task_id": task_id,
            "title": title,
            "status": log_status,
            "cost_seconds": t_cost,
            "resolution": f"{width}x{height}",
            "length_frames": length,
            "steps": steps,
            "seed": seed,
            "prompt": prompt_text,
            "saved_files": copied_files
        }
        with open(log_file, "a", encoding="utf-8") as lf:
            lf.write(json.dumps(log_record, ensure_ascii=False) + "\n")

        print("-" * 50)

    print("\n" + "=" * 68)
    print(f"🎉 全部批处理任务执行完毕！")
    print(f"📂 成片与日志均已持久化至: {out_path.resolve()}")
    print("=" * 68)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ComfyUI Batch Runner for MiniMax H3 (2.0)")
    parser.add_argument("--workflow", default="workflows/h3_t2v_pdd8_production.json", help="Path to workflow JSON")
    parser.add_argument("--prompts", default="configs/prompts.example.json", help="Path to prompts JSON")
    parser.add_argument("--output", default="/content/drive/MyDrive/MiniMax_Outputs", help="Output directory")
    parser.add_argument("--server", default="http://127.0.0.1:8188", help="ComfyUI server URL")
    parser.add_argument("--comfy-output", default="/content/ComfyUI/output", help="ComfyUI output dir")
    parser.add_argument("--timeout", type=int, default=900, help="Max seconds per shot")
    args = parser.parse_args()

    run_batch(
        workflow_path=args.workflow,
        prompts_path=args.prompts,
        output_dir=args.output,
        server_url=args.server,
        comfy_output_dir=args.comfy_output,
        timeout_per_shot=args.timeout
    )
