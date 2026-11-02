#!/usr/bin/env python3
"""
Headless Batch Runner for MiniMax H3 on ComfyUI.
Sends prompts to ComfyUI local API (127.0.0.1:8188), monitors generation progress,
handles timeout interrupts, and exports output videos/audio and metadata directly to Google Drive.
"""

import os
import sys
import json
import time
import shutil
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

def post_prompt(server_url: str, prompt_workflow: dict) -> Optional[str]:
    """Submits workflow JSON to ComfyUI API /prompt."""
    data = json.dumps({"prompt": prompt_workflow}).encode("utf-8")
    req = urllib.request.Request(f"{server_url}/prompt", data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            res = json.loads(response.read().decode("utf-8"))
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
        if elapsed - last_print >= 15:
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
        for category in ["images", "videos", "gifs", "audio"]:
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
    tasks_list: Optional[List[dict]] = None
):
    print("=" * 60)
    print("🎬 MiniMax H3 批处理队列执行器 (Batch Runner)")
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

    print(f"💾 最终输出落盘路径: {output_dir}")
    print("=" * 60)
    print(f"📊 共检测到 {len(tasks)} 个排产镜头任务\n")

    for idx, task in enumerate(tasks, 1):
        task_id = task.get("id", f"shot_{idx:02d}")
        title = task.get("title", "未命名镜头")
        prompt_text = task.get("prompt", "")
        neg_text = task.get("negative_prompt", "")
        steps = task.get("steps", 8)
        cfg = task.get("cfg", 3.0)
        seed = task.get("seed", int(time.time()))
        width = task.get("width", 1344)
        height = task.get("height", 768)
        length = task.get("length", 121)

        print(f"[{idx}/{len(tasks)}] 正在启动: {task_id} - {title}")
        print(f"   📐 规格: {width}x{height} | {length} 帧 | 步数 {steps} | 种子 {seed}")
        print(f"   📝 提示词: {prompt_text[:80]}...")

        # 克隆工作流并动态注入参数
        wf = json.loads(json.dumps(base_workflow))

        # 检查是否挂载加速 LoRA
        has_lora = any(
            isinstance(node, dict) and node.get("class_type") in ("LoraLoader", "LoraLoaderModelOnly")
            for node in wf.values()
        )
        force_steps = task.get("force_steps", False)
        actual_steps = steps

        # 遍历节点并匹配注入
        for node_id, node in wf.items():
            class_type = node.get("class_type")
            inputs = node.get("inputs", {})
            title_meta = node.get("_meta", {}).get("title", "")

            if class_type == "CLIPTextEncode" and (title_meta.startswith("Positive") or "Positive" in title_meta):
                inputs["text"] = prompt_text
            elif class_type == "CLIPTextEncode" and ("Negative" in title_meta or title_meta.startswith("Negative")):
                if neg_text:
                    inputs["text"] = neg_text
            elif class_type in ("EmptyMiniMaxH3LatentAV", "EmptyH3LatentVideo", "EmptyLatentVideo"):
                inputs["width"] = width
                inputs["height"] = height
                inputs["length"] = length
            elif class_type == "KSampler":
                if not has_lora and steps < 15 and not force_steps:
                    default_steps = inputs.get("steps", 20)
                    print(f"   ⚠️ 当前工作流未挂载加速 LoRA，保留工作流模板默认步数 ({default_steps}) 以防生成废片")
                    actual_steps = default_steps
                else:
                    inputs["steps"] = steps
                    actual_steps = steps
                inputs["cfg"] = cfg
                inputs["seed"] = seed
            elif class_type in ("SaveAnimatedWEBP", "VHS_VideoCombine", "SaveVideo"):
                inputs["filename_prefix"] = f"{task_id}_{seed}"
                if class_type == "SaveAnimatedWEBP":
                    inputs.setdefault("fps", 24.0)
                    inputs.setdefault("lossless", False)
                    inputs.setdefault("quality", 85)
                    inputs.setdefault("method", "default")
                elif class_type == "SaveVideo":
                    inputs.setdefault("format", "mp4")
            elif class_type == "CreateVideo":
                inputs.setdefault("fps", 24.0)

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
            "steps": actual_steps,
            "cfg": cfg,
            "seed": seed,
            "prompt": prompt_text,
            "saved_files": copied_files
        }
        with open(log_file, "a", encoding="utf-8") as lf:
            lf.write(json.dumps(log_record, ensure_ascii=False) + "\n")

        print("-" * 50)

    print("\n" + "=" * 60)
    print(f"🎉 全部批处理任务执行完毕！")
    print(f"📂 成片与日志均已持久化至: {out_path.resolve()}")
    print("=" * 60)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ComfyUI Batch Runner for MiniMax H3")
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
