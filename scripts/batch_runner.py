#!/usr/bin/env python3
"""
Headless Batch Runner for MiniMax H3 on ComfyUI.
Sends prompts to ComfyUI local API (127.0.0.1:8188), monitors generation progress,
and automatically exports output videos and metadata logs directly to Google Drive.
"""

import os
import sys
import json
import time
import shutil
import urllib.request
import urllib.parse
from datetime import datetime
from pathlib import Path

def post_prompt(server_url: str, prompt_workflow: dict) -> str:
    """Submits workflow JSON to ComfyUI API /prompt."""
    data = json.dumps({"prompt": prompt_workflow}).encode("utf-8")
    req = urllib.request.Request(f"{server_url}/prompt", data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as response:
            res = json.loads(response.read().decode("utf-8"))
            return res.get("prompt_id")
    except Exception as e:
        print(f"❌ 请求 ComfyUI API 失败: {e}")
        return None

def wait_for_completion(server_url: str, prompt_id: str, timeout_sec: int = 900) -> dict:
    """Polls ComfyUI /history/{prompt_id} until generation finishes."""
    start_time = time.time()
    last_print = 0
    while time.time() - start_time < timeout_sec:
        try:
            req = urllib.request.Request(f"{server_url}/history/{prompt_id}")
            with urllib.request.urlopen(req) as response:
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

    print(f"❌ 任务超时 ({timeout_sec}s)")
    return None

def run_batch(
    workflow_path: str,
    prompts_path: str,
    output_dir: str,
    server_url: str = "http://127.0.0.1:8188",
    comfy_output_dir: str = "/content/ComfyUI/output"
):
    print("=" * 60)
    print("🎬 MiniMax H3 批处理队列执行器 (Batch Runner)")
    print(f"📄 工作流模板: {workflow_path}")
    print(f"📋 任务清单: {prompts_path}")
    print(f"💾 最终输出落盘路径: {output_dir}")
    print("=" * 60)

    # 确保输出目录存在
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    log_file = out_path / "render_history.jsonl"

    with open(workflow_path, "r", encoding="utf-8") as f:
        base_workflow = json.load(f)

    with open(prompts_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)

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

        # 查找对应节点并替换参数
        for node_id, node in wf.items():
            class_type = node.get("class_type")
            inputs = node.get("inputs", {})

            if class_type == "CLIPTextEncode" and node.get("_meta", {}).get("title", "").startswith("Positive"):
                inputs["text"] = prompt_text
            elif class_type == "CLIPTextEncode" and "Negative" in node.get("_meta", {}).get("title", ""):
                if neg_text:
                    inputs["text"] = neg_text
            elif class_type == "EmptyH3LatentVideo":
                inputs["width"] = width
                inputs["height"] = height
                inputs["length"] = length
            elif class_type == "KSampler":
                inputs["steps"] = steps
                inputs["cfg"] = cfg
                inputs["seed"] = seed
            elif class_type in ("SaveAnimatedWEBP", "VHS_VideoCombine"):
                inputs["filename_prefix"] = f"{task_id}_{seed}"

        # 提交至 ComfyUI
        t_start = time.time()
        prompt_id = post_prompt(server_url, wf)
        if not prompt_id:
            print(f"❌ 提交任务 {task_id} 失败，跳过。")
            continue

        print(f"   🚀 已推入执行队列 (Prompt ID: {prompt_id})")
        res = wait_for_completion(server_url, prompt_id)
        t_cost = round(time.time() - t_start, 2)

        if res and "outputs" in res:
            print(f"   ✅ 渲染成功！耗时: {t_cost}s")

            # 寻找生成的文件并复制至落盘目录 (Google Drive)
            copied_files = []
            comfy_out = Path(comfy_output_dir)
            if comfy_out.exists():
                for f in comfy_out.glob(f"{task_id}_{seed}*"):
                    target_dst = out_path / f.name
                    shutil.copy2(f, target_dst)
                    copied_files.append(str(target_dst))
                    print(f"   💾 成片已同步至云盘: {target_dst.name}")

            # 写入日志
            log_record = {
                "timestamp": datetime.now().isoformat(),
                "task_id": task_id,
                "title": title,
                "cost_seconds": t_cost,
                "resolution": f"{width}x{height}",
                "length_frames": length,
                "steps": steps,
                "cfg": cfg,
                "seed": seed,
                "prompt": prompt_text,
                "saved_files": copied_files
            }
            with open(log_file, "a", encoding="utf-8") as lf:
                lf.write(json.dumps(log_record, ensure_ascii=False) + "\n")
        else:
            print(f"   ❌ 渲染未按预期完成或无输出。")

        print("-" * 50)

    print("\n" + "=" * 60)
    print(f"🎉 全部批处理任务执行完毕！")
    print(f"📂 成片与日志均已永久保存在: {out_path.resolve()}")
    print("=" * 60)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ComfyUI Batch Runner for MiniMax H3")
    parser.add_argument("--workflow", default="workflows/h3_t2v_pdd8_production.json", help="Path to workflow JSON")
    parser.add_argument("--prompts", default="configs/prompts.example.json", help="Path to prompts JSON")
    parser.add_argument("--output", default="/content/drive/MyDrive/MiniMax_Outputs", help="Output directory")
    parser.add_argument("--server", default="http://127.0.0.1:8188", help="ComfyUI server URL")
    args = parser.parse_args()

    run_batch(args.workflow, args.prompts, args.output, args.server)
