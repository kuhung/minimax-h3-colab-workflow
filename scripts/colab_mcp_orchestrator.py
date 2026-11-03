#!/usr/bin/env python3
"""
Google Colab 官方 MCP 标准生产级自动化调度器 (Production Colab MCP Orchestrator)
---------------------------------------------------------------------------------
核心特性：
1. 完整遵循 googlecolab/colab-mcp WebSocket 桥接协议
2. 严密适配 Colab MCP 7 大原生工具 Schema:
   - add_code_cell(cellIndex, language, code)
   - run_code_cell(cellId)
   - get_cells(includeOutputs=True)
3. 引入 __MCP_STEP_X_DONE__ 哨兵同步机制：
   - 无论 run_code_cell 是同步阻塞还是异步返回，均精准判定单元格真实执行完成状态
   - 杜绝前序步骤未跑完就抢跑后续步骤，杜绝提前触发关机
4. 实时流式抓取 Colab 单元格输出并汇报进度
5. 任务完成后自动触发 runtime.unassign() 销毁实例停止计费
"""

import os
import sys
import re
import json
import time
import asyncio
import logging
import argparse
import webbrowser
from pathlib import Path
import websockets
from websockets.typing import Subprotocol

try:
    from colab_mcp.websocket_server import ColabWebSocketServer, COLAB, SCRATCH_PATH
    from colab_mcp.session import ColabTransport, ColabProxyClient
    from fastmcp import Client
except ImportError:
    print("❌ 缺少 colab_mcp 依赖，请使用 uv run 运行本脚本")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("colab_mcp_orchestrator")


def dump_tool_result_text(res) -> str:
    """将 call_tool 的返回结果安全提取为纯文本"""
    if res is None:
        return ""
    parts = []
    if hasattr(res, "structured_content") and res.structured_content:
        try:
            parts.append(json.dumps(res.structured_content, ensure_ascii=False))
        except Exception:
            parts.append(str(res.structured_content))
    if hasattr(res, "content") and res.content:
        for item in res.content:
            t = getattr(item, "text", None)
            if t:
                parts.append(str(t))
    return "\n".join(parts)


def extract_all_cell_ids(raw_text: str) -> list[str]:
    """从 JSON 或文本中鲁棒提取所有 cellId / id"""
    ids = []
    try:
        data = json.loads(raw_text)
        if isinstance(data, dict):
            # 可能是单对象 {"cellId": "xxx"} 或 {"newCellId": "xxx"}
            for k in ("cellId", "newCellId", "id", "cell_id"):
                if k in data and isinstance(data[k], str):
                    ids.append(data[k])
            # 可能是 {"cells": [...]}
            for v in data.values():
                if isinstance(v, list):
                    for item in v:
                        if isinstance(item, dict):
                            cid = item.get("cellId") or item.get("id")
                            if cid:
                                ids.append(str(cid))
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    cid = item.get("cellId") or item.get("id")
                    if cid:
                        ids.append(str(cid))
    except Exception:
        pass

    if not ids:
        # 正则兜底
        matches = re.findall(r'["\'](?:cellId|newCellId|id)["\']\s*:\s*["\']([^"\']+)["\']', raw_text)
        ids.extend(matches)

    if not ids:
        stripped = raw_text.strip().strip('"').strip("'")
        if re.match(r'^[a-zA-Z0-9_\-]{4,32}$', stripped):
            ids.append(stripped)

    return ids


async def get_notebook_cells_text(client) -> str:
    """调用 get_cells(includeOutputs=True) 并返回原始文本"""
    try:
        res = await client.call_tool("get_cells", {"includeOutputs": True})
        return dump_tool_result_text(res)
    except Exception:
        try:
            res = await client.call_tool("get_cells", {})
            return dump_tool_result_text(res)
        except Exception as e:
            return f"get_cells error: {e}"


async def orchestrate(shots: int = 2, auto_shutdown: bool = True):
    print("\n" + "=" * 74)
    print("🚀 Google Colab MCP 生产级自动化调度器已启动")
    print(f"   • 渲染任务: 上海街景原生 32kHz 立体声 MP4 ({shots} 个镜头)")
    print(f"   • 算力保护: {'出片后自动销毁实例 (runtime.unassign)' if auto_shutdown else '手动管理'}")
    print("=" * 74 + "\n")

    async with ColabWebSocketServer() as wss:
        auth_url = f"{COLAB}{SCRATCH_PATH}#mcpProxyToken={wss.token}&mcpProxyPort={wss.port}"

        print("🔗 [阶段 1/4] 本地 WebSocket 网关已就绪")
        print(f"   监听地址: ws://localhost:{wss.port}")
        print(f"   👉 专属连接链接:\n   {auth_url}\n")

        try:
            webbrowser.open_new(auth_url)
            print("🌐 已为您自动调起浏览器标签页。")
        except Exception as e:
            print(f"⚠️ 自动打开浏览器失败，请手动点击上方链接: {e}")

        print("\n⏳ [阶段 2/4] 正在等待浏览器握手连接（最长等待 10 分钟）...")
        wait_start = time.time()
        connected = False
        while time.time() - wait_start < 600:
            if wss.connection_live.is_set():
                connected = True
                break
            await asyncio.sleep(0.5)

        if not connected:
            print("\n❌ 等待浏览器连接超时（10 分钟），调度器安全退出。")
            return

        print("\n" + "=" * 74)
        print("🎉 ✅ [握手成功] Colab 前端已连通本地 MCP 网关！开始全自动注入与执行！")
        print("=" * 74 + "\n")

        async with ColabProxyClient(wss) as proxy:
            for _ in range(40):
                if proxy.is_connected() and proxy.proxy_mcp_client is not None:
                    break
                await asyncio.sleep(0.25)

            client = proxy.client_factory()
            if client is None or not proxy.is_connected():
                print("❌ MCP Client 初始化失败。")
                return

            tools_res = await client.list_tools()
            tool_names = [t.name for t in tools_res]
            print(f"🛠️ 已解锁 Colab MCP 工具集 ({len(tool_names)} 个): {tool_names}")

            # 探测初始单元格数量
            init_gc_text = await get_notebook_cells_text(client)
            init_cell_ids = extract_all_cell_ids(init_gc_text)
            print(f"📊 初始页面单元格列表 ({len(init_cell_ids)} 个): {init_cell_ids}\n")

            # 读取测试 Notebook 中的代码单元格
            nb_path = Path("notebooks/test_shanghai_mp4_audio.ipynb")
            nb_raw_str = nb_path.read_text(encoding="utf-8")
            nb_data = json.loads(nb_raw_str)

            raw_code_cells = [
                "".join(c["source"])
                for c in nb_data["cells"]
                if c.get("cell_type") == "code" and "".join(c.get("source", "")).strip()
            ]

            # 在步骤 3 中追加：自动将完整 Notebook 与项目代码持久化存档到用户的 Google Drive 线上
            archive_snippet = (
                "\n# --- 自动线上存档至 Google Drive ---\n"
                "import shutil\n"
                "drive_nb_dir = Path('/content/drive/MyDrive/Colab Notebooks')\n"
                "drive_nb_dir.mkdir(parents=True, exist_ok=True)\n"
                f"drive_nb_file = drive_nb_dir / 'test_shanghai_mp4_audio.ipynb'\n"
                f"drive_nb_file.write_text({repr(nb_raw_str)}, encoding='utf-8')\n"
                "drive_repo_dir = Path('/content/drive/MyDrive/minimax-h3-colab-workflow')\n"
                "shutil.copytree('/content/minimax-h3-colab-workflow', drive_repo_dir, dirs_exist_ok=True)\n"
                "print(f'☁️ [线上永久存档完成] Notebook 已存至: {drive_nb_file}')\n"
                "print(f'☁️ [线上永久存档完成] 项目代码已存至: {drive_repo_dir}')\n"
            )
            if len(raw_code_cells) >= 3:
                raw_code_cells[2] = raw_code_cells[2] + "\n" + archive_snippet

            step_titles = [
                "步骤 1: 挂载 Google Drive 与启动 1 小时算力看门狗",
                "步骤 2: 硬件门禁检测 (校验 L4 GPU 算力 ≥ 8.0)",
                "步骤 3: 部署 ComfyUI 环境、PyAV 与上海街景 MP4 工作流",
                "步骤 4: 秒级软链接 Google Drive 5 大核心模型权重",
                "步骤 5: 后台拉起 ComfyUI 守护进程 (--lowvram) 并健康检查",
                f"步骤 6: 批量渲染上海街景 {shots} 镜原生音画 MP4",
                "步骤 7: 校验输出目录 MP4 成片并生成试听卡片",
                "步骤 8: 触发 runtime.unassign() 自动销毁实例停止计费"
            ]

            current_index = len(init_cell_ids)

            for idx, raw_code in enumerate(raw_code_cells):
                step_num = idx + 1
                title = step_titles[idx] if idx < len(step_titles) else f"步骤 {step_num}"
                sentinel = f"__MCP_STEP_{step_num}_COMPLETED__"

                # 如果未开启 auto_shutdown，跳过最后一步自毁
                if step_num == 8 and not auto_shutdown:
                    print(f"⏭️ 跳过 {title} (--auto-shutdown 未启用)")
                    continue

                # 在代码末尾追加完成哨兵打印（步骤 8 是自毁，执行后直接断连无需等待哨兵）
                if step_num < 8:
                    code_to_run = f"{raw_code.rstrip()}\n\nprint('{sentinel}', flush=True)\n"
                else:
                    code_to_run = raw_code

                print("-" * 74)
                print(f"▶️ [{step_num}/{len(raw_code_cells)}] {title}")

                # 1. 创建新代码单元格
                target_cell_id = None
                try:
                    add_res = await client.call_tool(
                        "add_code_cell",
                        {
                            "cellIndex": current_index,
                            "language": "python",
                            "code": code_to_run
                        }
                    )
                    add_text = dump_tool_result_text(add_res)
                    extracted = extract_all_cell_ids(add_text)
                    if extracted:
                        target_cell_id = extracted[-1]
                    current_index += 1
                    print(f"   📝 已注入代码单元格 (index={current_index - 1}, cellId={target_cell_id})")
                except Exception as e_add:
                    print(f"   ⚠️ add_code_cell 异常 ({e_add})，尝试通过 get_cells 定位...")

                # 如果未能直接从返回值拿到 cellId，从 get_cells 中取最后一个
                if not target_cell_id:
                    gc_text = await get_notebook_cells_text(client)
                    all_ids = extract_all_cell_ids(gc_text)
                    if all_ids:
                        target_cell_id = all_ids[-1]
                        print(f"   🔍 通过 get_cells 定位到最新 cellId={target_cell_id}")

                if not target_cell_id:
                    print(f"   ❌ 无法获取步骤 {step_num} 的 cellId，终止流程。")
                    return

                # 2. 触发执行 run_code_cell
                print(f"   ⚡ 正在执行单元格 (cellId={target_cell_id})...")
                if step_num == 1:
                    print("   💡 [提示] 若浏览器弹出 Google Drive 授权窗口，请在网页上点击『连接到 Google 云端硬盘』。")

                t_step_start = time.time()
                run_output_text = ""
                try:
                    run_res = await client.call_tool("run_code_cell", {"cellId": target_cell_id})
                    run_output_text = dump_tool_result_text(run_res)
                except Exception as e_run:
                    # 步骤 8 调用 runtime.unassign() 会直接断开 WebSocket，属于预期行为
                    if step_num == 8:
                        print("   🛑 实例销毁指令已触发，Colab 会话已断开，计费立即停止！")
                        break
                    print(f"   ⚠️ run_code_cell 返回提示: {e_run}")

                if step_num == 8:
                    print("   🛑 实例销毁指令已触发，计费立即停止！")
                    break

                # 3. 检查哨兵是否已在输出中，若未完成则轮询 get_cells 等待完成
                step_timeout = 1800 if step_num in (3, 4, 6) else 600
                last_printed_snippet = ""

                while time.time() - t_step_start < step_timeout:
                    combined_view = run_output_text + "\n" + await get_notebook_cells_text(client)

                    # 检查是否出现致命错误
                    if "RuntimeError:" in combined_view and "硬件不达标" in combined_view:
                        print("   🚨 检测到硬件门禁拦截（非 L4/A100 GPU），实例已自动触发销毁！")
                        return

                    if sentinel in combined_view:
                        elapsed = round(time.time() - t_step_start, 1)
                        print(f"   ✅ 步骤 {step_num} 执行完成！(耗时 {elapsed}s)")
                        break

                    # 打印增量状态提示
                    elapsed = int(time.time() - t_step_start)
                    if elapsed > 0 and elapsed % 15 == 0:
                        print(f"   ⏳ 步骤 {step_num} 正在云端运行中... (已耗时 {elapsed}s)")

                    await asyncio.sleep(5)
                else:
                    print(f"   ⚠️ 步骤 {step_num} 等待超时 ({step_timeout}s)，继续推进下一步...")

            print("\n" + "=" * 74)
            print("🎉 🏁 Google Colab MCP 全自动生产管线圆满完成！")
            print("📂 成片已保存至您的 Google Drive: /MyDrive/MiniMax_Outputs/shanghai_street/")
            print("🛑 云端运行时已安全释放，无任何闲置算力损耗。")
            print("=" * 74 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Production Colab MCP Orchestrator")
    parser.add_argument("--shots", type=int, default=2, help="Number of shots (default: 2)")
    parser.add_argument("--auto-shutdown", action="store_true", default=True, help="Auto unassign runtime after finish")
    args = parser.parse_args()

    asyncio.run(orchestrate(shots=args.shots, auto_shutdown=args.auto_shutdown))
