# MiniMax H3 Colab 纯本地视频生产工作流

> 基于 Google AI Pro 每月赠送的 200 Colab 计算单元，全流程**纯本地白嫖**运行 MiniMax H3（Hailuo 3.0）全模态视频模型。
> **硬约束**：不调用任何 MiniMax 外部商业 API、不产生二次计费、聚焦无脸视频（Faceless Video）高品质批量出片。

---

## 🏗️ 架构与工作流全景

```mermaid
flowchart TD
    A["Google AI Pro 账号 (200 计算单元)"] --> B["Colab 硬件门禁检测 (check_gpu.py)"]
    B -->|T4 (算力 7.5)| C["❌ 立即中断并销毁实例，防扣费"]
    B -->|L4 (24GB) / A100 (40GB)| D["✅ 硬件通过 (算力 >= 8.0)"]
    D --> E["Google Drive 完整性缓存挂载 (download_models.py)"]
    E --> F["ComfyUI 后台服务启动 (低显存卸载模式)"]
    F --> G["基准冒烟评测 (验证 NVFP4 与单步耗时)"]
    G --> H["无头批量生产队列 (PDD 8步 / batch_runner.py)"]
    H --> I["产物落盘: Google Drive (视频/动图 + 结构化日志)"]
    I --> J["🛑 三重关机保护: finally 销毁 + Watchdog 看门狗 (杜绝闲置扣费)"]
```

---

## 📁 仓库目录结构

- 📓 [notebooks/minimax_h3_l4_production.ipynb](file:///Users/kuhung/roy/3_Active_Labs/minimax-h3-colab-workflow/notebooks/minimax_h3_l4_production.ipynb) — **Colab 核心交互笔记本**（集成看门狗、门禁、完整性缓存、冒烟评测与批处理）。
- 🤖 [scripts/agent_colab_runner.py](file:///Users/kuhung/roy/3_Active_Labs/minimax-h3-colab-workflow/scripts/agent_colab_runner.py) — **Agent 自动化托管执行器**（支持三重防扣费兜底、看门狗熔断、自动环境安装与批量渲染）。
- ⚙️ [scripts/check_gpu.py](file:///Users/kuhung/roy/3_Active_Labs/minimax-h3-colab-workflow/scripts/check_gpu.py) — **硬件环境门禁脚本**（严格校验计算能力 ≥ 8.0、显存与磁盘；支持不达标立即销毁）。
- 📦 [scripts/download_models.py](file:///Users/kuhung/roy/3_Active_Labs/minimax-h3-colab-workflow/scripts/download_models.py) — **权重完整性校验与 Drive 缓存管理**（原子备份、最小体积门限拦截、杜绝破损缓存软链接）。
- 🎬 [scripts/batch_runner.py](file:///Users/kuhung/roy/3_Active_Labs/minimax-h3-colab-workflow/scripts/batch_runner.py) — **无头批量队列渲染器**（支持超时 `/interrupt` 显存释放、精准提取输出、自动写入 Drive）。
- 🧩 [workflows/h3_t2v_l4_baseline.json](file:///Users/kuhung/roy/3_Active_Labs/minimax-h3-colab-workflow/workflows/h3_t2v_l4_baseline.json) — ComfyUI 544p 快速冒烟基线工作流。
- 🧩 [workflows/h3_t2v_pdd8_production.json](file:///Users/kuhung/roy/3_Active_Labs/minimax-h3-colab-workflow/workflows/h3_t2v_pdd8_production.json) — ComfyUI 768p PDD 8步高画质出片工作流。
  > 💡 **重要说明**：`workflows/` 下的 JSON 节点名称需在首次启动 ComfyUI 后与官方最新 H3 节点模板核对（若节点签名有更新，可在 WebUI 中通过 `Save (API Format)` 导出替换）。此外，原生 `SaveAnimatedWEBP` 仅输出无声动图，如需合入 32kHz 音轨需配合 `ComfyUI-VideoHelperSuite (VHS_VideoCombine)`。
- 📋 [configs/prompts.example.json](file:///Users/kuhung/roy/3_Active_Labs/minimax-h3-colab-workflow/configs/prompts.example.json) — 无脸视频排期任务模板清单。
- 📖 [docs/COLAB_MCP_INTEGRATION.md](file:///Users/kuhung/roy/3_Active_Labs/minimax-h3-colab-workflow/docs/COLAB_MCP_INTEGRATION.md) — **Colab MCP 深度集成与 Agent 全自动化托管操作手册**。
- 📖 [docs/PROMPT_RECIPES.md](file:///Users/kuhung/roy/3_Active_Labs/minimax-h3-colab-workflow/docs/PROMPT_RECIPES.md) — H3 音画同出提示词结构公式与无脸镜头配方库。
- 📚 [docs/RESEARCH_SUMMARY.md](file:///Users/kuhung/roy/3_Active_Labs/minimax-h3-colab-workflow/docs/RESEARCH_SUMMARY.md) — 前期全网调研核查报告归档。

---

## ⚡ 核心避坑法则（保卫 200 计算单元）

| 规则 | 规范说明与处置方式 |
|---|---|
| **1. 算力单元消耗估算** | **参考费率估算（待首轮 Benchmark 核准）：L4 约 2.8~4.8 CU/h，A100 约 6.5~11.8 CU/h，具体请以 Colab 右上角「资源」面板实时显示费率为准**。 |
| **2. 严禁在 T4 上跑** | T4 算力 7.5，不支持 Int8 ConvRot 算子，强行跑会立即 OOM 崩溃。门禁脚本配置了自动阻断并调用 `runtime.unassign()` 关机。 |
| **3. 三重防闲置安全网** | ① 脚本内 `try...finally` 必定触发关机；② 后台 Watchdog 线程超时硬杀 (`--watchdog-hours`)；③ Agent 结束主动断开。 |
| **4. 缓存完整性校验** | 剪枝模型约 40GB，首次拉取需验证完整性门限（≥ 90% 实际体积），通过原子暂存文件备份到 Drive，防止中断导致损坏缓存。 |
| **5. 拒绝盲目抽卡** | 分辨率变动会导致噪声潜空间变化（544p 的 seed 无法 1:1 复制到 768p），初筛建议以固定分辨率快速小步数抽镜头。 |
| **6. 节点模板与音画配合** | `workflows/` 下的 JSON 节点名称需在首次启动 ComfyUI 后与官方最新 H3 节点模板核对（若节点签名有更新，可在 WebUI 中通过 `Save (API Format)` 导出替换）；原生 `SaveAnimatedWEBP` 仅输出无声动图，如需合入 32kHz 音轨需配合 `ComfyUI-VideoHelperSuite (VHS_VideoCombine)`。 |

---

## 🤖 方案 A：Colab MCP + Agent 自动化托管模式（强烈推荐）

> 借助官方 [googlecolab/colab-mcp](https://github.com/googlecolab/colab-mcp)，将繁重的操作全部交由 Agent 自主完成。详见 [docs/COLAB_MCP_INTEGRATION.md](file:///Users/kuhung/roy/3_Active_Labs/minimax-h3-colab-workflow/docs/COLAB_MCP_INTEGRATION.md)。

1. **环境已就绪**：MCP 配置文件 `~/.gemini/config/mcp_config.json` 已挂载 `colab-mcp`。
2. **对 Agent 说一句话即开始出片**：
   > “帮我用 L4 批量跑 `configs/prompts.example.json` 里的无脸镜头，开启自动关机与看门狗。”
3. **Agent 自主执行闭环**：
   - 自动桥接打开 Colab 会话并校验 L4 算力门禁（遇 T4 立即阻断并销毁保护算力）。
   - 自动软链接 Google Drive 模型缓存（原子校验，杜绝损坏文件）。
   - 自动拉起 ComfyUI 守护进程并执行批处理渲染（遇超时自动 `/interrupt` 释放显存）。
   - 实时轮询日志向您汇报出片进度，所有成片自动存入 Google Drive。
   - 任务无论成功还是异常，`finally` 强制触发 `runtime.unassign()` 销毁实例，保卫计算单元。

---

## 🛠️ 方案 B：传统人工交互 SOP（备用降级）

### 第一步：打开笔记本与挂载
1. 将 [minimax_h3_l4_production.ipynb](file:///Users/kuhung/roy/3_Active_Labs/minimax-h3-colab-workflow/notebooks/minimax_h3_l4_production.ipynb) 上传或导入至 Google Colab。
2. 运行时配置：**GPU -> L4 (高内存)**。
3. 顺序执行单元格 1（挂载 Google Drive 并激活后台 Watchdog 线程）与单元格 2（硬件门禁检测）。

### 第二步：一键环境与模型挂载
1. 执行单元格 3 与 4，脚本会自动从 HuggingFace 拉取剪枝量化最小集（约 40GB）或软链接 Drive 缓存：
   - 变压器：`minimax_h3_fl2va_pruned_int8_convrot.safetensors` (~20.97GB)
   - 文本编码器：`qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` (~15.69GB)
   - VAE & PDD 8步蒸馏 LoRA
2. 执行单元格 5 启动 ComfyUI 守护进程（开启动态显存卸载 `--lowvram`）。

### 第三步：冒烟实测与批量出片
1. 执行单元格 6 进行基准冒烟测试，实测单步耗时与实际显存占用。
2. 执行单元格 7 执行批处理渲染：
   ```bash
   python scripts/batch_runner.py \
       --workflow workflows/h3_t2v_pdd8_production.json \
       --prompts configs/prompts.example.json \
       --output /content/drive/MyDrive/MiniMax_Outputs
   ```
3. 执行单元格 8 彻底断开并销毁 Colab 运行时。
