# 开源生态借鉴清单与技术吸收明细

本文件记录 MiniMax H3 生态中优秀开源项目的核心闪光点，以及本项目吸收、改造和落地的具体方案。

---

## 一、 核心借鉴项目与吸收明细

### 1. `ModelTC/Minimax-H3-Turbo`
- **上游定位**：MiniMax H3 官方模型的对抗蒸馏（Turbo LoRA）源头。
- **核心价值**：将扩散去噪步数从原始 20 至 30 步压缩至 4 步 / 8 步，带来 4 到 5 倍端到端提速，同时原生保留 32kHz 立体声音画同步。
- **本项目吸收与落地**：
  - [x] 在 `scripts/download_models.py` 中将 8 步 Turbo LoRA 作为默认挂载权重。
  - [ ] 暴露 `--turbo-steps 4` 参数，为创作者提供 80 秒极速分镜快筛模式。
  - [x] 锁死单路无 CFG（`BasicGuider`, CFG=1.0），彻底消除蒸馏模型在外推 CFG 下的过曝与相位破音。

---

### 2. `ethanfel/ComfyUI-MiniMax-H3-Guide`
- **上游定位**：面向 H3 结构化提示词与媒体流原生路由的 ComfyUI 自定义节点套件。
- **核心价值**：
  - 总结了 H3 能够识别的原生提示词结构（`[Shot]`, `[Subject]`, `[Camera]`，以及 10 大官方风格嵌入语法）。
  - 验证了如何将多张参考图（1 至 9 张）的特征向量按主体、场景、风格正确传递给 DiT 模型，避免特征混杂。
- **本项目吸收与落地**：
  - [ ] 提取其多参考图媒体路由逻辑，直接转化为纯 Python / ComfyUI 原生节点的拓扑映射，不引入其第三方自定义节点包，保持零依赖。
  - [ ] 将其结构化 Prompt 语法（见 `docs/PROMPT_RECIPES.md`）集成进 `scripts/batch_runner.py` 的分镜组装模块中。

---

### 3. `RioShiina47/comfy-webui`
- **上游定位**：基于 Gradio 构建的 ComfyUI 流程服务化工具，重点包装了 Ref2VA（多模态参考链）。
- **核心价值**：
  - 彻底屏蔽了底层 Guider、Scheduler、VAE Decode 等节点噪音。
  - 抽象出面向创作者的最简输入表单：首帧图上传槽位 + 角色多视角参考图槽位 + 分镜描述。
- **本项目吸收与落地**：
  - [ ] 借鉴其输入抽象层设计，在 Colab Notebook 中构建极其简单的传图与分镜映射接口（例如建立 `inputs/references/` 文件夹，自动按文件名匹配分镜脚本），降低用户配置多图的认知负担。

---

### 4. `MiniMax-AI/MiniMax-H3` (官方) 与 `Comfy-Org/MiniMax-H3`
- **上游定位**：模型算法母体与 Comfy-Org 官方维护的运行拓扑。
- **核心价值**：
  - 定义了标准的原生节点：`MiniMaxH3ImageToVideo`、`VAEDecodeAudio`、`CreateVideo`、`SaveVideo`。
  - 提供了 17k + 5 的时域帧数对齐标准（如 124 帧 @ 24fps）。
- **本项目吸收与落地**：
  - [x] 100% 坚持 ComfyUI 官方原生链路，坚决不引入第三方视频编译扩展。
  - [x] 在渲染器中内置 `17k + 5` 帧数自动对齐逻辑，杜绝时域切片伪影。
