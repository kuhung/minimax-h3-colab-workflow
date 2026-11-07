# 交办：结合 SOTA 调研文档，给 MiniMax H3 项目出方案和路线图

交办人：爆炸头（Muse）｜日期：2026-11-07｜执行方：Codex

## 任务

读完下面拆分后的《2026 AI视频生成全场景SOTA工作流与MiniMax H3实战调研报告》要点，结合我们目前的 MiniMax H3 项目现状，输出两样东西：

1. **方案**：哪些最佳实践要采纳、哪些不采纳（不采纳要写理由）、每项落到哪个具体改动。
2. **路线图**：按阶段排好顺序，每阶段写清交付物、验收标准、大致工作量，更新 `docs/ROADMAP.md`。

原文（Google Doc，2026-11-04 更新）：《AI视频生成SOTA工作流调研》
https://docs.google.com/document/d/1XpQQIXkId9UhS8gdPdh2tZhFqOf28ulqYeQWH7PCxsY/edit?usp=drivesdk

注意：文档里的数字和结论是调研整理口径（带角标来源），不是我方实测。凡是影响成本、耗时、显存的数字，方案里要标“文档口径，待本地实测复核”，不要直接当定论写进 ROADMAP。

## 项目现状（2026-11-07，实测为准）

仓库：`/Users/kuhung/roy/3_Active_Labs/minimax-h3-colab-workflow`
结构：`configs/`、`workflows/`、`scripts/`、`notebooks/`、`docs/`（含 ROADMAP.md）。
Drive 侧同名文件夹 `minimax-h3-colab-workflow` 内有 configs（shanghai_street.json）、workflows（h3_t2v_pdd8_mp4.json）、scripts（test_shanghai_mp4.py、check_gpu.py、batch_runner.py、download_models.py）；另有 v1.0/v2.0 notebook 与 MiniMax_Models、MiniMax_Outputs 文件夹。

- 阶段一已完成：单路无 CFG（BasicGuider，CFG=1.0）、单镜推理 8 步、L4 基准稳态耗时 165 秒/镜、单镜 0.087 CU；ComfyUI 原生节点打包 32kHz 立体声 MP4，零第三方插件；硬件门禁（防 T4 误扣费）与看门狗自动关机已就位。
- 阶段二进行中，全部未勾选：视觉化 README 重构（成片演示＋Open in Colab 徽章、成本账目表、双语文档）、实操型工作流科普长文、推向 V2EX/X/即刻/Reddit、Issues 模板。
- 阶段三未开始：多参考图（I2V）拓扑接入、降低传图摩擦力、补充 1–2 组固定人物分镜的强叙事成片案例。
- v2.0 实测存证：三镜合计 574.6 秒、消耗 0.309 CU（单次 L4 记录，不能推成月产量保证）。
- 口径冲突待统一：ROADMAP 写的是 1.9 CU/h、0.087 CU/镜；后续实测文档已统一为 1.94 CU/h、约 0.089 CU/镜。发布前必须统一，方案里先定一个数。
- 11-03 咖啡 bug 教训：冒烟测试覆盖了生产模板、提示词替换静默失败。修复办法已定：试验文件与生产模板分开、从工作流连接定位节点、给提示词与分辨率加校验。任何新工作流改动都要守这条。

## 硬约束（不可违反）

- 全程不引入 MiniMax API 请求：纯本地路线，Google AI Pro 的 Colab 额度＋开源权重。文档里凡是走 API 的档位（H3 Lite/Highspeed 探路、完整版 H3 出 2K、H3-Regenerate-2K、H3-Context-IR）都不能直接照搬，只能取其思路，落到本地可实现的等价做法。
- 要现成教程、实践或 GitHub 仓库，不要从头自建工作流；现成 Colab 环境与显卡配置建议优先于自搭。
- 需外部 API 密钥的仓库同样排除。
- 2K 与托管加速档不在路线内。

## 文档拆分：各部分要点

### 1. 四大实操准则（文档第 4 段）
- Ref2VA 硬性配额：总文件 ≤12（图 ≤9、视频 ≤3、音频 ≤3）；视频与音频各自累计时长均 ≤15 秒；音频绝不可作为唯一输入；提示词用 @图片N、@视频N、@音频N 显式绑定控制维度。
- 档位分层：Lite（480p/768p）/Highspeed（768p）探路，定稿再用完整版 H3 出 2K＋32kHz 立体声；轻量档不生成配乐与声音，只用参考音频引导动作节奏与口型。——注意：这是 API 档位，本地路线只能借鉴“先低成本探路、再高成本定稿”的分层思路。
- 开源边界：官方只开源 768p 的 H3-Base 基础权重（自定义许可证）；H3-Context-IR（多模态编排）与 H3-Regenerate-2K（2K 超分）闭源 API 独占；本地部署缺编排能力，官方补法是提示词技能包（`npx skills add https://github.com/MiniMax-AI/MiniMax-H3 --skill h3-prompt-writing`，含 base-en.txt、ref-en.txt）或 MiniMax Hub 桌面端 8 大画布工作流。
- 提示词转向“制作时间线思维”：精确到毫秒的切镜时间戳（如 00:04.000 切换镜头）＋自然语言定点改片，低成本迭代。

### 2. 七大输入场景与提示词减法原则（文档第一、二节）
- 顺序：先定角色与场景静帧（T2I 定妆照多角度参考集 → I2I 派生分镜关键帧），再批量出动态分镜；直接 T2V 做多镜头剧情易出现面部漂移、服装突变、Logo 变形。
- I2V/FLF2V 提示词减法：首帧已定外貌服装环境，提示词只写“画面如何运动与变化”（动作、镜头轨迹、环境变化、速度）＋显式约束（五官服装一致、不新增人物、背景不变）。
- H3-Base-FL2VA 支持首帧、尾帧、首尾帧三种生视频模式。

### 3. 六步 SOTA 流水线（文档第三节）
- 结构化拆解（剧本→JSON 分镜：shot_id、duration_sec 2–5 秒、scene、character、action、dialogue、subtitle 每行 ≤15 汉字；60–90 秒短片约 10–14 镜，前 3 镜定调）→ 静态资产锁死 → 按镜头属性动态路由（保商品外观/Logo 走 H3 首帧 I2V；动作起止/场景转换走 FLF2V；多角色互动/复杂运镜/同步对白走 Ref2VA；大幅度肢体动作优先可灵）→ 长镜头尾帧续接与局部指令返工（尾帧无损导出作下一段首帧；局部瑕疵用自然语言改片，不整段重抽）→ 多轨声音与对口型（三层：对白/环境音/BGM；无声档出的画面另做对口型）。

### 4. H3 三模块架构（文档第五节 §1）
- H3-Context-IR（闭源编排）→ H3-Base（开源 768p 主干，33B 单流 Transformer，约 13B AdaLN 分支可预计算缓存、推理时不常驻；H3-Encoder 直接用 Qwen3-VL-32B 第 50 层隐藏状态；VAE 16 倍空间压缩、4 倍时间压缩、24 潜通道）→ H3-Regenerate-2K（闭源超分）。
- 本地只拿到中间一段，这是所有本地方案必须绕开的缺口。

### 5. 五大避坑指南（文档第五节 §2）
- ① Ref2VA 配额与音频红线（同 §1，另：视频每段 2–15 秒；部分平台前 5 张图含在基础计费）。
- ② Lite/Highspeed/H3 三档分层与“无生成音”机制（同 §1）。
- ③ 开源边界与本地提示词补偿（同 §1，另：ComfyUI、vLLM-Omni、SGLang 等已适配；华为昇腾、摩尔线程等硬件厂商已适配）。
- ④ 三代模型端点别混淆：一代 abab-video-1/I2V-01（720p、25fps、6 秒，15 种中括号运镜指令）；二代 Hailuo 02/2.3（约 10 秒、1080p 无声）；三代 H3 全面转自然语言时间线控制，支持 21:9、16:9、4:3、1:1、3:4、9:16 及全参考 Auto 构图。
- ⑤ 复杂片头/UI/文字渲染排他性规则：英文署名清晰可读、只出现一次、不重复职位人名、不新增中文、不乱码。

### 6. 导演级提示词工程（文档第五节 §3）
- 四段式公式：主体与场景 ＋ 定时动作 ＋ 镜头与景别设计 ＋ 风格、光影与声音。
- 单变量法则：每次迭代只改一个变量，模型、时长、宽高比、分辨率保持恒定。
- 迁移注意：可灵风格提示词迁到 H3 要补切镜时间戳与声音指令；Runway 静态视觉提示词要转成含速度与目标的完整运镜动作。文档另有六大标准模板与四大常见错误对照表（见原文 §3）。

### 7. 开源生态演进与生产引入（文档第五节 §4）
- Comfy-Org AdaLN 剪枝版（BF16 37.46 GB，约省 40%）与 pruned_int8_convrot（19.53 GiB）；配 qwen3vl_32b_minimax_h3_nvfp4_awq（14.61 GiB）或 INT8（25.28 GiB）文本编码器，24GB 显存可跑全功能。Winnougan W4A8 ConvRot 版扩散主干 12.5 GB。
- Unsloth GGUF 分级量化（2-bit 至 8-bit；推荐 Q4_K_M 或 Q8_0，普通 Q4_K 画质劣化明显）；运行时必须显式 --cfg-scale 1.0（高于 1.0 会中断）；可用 --backend te=cpu 与 --offload-to-cpu 卸载文本编码器。
- FastH3 Trim（50 层裁至 42 层，Rank-16 时间步条件，8 步采样，体积缩小 4.2 倍；ComfyUI 0.36.0+、res_multistep 采样器、MiniMaxH3SigmaShift 视频 10/音频 3）。
- Turbo LoRA（fl2v 8 步、ref2v 4 步）＋ TE-Speed 块缓存：6–8 步可消除动态涂抹感；社区实测 RTX 3060 12GB 上 8 步 Turbo LoRA＋新版 INT8 VAE 渲染 5 秒 864×480 仅需 4.5 分钟。
- FastH3（4 步）/FastH3 V2（8 步）蒸馏模型：仅支持文生音视频，不支持图生与全参考；高难度运动、精细细节、部分音频、跨种子口型/屏幕文字弱于基础版。
- DMAD 4/8 步蒸馏 LoRA：4 步曾有音质劣化反馈，声音要求高时倾向 8 步 Larry Turbo LoRA。
- ComfyUI 0.36.0 视频 VAE 2x 加速与 INT8 ConvRot VAE（2.81 GB）：编码提速 2.2 倍、解码 1.4–2.7 倍；注意 RTX 2080 Ti 等老卡用 int8_convrot VAE＋原生时空分块可能出全黑视频 Bug，回退 fp16 VAE。
- Fun-Controlnet-Union（约 6.8 GiB，Canny/Depth/HED/MLSD/Pose/视频局部重绘）与 PowerLoraStack（多 LoRA 无损挂载，解决量化模型合 LoRA 的噪声与维度不匹配问题）。
- 长视频拼接与色调补偿：H3 单次上限 15 秒；H3-Motion-Context 传递尾帧与音频保运动连贯；ToneCompensate 解决链式生成后续片段变暗与拼接亮度跳变（frame_shift 默认推荐；用 ComfyUI 原生加载器保 BT.709，避免 cv2 的 VHS 加载器色偏）。
- 24GB 显存主力配置（文档 §4（2））：ComfyUI ≥0.36.0，用内置模板初始化节点图，不要手搭连线；fl2va/ref2va pruned_int8_convrot 各 19.53 GiB 放 diffusion_models；qwen3vl_32b_nvfp4_awq 放 text_encoders；INT8 VAE 与 audio_vae_fp32 放 vae；挂 TE-Speed、PowerLoraStack、6–8 步 Turbo LoRA。
- Ref2VA 本地连线铁律：参考视频必须先转 24fps（ComfyUI 默认按无元数据 24fps 解析）；参考视频画面与音轨分开处理，ref_video_audio_N 必须与对应 ref_video_N 手动连线；音频编号排序上视频附带音轨优先靠前、再轮到独立音频；含音轨的参考视频同时连 vae 与 audio_vae，独立音频只连 audio_vae。先以 5 秒、16:9、固定种子做冒烟测试，验证运镜构图后再决定本地精渲。

## 验收标准

- 方案：逐项写清采纳/不采纳＋理由；不采纳的 API 档位要写清本地等价做法或明确放弃；凡引用的文档数字标来源与“待实测复核”。
- 路线图：阶段划分与现有 ROADMAP 衔接（阶段二、三的未勾选项优先）；每阶段有明确交付物与验收标准；CU 口径先统一成一个数再往下排。
- 输出位置：方案写进仓库 `docs/` 下的新文档；路线图直接更新 `docs/ROADMAP.md`（保留原有已完成项不动）。
- 不改动现有 notebook 与权重文件；不动 `main` 分支以外的任何发布动作。
