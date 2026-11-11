# 交办：按 SOTA 调研资料写 MiniMax H3 科普文章（分多期/多层）

交办人：爆炸头（Muse）｜日期：2026-11-07｜执行方：Antigravity
模型：Gemini 3.8 Flash（`gemini-3.8-flash`，Antigravity agent 默认模型）

## 任务

以《2026 AI视频生成全场景SOTA工作流与MiniMax H3实战调研报告》为唯一素材来源，写一套 MiniMax H3 科普文章。先按体量和逻辑定分期/分层方案，再按方案写成稿。

素材原文（Google Doc，2026-11-04 更新）：《AI视频生成SOTA工作流调研》
https://docs.google.com/document/d/1XpQQIXkId9UhS8gdPdh2tZhFqOf28ulqYeQWH7PCxsY/edit?usp=drivesdk

本地已有拆分参考：`/Users/kuhung/roy/3_Active_Labs/minimax-h3-colab-workflow/docs/sota-split-brief-20261107.md`（七部分要点拆分，含项目现状与硬约束）。

## 分期/分层要求

- 先出方案：按体量和逻辑把全文切成多期（或多层），每期写清主题、覆盖原文档哪几部分、篇幅、顺序、与前后期的衔接。
- 分层建议（可调整，但要说明理由）：
  - 入门层：H3 是什么、能做什么、四大实操准则、档位分层思路——零基础读者能直接上手。
  - 进阶层：七大输入场景、提示词减法原则、时间线提示词工程（四段式公式、单变量法则）。
  - 工程层：三模块架构、开源边界与本地部署、量化与加速生态、生产配置与 Ref2VA 连线铁律。
- 每期独立成篇，单篇能独立阅读；多期之间用“上期回顾＋下期预告”一句话衔接，不写跨期悬念。

## 写作要求

- 读者：零基础中文读者（Vibe Coding 课程读者）。讲清“是什么、怎么用”，不要堆术语；术语第一次出现要用人话解释。
- 忠实原文：只写文档里的内容和公开报道的原始说法，不自创类比、定位、标题党和升华金句；不编造没有的数据和案例。
- 数字与结论带出处：凡引用的文档数字（配额、分辨率、耗时、成本、显存）标明来源，凡是文档口径非我方实测的，写明“文档口径，待本地实测复核”。
- 本地路线口径：我们走的是纯本地路线（Google AI Pro 的 Colab 额度＋开源权重，不引入 MiniMax API）。凡是文档里走 API 的档位（H3 Lite/Highspeed 探路、完整版 H3 出 2K、H3-Regenerate-2K、H3-Context-IR），文章里要写清这是托管/API 档位，本地路线只能借鉴思路，不能写成“本地也能直接用”。
- 每期文末附“原文对照”：列出本期内容对应原文档的章节，方便读者回查。

## 交付物

- 分期/分层方案（1 份）：`docs/articles/plan.md`。
- 各期成稿（中文 Markdown，每期一个文件）：`docs/articles/` 下，按 `01-xxx.md`、`02-xxx.md` 顺序编号。
- 输出位置：仓库 `/Users/kuhung/roy/3_Active_Labs/minimax-h3-colab-workflow/docs/articles/`。
- 不改动仓库现有 notebook、权重文件与 ROADMAP.md。

## 验收标准

- 方案：切分覆盖原文档全部七部分，无遗漏、无重复；每期主题与篇幅合理。
- 成稿：每期事实与原文一致，数字带出处；零基础读者能读懂；无自创类比与标题党；API 与本地路线的口径没有混淆。
