# MiniMax H3 (Hailuo 3.0) 提示词配方与官方结构化规范指南

MiniMax H3 采用音画同频联合扩散（Joint Audio-Visual Latent Denoising）架构。在提示词中精确描述声音、镜头时序与空间运动，不仅能驱动原生 32kHz 高保真立体声音频生成，还能大幅强化画面时间与因果一致性。

---

## 一、 官方原生结构化多镜头语法（Native Structured Prompt Spec）

MiniMax H3 官方原生支持**多镜头时间轴编排**与**全模态声音解耦描述**。在单个任务或长镜头（如 124 帧 @ 24fps ≈ 5.16s）中，可使用以下结构化标记：

### 1. 结构化标记字段速查表

```yaml
integrated_multimodal_description: 
  [Shot 1] At 00:00.000, the camera begins with ...
  [Shot 2] At 00:02.500, the camera cuts to / pans to ...

overall_soundscape: 
  Continuous ambient tone, environmental reverberation, diegetic sound effects.

non_diegetic_music: 
  Background musical score, instrumental genre, tempo, mood.
```

### 2. 字段详细规范与示例

#### (1) `integrated_multimodal_description`
- **镜头时序控制**：每个镜头使用 `[Shot N] At MM:SS.mmm, the camera [action] ...` 明确切镜（Cut）或运镜（Tracking/Pan/Tilt）。
- **示例**：
  ```
  integrated_multimodal_description: [Shot 1] At 00:00.000, slow cinematic tracking shot entering a dimly lit server room with humming racks and icy mist. [Shot 2] At 00:02.600, the camera smoothly cuts to an extreme close-up of amber LED indicators flickering rapidly, reflecting on polished metallic casing.
  ```

#### (2) `overall_soundscape`（现场环境沉浸声 / Diegetic Sound）
- 描述画面内物体产生的自然物理声与环境底噪，H3 的 Audio VAE 会自动将动作与音效毫秒级对齐。
- **示例**：
  ```
  overall_soundscape: Deep continuous industrial bass hum of ventilation cooling fans, rhythmic electronic chirp echoes, subtle soft air rush of cryogenic cooling mist.
  ```

#### (3) `non_diegetic_music`（背景音乐配乐 / Non-diegetic Score）
- 描述用于烘托氛围的情绪化配乐（非画面内物体发声）。
- **示例**：
  ```
  non_diegetic_music: Atmospheric synthwave drone with minimal pulsing sub-bass, mysterious and futuristic mood.
  ```

#### (4) 对话与人声标签 `<d>[Language] ...</d>`
- 当场景需要微弱人声或拟真对话时，使用 `<d>[Language] 文本</d>` 包装：
  ```
  [Shot 1] At 00:01.000, a researcher wearing a headset whispers <d>[English] System compilation complete, initiating sequence.</d> with natural room reverberation.
  ```
  *(注：无脸视频建议聚焦环境拟音声效，避免过长口播导致语速漂移)*。

---

## 二、 官方 10 大原生特效与运镜 Embedding（`embedding:minimaxh3_*`）

MiniMax 官方在 HuggingFace 仓库 `Comfy-Org/MiniMax-H3/tree/main/embeddings` 中发布了 10 个官方原生微调的视觉特效与运镜 Embedding 文件（需存放在 `ComfyUI/models/embeddings/` 目录下调用）。在 Prompt 中输入对应触发词即可唤醒特定的高难度电影级特效或转场运镜：

| 官方 Embedding 触发词 | 精确文件大小 | 核心视觉动力学 / 特效定义 | 典型应用场景 |
|---|---|---|---|
| `embedding:minimaxh3_bullet_time` | `962,680` 字节 | 经典黑客帝国「子弹时间」，画面主体极慢速甚至冻结，摄像机高速多角度环绕旋转 | 动作高潮、微距水滴凝滞、科技悬停展示 |
| `embedding:minimaxh3_spiral_ascent` | `1,341,560` 字节 | 螺旋升降镜头（Corkscrew / Spiral Crane），摄像机沿垂直中轴螺旋盘旋上升 | 宏大建筑俯瞰、深林树冠升腾、都市巡航 |
| `embedding:minimaxh3_kiss_camera` | `993,400` 字节 | 双人近景推进、情侣亲吻特写、电影感浪漫景深对焦 | 剧情短片情感节点、婚礼与浪漫叙事 |
| `embedding:minimaxh3_four_seasons` | `1,454,200` 字节 | 四季更替流转（春夏秋冬植被色彩与天候快速渐变） | 时间流逝蒙太奇、自然景观岁月感呈现 |
| `embedding:minimaxh3_blooming_flowers` | `1,259,640` 字节 | 花卉花瓣延时绽放（Time-lapse Blooming），细微脉络舒展 | 植物生长延时、微观生命力、治愈系自然短片 |
| `embedding:minimaxh3_art_is_explosion` | `512,120` 字节 | 戏剧性爆炸冲击波与尘埃碎片散射（Art is Explosion） | 动作大片特效、微观能量释放、科幻引爆 |
| `embedding:minimaxh3_fire_breath` | `1,208,440` 字节 | 烈焰喷吐与高温火舌等离子体流体粒子动态 | 奇幻生物吐息、工业熔炉喷火、魔法对决 |
| `embedding:minimaxh3_dark_magic` | `604,280` 字节 | 暗黑暗物质魔法能量流、紫黑雾气缭绕消散 | 奇幻暗黑世界观、神秘科技失控 |
| `embedding:minimaxh3_storm_magic` | `1,403,000` 字节 | 暴风雨闪电交加、电弧激荡与狂暴气流旋涡 | 极端天候灾难、雷电能量激发、风暴降临 |
| `embedding:minimaxh3_truman_show` | `921,720` 字节 | 「楚门的世界」式世界观撕裂、现实解构与穹顶窥视感 | 哲学思考、虚拟现实破壁、反乌托邦剧情 |

> 💡 **使用提示**：无脸视频日常生产一般依赖纯 Prompt 描述即可出高质感成片；如需特定特效（如打字微距子弹时间冻结、航拍螺旋上升），可下载对应 `.safetensors` 至 `models/embeddings/` 并在提示词中添加触发词。

---

## 三、 无脸视频（Faceless Video）经典三段式公式

```
[主体与材质细节] + [镜头运动与光影质感] + [环境声音与拟声音效]
```

### 1. 主体与视觉细节（Visual Subject）
- **核心原则**：无脸视频杜绝真人露脸特写，侧重于：
  - **局部动作**：手部操作（机械键盘盲打、精密手柄装配、咖啡手冲滴落）
  - **宏观场景**：雨夜都市天际线、深林晨雾、数据中心服务器阵列、外滩霓虹江景
  - **微观材质**：芯片微结构、微距咖啡液滴溅起、机械齿轮运转
- **画质修饰词**：`photorealistic, cinematic lighting, shallow depth of field, finely detailed textures, 8k resolution`.

### 2. 镜头运动与构图（Camera Motion）
- `Slow cinematic push in`（缓慢推近）
- `Smooth orbit tracking shot`（平滑环绕跟拍）
- `Slow motion drone establishing shot`（慢动作无人机俯瞰）
- `Handheld macro shot with subtle camera shake`（微距手持微颤）
- `Static locked-off shot with passing shadows`（静止固定机位，光影流动）

### 3. 环境音与音效（Audio Scene）与原生音频生成
- **核心原则**：将 H3 原生音频作为 **高端环境沉浸声（Ambient Sound FX）**。
- **声音词汇示例**：
  - `Ambient sound of heavy rain tapping on window glass, distant rolling thunder.`
  - `Subtle hum of server fans in a cold room, intermittent rhythmic electronic chirps.`
  - `Crisp mechanical keyboard clicking, soft page-turning sound in a quiet room.`
  - `Sizzling sound of frying buns in bubbling hot oil, gentle iron lid clinking.`

---

## 四、 经典无脸场景配方库

### 配方 1：极客破晓 / 算力心脏（Tech & Data Center）
- **Prompt**:
  > integrated_multimodal_description:
  > [Shot 1] Cinematic slow forward dolly tracking shot moving through an endless corridor of high-density AI server racks. Blue and amber LED activity lights flickering rhythmically in deep shadows. Volumetric cold mist and cryo-cooling vapor drifting across the floor grates.
  >
  > overall_soundscape: Deep industrial bass hum of ventilation fans, intermittent electronic beep echoes in an enclosed server hall, faint acoustic hiss of cold cooling vapor.
  >
  > non_diegetic_music: Low-frequency futuristic ambient drone, dark and focused.
- **推荐规格**：960×544，124 帧（5.16 秒），Turbo 8 步。*(可选：如需环绕旋转，可追加 `embedding:minimaxh3_bullet_time`)*

### 配方 2：上海街景 / 弄堂生煎早市（Shanghai Lane & Food Macro）
- **Prompt**:
  > integrated_multimodal_description:
  > [Shot 1] Cinematic close-up macro shot in an old Shanghai lane alley in front of a Shikumen red brick archway. A traditional cast-iron flat pan sizzling with round shengjian mantou buns, crispy golden-brown sesame bottoms, sprinkled with chopped green scallions. Dense fragrant white steam clouds rising softly in the cool dawn light.
  >
  > overall_soundscape: Crisp high-frequency sizzling of frying buns in hot oil, gentle iron lid clinking, distant charming acoustic chime of a vintage bicycle bell echoing down the narrow lane.
- **推荐规格**：960×544，124 帧，Turbo 8 步。

### 配方 3：极简办公 / 深度心流（Deep Work & ASMR）
- **Prompt**:
  > integrated_multimodal_description:
  > [Shot 1] Top-down bird-eye view of a minimalist dark walnut desk. Slender anonymous hands swiftly typing on a custom mechanical keyboard with brass plate. In the corner, steam gracefully rising from a handmade ceramic coffee mug. Soft natural morning dawn light pouring in through blinds.
  >
  > overall_soundscape: Crisp tactile mechanical switch clacking with satisfying solid bottom-out sound, subtle sleeve cloth rustle, quiet room morning silence.
- **推荐规格**：960×544 或 1344×768，124 帧，Turbo 8 步。

---

## 五、 后期自动化音视频处理流水线

1. **画面（Colab H3 2.0）**：批量输出 124 帧（5.16 秒）高质量 B-roll 原生 MP4（内置 32kHz 立体声环境音）。
2. **音频分轨分离**：
   ```bash
   ffmpeg -i shot_01.mp4 -vn -c:a copy ambient_32k.wav
   ```
3. **独立 TTS 旁白生成**：
   使用独立的 TTS（如 Edge-TTS、CosyVoice、Fish Audio）生成清晰的解说口播配音。
4. **音频终混（自动化脚本）**：
   - 旁白轨（Voiceover）：主音量设为 `0dB`；
   - H3 原生环境拟音轨（Ambient FX）：音量衰减至 `-14dB ~ -18dB` 作为高质量背景底噪；
   - 结合自动字幕生成，形成 100% 独家质感的工业级无脸视频。
