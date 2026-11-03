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

## 二、 官方 10 大风格 Embedding（`embedding:minimaxh3_*`）

MiniMax 官方提供了针对 H3 调优的原生风格 Embedding 触发词。在 Prompt 中加入以下 embedding 标记可直接唤醒特定视觉质感：

| 风格 Embedding | 视觉风格定义 | 最佳应用场景 |
|---|---|---|
| `embedding:minimaxh3_cinematic` | 电影级胶片质感、宽动态范围、变形宽银幕光斑 | 剧情短片、科技宣传片、大片感建立镜头 |
| `embedding:minimaxh3_cyberpunk` | 高饱和赛博霓虹、夜雨倒影、冷暖对比光 | 赛博朋克都市、未来科技、数据中心 |
| `embedding:minimaxh3_documentary` | 真实自然纪实感、柔和自然光散射、低调色感 | 市井烟火、弄堂老街、人文微观察 |
| `embedding:minimaxh3_macro` | 极浅景深、微米级材质表面纹理反射、边缘圆润光斑 | 芯片电路、咖啡油脂拉花、机械键盘轴体 |
| `embedding:minimaxh3_anime` | 新海诚风日式动画光影、高透明感天际线 | 二次元概念场景、青春唯美风 |
| `embedding:minimaxh3_noir` | 经典黑白黑色电影、高反差明暗对比（Chiaroscuro） | 悬疑犯罪、侦探推理、戏剧性独白 |
| `embedding:minimaxh3_vintage` | 复古 80 年代胶片噪点、微弱色相偏置、暖黄色调 | 怀旧记忆、老上海风情、复古黑胶 |
| `embedding:minimaxh3_hyperlapse` | 丝滑大范围移动延时、光轨流动与云卷云舒 | 都市车流、天际线日出日落、长跨度穿梭 |
| `embedding:minimaxh3_nature` | 自然纪录片质感（BBC 风格）、高动态绿色生机 | 森林晨雾、雪山航拍、海浪拍岸 |
| `embedding:minimaxh3_commercial` | 现代高端商业静物广告、无暇洁净布光、超清晰反光 | 消费电子、腕表首饰、美妆工业品 |

---

## 三、 无脸视频（Faceless Video）经典三段式公式

```
[主体与材质细节] + [镜头运动与光影质感] + [环境声音与拟声音效] + [风格 Embedding]
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

### 3. 环境音与音效（Audio Scene）—— H3 的秘密武器
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
  > integrated_multimodal_description: [Shot 1] At 00:00.000, cinematic slow forward dolly tracking shot moving through an endless corridor of high-density AI server racks. Blue and amber LED activity lights flickering rhythmically in deep shadows. Volumetric cold mist and cryo-cooling vapor drifting across the floor grates. embedding:minimaxh3_cinematic
  >
  > overall_soundscape: Deep industrial bass hum of ventilation fans, intermittent electronic beep echoes in an enclosed server hall, faint acoustic hiss of cold cooling vapor.
  >
  > non_diegetic_music: Low-frequency futuristic ambient drone, dark and focused.
- **推荐规格**：960×544，124 帧（5.16 秒），Turbo 8 步。

### 配方 2：上海街景 / 弄堂生煎早市（Shanghai Lane & Food Macro）
- **Prompt**:
  > integrated_multimodal_description: [Shot 1] At 00:00.000, cinematic close-up macro shot in an old Shanghai lane alley in front of a Shikumen red brick archway. A traditional cast-iron flat pan sizzling with round shengjian mantou buns, crispy golden-brown sesame bottoms, sprinkled with chopped green scallions. Dense fragrant white steam clouds rising softly in the cool dawn light. embedding:minimaxh3_documentary
  >
  > overall_soundscape: Crisp high-frequency sizzling of frying buns in hot oil, gentle iron lid clinking, distant charming acoustic chime of a vintage bicycle bell echoing down the narrow lane.
- **推荐规格**：960×544，124 帧，Turbo 8 步。

### 配方 3：极简办公 / 深度心流（Deep Work & ASMR）
- **Prompt**:
  > integrated_multimodal_description: [Shot 1] At 00:00.000, top-down bird-eye view of a minimalist dark walnut desk. Slender anonymous hands swiftly typing on a custom mechanical keyboard with brass plate. In the corner, steam gracefully rising from a handmade ceramic coffee mug. Soft natural morning dawn light pouring in through blinds. embedding:minimaxh3_macro
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
