# MiniMax H3 无脸视频（Faceless Video）提示词配方与结构指南

MiniMax H3 采用音画同频扩散（Joint Audio-Visual Denoising），因此提示词结构与单纯文生图/文生视频模型存在本质差异。在提示词中明确描述声音，不仅能生成 32kHz 原生立体声，还能强化画面的时空一致性。

---

## 一、三段式提示词公式

```
[主体与视觉细节] + [镜头运动与光影质感] + [环境声音与拟声音效]
```

### 1. 主体与视觉细节（Visual Subject）
- **核心原则**：无脸视频杜绝真人露脸特写，侧重于：
  - 局部动作（手部操作、键盘打字、翻书、手握咖啡杯）
  - 宏观场景（雨夜都市天际线、深林晨雾、数据中心服务器阵列）
  - 微观与材质（芯片微观结构、咖啡液滴溅起、机械齿轮运转）
- **画质修饰**：`photorealistic, cinematic lighting, shallow depth of field, 8k resolution, finely detailed textures`.

### 2. 镜头运动与构图（Camera Motion）
- **运镜词汇**：
  - `Slow cinematic push in`（缓慢推近）
  - `Smooth orbit tracking shot`（平滑环绕跟拍）
  - `Slow motion drone establishing shot`（慢动作无人机俯瞰）
  - `Handheld macro shot with subtle camera shake`（微距手持微颤）
  - `Static locked-off shot with passing shadows`（静止固定机位，光影流动）

### 3. 环境音与音效（Audio Scene）—— H3 的秘密武器
- **核心原则**：不建议在 H3 中赌复杂的旁白口播对白（容易语无伦次或语速不均）。把 H3 原生音频作为 **高端环境沉浸声（Ambient Sound FX）**。
- **声音词汇示例**：
  - `Ambient sound of heavy rain tapping on window glass, distant rolling thunder.`
  - `Subtle hum of server fans in a cold room, intermittent rhythmic electronic chirps.`
  - `Crisp mechanical keyboard clicking, soft page-turning sound in a quiet room.`
  - `Crackling fireplace warmth, wind howling outside.`

---

## 二、经典无脸场景模板库

### 模板 1：极客 / 科技科普（Tech & Coding Focus）
- **Prompt**:
  > A close-up cinematic shot of mechanical hands typing on an ergonomic split keyboard, cyan ambient underglow illuminating the desk mat. Multiple curved ultrawide monitors display scrolling Python and CUDA code. Smooth slow dolly forward. Crisp mechanical clicky switch sounds, soft hum of PC cooling fans, quiet room acoustic ambiance.
- **推荐规格**：1344×768（或 960×544 筛选），5–8 秒，PDD 8 步。

### 模板 2：商业金融 / 宏观趋势（Finance & Abstract Concept）
- **Prompt**:
  > Aerial drone view slowly flying forward over a hypermodern financial district at twilight. Golden hour reflections on glass skyscraper facades, traffic light trails flowing smoothly along the highway below. Cinematic volumetric fog. Gentle evening breeze audio, deep low-frequency atmospheric city hum, distant muffled traffic rumble.
- **推荐规格**：1344×768，8 秒，PDD 8 步。

### 模板 3：自我提升 / 深度工作 / ASMR（Deep Work & Productivity）
- **Prompt**:
  > Overhead top-down shot of a minimalist wooden desk. A fountain pen steadily sketching flowcharts in a leather-bound notebook, next to a steaming ceramic mug of pour-over coffee. Natural soft morning sunlight drifting across the table. Scratching sound of pen nib on textured paper, delicate liquid pour and clinking mug sound, peaceful morning silence.
- **推荐规格**：960×544，5 秒，Turbo 4 步（快筛首选）。

---

## 三、剪辑与成片工业流

1. **画面（Colab H3）**：批量输出 5–8 秒的高质量 B-roll 镜头（无任何水印与真人露脸）。
2. **音频分轨分离**：
   - 提取 H3 生成的 MP4 中自带的音频轨道（使用 `ffmpeg -i video.mp4 -vn -c:a copy ambient.wav`）。
   - 该音频轨道天然与画面动作（如敲键盘、雨滴、倒水）对齐。
3. **旁白生成**：
   - 使用独立的 TTS（如 Edge-TTS、CosyVoice、Fish Audio）生成清晰的讲解旁白。
4. **后期合成（CapCut / Premiere / FFMPEG 自动化）**：
   - 旁白轨（Voiceover）置于主音量（0dB）。
   - H3 原生环境轨（Ambient FX）置于次音量（-12dB 到 -18dB），作为声场铺底。
   - 自动生成字幕并导出，完成 100% 纯原创、高质感、无脸视频。
