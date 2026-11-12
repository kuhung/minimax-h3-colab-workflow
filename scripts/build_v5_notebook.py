"""Build a self-contained three-shot watch notebook using verified V3 setup."""
import json
from pathlib import Path
import build_v3_notebooks as v3

ROOT = Path(__file__).resolve().parent.parent

def build():
    source = json.loads((ROOT/'notebooks/v3.1_minimax_h3_i2v.ipynb').read_text())
    watch = next(''.join(c['source']) for c in source['cells'] if 'def generate_default_watch_anchor' in ''.join(c['source']))
    workflows = {mode: json.loads((ROOT/f'workflows/h3_{mode}_production.json').read_text()) for mode in ('i2v','flf2v')}
    shots = [
        dict(id='01_reveal', mode='i2v', start='wide', seed=501, prompt='Slow gentle camera push toward the watch. A warm highlight moves across the metal bezel. Preserve the dial, hands and watch geometry. No dialogue.'),
        dict(id='02_detail', mode='i2v', start='detail', seed=502, prompt='Macro close-up of the same watch dial. The second hand moves clockwise, a soft light sweeps across the crystal. Preserve all dial markings and proportions. No dialogue.'),
        dict(id='03_finish', mode='flf2v', start='detail', end='wide', seed=503, prompt='The camera smoothly pulls back from the dial close-up to reveal the complete watch, ending in a steady product shot. Preserve the same watch throughout. No dialogue.'),
    ]
    (ROOT/'configs/watch_storyboard.json').write_text(json.dumps(shots,indent=2,ensure_ascii=False)+'\n')
    assets = watch[:watch.index('active_candidate =')]
    assets += '''
# 可填写 v4 生成或修图后保存到 Drive 的腕表图片；留空运行程序化素材基线。
WATCH_IMAGE = ""
from PIL import ImageOps
from IPython.display import display
if WATCH_IMAGE:
    if not Path(WATCH_IMAGE).is_file():
        raise FileNotFoundError(WATCH_IMAGE)
    master = ImageOps.exif_transpose(Image.open(WATCH_IMAGE)).convert("RGB")
else:
    generate_default_watch_anchor(START_IMAGE_PATH)
    master = Image.open(START_IMAGE_PATH).convert("RGB")
wide = ImageOps.fit(master, (960, 544), method=Image.Resampling.LANCZOS)
w, h = wide.size
# 所有关键帧由同一张主图派生，避免换表；细节镜头是裁剪，非新增细节。
detail = wide.crop((w//4, h//4, 3*w//4, 3*h//4)).resize((960,544), Image.Resampling.LANCZOS)
COMFY_INPUT = COMFY_ROOT / "input"
COMFY_INPUT.mkdir(parents=True, exist_ok=True)
for name, image in (("wide", wide), ("detail", detail)):
    image.save(COMFY_INPUT / f"watch_{name}.png")
    display(image)
'''
    render = '''
import copy, uuid
from IPython.display import Video, display
RUN_ID = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
RUN_DIR = DRIVE_OUTPUTS / ("watch_storyboard_" + RUN_ID)
RUN_DIR.mkdir(parents=True, exist_ok=True)
records, clips = [], []
for key in ("wide", "detail"):
    shutil.copy2(COMFY_ROOT / "input" / f"watch_{key}.png", RUN_DIR)
try:
    for shot in SHOTS:
        wf = copy.deepcopy(WORKFLOWS[shot["mode"]])
        wf["1"]["inputs"]["unet_name"] = DIFFUSION_FILE
        wf["4"]["inputs"]["vae_name"] = VIDEO_VAE_FILE
        wf["10"]["inputs"]["lora_name"] = LORA_FILE
        wf["14"]["inputs"]["noise_seed"] = shot["seed"]
        wf["16"]["inputs"]["steps"] = 8
        wf["20"]["inputs"]["image"] = "watch_" + shot["start"] + ".png"
        wf["6"]["inputs"].update(prompt=shot["prompt"], width=960, height=544, length=124, first_frame=["20",0])
        if shot["mode"] == "flf2v":
            wf["21"]["inputs"]["image"] = "watch_" + shot["end"] + ".png"
            wf["6"]["inputs"]["last_frame"] = ["21",0]
        wf["9"]["inputs"]["filename_prefix"] = RUN_ID + "_" + shot["id"]
        (RUN_DIR / (shot["id"] + "_workflow.json")).write_text(json.dumps(wf,indent=2))
        started = time.time()
        response = post_json("/prompt", {"prompt":wf}, timeout=30)
        if response.get("error") or response.get("node_errors"):
            raise RuntimeError(str(response))
        pid = response["prompt_id"]
        while time.time() - started < 1200:
            history = post_json(f"/history/{pid}", timeout=30)
            if pid in history:
                result = history[pid]
                if result.get("status",{}).get("status_str") != "success":
                    raise RuntimeError(str(result.get("status")))
                break
            time.sleep(3)
        else:
            post_json("/interrupt", {})
            raise TimeoutError(shot["id"])
        # 唯一批次前缀，避免误取旧成片。
        files = list(LOCAL_OUT.rglob(wf["9"]["inputs"]["filename_prefix"] + "*.mp4"))
        if len(files) != 1:
            raise RuntimeError(f"输出文件数量异常: {files}")
        local = files[0]
        streams = json.loads(subprocess.check_output(["ffprobe","-v","error","-show_streams","-of","json",str(local)]))["streams"]
        if not any(s["codec_type"]=="video" for s in streams):
            raise RuntimeError("缺少视频轨")
        target = RUN_DIR / (shot["id"] + ".mp4")
        shutil.copy2(local,target)
        elapsed = time.time()-started
        records.append(dict(shot=shot, seconds=elapsed, estimated_cu=elapsed*1.94/3600, streams=streams))
        (RUN_DIR/"metrics.json").write_text(json.dumps(records,indent=2))
        clips.append(local)
        log(f"{shot['id']} 完成: {elapsed:.1f}s，已备份 Drive")
    concat = LOCAL_OUT / (RUN_ID + "_concat.txt")
    concat.write_text("".join(f"file '{p.as_posix()}'\\n" for p in clips))
    final = LOCAL_OUT / (RUN_ID + "_watch_final.mp4")
    subprocess.run(["ffmpeg","-y","-v","error","-f","concat","-safe","0","-i",str(concat),"-c:v","libx264","-pix_fmt","yuv420p","-c:a","aac","-ar","32000","-ac","2","-movflags","+faststart",str(final)],check=True)
    shutil.copy2(final,RUN_DIR/"watch_final.mp4")
    display(Video(str(final),embed=True))
    log(f"三镜头成片已保存: {RUN_DIR}")
finally:
    if AUTO_SHUTDOWN:
        safe_shutdown("多镜头任务结束；已完成镜头保存在 Drive")
'''
    cells=[v3.md('# MiniMax H3 v5.0：腕表三镜头短片\n\n依次运行。默认生成统一腕表素材；可在关键帧单元填入 v4 图片的 Drive 路径。三镜分别为全貌推进、表盘细节、拉远定格，每镜124帧，成片约15.5秒。输入为同图派生关键帧，镜头间采用直接剪切。此版本待 Colab 实测。'),v3.code(v3.get_cell_config('i2v',8)),v3.code(v3.CELL_DRIVE),v3.code(v3.CELL_GPU),v3.md('## 关键帧与分镜\n检查下方全貌和细节图，确认主体可用后再运行后续单元。'),v3.code(assets),v3.code('SHOTS = '+repr(shots)+'\nWORKFLOWS = '+repr(workflows)),v3.code(v3.CELL_COMFY),v3.code(v3.get_cell_models('i2v',8)),v3.code(v3.CELL_START_COMFY),v3.code(render)]
    nb=dict(nbformat=4,nbformat_minor=0,metadata={'colab':{'name':'v5.0_minimax_h3_watch_storyboard.ipynb','gpuType':'L4'}},cells=cells)
    (ROOT/'notebooks/v5.0_minimax_h3_watch_storyboard.ipynb').write_text(json.dumps(nb,indent=2,ensure_ascii=False)+'\n')
if __name__=='__main__':build()
