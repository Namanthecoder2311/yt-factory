"""Runs on a free Kaggle GPU: turns still scene images into short motion clips with
Wan 2.2 TI2V-5B (Apache 2.0). Inputs come from the attached dataset (jobs.json + images);
clips and report.json are written to /kaggle/working for the GitHub runner to download.
"""
import gc
import glob
import json
import os
import subprocess
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "diffusers", "transformers",
                "accelerate", "ftfy", "sentencepiece", "imageio", "imageio-ffmpeg", "spandrel"],
               check=True)

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from diffusers import AutoencoderKLWan  # noqa: E402
from diffusers.utils import export_to_video  # noqa: E402

MODEL = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
NEG = ("blurry, distorted face, extra limbs, deformed hands, text, watermark, logo, "
       "flicker, morphing, low quality, static image")
OUT = "/kaggle/working"

found = glob.glob("/kaggle/input/**/jobs.json", recursive=True)
if not found:  # show what Kaggle mounted, then stop
    for d, _, files in os.walk("/kaggle/input"):
        print(d, files[:10])
    raise SystemExit("jobs.json not found under /kaggle/input")
jobs_path = found[0]
root = os.path.dirname(jobs_path)
jobs = json.load(open(jobs_path))
props = torch.cuda.get_device_properties(0)
# T4 (sm_75) only emulates bf16, which is slow; use native fp16 there.
dtype = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
print(f"GPU {props.name} {props.total_memory / 1e9:.1f} GB, dtype {dtype}", flush=True)

t_load = time.time()
vae = AutoencoderKLWan.from_pretrained(MODEL, subfolder="vae", torch_dtype=torch.float32)
try:
    from diffusers import WanImageToVideoPipeline
    pipe = WanImageToVideoPipeline.from_pretrained(MODEL, vae=vae, torch_dtype=dtype)
    kind = "WanImageToVideoPipeline"
except Exception as e:  # the 5B checkpoint may only load as the text+image WanPipeline
    print("I2V pipeline failed, falling back to WanPipeline:", repr(e)[:300], flush=True)
    from diffusers import WanPipeline
    pipe = WanPipeline.from_pretrained(MODEL, vae=vae, torch_dtype=dtype)
    kind = "WanPipeline"
# The UMT5 text encoder (~11 GB) and the 5B transformer (~10 GB) can't share a 15 GB T4, so
# stream weights layer by layer instead of whole models (slower, but fits).
if props.total_memory < 30e9:
    pipe.enable_sequential_cpu_offload()
else:
    pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()
print(f"loaded {kind} in {time.time() - t_load:.0f}s", flush=True)

_upscaler = None
ESRGAN_URL = ("https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/"
              "RealESRGAN_x4plus_anime_6B.pth")  # BSD-3, tuned for cartoons


def upscale(frames):
    """4x Real-ESRGAN (anime) per frame; returns float RGB frames in [0, 1]."""
    global _upscaler
    import urllib.request

    import numpy as np
    from spandrel import ModelLoader

    if _upscaler is None:
        path = "/tmp/esrgan_anime.pth"
        if not os.path.exists(path):
            urllib.request.urlretrieve(ESRGAN_URL, path)
        _upscaler = ModelLoader().load_from_file(path).model.cuda().eval().half()
    out = []
    with torch.no_grad():
        for f in frames:
            a = np.asarray(f, dtype=np.float32)
            if a.max() > 1.5:
                a = a / 255.0
            x = torch.from_numpy(a).permute(2, 0, 1)[None].cuda().half()
            y = _upscaler(x).clamp(0, 1)[0].permute(1, 2, 0).float().cpu().numpy()
            out.append(y)
    return out


os.makedirs(f"{OUT}/clips", exist_ok=True)
report = {"gpu": props.name, "pipeline": kind, "dtype": str(dtype), "clips": []}
for job in jobs["clips"]:
    w, h = job.get("width", 640), job.get("height", 352)
    img = Image.open(os.path.join(root, job["image"])).convert("RGB")
    scale = max(w / img.width, h / img.height)
    img = img.resize((round(img.width * scale), round(img.height * scale)))
    left, top = (img.width - w) // 2, (img.height - h) // 2
    img = img.crop((left, top, left + w, top + h))
    t = time.time()
    entry = {"id": job["id"]}
    try:
        result = pipe(image=img, prompt=job["prompt"], negative_prompt=NEG, height=h, width=w,
                      num_frames=job.get("frames", 49), num_inference_steps=job.get("steps", 30),
                      guidance_scale=job.get("guidance", 5.0))
        frames = result.frames[0]
        if job.get("upscale"):
            try:
                frames = upscale(frames)
                entry["upscaled"] = True
            except Exception as e:  # keep the clip even if upscaling fails
                entry["upscale_error"] = repr(e)[:300]
        export_to_video(frames, f"{OUT}/clips/{job['id']}.mp4", fps=24)
        entry.update(ok=True, seconds=round(time.time() - t))
    except Exception as e:
        entry.update(ok=False, error=repr(e)[:800], seconds=round(time.time() - t))
        result = None
        gc.collect()
        torch.cuda.empty_cache()
    report["clips"].append(entry)
    json.dump(report, open(f"{OUT}/report.json", "w"), indent=1)
    print(entry, flush=True)
