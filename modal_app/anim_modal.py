"""Image-to-video clips on Modal's GPUs (free monthly credits) with Wan 2.2 TI2V-5B (Apache 2.0).

    modal run modal_app/anim_modal.py --jobs tests/anim/jobs.json --out anim_out

Same jobs.json format as `factory.py herojobs`. A 48 GB L40S holds the text encoder and the
5B transformer together, so nothing is offloaded and clips render several times faster than on
a T4. Model weights are cached in a Modal Volume after the first run.
"""
import json
import os
import time
from pathlib import Path

import modal

MODEL = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
# L40S/A100 need a card on file; A10G/L4/T4 run on the free credits alone.
GPU = os.environ.get("ANIM_GPU", "A10G")
BIG_GPU = GPU.upper().split(":")[0] in {"L40S", "A100", "A100-80GB", "H100", "H200", "B200"}
NEG = ("blurry, distorted face, extra limbs, deformed hands, text, watermark, logo, "
       "flicker, morphing, low quality, static image")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install("torch", "diffusers", "transformers", "accelerate", "ftfy", "sentencepiece",
                 "imageio", "imageio-ffmpeg", "pillow")
    .env({"HF_HOME": "/models/hf", "ANIM_GPU": GPU})
)
models = modal.Volume.from_name("yt-anim-models", create_if_missing=True)
app = modal.App("yt-anim", image=image)


@app.cls(gpu=GPU, volumes={"/models": models}, timeout=60 * 60, scaledown_window=60)
class Animator:
    @modal.enter()
    def load(self):
        import torch
        from diffusers import AutoencoderKLWan, WanImageToVideoPipeline

        t = time.time()
        vae = AutoencoderKLWan.from_pretrained(MODEL, subfolder="vae", torch_dtype=torch.float32)
        self.pipe = WanImageToVideoPipeline.from_pretrained(MODEL, vae=vae,
                                                            torch_dtype=torch.bfloat16)
        if BIG_GPU:
            self.pipe.to("cuda")
        else:  # 24 GB: keep only the active model on the GPU
            self.pipe.enable_model_cpu_offload()
            self.pipe.vae.enable_tiling()
        models.commit()  # keep downloaded weights for the next run
        print(f"model ready in {time.time() - t:.0f}s", flush=True)

    @modal.method()
    def animate(self, img_bytes: bytes, prompt: str, width: int, height: int, frames: int,
                steps: int, guidance: float) -> bytes:
        import io
        import tempfile

        from diffusers.utils import export_to_video
        from PIL import Image

        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        scale = max(width / img.width, height / img.height)
        img = img.resize((round(img.width * scale), round(img.height * scale)))
        left, top = (img.width - width) // 2, (img.height - height) // 2
        img = img.crop((left, top, left + width, top + height))
        out = self.pipe(image=img, prompt=prompt, negative_prompt=NEG, height=height, width=width,
                        num_frames=frames, num_inference_steps=steps, guidance_scale=guidance)
        with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
            export_to_video(out.frames[0], f.name, fps=24)
            return Path(f.name).read_bytes()


@app.local_entrypoint()
def main(jobs: str = "tests/anim/jobs.json", out: str = "anim_out"):
    spec = json.loads(Path(jobs).read_text())
    Path(out, "clips").mkdir(parents=True, exist_ok=True)
    animator = Animator()
    report = {"gpu": GPU, "clips": []}
    for job in spec["clips"]:
        t = time.time()
        entry = {"id": job["id"]}
        try:
            # Modal defaults: 720p-class, ~5 s at 24 fps (frames must be 4k+1).
            video = animator.animate.remote(
                Path(job["image"]).read_bytes(), job["prompt"],
                job.get("modal_width", 1280), job.get("modal_height", 704),
                job.get("modal_frames", 121), job.get("steps", 30), job.get("guidance", 5.0))
            Path(out, "clips", f"{job['id']}.mp4").write_bytes(video)
            entry.update(ok=True, seconds=round(time.time() - t))
        except Exception as e:
            entry.update(ok=False, error=repr(e)[:800], seconds=round(time.time() - t))
        report["clips"].append(entry)
        print(entry, flush=True)
    Path(out, "report.json").write_text(json.dumps(report, indent=1))
    if not any(c["ok"] for c in report["clips"]):
        raise SystemExit("all clips failed")
