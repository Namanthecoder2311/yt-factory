"""Wigglo Toons feasibility test on a free Kaggle GPU.

1. Blender (GPL; output is ours) renders 10 s of the procedural 3D Wigglo, timing each frame.
2. ACE-Step 1.5 (MIT) generates an original 60 s kids song with vocals from our lyrics.
3. The song is laid under the render as a preview.
Results land in /kaggle/working: wigglo_3d_test.mp4, song.*, wigglo_preview.mp4, report.json.
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import time

OUT = "/kaggle/working"
BLENDER_URL = "https://download.blender.org/release/Blender4.2/blender-4.2.3-linux-x64.tar.xz"
report = {}


def save():
    json.dump(report, open(f"{OUT}/report.json", "w"), indent=1)


def sh(cmd, **kw):
    print("+", cmd[:200], flush=True)
    return subprocess.run(cmd, shell=True, text=True, capture_output=True, **kw)


inputs = os.path.dirname(glob.glob("/kaggle/input/**/wigglo_scene.py", recursive=True)[0])
ffmpeg = shutil.which("ffmpeg")
if not ffmpeg:
    sh(f"{sys.executable} -m pip install -q imageio-ffmpeg")
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
gpu = sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader").stdout.strip()
report["gpu"] = gpu
print("GPU:", gpu, flush=True)

# ---------------------------------------------------------------- 1. Blender 3D render
try:
    if os.path.exists(f"{inputs}/skip_blender"):
        raise RuntimeError("skipped (skip_blender flag)")
    t = time.time()
    sh(f"curl -sSL {BLENDER_URL} -o /tmp/blender.tar.xz && tar -xf /tmp/blender.tar.xz -C /tmp")
    blender = glob.glob("/tmp/blender-*/blender")[0]
    r = sh(f"{blender} -b -P {inputs}/wigglo_scene.py -- --out /tmp/frames --seconds 10 "
           f"--res 1280x720 --samples 32")
    log = r.stdout + r.stderr
    print(log[-3000:], flush=True)
    stats = [l for l in log.splitlines() if "RENDER_STATS" in l or "render device" in l]
    sh(f"{ffmpeg} -y -loglevel error -framerate 24 -i /tmp/frames/f_%04d.png -c:v libx264 "
       f"-pix_fmt yuv420p -crf 18 {OUT}/wigglo_3d_test.mp4")
    report["blender"] = {"ok": os.path.exists(f"{OUT}/wigglo_3d_test.mp4"), "stats": stats,
                         "total_seconds": round(time.time() - t)}
except Exception as e:
    report["blender"] = {"ok": False, "error": repr(e)[:800]}
save()

# ---------------------------------------------------------------- 2. ACE-Step song
try:
    t = time.time()
    song = json.load(open(f"{inputs}/song_test.json"))
    sh("git clone -q --depth 1 https://github.com/ACE-Step/ACE-Step-1.5.git /tmp/ace")
    # Kaggle's Python is newer than the project's pinned version, so install while ignoring
    # the pin, plus its requirements file if it ships one.
    r = sh(f"{sys.executable} -m pip install -q --ignore-requires-python -e /tmp/ace")
    print(r.stdout[-1500:], r.stderr[-1500:], flush=True)
    import tomllib  # install every declared dependency except torch (keep Kaggle's CUDA build)

    deps = tomllib.load(open("/tmp/ace/pyproject.toml", "rb")).get("project", {}).get("dependencies", [])
    skip = ("torch", "mlx", "nano-vllm", "gradio", "fastapi", "uvicorn", "tensorboard", "lightning",
            "lycoris", "modelscope", "python-multipart")
    keep = [d.split(";")[0].strip() for d in deps if not d.lower().startswith(skip)]
    failed = []
    for d in keep + ["loguru", "vector_quantize_pytorch"]:  # one at a time: a bad pin can't block the rest
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", d], capture_output=True, text=True)
        if r.returncode:
            failed.append(d)
    print("deps failed:", failed, flush=True)
    report["song_deps_failed"] = failed
    sys.path.insert(0, "/tmp/ace")
    from acestep.handler import AceStepHandler
    from acestep.inference import GenerationConfig, GenerationParams, generate_music
    from acestep.llm_inference import LLMHandler

    dit = AceStepHandler()
    dit.initialize_service(project_root="/tmp/ace", config_path="acestep-v15-turbo", device="cuda")
    llm = LLMHandler()
    for backend in ("pt", "transformers", "vllm"):
        try:
            llm.initialize(checkpoint_dir="/tmp/ace/checkpoints", lm_model_path="acestep-5Hz-lm-0.6B",
                           backend=backend, device="cuda")
            report["song_lm_backend"] = backend
            break
        except Exception as e:
            print("LM backend", backend, "failed:", repr(e)[:300], flush=True)
    params = GenerationParams(caption=song["caption"], lyrics=song["lyrics"],
                              vocal_language=song["language"], duration=song["duration"],
                              inference_steps=8)
    res = generate_music(dit, llm, params, GenerationConfig(batch_size=1, audio_format="wav"),
                         save_dir=f"{OUT}/song")
    if not res.success:
        raise RuntimeError(res.error)
    path = res.audios[0]["path"]
    report["song"] = {"ok": True, "file": os.path.relpath(path, OUT), "seconds": round(time.time() - t)}
except Exception as e:
    report["song"] = {"ok": False, "error": repr(e)[:1200], "seconds": round(time.time() - t)}
save()

# ---------------------------------------------------------------- 3. preview with song
if report.get("blender", {}).get("ok") and report.get("song", {}).get("ok"):
    sh(f"{ffmpeg} -y -loglevel error -stream_loop 5 -i {OUT}/wigglo_3d_test.mp4 "
       f"-i {OUT}/{report['song']['file']} -t 30 -c:v libx264 -pix_fmt yuv420p -c:a aac "
       f"-shortest {OUT}/wigglo_preview.mp4")
    report["preview"] = os.path.exists(f"{OUT}/wigglo_preview.mp4")
save()
print(json.dumps(report, indent=1))
