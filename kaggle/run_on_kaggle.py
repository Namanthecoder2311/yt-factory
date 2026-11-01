"""Send animation jobs to a free Kaggle GPU and bring the clips back.

    python kaggle/run_on_kaggle.py jobs.json out_dir

jobs.json: {"clips": [{"id": "...", "image": "path/to/img.png", "prompt": "...", ...}]}
Needs KAGGLE_USERNAME and KAGGLE_KEY in the environment (phone-verified Kaggle account,
so the kernel may use a GPU and the internet).
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

USER = os.environ["KAGGLE_USERNAME"]
DATASET = f"{USER}/yt-anim-input"
KERNEL = f"{USER}/yt-anim-runner"
HERE = Path(__file__).parent


def kaggle(*args, check=True):
    print("+ kaggle", " ".join(args), flush=True)
    r = subprocess.run(["kaggle", *args], capture_output=True, text=True)
    print((r.stdout + r.stderr).strip()[-1500:], flush=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"kaggle {args[0]} {args[1]} failed")
    return r


def upload_inputs(jobs_file):
    jobs = json.loads(Path(jobs_file).read_text())
    ds = Path("kaggle_ds")
    shutil.rmtree(ds, ignore_errors=True)
    ds.mkdir()
    for job in jobs["clips"]:
        name = f"{job['id']}{Path(job['image']).suffix}"
        shutil.copy(job["image"], ds / name)
        job["image"] = name
    (ds / "jobs.json").write_text(json.dumps(jobs, indent=1))
    (ds / "dataset-metadata.json").write_text(json.dumps(
        {"title": "yt-anim-input", "id": DATASET, "licenses": [{"name": "CC0-1.0"}]}))
    exists = kaggle("datasets", "status", DATASET, check=False).returncode == 0
    if exists:
        kaggle("datasets", "version", "-p", str(ds), "-m", f"run {int(time.time())}", "-r", "zip")
    else:
        kaggle("datasets", "create", "-p", str(ds), "-r", "zip")
    for _ in range(60):  # wait until Kaggle has processed the new version
        if "ready" in kaggle("datasets", "status", DATASET, check=False).stdout:
            return
        time.sleep(10)
    raise RuntimeError("dataset never became ready")


def push_kernel():
    kd = Path("kaggle_kernel")
    shutil.rmtree(kd, ignore_errors=True)
    kd.mkdir()
    shutil.copy(HERE / "anim_kernel.py", kd / "anim_kernel.py")
    (kd / "kernel-metadata.json").write_text(json.dumps({
        "id": KERNEL, "title": "yt-anim-runner", "code_file": "anim_kernel.py",
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": True, "enable_internet": True, "dataset_sources": [DATASET],
        "competition_sources": [], "kernel_sources": [], "model_sources": []}))
    kaggle("kernels", "push", "-p", str(kd))


def wait_and_fetch(out_dir, max_hours=5.5):
    deadline = time.time() + max_hours * 3600
    time.sleep(60)
    while time.time() < deadline:
        status = kaggle("kernels", "status", KERNEL, check=False).stdout.lower()
        if "complete" in status or "error" in status or "cancel" in status:
            break
        time.sleep(90)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    kaggle("kernels", "output", KERNEL, "-p", out_dir, check=False)
    report = Path(out_dir) / "report.json"
    print(report.read_text() if report.exists() else "no report.json (see Kaggle log above)")
    if "error" in status:
        sys.exit(1)


if __name__ == "__main__":
    upload_inputs(sys.argv[1])
    push_kernel()
    wait_and_fetch(sys.argv[2])
