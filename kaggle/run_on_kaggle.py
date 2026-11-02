"""Run a script on a free Kaggle GPU and bring its /kaggle/working output back.

    python kaggle/run_on_kaggle.py jobs.json out_dir                  # animation clips
    python kaggle/run_on_kaggle.py --kernel kaggle/kids_test_kernel.py --inputs kids \
        --name kids-test out_dir                                     # any kernel + input folder

jobs.json: {"clips": [{"id": "...", "image": "path/to/img.png", "prompt": "...", ...}]}
Needs KAGGLE_USERNAME and KAGGLE_KEY in the environment (phone-verified Kaggle account,
so the kernel may use a GPU and the internet).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

USER = os.environ["KAGGLE_USERNAME"]
HERE = Path(__file__).parent


def kaggle(*args, check=True):
    print("+ kaggle", " ".join(args), flush=True)
    r = subprocess.run(["kaggle", *args], capture_output=True, text=True)
    print((r.stdout + r.stderr).strip()[-1500:], flush=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"kaggle {args[0]} {args[1]} failed")
    return r


def anim_inputs(jobs_file, ds):
    jobs = json.loads(Path(jobs_file).read_text())
    for job in jobs["clips"]:
        name = f"{job['id']}{Path(job['image']).suffix}"
        shutil.copy(job["image"], ds / name)
        job["image"] = name
    (ds / "jobs.json").write_text(json.dumps(jobs, indent=1))


def upload(name, fill):
    dataset = f"{USER}/yt-{name}-input"
    ds = Path("kaggle_ds")
    shutil.rmtree(ds, ignore_errors=True)
    ds.mkdir()
    fill(ds)
    (ds / "dataset-metadata.json").write_text(json.dumps(
        {"title": f"yt-{name}-input", "id": dataset, "licenses": [{"name": "CC0-1.0"}]}))
    exists = kaggle("datasets", "status", dataset, check=False).returncode == 0
    if exists:
        kaggle("datasets", "version", "-p", str(ds), "-m", f"run {int(time.time())}", "-r", "zip")
    else:
        kaggle("datasets", "create", "-p", str(ds), "-r", "zip")
    for _ in range(60):  # wait until Kaggle has processed the new version
        if "ready" in kaggle("datasets", "status", dataset, check=False).stdout:
            return dataset
        time.sleep(10)
    raise RuntimeError("dataset never became ready")


def push_kernel(name, script, dataset):
    kernel = f"{USER}/yt-{name}-runner"
    kd = Path("kaggle_kernel")
    shutil.rmtree(kd, ignore_errors=True)
    kd.mkdir()
    shutil.copy(script, kd / Path(script).name)
    (kd / "kernel-metadata.json").write_text(json.dumps({
        "id": kernel, "title": f"yt-{name}-runner", "code_file": Path(script).name,
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": True, "enable_internet": True, "dataset_sources": [dataset],
        "competition_sources": [], "kernel_sources": [], "model_sources": []}))
    kaggle("kernels", "push", "-p", str(kd))
    return kernel


def wait_and_fetch(kernel, out_dir, max_hours=5.5):
    deadline = time.time() + max_hours * 3600
    time.sleep(60)
    status = ""
    while time.time() < deadline:
        status = kaggle("kernels", "status", kernel, check=False).stdout.lower()
        if "complete" in status or "error" in status or "cancel" in status:
            break
        time.sleep(90)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    kaggle("kernels", "output", kernel, "-p", out_dir, check=False)
    report = Path(out_dir) / "report.json"
    print(report.read_text() if report.exists() else "no report.json")
    if "error" in status:
        show_log(out_dir)
        sys.exit(1)


def show_log(out_dir, lines=80):
    """Print the end of the kernel's own log, so a failed run shows its real error here."""
    for log in Path(out_dir).glob("*.log"):
        try:
            entries = json.loads(log.read_text(encoding="utf-8", errors="replace"))
            text = "".join(e.get("data", "") for e in entries)
        except ValueError:
            text = log.read_text(encoding="utf-8", errors="replace")
        print(f"--- end of {log.name} ---")
        print("
".join(text.splitlines()[-lines:]), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("args", nargs="+", help="[jobs.json] out_dir")
    ap.add_argument("--kernel")
    ap.add_argument("--inputs")
    ap.add_argument("--name", default="anim")
    ap.add_argument("--log-only", action="store_true", help="only fetch and show the kernel's log")
    a = ap.parse_args()
    if a.log_only:
        kaggle("kernels", "output", f"{USER}/yt-{a.name}-runner", "-p", a.args[-1], check=False)
        show_log(a.args[-1], lines=200)
        sys.exit(0)
    if a.kernel:
        out = a.args[-1]
        dataset = upload(a.name, lambda ds: shutil.copytree(a.inputs, ds, dirs_exist_ok=True))
        kernel = push_kernel(a.name, a.kernel, dataset)
    else:
        jobs, out = a.args
        dataset = upload(a.name, lambda ds: anim_inputs(jobs, ds))
        kernel = push_kernel(a.name, HERE / "anim_kernel.py", dataset)
    wait_and_fetch(kernel, out)
