"""yt-factory v2: multi-story animated horror videos on free cloud CPUs.

    python factory.py story    episodes/horror/x.yaml --index 0   # render one story
    python factory.py assemble episodes/horror/x.yaml             # join stories, thumbnail

Per story: images (Cloudflare FLUX or placeholder) -> depth (Depth Anything V2 Small)
-> fg/bg layers -> narration + character voices (Kokoro) -> ambience/SFX (numpy)
-> frames (parallax shots, scares, fog/rain, flicker, grain) piped to ffmpeg with word captions.
Env (optional): CF_ACCOUNT_ID, CF_API_TOKEN.
"""
import argparse
import base64
import json
import os
import re
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import requests
import soundfile as sf
import yaml
from PIL import Image, ImageDraw, ImageEnhance, ImageFont

SR = 24000
FPS = 30
SHOT_LEN = 3.6
LINE_GAP = 0.3
SCENE_GAP = 0.5
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
CF_MODEL = "@cf/black-forest-labs/flux-1-schnell"
# (render size, output size); rendering below output size is ~2x faster and the 1024px
# source images hold no extra detail anyway. ffmpeg upscales and draws crisp captions.
SIZES = {"long": ((1280, 720), (1920, 1080)), "short": ((720, 1280), (1080, 1920))}
MARKER = re.compile(r"\[(SCARE|SFX:(\w+))\]")


def run(cmd):
    print("+", " ".join(str(c) for c in cmd)[:240], flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def load(ep_path):
    ep = yaml.safe_load(Path(ep_path).read_text(encoding="utf-8"))
    ep["slug"] = Path(ep_path).stem
    ep["out"] = Path("out") / ep["slug"]
    ep["out"].mkdir(parents=True, exist_ok=True)
    return ep


# ---------------------------------------------------------------- images

def cf_image(prompt, path):
    acct, token = os.environ["CF_ACCOUNT_ID"], os.environ["CF_API_TOKEN"]
    url = f"https://api.cloudflare.com/client/v4/accounts/{acct}/ai/run/{CF_MODEL}"
    for attempt in range(5):
        r = requests.post(url, headers={"Authorization": f"Bearer {token}"},
                          json={"prompt": prompt, "steps": 4}, timeout=120)
        if r.ok and r.json().get("success"):
            path.write_bytes(base64.b64decode(r.json()["result"]["image"]))
            return
        print(f"image attempt {attempt + 1}: {r.status_code} {r.text[:200]}", flush=True)
        time.sleep(6 * (attempt + 1))
    raise RuntimeError(f"Cloudflare image failed: {prompt[:80]}")


def placeholder_image(prompt, label, path):
    rng = np.random.default_rng(abs(hash(prompt)) % 2**32)
    w = h = 1024
    grad = np.linspace(0, 1, h)[:, None, None] * np.array([30, 36, 50]) + np.array([8, 10, 18])
    img = Image.fromarray(np.broadcast_to(grad, (h, w, 3)).astype("uint8"))
    d = ImageDraw.Draw(img)
    for _ in range(6):  # background "buildings" so parallax is visible
        x = int(rng.integers(0, w - 150))
        d.rectangle([x, int(rng.integers(350, 600)), x + int(rng.integers(80, 200)), h], fill=(18, 20, 30))
    cx = int(rng.integers(300, 700))
    d.ellipse([cx - 60, 520, cx + 60, 640], fill=(4, 4, 6))       # foreground figure
    d.rectangle([cx - 80, 630, cx + 80, h], fill=(4, 4, 6))
    font = ImageFont.truetype(FONT, 30)
    d.text((30, 30), f"{label} (placeholder)", font=font, fill=(200, 60, 60))
    d.text((30, 70), prompt[:60], font=font, fill=(170, 170, 180))
    img.save(path)


def story_images(ep, si):
    use_cf = bool(os.environ.get("CF_ACCOUNT_ID") and os.environ.get("CF_API_TOKEN"))
    print(f"images: {'cloudflare' if use_cf else 'placeholder'}", flush=True)
    paths = []
    for k, scene in enumerate(ep["stories"][si]["scenes"]):
        p = ep["out"] / f"img_s{si}_{k:02d}.png"
        if not p.exists():
            prompt = f"{scene['image']}, {ep.get('style', '')}".strip(", ")
            if use_cf:
                cf_image(prompt, p)
            else:
                placeholder_image(scene["image"], f"S{si + 1} scene {k + 1}", p)
        paths.append(p)
    return paths


# ---------------------------------------------------------------- voice

class Voices:
    def __init__(self, ep):
        from kokoro import KPipeline

        self._KPipeline = KPipeline
        self.pipes = {}
        self.cast = {"Narrator": "am_michael", **(ep.get("voices") or {})}
        self.speed = ep.get("speed", 0.92)

    def say(self, text, speaker):
        voice = self.cast.get(speaker, self.cast["Narrator"])
        lang = voice[0]
        if lang not in self.pipes:
            self.pipes[lang] = self._KPipeline(lang_code=lang)
        parts, words, offset = [], [], 0.0
        for r in self.pipes[lang](text, voice=voice, speed=self.speed):
            audio = r.audio
            if audio is None:
                continue
            audio = audio.numpy() if hasattr(audio, "numpy") else np.asarray(audio)
            chunk_words = []
            for t in r.tokens or []:
                if t.start_ts is None or t.end_ts is None:
                    continue
                if re.search(r"\w", t.text):
                    chunk_words.append([t.text, offset + t.start_ts, offset + t.end_ts])
                elif chunk_words:
                    chunk_words[-1][0] += t.text
            if not chunk_words:  # no timestamps: spread words by length
                ws, dur = r.graphemes.split(), len(audio) / SR
                total, cur = sum(len(w) for w in ws) or 1, offset
                for w in ws:
                    seg = dur * len(w) / total
                    chunk_words.append([w, cur, cur + seg])
                    cur += seg
            words += chunk_words
            parts.append(audio)
            offset += len(audio) / SR
        wav = np.concatenate(parts) if parts else np.zeros(int(0.2 * SR), dtype=np.float32)
        return wav.astype(np.float32), words


def parse_line(line):
    speaker, text = ("Narrator", line) if isinstance(line, str) else next(iter(line.items()))
    scare = sfx = None
    for m in MARKER.finditer(text):
        if m.group(1) == "SCARE":
            scare = True
        else:
            sfx = m.group(2)
    return speaker, " ".join(MARKER.sub("", text).split()), bool(scare), sfx


def story_timeline(ep, si, voices):
    """Speak every line; return narration audio, caption words, scene spans, events."""
    story = ep["stories"][si]
    chunks, words, scenes, events, t = [], [], [], [], 0.0

    def add(wav, ws, gap):
        nonlocal t
        start = t
        words.extend([w, start + a, start + b] for w, a, b in ws)
        chunks.append(wav)
        chunks.append(np.zeros(int(gap * SR), dtype=np.float32))
        t += len(wav) / SR + gap
        return start

    title_wav, _ = voices.say(story["title"] + ".", "Narrator")
    add(title_wav, [], 1.0)
    scenes.append({"start": 0.0, "end": t, "title": story["title"]})

    for scene in story["scenes"]:
        s0 = t
        for line in scene["lines"]:
            speaker, text, scare, sfx = parse_line(line)
            wav, ws = voices.say(text, speaker)
            start = add(wav, ws, LINE_GAP)
            if scare:
                events.append(("scare", start))
            if sfx:
                events.append((sfx, start))
        t += SCENE_GAP
        chunks.append(np.zeros(int(SCENE_GAP * SR), dtype=np.float32))
        scenes.append({"start": s0, "end": t, "scene": scene})
    return np.concatenate(chunks), words, scenes, events, t


# ---------------------------------------------------------------- sound design

def lowpass(x, taps):
    """Moving average via cumulative sum, O(n) for any window length."""
    c = np.cumsum(np.concatenate([[0.0], x]))
    half = taps // 2
    idx = np.arange(len(x))
    lo, hi = np.clip(idx - half, 0, len(x)), np.clip(idx + half + 1, 0, len(x))
    return ((c[hi] - c[lo]) / (hi - lo)).astype(np.float32)


def ambience(dur, rng):
    n = int(dur * SR)
    brown = np.cumsum(rng.normal(0, 1, n)).astype(np.float32)
    brown -= lowpass(brown, 4801)                 # remove drift
    brown = lowpass(brown, 60)
    brown /= np.abs(brown).max() + 1e-6
    t = np.arange(n) / SR
    drone = (0.5 * np.sin(2 * np.pi * 55 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.07 * t))
             + 0.3 * np.sin(2 * np.pi * 82.4 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.11 * t + 1)))
    amb = 0.55 * brown + 0.45 * drone
    fade = min(n, 2 * SR)
    amb[:fade] *= np.linspace(0, 1, fade)
    return (amb * 0.09).astype(np.float32)


def sfx_hit(rng):
    n = int(1.2 * SR)
    t = np.arange(n) / SR
    sweep = np.sin(2 * np.pi * (70 * t - 15 * t * t)) * np.exp(-t * 3.2)
    noise = lowpass(rng.normal(0, 1, n), 25) * np.exp(-t * 9)
    return (0.55 * sweep + 0.35 * noise / (np.abs(noise).max() + 1e-6)).astype(np.float32)


def sfx_knock(rng):
    one = int(0.09 * SR)
    t = np.arange(one) / SR
    knock = (lowpass(rng.normal(0, 1, one), 8) * np.exp(-t * 60) * 0.6
             + np.sin(2 * np.pi * 170 * t) * np.exp(-t * 45))
    out = np.zeros(int(1.0 * SR), dtype=np.float32)
    for i in range(3):
        s = int(i * 0.33 * SR)
        out[s:s + one] += knock
    return out / (np.abs(out).max() + 1e-6) * 0.7


def sfx_whisper(rng):
    n = int(1.4 * SR)
    t = np.arange(n) / SR
    hiss = rng.normal(0, 1, n) - lowpass(rng.normal(0, 1, n), 4)
    env = np.sin(np.pi * t / t[-1]) ** 2 * (0.6 + 0.4 * np.sin(2 * np.pi * 7 * t))
    return (hiss * env * 0.18).astype(np.float32)


SFX = {"scare": sfx_hit, "knock": sfx_knock, "whisper": sfx_whisper}


def mix_audio(narr, dur, events, path, seed):
    rng = np.random.default_rng(seed)
    n = int(dur * SR) + SR
    mix = np.zeros(n, dtype=np.float32)
    mix[:len(narr)] += narr * 1.0
    mix[:n] += ambience(n / SR, rng)[:n]
    for kind, at in events:
        fx = SFX.get(kind, sfx_hit)(rng)
        s = int(max(0, at - (0.05 if kind == "scare" else 0)) * SR)
        e = min(n, s + len(fx))
        mix[s:e] += fx[:e - s] * 0.8
    mix /= max(1.0, np.abs(mix).max() / 0.95)
    sf.write(path, mix[:int(dur * SR)], SR)


# ---------------------------------------------------------------- layers & frames

_depth = None


def depth_of(img):
    global _depth
    if _depth is None:
        from transformers import pipeline
        _depth = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf",
                          device="cpu")
    d = np.asarray(_depth(Image.fromarray(img))["depth"], dtype=np.float32)
    d = cv2.resize(d, (img.shape[1], img.shape[0]))
    return (d - d.min()) / (d.max() - d.min() + 1e-6)          # 1 = nearest


def grade(img):
    f = img.astype(np.float32)
    gray = f.mean(axis=2, keepdims=True)
    f = gray + (f - gray) * 0.7                                 # desaturate
    f = (f - 128) * 1.12 + 120                                  # contrast, slightly darker
    f *= np.array([0.92, 0.97, 1.08], dtype=np.float32)         # cold tint
    return np.clip(f, 0, 255).astype(np.uint8)


class Layers:
    def __init__(self, path):
        img = grade(cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB))
        h, w = img.shape[:2]
        d = depth_of(img)
        thr = np.quantile(d, 0.72)
        a = cv2.GaussianBlur(np.clip((d - thr) / 0.08, 0, 1), (0, 0), 3).astype(np.float32)
        mask = cv2.dilate((a > 0.05).astype(np.uint8) * 255, np.ones((21, 21), np.uint8))
        small = cv2.inpaint(cv2.resize(img, (w // 2, h // 2)), cv2.resize(mask, (w // 2, h // 2)),
                            5, cv2.INPAINT_TELEA)
        bg = np.where(mask[..., None] > 0, cv2.resize(small, (w, h)), img)
        self.fg, self.bg, self.alpha, self.w, self.h = img, bg, a, w, h
        ys, xs = np.nonzero(a > 0.5)
        self.focus = (xs.mean() / w, ys.mean() / h) if len(xs) > 2000 else (0.5, 0.45)


class Effects:
    def __init__(self, size, seed):
        w, h = size
        rng = np.random.default_rng(seed)
        self.rng = rng
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r = ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2
        self.vignette = np.clip(1 - 0.45 * r, 0.25, 1)[..., None].astype(np.float32)
        self.grain = [rng.normal(0, 5, (h, w, 1)).astype(np.float32) for _ in range(6)]
        fog = cv2.resize(rng.random((h // 16, w // 8)).astype(np.float32), (2 * w, h),
                         interpolation=cv2.INTER_CUBIC)
        fog = cv2.GaussianBlur(fog, (0, 0), 25)
        self.fog = ((fog - fog.min()) / (fog.max() - fog.min() + 1e-6))[..., None]
        rain = np.zeros((2 * h, w), np.float32)
        for _ in range(int(w * h / 1800)):
            x, y = int(rng.integers(0, w)), int(rng.integers(0, 2 * h))
            cv2.line(rain, (x, y), (x - 6, y + 34), float(rng.uniform(0.3, 1)), 1)
        self.rain = cv2.GaussianBlur(rain, (0, 0), 0.8)[..., None]
        self.w, self.h = w, h


def ease(p):
    return p * p * (3 - 2 * p)


def shot_plan(kind, focus, flip):
    """(start, end) of (cx, cy, scale, parallax) in normalized crop units."""
    fx, fy = focus
    if kind == "wide":
        return (0.5, 0.5, 1.0, -0.025), (0.5, 0.47, 0.88, 0.025)
    if kind == "close":
        return (fx, fy, 0.62, 0.0), (fx, fy, 0.52, 0.015 * (1 if flip else -1))
    a, b = (0.38, 0.62) if flip else (0.62, 0.38)
    return (a, 0.5, 0.8, -0.03), (b, 0.5, 0.8, 0.03)


def warp(layer, cx, cy, s, cw_max, ch_max, size):
    w, h = size
    cw, ch = cw_max * s, ch_max * s
    sx = w / cw
    x0 = cx * layer.shape[1] - cw / 2
    y0 = cy * layer.shape[0] - ch / 2
    m = np.float32([[sx, 0, -x0 * sx], [0, sx, -y0 * sx]])
    return cv2.warpAffine(layer, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def render_scene(proc, lay, scene, f0, f1, t0, scares, fx, size, rng, idx):
    w, h = size
    aspect = w / h
    cw_max = min(lay.w, lay.h * aspect)
    ch_max = cw_max / aspect
    n = f1 - f0
    shots = max(1, round(n / FPS / SHOT_LEN))
    kinds = ["wide", "close", "pan"]
    weather = scene.get("weather", "none")
    for i in range(n):
        t = t0 + i / FPS
        si = min(shots - 1, i * shots // n)
        p = ease((i - si * n / shots) / (n / shots))
        a, b = shot_plan(kinds[(idx + si) % 3], lay.focus, (idx + si) % 2 == 0)
        cx, cy, s, par = (x + (y - x) * p for x, y in zip(a, b))
        for at in scares:                                      # punch-in + shake
            dt = t - at
            if 0 <= dt < 0.8:
                s *= 1 - 0.14 * min(dt / 0.12, 1)
                amp = 0.012 * (1 - dt / 0.8)
                cx += rng.uniform(-amp, amp)
                cy += rng.uniform(-amp, amp)
        half_x = cw_max * s / 2 / lay.w
        half_y = ch_max * s / 2 / lay.h
        cx = min(max(cx, half_x), 1 - half_x)
        cy = min(max(cy, half_y), 1 - half_y)
        bg = warp(lay.bg, cx - par * 0.4, cy, s, cw_max, ch_max, size).astype(np.float32)
        fg = warp(lay.fg, cx + par * 0.6, cy, s, cw_max, ch_max, size).astype(np.float32)
        al = warp(lay.alpha, cx + par * 0.6, cy, s, cw_max, ch_max, size)[..., None]
        frame = bg + (fg - bg) * al
        if weather == "fog":
            off = int(t * 25) % w
            f = fx.fog[:, off:off + w]
            frame = frame * (1 - 0.22 * f) + 190 * 0.22 * f
        elif weather == "rain":
            off = (fx.h - int(t * 1100) % fx.h)
            frame = frame * 0.85 + fx.rain[off:off + h] * 140
        if rng.random() < 0.015:
            frame *= 0.6                                       # light flicker
        frame = frame * fx.vignette + fx.grain[i % 6]
        for at in scares:
            if 0 <= t - at < 0.1:
                frame = frame * 0.55 + np.array([170, 0, 0], np.float32) * 0.45
        edge = min(i, n - 1 - i) / FPS
        if edge < 0.25:
            frame *= edge / 0.25
        proc.stdin.write(np.clip(frame, 0, 255).astype(np.uint8).tobytes())


def render_title(proc, f0, f1, fx):
    for i in range(f1 - f0):
        frame = np.full((fx.h, fx.w, 3), 6, np.float32) + fx.grain[i % 6]
        proc.stdin.write(np.clip(frame, 0, 255).astype(np.uint8).tobytes())


# ---------------------------------------------------------------- captions

def ass_time(t):
    return f"{int(t // 3600)}:{int(t % 3600 // 60):02d}:{t % 60:05.2f}"


def write_ass(words, titles, out_size, path, per_line):
    w, h = out_size
    vertical = h > w
    fs, title_fs = (82, 110) if vertical else (54, 96)
    margin = int(h * (0.28 if vertical else 0.07))
    lines = ["[Script Info]", "ScriptType: v4.00+", f"PlayResX: {w}", f"PlayResY: {h}", "",
             "[V4+ Styles]",
             "Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, Bold, "
             "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV",
             f"Style: Cap,DejaVu Sans,{fs},&H00FFFFFF,&H00000000,&H90000000,1,1,5,2,2,60,60,{margin}",
             f"Style: Title,DejaVu Sans,{title_fs},&H00E0E0E0,&H00000000,&H00000000,1,1,4,0,5,80,80,0",
             "", "[Events]", "Format: Layer, Start, End, Style, Text"]
    for start, end, text in titles:
        lines.append(f"Dialogue: 1,{ass_time(start)},{ass_time(end)},Title,"
                     f"{{\\fad(400,400)}}{text.upper()}")
    groups, cur = [], []
    for wd in words:
        if cur and (len(cur) >= per_line or wd[1] - cur[-1][2] > 0.6):
            groups.append(cur)
            cur = []
        cur.append(wd)
    if cur:
        groups.append(cur)
    for g in groups:
        for j, (word, a, b) in enumerate(g):
            end = g[j + 1][1] if j + 1 < len(g) else b + 0.15
            text = " ".join(("{\\c&H0000E6FF&}" + x[0] + "{\\c&H00FFFFFF&}") if k == j else x[0]
                            for k, x in enumerate(g))
            lines.append(f"Dialogue: 0,{ass_time(a)},{ass_time(end)},Cap,{text}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- commands

def cmd_story(ep, si):
    fmt = ep.get("format", "long")
    size, out_size = SIZES[fmt]
    out = ep["out"]
    images = story_images(ep, si)
    narr, words, scenes, events, dur = story_timeline(ep, si, Voices(ep))
    audio = out / f"audio_s{si}.wav"
    mix_audio(narr, dur, events, audio, seed=si)
    titles = [(0.2, scenes[0]["end"] - 0.2, ep["stories"][si]["title"])]
    ass = out / f"subs_s{si}.ass"
    write_ass(words, titles, out_size, ass, per_line=3 if fmt == "short" else 6)

    total = round(dur * FPS)
    final = out / f"story_{si}.mp4"
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{size[0]}x{size[1]}", "-r", str(FPS), "-i", "-", "-i", str(audio),
         "-vf", f"scale={out_size[0]}:{out_size[1]}:flags=lanczos,subtitles={ass.as_posix()}",
         "-af", "aecho=0.8:0.6:45:0.12", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
         "-maxrate", "6M", "-bufsize", "12M", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k",
         "-ar", "48000", "-ac", "2", "-frames:v", str(total), "-movflags", "+faststart", str(final)],
        stdin=subprocess.PIPE)
    fx = Effects(size, seed=si)
    rng = np.random.default_rng(100 + si)
    scares = [at for kind, at in events if kind == "scare"]
    for k, sc in enumerate(scenes):
        f0, f1 = round(sc["start"] * FPS), round(sc["end"] * FPS)
        if "title" in sc:
            render_title(proc, f0, f1, fx)
        else:
            lay = Layers(images[k - 1])
            render_scene(proc, lay, sc["scene"], f0, f1, sc["start"], scares, fx, size, rng, k)
        print(f"story {si}: scene {k}/{len(scenes) - 1} done ({sc['end']:.0f}s)", flush=True)
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    print(json.dumps({"story": si, "duration_sec": round(dur, 1), "scenes": len(images)}))


def make_thumbnail(ep, out):
    th = ep.get("thumbnail", {})
    src = out / f"img_s{th.get('story', 1) - 1}_{th.get('scene', 1) - 1:02d}.png"
    img = Image.open(src).convert("RGB")
    tw, tht = 1280, 720
    scale = max(tw / img.width, tht / img.height)
    img = img.resize((int(img.width * scale) + 1, int(img.height * scale) + 1), Image.LANCZOS)
    left, top = (img.width - tw) // 2, (img.height - tht) // 2
    img = ImageEnhance.Contrast(img.crop((left, top, left + tw, top + tht))).enhance(1.25)
    img = ImageEnhance.Brightness(img).enhance(0.8)
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT, 110)
    words, rows, row = th.get("text", ep["title"]).upper().split(), [], ""
    for word in words:
        if row and d.textlength(row + " " + word, font=font) > tw * 0.62:
            rows.append(row)
            row = word
        else:
            row = (row + " " + word).strip()
    rows.append(row)
    y = 50
    for r in rows:
        d.text((50, y), r, font=font, fill=(255, 230, 0), stroke_width=9, stroke_fill=(0, 0, 0))
        y += 122
    d.rectangle([0, tht - 14, tw, tht], fill=(200, 0, 0))
    img.save(out / "thumbnail.jpg", quality=92)


def cmd_assemble(ep):
    out = ep["out"]
    parts = [out / f"story_{i}.mp4" for i in range(len(ep["stories"]))]
    (out / "parts.txt").write_text("".join(f"file '{p.name}'\n" for p in parts))
    final = out / f"{ep['slug']}.mp4"
    run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", out / "parts.txt",
         "-c", "copy", "-movflags", "+faststart", final])
    make_thumbnail(ep, out)
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                            "-of", "csv=p=0", str(final)], capture_output=True, text=True)
    dur = float(probe.stdout.strip() or 0)
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", final, "-vf",
         f"fps=1/{max(1, dur / 16):.2f},scale=480:-2,tile=4x4", "-frames:v", "1",
         out / "contact_sheet.png"])
    meta = {"title": ep["title"], "duration_min": round(dur / 60, 2), "stories": len(parts),
            "size_mb": round(final.stat().st_size / 1e6, 1), "file": final.name}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["story", "assemble", "count"])
    ap.add_argument("episode")
    ap.add_argument("--index", type=int, default=0)
    a = ap.parse_args()
    episode = load(a.episode)
    if a.command == "count":
        print(json.dumps(list(range(len(episode["stories"])))))
    elif a.command == "story":
        cmd_story(episode, a.index)
    else:
        cmd_assemble(episode)
