"""Build one faceless video from an episode YAML.

Steps: images (Cloudflare FLUX or placeholder) -> narration (Kokoro) -> ambience (ffmpeg)
-> animated scenes (ffmpeg zoompan + grain/vignette/flicker) -> subtitles -> thumbnail.

Usage: python build.py episodes/horror/test-short-001.yaml
Env (optional): CF_ACCOUNT_ID, CF_API_TOKEN. Without them, placeholder images are used.
"""
import base64
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import requests
import soundfile as sf
import yaml
from PIL import Image, ImageDraw, ImageEnhance, ImageFont

SR = 24000
FPS = 30
SENTENCE_GAP = 0.25
SCENE_GAP = 0.6
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
CF_MODEL = "@cf/black-forest-labs/flux-1-schnell"


def run(cmd):
    print("+", " ".join(str(c) for c in cmd)[:300], flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


# ---------- images ----------

def cf_image(prompt, out_path):
    acct, token = os.environ["CF_ACCOUNT_ID"], os.environ["CF_API_TOKEN"]
    url = f"https://api.cloudflare.com/client/v4/accounts/{acct}/ai/run/{CF_MODEL}"
    for attempt in range(4):
        r = requests.post(url, headers={"Authorization": f"Bearer {token}"},
                          json={"prompt": prompt, "steps": 8}, timeout=120)
        if r.ok and r.json().get("success"):
            out_path.write_bytes(base64.b64decode(r.json()["result"]["image"]))
            return
        print(f"image attempt {attempt + 1} failed: {r.status_code} {r.text[:200]}", flush=True)
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Cloudflare image generation failed for: {prompt[:80]}")


def placeholder_image(prompt, idx, out_path):
    w, h = 1024, 1024
    top, bottom = np.array([10, 14, 24]), np.array([40, 48, 64])
    grad = np.linspace(0, 1, h)[:, None, None] * (bottom - top) + top
    img = Image.fromarray(np.broadcast_to(grad, (h, w, 3)).astype("uint8"))
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT, 34)
    d.text((40, 40), f"SCENE {idx + 1} (placeholder)", font=font, fill=(200, 60, 60))
    words, line, y = prompt.split(), "", 120
    for word in words:
        if len(line) + len(word) > 38:
            d.text((40, y), line, font=font, fill=(200, 200, 210))
            line, y = "", y + 46
        line += word + " "
    d.text((40, y), line, font=font, fill=(200, 200, 210))
    img.save(out_path)


def make_images(ep, out):
    use_cf = bool(os.environ.get("CF_ACCOUNT_ID") and os.environ.get("CF_API_TOKEN"))
    print(f"images: {'cloudflare' if use_cf else 'placeholder'}", flush=True)
    paths = []
    for i, scene in enumerate(ep["scenes"]):
        p = out / f"scene_{i:02d}.png"
        prompt = f"{scene['image']}, {ep.get('style', '')}".strip(", ")
        if use_cf:
            cf_image(prompt, p)
        else:
            placeholder_image(scene["image"], i, p)
        paths.append(p)
    return paths


# ---------- narration ----------

def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", " ".join(text.split())) if s.strip()]


def make_narration(ep, out):
    from kokoro import KPipeline

    pipe = KPipeline(lang_code="a")
    voice, speed = ep.get("voice", "am_michael"), ep.get("speed", 0.92)
    chunks, cues, scene_durs, t = [], [], [], 0.0
    for scene in ep["scenes"]:
        start = t
        for sentence in split_sentences(scene["narration"]):
            parts = []
            for _, _, audio in pipe(sentence, voice=voice, speed=speed):
                if audio is not None:
                    parts.append(audio.numpy() if hasattr(audio, "numpy") else np.asarray(audio))
            wav = np.concatenate(parts) if parts else np.zeros(1)
            dur = len(wav) / SR
            cues.append((t, t + dur, sentence))
            chunks += [wav, np.zeros(int(SENTENCE_GAP * SR))]
            t += dur + SENTENCE_GAP
        chunks.append(np.zeros(int(SCENE_GAP * SR)))
        t += SCENE_GAP
        scene_durs.append(t - start)
    path = out / "narration.wav"
    sf.write(path, np.concatenate(chunks).astype("float32"), SR)
    return path, cues, scene_durs


# ---------- audio mix ----------

def make_audio(narration, total, out):
    ambience = out / "ambience.wav"
    run(["ffmpeg", "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", f"anoisesrc=color=brown:amplitude=0.25:d={total}",
         "-f", "lavfi", "-i", f"sine=frequency=55:duration={total}",
         "-f", "lavfi", "-i", f"sine=frequency=82:duration={total}",
         "-filter_complex",
         "[0]lowpass=f=260,volume=0.5[n];[1]tremolo=f=0.15:d=0.6,volume=0.35[a];"
         "[2]tremolo=f=0.11:d=0.7,volume=0.2[b];[n][a][b]amix=inputs=3:normalize=0,"
         "afade=t=in:d=2,volume=0.55[o]",
         "-map", "[o]", "-ar", str(SR), ambience])
    mixed = out / "mix.wav"
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", narration, "-i", ambience,
         "-filter_complex",
         "[0]aecho=0.8:0.6:60:0.18,volume=1.4[v];[1]volume=0.35[bg];"
         "[v][bg]amix=inputs=2:normalize=0:duration=longest,alimiter=limit=0.95[o]",
         "-map", "[o]", mixed])
    return mixed


# ---------- video ----------

def make_clips(images, durs, size, out):
    w, h = size
    clips = []
    for i, (img, dur) in enumerate(zip(images, durs)):
        frames = max(1, round(dur * FPS))
        zoom = (f"1+0.12*on/{frames}" if i % 2 == 0 else f"1.12-0.12*on/{frames}")
        pan_x = "iw/2-(iw/zoom/2)" + ("+on*0.15" if i % 3 == 1 else "")
        vf = (f"scale={2 * w}:{2 * h}:force_original_aspect_ratio=increase,crop={2 * w}:{2 * h},"
              f"zoompan=z='{zoom}':x='{pan_x}':y='ih/2-(ih/zoom/2)':d={frames}:s={w}x{h}:fps={FPS},"
              "hue=b='-1.5*lt(random(0),0.025)',"
              "noise=alls=12:allf=t+u,vignette=PI/4,"
              f"fade=t=in:st=0:d=0.35,fade=t=out:st={max(0, dur - 0.35):.3f}:d=0.35,format=yuv420p")
        clip = out / f"clip_{i:02d}.mp4"
        run(["ffmpeg", "-y", "-loglevel", "error", "-i", img, "-vf", vf, "-frames:v", frames,
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-r", FPS, clip])
        clips.append(clip)
    listing = out / "clips.txt"
    listing.write_text("".join(f"file '{c.name}'\n" for c in clips))
    silent = out / "video_silent.mp4"
    run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", listing,
         "-c", "copy", silent])
    return silent


def ass_time(t):
    h, m = int(t // 3600), int(t % 3600 // 60)
    return f"{h}:{m:02d}:{t % 60:05.2f}"


def make_subs(cues, size, out, words_per_line):
    w, h = size
    vertical = h > w
    font_size = 78 if vertical else 58
    margin_v = int(h * (0.30 if vertical else 0.08))
    lines = [
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {w}", f"PlayResY: {h}", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, Bold, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV",
        f"Style: Default,DejaVu Sans,{font_size},&H00FFFFFF,&H00000000,&H80000000,1,1,5,2,2,"
        f"60,60,{margin_v}",
        "", "[Events]", "Format: Layer, Start, End, Style, Text",
    ]
    for start, end, text in cues:
        words = text.split()
        groups = [words[i:i + words_per_line] for i in range(0, len(words), words_per_line)]
        span, cursor = end - start, start
        for g in groups:
            seg = span * len(g) / len(words)
            lines.append(f"Dialogue: 0,{ass_time(cursor)},{ass_time(cursor + seg)},Default,"
                         + " ".join(g).replace("{", "(").replace("}", ")"))
            cursor += seg
    path = out / "subs.ass"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def make_thumbnail(ep, images, out):
    th = ep.get("thumbnail", {})
    img = Image.open(images[min(th.get("scene", 1) - 1, len(images) - 1)]).convert("RGB")
    tw, tht = 1280, 720
    scale = max(tw / img.width, tht / img.height)
    img = img.resize((int(img.width * scale) + 1, int(img.height * scale) + 1))
    left, top = (img.width - tw) // 2, (img.height - tht) // 2
    img = ImageEnhance.Brightness(img.crop((left, top, left + tw, top + tht))).enhance(0.7)
    d = ImageDraw.Draw(img)
    text = th.get("text", ep["title"]).upper()
    font = ImageFont.truetype(FONT, 96)
    words, rows, row = text.split(), [], ""
    for word in words:
        if row and d.textlength(row + " " + word, font=font) > tw - 120:
            rows.append(row)
            row = word
        else:
            row = (row + " " + word).strip()
    rows.append(row)
    y = tht - 60 - len(rows) * 110
    for r in rows:
        d.text((60, y), r, font=font, fill=(255, 230, 0), stroke_width=8, stroke_fill=(0, 0, 0))
        y += 110
    path = out / "thumbnail.jpg"
    img.save(path, quality=92)
    return path


def main(ep_path):
    ep = yaml.safe_load(Path(ep_path).read_text(encoding="utf-8"))
    slug = Path(ep_path).stem
    out = Path("out") / slug
    out.mkdir(parents=True, exist_ok=True)
    size = (1080, 1920) if ep.get("format", "long") == "short" else (1920, 1080)

    images = make_images(ep, out)
    narration, cues, durs = make_narration(ep, out)
    total = sum(durs)
    audio = make_audio(narration, total, out)
    silent = make_clips(images, durs, size, out)
    subs = make_subs(cues, size, out, words_per_line=4 if size[1] > size[0] else 8)

    final = out / f"{slug}.mp4"
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", silent, "-i", audio,
         "-vf", f"subtitles={subs.as_posix()}", "-c:v", "libx264", "-preset", "veryfast",
         "-crf", "20", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart",
         final])
    make_thumbnail(ep, images, out)
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", final, "-vf",
         f"fps=1/{max(1, total / 12):.2f},scale=360:-2,tile=4x3", "-frames:v", "1",
         out / "contact_sheet.png"])

    meta = {"title": ep["title"], "duration_sec": round(total, 1), "size": size,
            "scenes": len(images), "file": final.name}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    for f in out.glob("clip_*.mp4"):
        f.unlink()
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
