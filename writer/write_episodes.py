"""Write new Shorts scripts into a channel's queue with a free LLM API (no Claude needed).

    python writer/write_episodes.py horror --count 7
    python writer/write_episodes.py kids --count 7

Uses GEMINI_API_KEY (Google AI Studio free tier) or, if absent/failing, GROQ_API_KEY.
The model only writes the story parts (titles, hook, scenes, lines, lyrics). Characters, art
style and voices are fixed here, so every episode keeps the same look. Every result is
validated; anything that breaks a rule is dropped, never queued.
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parent.parent
CTX = Path(__file__).resolve().parent / "context"

SERIES = {
    "horror": {
        "style": ("dark 2D illustrated horror cartoon, clean bold outlines, muted teal and grey "
                  "palette with sickly fluorescent green light, heavy shadows, cinematic vertical "
                  "composition, no text, no letters, no logos, no gore"),
        "characters": {
            "Daniel": {"look": "a 19 year old skinny young man with messy dark brown hair, tired "
                               "eyes, wearing a blue store employee vest over a grey hoodie and a name tag"},
            "Clerk": {"look": "the Night Clerk, an extremely tall thin faceless shadow figure in a "
                              "dark store vest, two small glowing white eyes, long arms hanging past its knees"},
            "Mia": {"look": "a young woman in her twenties with a dark ponytail, wearing a yellow "
                            "rain jacket"},
        },
        "voices": {"Narrator": "am_michael", "Intercom": "am_echo", "Stranger": "am_onyx",
                   "Mia": "af_bella", "Daniel": "am_puck"},
        "words": (40, 85), "scenes": (5, 7),
    },
    "kids": {
        "style": ("cute 3D animated cartoon render for toddlers, soft studio lighting, bright cheerful "
                  "pastel colors, rounded friendly shapes, Pixar quality, vertical composition, "
                  "no text, no letters, no logos"),
        "characters": {
            "Wigglo": {"look": "Wigglo, an adorable round lime-green baby caterpillar with big "
                               "sparkly brown eyes, rosy cheeks, a happy smile, tiny antennae with "
                               "yellow pom-poms and a small orange beanie hat"},
            "Bunny": {"look": "a fluffy pink baby bunny with long ears, big shiny eyes and a tiny white tail"},
            "Birdie": {"look": "a round little sky-blue baby bird with a yellow beak and soft fluffy feathers"},
        },
        "words": (20, 70), "scenes": (4, 6),
    },
}

BANNED = re.compile(r"\b(true story|based on real|cocomelon|jj|walmart|disney|pixar|gore|blood|"
                    r"kill|suicide|subscribe now)\b", re.I)


def existing_titles(channel):
    titles = []
    for f in (ROOT / "episodes" / channel).rglob("*.yaml"):
        try:
            titles.append(yaml.safe_load(f.read_text(encoding="utf-8"))["title"])
        except Exception:
            pass
    return titles


def prompt_for(channel, count, avoid):
    s = SERIES[channel]
    voice = (CTX / f"{channel}-voice.md").read_text(encoding="utf-8")
    research = (CTX / "horror-research.md").read_text(encoding="utf-8") if channel == "horror" else ""
    example = (ROOT / "episodes" / channel / "queue").glob("*.yaml")
    sample = next(iter(sorted(example)), None) or next((ROOT / "episodes" / channel).rglob("short*.yaml"), None)
    sample_txt = sample.read_text(encoding="utf-8") if sample else ""
    chars = ", ".join(s["characters"])
    lo, hi = s["words"]
    smin, smax = s["scenes"]
    rules = {
        "horror": (
            "Write YouTube Shorts horror micro-stories for a US 16-30 audience. Hook in the very "
            "first line, a twist in the last line, and the last line should loop back to the first. "
            "Original stories only, set in everyday US places (stores, parking lots, night shifts, "
            "drive-thrus, motels, elevators, gas stations, babysitting, apartments). Never claim a "
            "story is true. No gore, no real crimes, nothing for kids. Allowed markers at the start "
            "of a line: [SCARE], [SFX:knock], [SFX:whisper]. Dialogue lines are objects like "
            '{"Intercom": "text"}; allowed speakers: Narrator, Intercom, Stranger, Mia, Daniel.'),
        "kids": (
            "Write YouTube Shorts kids songs for ages 1-5. Each teaches ONE thing (counting, "
            "colors, shapes, animals and their sounds, body parts, manners, brushing teeth, "
            "bedtime, washing hands). Original lyrics only, short simple words, lots of "
            "repetition, happy and safe, no villains, no brands, no calls to action. The song has "
            "a [Verse] and a [Chorus], 6-10 lines total, about 30 seconds. Every lyric line must "
            "appear, in order, in the scenes' lines."),
    }[channel]
    schema = {
        "title": "short internal title",
        "hook": "on-screen hook text, max 38 characters, uppercase",
        "youtube": {"title": "YouTube title, max 90 chars, ends with ' #shorts'",
                    "description": "1-2 sentences", "tags": ["6-8 search tags"]},
        **({"song": {"caption": "music style for the song generator",
                     "lyrics": "[Verse]\\n...\\n\\n[Chorus]\\n..."}} if channel == "kids" else {}),
        "scenes": [{"image": "one vivid visual description of a still frame",
                    "cast": [f"subset of: {chars} (or empty)"],
                    "hero": "the motion that happens in this 3-second shot",
                    "weather": "fog | rain | none",
                    "lines": ["spoken line(s) or lyric line(s), 1-2 per scene"]}],
    }
    return f"""{rules}

CHANNEL VOICE (follow it):
{voice}

{('WHAT WORKS IN THIS NICHE:' + chr(10) + research[:3000]) if research else ''}

RECURRING CHARACTERS (use their names in "cast" when they appear; describe their actions, not their looks): {chars}

CONSTRAINTS: {smin}-{smax} scenes; {lo}-{hi} total words across all lines; every scene has a
"hero" motion; titles must be new and different from these existing ones: {json.dumps(avoid[-40:])}

EXAMPLE EPISODE (YAML, for format only; do not copy its story):
{sample_txt[:2500]}

Return ONLY JSON: {{"episodes": [ {count} objects shaped like {json.dumps(schema)} ]}}"""


def gemini_model(key):
    """Newest text 'flash' model this key can call. Google retires model names over time, so
    pick from the live list instead of hardcoding one (GEMINI_MODEL overrides)."""
    if os.environ.get("GEMINI_MODEL"):
        return os.environ["GEMINI_MODEL"]
    r = requests.get("https://generativelanguage.googleapis.com/v1beta/models",
                     params={"key": key, "pageSize": 200}, timeout=60)
    r.raise_for_status()
    names = [m["name"].split("/", 1)[1] for m in r.json().get("models", [])
             if "generateContent" in m.get("supportedGenerationMethods", [])]
    skip = ("image", "tts", "audio", "live", "embedding", "vision", "thinking-exp", "lite")
    flash = [n for n in names if "flash" in n and not any(s in n for s in skip)]
    pool = flash or [n for n in names if "gemini" in n and not any(s in n for s in skip)]
    if not pool:
        raise RuntimeError(f"no usable Gemini model in {names[:20]}")

    def version(n):
        m = re.search(r"gemini-(\d+(?:\.\d+)?)", n)
        return (float(m.group(1)) if m else 0, "preview" not in n and "exp" not in n, "latest" in n)
    best = max(pool, key=version)
    print("gemini model:", best, flush=True)
    return best


def ask_gemini(prompt):
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("no GEMINI_API_KEY")
    model = gemini_model(key)
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        params={"key": key}, timeout=180,
        json={"contents": [{"parts": [{"text": prompt}]}],
              "generationConfig": {"responseMimeType": "application/json", "temperature": 1.0}})
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]


def ask_groq(prompt):
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("no GROQ_API_KEY")
    r = requests.post("https://api.groq.com/openai/v1/chat/completions", timeout=180,
                      headers={"Authorization": f"Bearer {key}"},
                      json={"model": os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
                            "messages": [{"role": "user", "content": prompt}],
                            "response_format": {"type": "json_object"}, "temperature": 1.0})
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def line_text(line):
    return line if isinstance(line, str) else str(next(iter(line.values())))


def validate(channel, ep, avoid):
    """Return a list of problems; empty means the episode can be queued."""
    s, errs = SERIES[channel], []
    scenes = ep.get("scenes") or []
    if not (s["scenes"][0] <= len(scenes) <= s["scenes"][1]):
        errs.append(f"scene count {len(scenes)}")
    words = sum(len(re.sub(r"\[.*?\]", "", line_text(l)).split()) for sc in scenes for l in sc.get("lines", []))
    if not (s["words"][0] <= words <= s["words"][1]):
        errs.append(f"word count {words}")
    for sc in scenes:
        if not (sc.get("image") and sc.get("hero") and sc.get("lines")):
            errs.append("scene missing image/hero/lines")
        bad = set(sc.get("cast") or []) - set(s["characters"])
        if bad:
            errs.append(f"unknown cast {bad}")
        for l in sc.get("lines", []):
            if isinstance(l, dict) and next(iter(l)) not in s.get("voices", {}):
                errs.append(f"unknown speaker {next(iter(l))}")
    yt = ep.get("youtube") or {}
    if not yt.get("title") or len(yt["title"]) > 100:
        errs.append("youtube title")
    if len(ep.get("hook", "")) > 40 or not ep.get("hook"):
        errs.append("hook length")
    if ep.get("title") in avoid:
        errs.append("duplicate title")
    blob = json.dumps(ep)
    if BANNED.search(blob):
        errs.append(f"banned word: {BANNED.search(blob).group(0)}")
    if channel == "kids":
        lyrics = (ep.get("song") or {}).get("lyrics", "")
        norm = lambda x: re.sub(r"[^a-z0-9 ]", "", x.lower()).split()  # noqa: E731
        lyric_lines = [norm(x) for x in lyrics.splitlines() if x.strip() and not x.strip().startswith("[")]
        scene_lines = [norm(line_text(l)) for sc in scenes for l in sc.get("lines", [])]
        if not (6 <= len(lyric_lines) <= 10) or lyric_lines != scene_lines:
            errs.append("lyrics must match scene lines, 6-10 lines")
    return errs


def to_yaml(channel, ep):
    s = SERIES[channel]
    out = {"title": ep["title"], "channel": channel, "format": "short", "hook": ep["hook"].upper(),
           "youtube": ep["youtube"], "style": s["style"],
           "characters": {k: v for k, v in s["characters"].items()
                          if any(k in (sc.get("cast") or []) for sc in ep["scenes"])}}
    if channel == "horror":
        out["speed"] = 1.0
        out["voices"] = s["voices"]
    else:
        out["song"] = {"caption": ep["song"]["caption"], "lyrics": ep["song"]["lyrics"],
                       "duration": 30, "language": "en"}
    scenes = []
    for sc in ep["scenes"]:
        item = {"image": sc["image"], "hero": sc["hero"], "lines": sc["lines"]}
        if sc.get("cast"):
            item["cast"] = sc["cast"]
        if sc.get("weather") in ("fog", "rain"):
            item["weather"] = sc["weather"]
        scenes.append(item)
    out["stories"] = [{"title": ep["title"], "scenes": scenes}]
    return yaml.safe_dump(out, sort_keys=False, allow_unicode=True, width=100)


def next_number(channel):
    nums = [int(m.group(1)) for f in (ROOT / "episodes" / channel).rglob("short*.yaml")
            if (m := re.match(r"short(\d+)", f.stem))]
    return max(nums, default=0) + 1


def main(channel, count):
    avoid = existing_titles(channel)
    prompt = prompt_for(channel, count, avoid)
    raw = None
    for ask in (ask_gemini, ask_groq):
        for attempt in range(2):
            try:
                raw = ask(prompt)
                break
            except Exception as e:
                print(f"{ask.__name__} attempt {attempt + 1} failed: {repr(e)[:300]}", flush=True)
                time.sleep(10)
        if raw:
            break
    if not raw:
        sys.exit("no LLM answered")
    episodes = json.loads(raw).get("episodes", [])
    queue = ROOT / "episodes" / channel / "queue"
    queue.mkdir(parents=True, exist_ok=True)
    n, written = next_number(channel), 0
    for ep in episodes:
        errs = validate(channel, ep, avoid)
        if errs:
            print(f"dropped '{ep.get('title')}': {errs}", flush=True)
            continue
        slug = re.sub(r"[^a-z0-9]+", "-", ep["title"].lower()).strip("-")[:30]
        path = queue / f"short{n:03d}-{slug}.yaml"
        path.write_text(to_yaml(channel, ep), encoding="utf-8")
        avoid.append(ep["title"])
        print("queued", path.relative_to(ROOT), flush=True)
        n, written = n + 1, written + 1
    print(f"{written}/{len(episodes)} episodes queued for {channel}")
    if written == 0:
        sys.exit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("channel", choices=["horror", "kids"])
    ap.add_argument("--count", type=int, default=7)
    a = ap.parse_args()
    main(a.channel, a.count)
