"""Find the biggest channels in a niche and the videos that beat their own channel's median.

Usage: python research/find_channels.py "animated horror stories" "scary stories animated" ...
Writes research_out/{channels.json, collected.json, report.md}. Public listings only, no login.
"""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import yt_dlp

SEARCH_PER_QUERY = 60
TOP_CHANNELS = 15
VIDEOS_PER_CHANNEL = 60
OUT = Path("research_out")


def ydl():
    return yt_dlp.YoutubeDL({"extract_flat": "in_playlist", "quiet": True, "no_warnings": True,
                             "ignoreerrors": True, "skip_download": True})


def search(queries):
    hits = defaultdict(lambda: {"hits": 0, "views": 0, "url": None, "name": None, "examples": []})
    with ydl() as y:
        for q in queries:
            info = y.extract_info(f"ytsearch{SEARCH_PER_QUERY}:{q}", download=False) or {}
            for e in info.get("entries") or []:
                if not e or not e.get("channel_id"):
                    continue
                c = hits[e["channel_id"]]
                c["hits"] += 1
                c["views"] += e.get("view_count") or 0
                c["url"] = e.get("channel_url") or f"https://www.youtube.com/channel/{e['channel_id']}"
                c["name"] = e.get("channel") or e.get("uploader")
                if len(c["examples"]) < 3:
                    c["examples"].append(e.get("title"))
    return hits


def channel_videos(url):
    with ydl() as y:
        y.params["playlistend"] = VIDEOS_PER_CHANNEL
        info = y.extract_info(url.rstrip("/") + "/videos", download=False) or {}
    vids = [e for e in info.get("entries") or [] if e]
    return info.get("channel_follower_count"), vids


def fmt(n):
    if n is None:
        return "?"
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= div:
            return f"{n / div:.1f}{unit}"
    return str(int(n))


def main(queries):
    OUT.mkdir(exist_ok=True)
    hits = search(queries)
    ranked = sorted(hits.values(), key=lambda c: (-c["views"]))[:TOP_CHANNELS]

    channels, collected = [], []
    for c in ranked:
        subs, vids = channel_videos(c["url"])
        views = [v.get("view_count") or 0 for v in vids]
        durs = [v.get("duration") or 0 for v in vids if v.get("duration")]
        if not views:
            continue
        row = {"name": c["name"], "url": c["url"], "subs": subs, "sampled": len(vids),
               "median_views": int(statistics.median(views)), "max_views": max(views),
               "avg_minutes": round(statistics.mean(durs) / 60, 1) if durs else None,
               "search_hits": c["hits"], "examples": c["examples"]}
        channels.append(row)
        for v in vids:
            collected.append({"channel": c["name"], "title": v.get("title"),
                              "views": v.get("view_count") or 0, "duration": v.get("duration"),
                              "url": v.get("url") or f"https://www.youtube.com/watch?v={v.get('id')}"})

    channels.sort(key=lambda r: -r["median_views"])
    (OUT / "channels.json").write_text(json.dumps(channels, indent=1))
    (OUT / "collected.json").write_text(json.dumps(collected, indent=1))

    lines = [f"# Niche research: {', '.join(queries)}", "",
             "## Channels (sorted by median views of their latest videos)", "",
             "| Channel | Subs | Median views | Best video | Avg length (min) | URL |",
             "|---|---|---|---|---|---|"]
    for r in channels:
        lines.append(f"| {r['name']} | {fmt(r['subs'])} | {fmt(r['median_views'])} | "
                     f"{fmt(r['max_views'])} | {r['avg_minutes']} | {r['url']} |")
    top = sorted(collected, key=lambda v: -v["views"])[:30]
    lines += ["", "## Top 30 videos by views", "", "| Views | Min | Channel | Title |", "|---|---|---|---|"]
    for v in top:
        mins = round((v["duration"] or 0) / 60, 1)
        lines.append(f"| {fmt(v['views'])} | {mins} | {v['channel']} | [{v['title']}]({v['url']}) |")
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1:] or ["animated horror stories"])
