"""Upload a finished video to YouTube as PRIVATE, optionally scheduled to go public later.

    python uploader/youtube_upload.py out/ep001-walmart --channel horror [--publish-at 2026-11-06T23:00:00Z]

Reads upload.json (written by `factory.py assemble`) for title, description, tags, category and
the Made for Kids flag. Credentials come from the environment:
    YT_CLIENT_ID, YT_CLIENT_SECRET, YT_REFRESH_TOKEN_<CHANNEL>   (e.g. YT_REFRESH_TOKEN_HORROR)

Videos are always uploaded private: nothing becomes public without the owner's approval
(publishAt only schedules a video the owner already approved). Until Google's API audit passes,
uploads from this project stay private-locked anyway.
"""
import argparse
import json
import os
import sys
from pathlib import Path

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

SCOPES = ["https://www.googleapis.com/auth/youtube.upload",
          "https://www.googleapis.com/auth/youtube"]


def client(channel):
    creds = Credentials(None, refresh_token=os.environ[f"YT_REFRESH_TOKEN_{channel.upper()}"],
                        client_id=os.environ["YT_CLIENT_ID"],
                        client_secret=os.environ["YT_CLIENT_SECRET"],
                        token_uri="https://oauth2.googleapis.com/token", scopes=SCOPES)
    return build("youtube", "v3", credentials=creds, cache_discovery=False)


def main(folder, channel, publish_at=None, expect_handle=None):
    folder = Path(folder)
    pack = json.loads((folder / "upload.json").read_text())
    yt = client(channel)

    me = yt.channels().list(part="snippet", mine=True).execute()["items"][0]["snippet"]
    print("uploading to channel:", me["title"], me.get("customUrl"), flush=True)
    if expect_handle and me.get("customUrl", "").lower() != expect_handle.lower():
        sys.exit(f"wrong channel: token belongs to {me.get('customUrl')}, expected {expect_handle}")

    status = {"privacyStatus": "private", "selfDeclaredMadeForKids": pack["madeForKids"]}
    if publish_at:
        status["publishAt"] = publish_at
    body = {"snippet": {"title": pack["title"], "description": pack["description"],
                        "tags": pack["tags"], "categoryId": pack["categoryId"],
                        "defaultLanguage": pack["language"],
                        "defaultAudioLanguage": pack["audioLanguage"]},
            "status": status}
    req = yt.videos().insert(part="snippet,status", body=body,
                             media_body=MediaFileUpload(str(folder / pack["file"]),
                                                        chunksize=8 * 1024 * 1024, resumable=True))
    resp = None
    while resp is None:
        prog, resp = req.next_chunk()
        if prog:
            print(f"uploaded {prog.progress() * 100:.0f}%", flush=True)
    vid = resp["id"]
    print("video id:", vid, flush=True)
    # YouTube ignored selfDeclaredMadeForKids on insert for the first Wigglo Toons Shorts, so the
    # audience is checked and, if needed, set again before the Short goes live
    got = yt.videos().list(part="status", id=vid).execute()["items"][0]["status"]
    if got.get("madeForKids") != pack["madeForKids"]:
        fix = {k: got[k] for k in ("privacyStatus", "embeddable", "license", "publicStatsViewable")}
        fix["selfDeclaredMadeForKids"] = pack["madeForKids"]
        if publish_at:
            fix["publishAt"] = publish_at
        yt.videos().update(part="status", body={"id": vid, "status": fix}).execute()
        got = yt.videos().list(part="status", id=vid).execute()["items"][0]["status"]
        print("made-for-kids set again, now:", got.get("madeForKids"), flush=True)
        if got.get("madeForKids") != pack["madeForKids"]:
            sys.exit(f"audience setting is {got.get('madeForKids')}, expected {pack['madeForKids']}: "
                     f"fix it in YouTube Studio for https://youtu.be/{vid}")
    thumb = folder / pack["thumbnail"]
    if thumb.exists():
        try:  # custom thumbnails need a phone-verified channel
            yt.thumbnails().set(videoId=vid, media_body=MediaFileUpload(str(thumb))).execute()
        except Exception as e:
            print("thumbnail not set:", repr(e)[:300], flush=True)
    result = {"videoId": vid, "url": f"https://youtu.be/{vid}", "privacy": "private",
              "publishAt": publish_at, "channel": me.get("customUrl")}
    (folder / "uploaded.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--channel", required=True, choices=["horror", "kids"])
    ap.add_argument("--publish-at")
    a = ap.parse_args()
    handles = {"horror": "@midnightaisletales", "kids": "@wigglotoonstv"}
    main(a.folder, a.channel, a.publish_at, handles[a.channel])
