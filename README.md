# yt-factory

Cloud pipeline for two faceless YouTube channels, **Midnight Aisle Tales** (horror) and
**Wigglo Toons** (kids). Everything runs on free tiers: GitHub Actions and Cloudflare Workers AI.
No video tool is connected right now, so no Shorts are being built. Nothing renders on the owner's laptop.

## Live status

The badges update on every run. Click one to see its run history.

| Workflow | Live status | What it does |
|---|---|---|
| Channel branding | [![Branding](https://github.com/Namanthecoder2311/yt-factory/actions/workflows/branding.yml/badge.svg)](https://github.com/Namanthecoder2311/yt-factory/actions/workflows/branding.yml) | Profile picture + banner from a branding YAML |
| Niche research | [![Research](https://github.com/Namanthecoder2311/yt-factory/actions/workflows/research.yml/badge.svg)](https://github.com/Namanthecoder2311/yt-factory/actions/workflows/research.yml) | Top channels + outlier videos in a niche (yt-dlp) |
| Factory schedule | [![Schedule](https://github.com/Namanthecoder2311/yt-factory/actions/workflows/factory-cron.yml/badge.svg)](https://github.com/Namanthecoder2311/yt-factory/actions/workflows/factory-cron.yml) | Sun + Wed: builds the next episode in each channel's queue |
| Upload to YouTube | [![Upload](https://github.com/Namanthecoder2311/yt-factory/actions/workflows/publish.yml/badge.svg)](https://github.com/Namanthecoder2311/yt-factory/actions/workflows/publish.yml) | Private upload + scheduled publish after owner approval; **needs Google Cloud OAuth secrets** |
| Animate on Modal | [![Modal](https://github.com/Namanthecoder2311/yt-factory/actions/workflows/animate-modal.yml/badge.svg)](https://github.com/Namanthecoder2311/yt-factory/actions/workflows/animate-modal.yml) | Faster GPU animation; **blocked until a card is on the Modal account** |

## Component status (last checked 1 Nov 2026)

| Component | Status | Proof |
|---|---|---|
| Niche research (yt-dlp, outlier ranking) | ✅ Working | 15 channels, 868 videos analysed |
| Character-consistent images (Cloudflare FLUX.2 klein + refs) | ✅ Working | Both channels' DP and banner |
| Narration + 6 character voices (Kokoro) | ✅ Working | Episode 001 |
| Sound design (ambience, knocks, whispers, scare hits) | ✅ Working | Episode 001 |
| Parallax animation, fog, rain, scares, word captions | ✅ Working | Episode 001 |
| Multi-story parallel render + assembly + thumbnail | ✅ Working | Episode 001 (7.4 min) |
| Channel branding generator | ✅ Working | Wigglo Toons + Midnight Aisle Tales live |
| Modal GPU (L40S/A10G/T4) | ⛔ Blocked | Needs a payment method on Modal |
| Upload pack (title, description, tags, chapters, kids flag) | 🔨 Built, first real use next build | `upload.json` |
| Episode queue + twice-weekly schedule | 🔨 Built | `episodes/<channel>/queue/` |
| YouTube API uploader | 🔨 Built, waiting for OAuth secrets | `uploader/youtube_upload.py` |
| Public release | ⏸️ Owner approves every video | Hard rule |

## How to run

```
gh workflow run build.yml -f episode=episodes/horror/pilot-60s.yaml -f animate=true
gh workflow run kids-test.yml
gh workflow run branding.yml -f config=branding/midnight-aisle.yaml
```

Secrets: `CF_ACCOUNT_ID`, `CF_API_TOKEN`,
`MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`.

## Licences of what we generate with

FLUX.2 klein 4B (Apache 2.0), Wan 2.2 TI2V-5B (Apache 2.0), Kokoro (Apache 2.0),
Depth Anything V2 Small (Apache 2.0), ACE-Step 1.5 (MIT), Blender (GPL; renders are ours),
Fredoka and Creepster fonts (SIL OFL).
