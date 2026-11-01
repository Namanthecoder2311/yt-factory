# yt-factory

Cloud pipeline for two faceless YouTube channels, **Midnight Aisle Tales** (horror) and
**Wigglo Toons** (kids). Everything runs on free tiers: GitHub Actions, Cloudflare Workers AI and
Kaggle GPUs. Nothing renders on the owner's laptop.

## Live status

The badges update on every run. Click one to see its run history.

| Workflow | Live status | What it does |
|---|---|---|
| Build video | [![Build video](https://github.com/Namanthecoder2311/yt-factory/actions/workflows/build.yml/badge.svg)](https://github.com/Namanthecoder2311/yt-factory/actions/workflows/build.yml) | Episode YAML → images → Kaggle hero clips → voices → render → final MP4 + thumbnail |
| Animate on Kaggle | [![Animate on Kaggle](https://github.com/Namanthecoder2311/yt-factory/actions/workflows/animate.yml/badge.svg)](https://github.com/Namanthecoder2311/yt-factory/actions/workflows/animate.yml) | Still image → 2 s motion clip (Wan 2.2) on a free T4 |
| Kids 3D + song test | [![Kids test](https://github.com/Namanthecoder2311/yt-factory/actions/workflows/kids-test.yml/badge.svg)](https://github.com/Namanthecoder2311/yt-factory/actions/workflows/kids-test.yml) | Blender 3D Wigglo render + ACE-Step song on Kaggle |
| Channel branding | [![Branding](https://github.com/Namanthecoder2311/yt-factory/actions/workflows/branding.yml/badge.svg)](https://github.com/Namanthecoder2311/yt-factory/actions/workflows/branding.yml) | Profile picture + banner from a branding YAML |
| Niche research | [![Research](https://github.com/Namanthecoder2311/yt-factory/actions/workflows/research.yml/badge.svg)](https://github.com/Namanthecoder2311/yt-factory/actions/workflows/research.yml) | Top channels + outlier videos in a niche (yt-dlp) |
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
| Real AI motion clips on Kaggle (Wan 2.2, T4) | ✅ Working | 3 test clips, ~7 min per 2 s clip |
| Channel branding generator | ✅ Working | Wigglo Toons + Midnight Aisle Tales live |
| Hero clips inside the full video | 🔄 First run in progress | `pilot-60s.yaml` |
| 3D Wigglo in Blender on Kaggle | 🔄 First run in progress | Kids test |
| Kids songs with vocals (ACE-Step 1.5) | 🔄 First run in progress | Kids test |
| Modal GPU (L40S/A10G/T4) | ⛔ Blocked | Needs a payment method on Modal |
| YouTube upload | ⏸️ Manual | Owner approves every upload |

## How to run

```
gh workflow run build.yml -f episode=episodes/horror/pilot-60s.yaml -f animate=true
gh workflow run kids-test.yml
gh workflow run branding.yml -f config=branding/midnight-aisle.yaml
```

Secrets: `CF_ACCOUNT_ID`, `CF_API_TOKEN`, `KAGGLE_USERNAME`, `KAGGLE_KEY`,
`MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`.

## Licences of what we generate with

FLUX.2 klein 4B (Apache 2.0), Wan 2.2 TI2V-5B (Apache 2.0), Kokoro (Apache 2.0),
Depth Anything V2 Small (Apache 2.0), ACE-Step 1.5 (MIT), Blender (GPL; renders are ours),
Fredoka and Creepster fonts (SIL OFL).
