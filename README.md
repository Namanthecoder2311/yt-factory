# yt-factory

Cloud pipeline that turns an episode YAML into a finished faceless video on GitHub Actions.

- Images: Cloudflare Workers AI, FLUX.1-schnell (Apache 2.0). Falls back to placeholders when secrets are missing.
- Voice: Kokoro TTS (Apache 2.0)
- Ambience, animation, subtitles: FFmpeg
- Thumbnail: Pillow

Run: Actions → "Build video" → episode path, or `gh workflow run build.yml -f episode=episodes/horror/test-short-001.yaml`.
Secrets: `CF_ACCOUNT_ID`, `CF_API_TOKEN` (Workers AI read/edit).
