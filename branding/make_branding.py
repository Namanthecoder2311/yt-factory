"""Channel profile picture (800x800) and banner (2560x1440) from a branding YAML.

    python branding/make_branding.py branding/wigglo.yaml out_dir

The mascot reference is generated first and reused, so DP and banner show the same character.
Text is drawn with Pillow (AI models garble lettering). Font: Fredoka (SIL OFL) if present.
"""
import sys
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from factory import cf_image  # noqa: E402

FONT = Path("fonts/Fredoka.ttf")
FALLBACK = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def font(size, path=FONT):
    path = Path(path)
    if path.exists():
        f = ImageFont.truetype(str(path), size)
        try:
            f.set_variation_by_name("Bold")
        except Exception:
            pass
        return f
    return ImageFont.truetype(FALLBACK, size)


def outlined(draw, xy, text, f, fill, stroke, width, anchor="mm"):
    draw.text(xy, text, font=f, fill=fill, stroke_width=width, stroke_fill=stroke, anchor=anchor)


def main(cfg_path, out):
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    mascot = out / "mascot_ref.png"
    if not mascot.exists():
        cf_image(f"character design sheet, full body, front view, plain white background, "
                 f"{cfg['mascot']}, {cfg['style']}", mascot, size=(768, 768))
    ref = out / "mascot_ref_small.png"
    Image.open(mascot).convert("RGB").resize((496, 496), Image.LANCZOS).save(ref)
    lead = f"Image 0 shows {cfg['mascot'].split(',')[0]}. Keep the character exactly as in image 0."

    dp_raw = out / "dp_raw.png"
    cf_image(f"{lead} {cfg['dp_prompt']}, {cfg['style']}", dp_raw, size=(1024, 1024), refs=[ref])
    Image.open(dp_raw).convert("RGB").resize((800, 800), Image.LANCZOS).save(out / "profile_800.png")

    banner_raw = out / "banner_raw.png"
    cf_image(f"{lead} {cfg['banner_prompt']}, {cfg['style']}", banner_raw, size=(1920, 1080),
             refs=[ref])
    banner = Image.open(banner_raw).convert("RGB").resize((2560, 1440), Image.LANCZOS)
    banner = banner.filter(ImageFilter.UnsharpMask(radius=2, percent=60))
    d = ImageDraw.Draw(banner)
    cx, cy = 1280, 720                                   # centre of the 1546x423 safe area
    color = lambda key, default: tuple(cfg.get(key, default))  # noqa: E731
    name_font = font(190, cfg.get("font", FONT))
    while d.textlength(cfg["name"], font=name_font) > 1500 and name_font.size > 90:
        name_font = font(name_font.size - 10, cfg.get("font", FONT))
    outlined(d, (cx, cy - 40), cfg["name"], name_font, color("name_fill", (255, 255, 255)),
             color("name_stroke", (235, 90, 40)), 16)
    outlined(d, (cx, cy + 120), cfg["tagline"], font(78, cfg.get("tagline_font", FONT)),
             color("tagline_fill", (255, 214, 60)), color("tagline_stroke", (60, 60, 140)), 8)
    banner.save(out / "banner_2560x1440.png")

    preview = banner.copy()                              # shows the safe area for review
    pd = ImageDraw.Draw(preview)
    pd.rectangle([cx - 773, cy - 211, cx + 773, cy + 211], outline=(255, 0, 0), width=6)
    preview.resize((1280, 720)).save(out / "banner_preview_safe_area.png")
    print("branding done:", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
