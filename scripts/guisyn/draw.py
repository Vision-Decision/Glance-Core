"""Draw boxes on a screenshot: python draw.py dump.json shot.png out.png  (or import draw())"""
import json, sys
from PIL import Image, ImageDraw, ImageFont


def draw(img, items, maxw=1920):
    """items: [(box xyxy, label, color)]"""
    img = img.convert("RGB")
    s = min(1.0, maxw / img.width)
    if s < 1:
        img = img.resize((int(img.width * s), int(img.height * s)), Image.LANCZOS)
    d = ImageDraw.Draw(img)
    try:
        f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except Exception:
        f = ImageFont.load_default()
    for b, lab, col in items:
        bb = [v * s for v in b]
        d.rectangle(bb, outline=col, width=2)
        if lab:
            d.text((bb[0] + 1, max(0, bb[1] - 12)), lab[:40], fill=col, font=f)
    return img


if __name__ == "__main__":
    j = json.load(open(sys.argv[1]))
    im = Image.open(sys.argv[2])
    items = [(e["box"], f"{e['role'][:6]}:{e['name']}", "red" if not e.get("uncaptioned") else "blue") for e in j["kept"]]
    draw(im, items).save(sys.argv[3])
