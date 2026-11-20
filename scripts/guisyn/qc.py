"""QC viewer: random samples from built shards -> crop sheets (positives) + full-frame sheets (negatives) + index.html.

  python qc.py --stage /data/guisyn/stage --out /data/guisyn/qc --n 100 --source desktop [--app inkscape] [--seed 0]
Positives: each tile = native-resolution crop around the box (box drawn in red) with the expression underneath.
Negatives: each sheet = one downscaled screenshot + its negative expressions (check none is visible).
"""
import argparse, base64, glob, io, json, os, random, tarfile, textwrap
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def font(n):
    try:
        return ImageFont.truetype(FONT, n)
    except Exception:
        return ImageFont.load_default()


def load_samples(stage, source, app, n, seed):
    recs = []
    for f in sorted(glob.glob(f"{stage}/shards/{source}/*.jsonl")):
        for l in open(f):
            r = json.loads(l)
            if app and r["app"] != app:
                continue
            r["_shard"] = f[:-len(".jsonl")]
            recs.append(r)
    rng = random.Random(seed)
    rng.shuffle(recs)
    return recs, rng


def image_of(r, cache={}):
    tf = r["_shard"] + ".images.tar"
    if tf not in cache:
        cache[tf] = tarfile.open(tf)
    return Image.open(io.BytesIO(cache[tf].extractfile(r["image"]).read())).convert("RGB")


def crop_tile(img, box, text, tw=480, th=300):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = max(tw, (x2 - x1) * 1.6), max(th, (y2 - y1) * 1.6)
    s = min(tw / w, th / h)
    cx0, cy0 = max(0, min(img.width - w, cx - w / 2)), max(0, min(img.height - h, cy - h / 2))
    c = img.crop((int(cx0), int(cy0), int(cx0 + w), int(cy0 + h))).resize((int(w * s), int(h * s)), Image.LANCZOS)
    tile = Image.new("RGB", (tw, th + 44), "white")
    tile.paste(c, (0, 0))
    d = ImageDraw.Draw(tile)
    d.rectangle([(x1 - cx0) * s - 1, (y1 - cy0) * s - 1, (x2 - cx0) * s + 1, (y2 - cy0) * s + 1], outline=(255, 0, 0), width=2)
    for k, line in enumerate(textwrap.wrap(text, 62)[:3]):
        d.text((3, th + 2 + 14 * k), line, fill="black", font=font(12))
    return tile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="/data/guisyn/stage")
    ap.add_argument("--out", default="/data/guisyn/qc")
    ap.add_argument("--source", default="desktop")
    ap.add_argument("--app", default="")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--neg_imgs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    tag = a.source + (f"_{a.app}" if a.app else "")
    od = f"{a.out}/{tag}"
    os.makedirs(od, exist_ok=True)
    recs, rng = load_samples(a.stage, a.source, a.app, a.n, a.seed)
    # positives: one random question from each of n random screenshots (spreads samples over states/apps)
    picks = []
    for r in recs:
        pos = [q for q in r["questions"] if q["present"]]
        if pos:
            picks.append((r, rng.choice(pos)))
        if len(picks) >= a.n:
            break
    tiles, manifest = [], []
    for k, (r, q) in enumerate(picks):
        img = image_of(r)
        t = crop_tile(img, q["box"], f"[{k}] {r['app']} {q['kind']}: {q['text']}")
        tiles.append(t)
        manifest.append(dict(k=k, id=r["meta"]["id"], app=r["app"], W=r["W"], H=r["H"], kind=q["kind"], text=q["text"],
                             box=q["box"], role=q["meta"].get("role")))
    per = 12
    sheets = []
    for s in range(0, len(tiles), per):
        grid = tiles[s:s + per]
        cols = 4
        rows = (len(grid) + cols - 1) // cols
        sh = Image.new("RGB", (cols * 484, rows * 348), (200, 200, 200))
        for i, t in enumerate(grid):
            sh.paste(t, ((i % cols) * 484, (i // cols) * 348))
        p = f"{od}/pos_{s // per:02d}.png"
        sh.save(p); sheets.append(p)
    # negatives
    nsheets = []
    for k, r in enumerate([r for r in recs if any(not q["present"] for q in r["questions"])][:a.neg_imgs]):
        img = image_of(r)
        s = 1600 / img.width
        im = img.resize((1600, int(img.height * s)), Image.LANCZOS)
        negs = [q for q in r["questions"] if not q["present"]]
        canvas = Image.new("RGB", (1600, im.height + 16 * (len(negs) + 1) + 8), "white")
        canvas.paste(im, (0, 0))
        d = ImageDraw.Draw(canvas)
        d.text((4, im.height + 2), f"[N{k}] {r['app']} {r['meta']['id']}  {r['W']}x{r['H']}", fill="black", font=font(13))
        for j, q in enumerate(negs):
            d.text((4, im.height + 18 + 16 * j), f"{q['kind']}: {q['text']}", fill=(160, 0, 0), font=font(13))
            manifest.append(dict(k=f"N{k}.{j}", id=r["meta"]["id"], app=r["app"], kind=q["kind"], text=q["text"]))
        p = f"{od}/neg_{k:02d}.png"
        canvas.save(p); nsheets.append(p)
    json.dump(manifest, open(f"{od}/manifest.json", "w"), indent=0)
    with open(f"{od}/index.html", "w") as f:
        f.write(f"<!doctype html><meta charset=utf-8><title>guisyn QC {tag}</title><body style='font-family:sans-serif'>"
                f"<h2>guisyn QC: {tag}</h2><p>{len(picks)} positive samples, {len(nsheets)} negative frames</p>")
        for p in sheets + nsheets:
            f.write(f"<p>{os.path.basename(p)}</p><img src='{os.path.basename(p)}' style='max-width:100%'>")
    print(od, len(sheets), len(nsheets))


if __name__ == "__main__":
    main()
