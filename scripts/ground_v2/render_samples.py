"""Spot-check sheets: per qtype sample K questions and render image + GT boxes + question text (4 panels per PNG).

  python scripts/ground_v2/render_samples.py --data out/train.jsonl --img_root ~/gv2/img --out_dir sheets/ --k 8
Writes sheets/<qtype>_<i>.png and sheets/samples.json (the sampled questions, for the verdict table).
"""
import argparse, collections, json, os, random, textwrap

from PIL import Image, ImageDraw, ImageFont


def font(sz):
    for p in ["/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
              "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"]:
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default(size=sz)


def panel(rec, q, root, w=560):
    p = f"{root}/{rec['img'].split('/', 1)[1]}"
    if not os.path.exists(p):  # fetch the few sampled COCO images on demand
        import urllib.request
        os.makedirs(os.path.dirname(p), exist_ok=True)
        urllib.request.urlretrieve("http://images.cocodataset.org/" + rec["img"].split("/", 1)[1], p)
    img = Image.open(p).convert("RGB")
    s = w / img.width
    img = img.resize((w, round(img.height * s)))
    d = ImageDraw.Draw(img)
    for b in q["boxes"]:
        d.rectangle([v * s for v in b], outline=(0, 255, 0), width=4)
        d.rectangle([v * s for v in b], outline=(0, 0, 0), width=1)
    f = font(20)
    head = f"[{q['qtype']}] {'PRESENT x%d' % q['n_targets'] if q['present'] else 'ABSENT'}"
    lines = [head] + textwrap.wrap(q["text"], 44 if q["lang"] == "en" else 24)
    th = 26 * len(lines) + 8
    out = Image.new("RGB", (w, img.height + th), (255, 255, 255))
    out.paste(img, (0, th))
    d = ImageDraw.Draw(out)
    for i, t in enumerate(lines):
        d.text((6, 4 + 26 * i), t, fill=(200, 0, 0) if i == 0 else (0, 0, 0), font=f)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--img_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--qtypes", default="")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    os.makedirs(a.out_dir, exist_ok=True)
    by = collections.defaultdict(list)
    for l in open(a.data):
        r = json.loads(l)
        for q in r["questions"]:
            by[q["qtype"]].append((r, q))
    want = a.qtypes.split(",") if a.qtypes else sorted(by)
    samples = {}
    for qt in want:
        items = rng.sample(by[qt], min(a.k, len(by[qt])))
        samples[qt] = [dict(image_id=r["image_id"], **q) for r, q in items]
        for i in range(0, len(items), 4):
            ps = [panel(r, q, a.img_root) for r, q in items[i:i + 4]]
            W = 2 * 560 + 10; rows = [ps[j:j + 2] for j in range(0, len(ps), 2)]
            H = sum(max(p.height for p in row) for row in rows) + 10 * len(rows)
            sheet = Image.new("RGB", (W, H), (128, 128, 128)); y = 0
            for row in rows:
                for j, p in enumerate(row):
                    sheet.paste(p, (j * 570, y))
                y += max(p.height for p in row) + 10
            sheet.save(f"{a.out_dir}/{qt}_{i // 4}.png")
    json.dump(samples, open(f"{a.out_dir}/samples.json", "w"), ensure_ascii=False, indent=1)
    print({k: len(v) for k, v in samples.items()})


if __name__ == "__main__":
    main()
