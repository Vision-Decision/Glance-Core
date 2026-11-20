"""Spot-check sheets for the category-free set: K samples per qtype, image + target box (green) + relation anchor (blue)
+ query; negatives show only the anchor (if any) so the absent claim is judged on the raw photo.

  python scripts/v12/render_check.py --data out/catfree.jsonl --img_root ~/gv2/img --out_dir sheets --k 8
Writes sheets/<qtype>_<i>.png (4 panels each) and sheets/samples.json (for the verdict table).
"""
import argparse, collections, json, os, random, sys

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ground_v2"))
from render_samples import font  # noqa: E402


def panel(rec, q, root, w=560):
    p = f"{root}/{rec['img'].split('/', 1)[1]}"
    if not os.path.exists(p):
        import urllib.request
        os.makedirs(os.path.dirname(p), exist_ok=True)
        urllib.request.urlretrieve("http://images.cocodataset.org/" + rec["img"].split("/", 1)[1], p)
    img = Image.open(p).convert("RGB")
    s = w / img.width
    img = img.resize((w, round(img.height * s)))
    d = ImageDraw.Draw(img)
    if q.get("ref_box"):
        d.rectangle([v * s for v in q["ref_box"]], outline=(0, 120, 255), width=4)
    for b in q.get("boxes") or []:
        d.rectangle([v * s for v in b], outline=(0, 255, 0), width=4)
        d.rectangle([v * s for v in b], outline=(0, 0, 0), width=1)
    f = font(20)
    head = f"[{q['qtype']}] {'PRESENT' if q['present'] else 'ABSENT'}" + (f"  anchor={q['ref_cat']}" if q.get("ref_cat") else "")
    lines = [head, q["text"]]
    out = Image.new("RGB", (w, img.height + 60), (255, 255, 255))
    out.paste(img, (0, 60))
    d = ImageDraw.Draw(out)
    for i, t in enumerate(lines):
        d.text((6, 4 + 27 * i), t, fill=(200, 0, 0) if i == 0 else (0, 0, 0), font=f)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--img_root", required=True)
    ap.add_argument("--out_dir", required=True); ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    os.makedirs(a.out_dir, exist_ok=True)
    by = collections.defaultdict(list)
    for l in open(a.data):
        r = json.loads(l)
        for q in r["questions"]:
            by[q["qtype"]].append((r, q))
    samples = {}
    for qt in sorted(by):
        items = rng.sample(by[qt], min(a.k, len(by[qt])))
        samples[qt] = [dict(image_id=r["image_id"], text=q["text"], present=q["present"]) for r, q in items]
        for i in range(0, len(items), 4):
            ps = [panel(r, q, a.img_root) for r, q in items[i:i + 4]]
            rows = [ps[j:j + 2] for j in range(0, len(ps), 2)]
            H = sum(max(p.height for p in row) for row in rows) + 10 * len(rows)
            sheet = Image.new("RGB", (2 * 570, H), (128, 128, 128)); y = 0
            for row in rows:
                for j, p in enumerate(row):
                    sheet.paste(p, (j * 570, y))
                y += max(p.height for p in row) + 10
            sheet.save(f"{a.out_dir}/{qt}_{i // 4}.png")
    json.dump(samples, open(f"{a.out_dir}/samples.json", "w"), ensure_ascii=False, indent=1)
    print({k: len(v) for k, v in samples.items()})


if __name__ == "__main__":
    main()
