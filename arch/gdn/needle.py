"""Needle-in-image multi-item recall benchmark (synthetic GUI canvases).

Fixed canvas (1600x1000 px = 50x31 visual tokens at 32 px/token, so the prefix length is the SAME for every N), a 16x32
grid of 100x31 px cells, N in {8..512} cells hold a labelled button with a unique pseudo-word; the rest are blank.
Widget size, font and token count are constant: only the number of distinct items the image prefix must store varies.
Per canvas 8 questions: 6 present ('the "<label>" button'), 2 absent (a label that is not on the canvas).
Theory for recurrent/linear layers (Zoology MQAR, Just-Read-Twice) predicts recall degrading with the number of distinct
items when the query arrives after the context -- exactly Glance's cache order.

  python arch/gdn/needle.py --out /content/needle --n 8,16,32,64,128,256,512 --per 25
writes /content/needle/needle_<N>.jsonl in the GUI bench format (evals/gui_data.py) + PNGs.
"""
import argparse, json, os, random
from PIL import Image, ImageDraw, ImageFont

W, H, CW, CH = 1600, 1000, 100, 31
COLS, ROWS = W // CW, 992 // CH          # 16 x 32 = 512
CONS, VOW = "bcdfghjklmnprstvz", "aeiou"


def word(rng):
    return "".join(rng.choice(CONS) + rng.choice(VOW) for _ in range(3)).capitalize()


def font(sz):
    for p in ["/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default(size=sz)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", default="8,16,32,64,128,256,512")
    ap.add_argument("--per", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    F = font(15)
    os.makedirs(f"{a.out}/img", exist_ok=True)
    for N in map(int, a.n.split(",")):
        rng = random.Random(a.seed * 1000 + N)
        recs = []
        for k in range(a.per):
            vocab = set()
            while len(vocab) < N + 2:
                vocab.add(word(rng))
            vocab = sorted(vocab); rng.shuffle(vocab)
            labels, absent = vocab[:N], vocab[N:N + 2]
            cells = rng.sample(range(COLS * ROWS), N)
            img = Image.new("RGB", (W, H), (245, 246, 248))
            d = ImageDraw.Draw(img)
            boxes = {}
            for lab, c in zip(labels, cells):
                r, q = divmod(c, COLS)
                x1, y1 = q * CW + 4, r * CH + 3
                x2, y2 = x1 + CW - 8, y1 + CH - 6
                d.rounded_rectangle([x1, y1, x2, y2], radius=4, fill=(255, 255, 255), outline=(150, 155, 165))
                tw = d.textlength(lab, font=F)
                d.text(((x1 + x2 - tw) / 2, y1 + 4), lab, fill=(30, 30, 35), font=F)
                boxes[lab] = [x1, y1, x2, y2]
            fn = f"img/n{N}_{k:03d}.png"
            img.save(f"{a.out}/{fn}")
            pres = rng.sample(labels, min(6, N))
            qs = [dict(text=f'the "{l}" button', present=True, box=boxes[l], meta=dict(id=f"n{N}_{k}_{l}", N=N)) for l in pres]
            qs += [dict(text=f'the "{l}" button', present=False, box=None, meta=dict(id=f"n{N}_{k}_{l}_abs", N=N)) for l in absent]
            recs.append(dict(file=fn, image_root=a.out, wh=[W, H], questions=qs))
        with open(f"{a.out}/needle_{N}.jsonl", "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        print("needle", N, len(recs), flush=True)


if __name__ == "__main__":
    main()
