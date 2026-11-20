"""Round-trip filter for the Qwen Chinese translations: keep a pair only if the back-translation (backtrans.py) still says the
same thing as the original English.
  1. embedding cosine(original, back-translation) >= --min_cos   (sentence-transformers, CPU)
  2. attribute words kept: every colour, number, position / ordinal / size word of the original appears in the back-
     translation (light normalisation: plural s, British/US spelling, synonyms like "middle"/"center")
  3. head noun kept: the original's COCO category word (or a synonym of the same category) appears in the back-translation,
     or the original has none (e.g. "left one")
Opus translations are passed through unchanged (they are QC-sampled separately).

  python scripts/v13/filter_bt.py --bt zh_bt.jsonl --opus zh_opus.jsonl --out zh_filtered.jsonl
"""
import argparse, collections, json, os, re, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ground_v2"))
from lexicon import EN, names_of  # noqa: E402

COLORS = {"white", "black", "red", "blue", "green", "yellow", "orange", "pink", "purple", "brown", "gray", "grey", "tan", "beige",
          "silver", "gold", "dark", "light", "striped", "plaid"}
NUMS = {"one", "two", "three", "four", "five", "six", "first", "second", "third", "fourth", "half", "both", "single"}
POS = {"left", "right", "top", "bottom", "middle", "center", "front", "back", "behind", "far", "near", "closest", "nearest",
       "furthest", "farthest", "upper", "lower", "corner", "big", "biggest", "small", "smallest", "large", "largest", "tall",
       "tallest", "short", "little"}
EQ = {"grey": "gray", "centre": "center", "middle": "center", "biggest": "largest", "big": "large", "little": "small",
      "nearest": "closest", "farthest": "furthest", "behind": "back", "upper": "top", "lower": "bottom"}
CATW = {c: {w.lower() for w in names_of(c) if re.fullmatch(r"[a-z ]+", w.lower())} for c in EN}


def toks(s):
    return [EQ.get(w, w) for w in (w.rstrip("s") if len(w) > 3 else w for w in re.findall(r"[a-z]+", s.lower()))]


def attrs(s):
    t = set(toks(s))
    return {w for w in t if w in COLORS | NUMS | POS or EQ.get(w) in POS}


def cats(s):
    s = " " + s.lower() + " "
    return {c for c, ws in CATW.items() if any(re.search(rf"\b{re.escape(w)}s?\b", s) for w in ws)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bt", required=True); ap.add_argument("--opus", default=""); ap.add_argument("--out", required=True)
    ap.add_argument("--min_cos", type=float, default=0.75)
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.bt)]
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(a.model, device="cpu")
    e1 = m.encode([r["en"] for r in rows], batch_size=512, normalize_embeddings=True, show_progress_bar=False)
    e2 = m.encode([r["bt"] for r in rows], batch_size=512, normalize_embeddings=True, show_progress_bar=False)
    cos = (e1 * e2).sum(-1)
    st, kept = collections.Counter(), []
    for r, c in zip(rows, cos):
        why = None
        if c < a.min_cos:
            why = "low_cos"
        elif not attrs(r["en"]) <= attrs(r["bt"]):
            why = "attribute_lost"
        else:
            co = cats(r["en"])
            if co and not (co & cats(r["bt"])):
                why = "head_noun_lost"
        st[why or "kept"] += 1
        if not why:
            kept.append(dict(r, cos=round(float(c), 3)))
    opus = [json.loads(l) for l in open(a.opus)] if a.opus and os.path.exists(a.opus) else []
    with open(a.out, "w") as f:
        for r in opus + kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    st["opus_passthrough"] = len(opus)
    print(json.dumps(dict(st)))
    json.dump(dict(st), open(a.out.replace(".jsonl", "_stats.json"), "w"))


if __name__ == "__main__":
    main()
